"""
train_image_cond.py
===================
Treino do LDM condicional com condicionamento por IMAGEM DE REFERÊNCIA.

Substitui o ArcFaceEncoder + IdentityAdapter por um ImageConditionEncoder
(ResNet-18 truncado + projeção treinável) seguindo a filosofia do IP-Adapter.

Condicionamento da U-Net
------------------------
  context = cat(attr_tokens, img_tokens, dim=-1)
           = [B, 512, 40 + num_img_tokens]

  Q = hidden states da U-Net          → shape [B, C, H*W]
  K, V = context projetado            → shape [B, C, 40+N]
  atenção: softmax(QKᵀ/√d) · V

Os atributos CelebA controlam sorriso, óculos, cabelo, etc.
A imagem de referência preserva a identidade (formato do rosto, olhos, nariz…).

CFG dropout independente em ambas as condições durante treino
(cfg_dropout_attr e cfg_dropout_img) para habilitar guidance multicondicional
na inferência.
"""

import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import torch.nn as nn
import torch.optim as optim

import logging
from tqdm import tqdm
import argparse
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid

from board import Board
from utils.utils_celeba import get_data_imagecond, save_images, setup_logging
from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
from models.modules import AttributeEmbedder
from models.encoders import ImageConditionEncoder
from vae.modules import VAE

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)

import contextlib


# =========================================================
# DDP
# =========================================================

def setup_ddp():

    if "RANK" not in os.environ:
        os.environ["RANK"]        = "0"
        os.environ["LOCAL_RANK"]  = "0"
        os.environ["WORLD_SIZE"]  = "1"
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="nccl")

    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank, global_rank


# =========================================================
# EMA
# =========================================================

def update_ema(ema_model, model, decay=0.9999):

    ema_model.eval()

    with torch.no_grad():

        for ema_param, param in zip(
            ema_model.parameters(),
            model.parameters(),
        ):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1 - decay))


# =========================================================
# BUILD CONTEXT
#
# Combina tokens de atributo [B, 512, 40]
#           e tokens de imagem [B, 512, num_img_tokens]
# em context [B, 512, 40 + num_img_tokens].
#
# CFG dropout independente:
#   - zera attr_ctx  com prob cfg_dropout_attr
#   - zera img_tokens com prob cfg_dropout_img
# → 4 regimes (ambos / só attr / só imagem / nenhum)
# =========================================================

def build_context(
    attribute_embedder,
    image_encoder,
    attrs,
    ref_img,
    cfg_dropout_attr: float = 0.1,
    cfg_dropout_img:  float = 0.1,
    training: bool = True,
):
    """
    Parâmetros
    ----------
    attrs   : [B, 40]          atributos binários CelebA
    ref_img : [B, 3, H, W]     imagem de referência em [-1, 1]

    Retorna
    -------
    context : [B, 512, 40 + num_img_tokens]
    """

    attr_ctx   = attribute_embedder(attrs)    # [B, 512, 40]
    img_tokens = image_encoder(ref_img)        # [B, 512, num_img_tokens]

    if training:
        if torch.rand(1).item() < cfg_dropout_attr:
            attr_ctx = torch.zeros_like(attr_ctx)
        if torch.rand(1).item() < cfg_dropout_img:
            img_tokens = torch.zeros_like(img_tokens)

    return torch.cat([attr_ctx, img_tokens], dim=-1)   # [B, 512, 40+N]


# =========================================================
# TRAIN
# =========================================================

def train(args):

    local_rank, global_rank = setup_ddp()

    is_master = (global_rank == 0)

    device = f"cuda:{local_rank}"

    if is_master:
        setup_logging(args.run_name)
        board       = Board(run_name=args.run_name, enabled=True)
        global_step = 0

    # =====================================================
    # DATA
    # Sempre usa imagens brutas (arcface_dir=None forçado).
    # id_data = [B, 3, 256, 256] em [-1, 1]
    # =====================================================

    train_loader, val_loader, _, train_sampler = get_data_imagecond(
        args,
        is_distributed=True,
    )

    # =====================================================
    # CONFIG
    # =====================================================

    latent_dim   = 4
    context_dim  = 512
    num_img_tokens = args.img_tokens   # tokens espaciais da imagem ref (padrão 16)

    # context total = 40 atributos + num_img_tokens imagem
    # CrossAttention aceita comprimento variável → sem mudanças na UNet

    # =====================================================
    # VAE (frozen)
    # =====================================================

    vae = VAE(in_channels=3, latent_dim=latent_dim).to(device)

    vae.load_state_dict(
        torch.load("vae/vae_epoch_62.pt", map_location=device)
    )

    vae.eval()

    for p in vae.parameters():
        p.requires_grad = False

    # =====================================================
    # UNET
    # =====================================================

    model = UNet_cond(
        in_channels=latent_dim,
        out_channels=latent_dim,
        context_dim=context_dim,
    ).to(device)

    # =====================================================
    # ATTRIBUTE EMBEDDER
    # =====================================================

    attribute_embedder = AttributeEmbedder(
        num_attributes=40,
        context_dim=context_dim,
    ).to(device)

    # =====================================================
    # IMAGE CONDITION ENCODER
    #   backbone ResNet-18 até layer3: FROZEN por padrão
    #   projeção (LayerNorm + Linear + GELU + Linear): treinável
    #
    # Fluxo de tensores:
    #   ref_img [B,3,256,256] → backbone → [B,256,16,16]
    #                         → pool(4,4) → [B,256,4,4]
    #                         → flatten  → [B,16,256]
    #                         → proj     → [B,16,512]
    #                         → permute  → [B,512,16]  ← img_tokens
    # =====================================================

    image_encoder = ImageConditionEncoder(
        context_dim=context_dim,
        num_tokens=num_img_tokens,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # =====================================================
    # DIFFUSION
    # =====================================================

    diffusion = Diffusion_conditional(
        img_size=args.image_size // 8,
        device=device,
    )

    # =====================================================
    # EMA
    # =====================================================

    ema_model         = deepcopy(model).eval()
    ema_embedder      = deepcopy(attribute_embedder).eval()
    ema_image_encoder = deepcopy(image_encoder).eval()

    for p in ema_model.parameters():
        p.requires_grad = False

    for p in ema_embedder.parameters():
        p.requires_grad = False

    for p in ema_image_encoder.parameters():
        p.requires_grad = False

    # =====================================================
    # DDP
    # =====================================================

    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    attribute_embedder = DDP(attribute_embedder, device_ids=[local_rank])

    # find_unused_parameters=True porque o backbone frozen não contribui
    # com gradiente mas ainda é parte do módulo DDP.
    image_encoder = DDP(
        image_encoder,
        device_ids=[local_rank],
        find_unused_parameters=True,
    )

    # =====================================================
    # OPTIMIZER
    # Treina: UNet + AttributeEmbedder + proj do ImageEncoder
    # NÃO treina: VAE, backbone frozen do ImageEncoder
    # =====================================================

    trainable_image_params = [
        p for p in image_encoder.parameters() if p.requires_grad
    ]

    optimizer = optim.AdamW(
        list(model.parameters()) +
        list(attribute_embedder.parameters()) +
        trainable_image_params,
        lr=args.lr,
        weight_decay=1e-4,
    )

    # =====================================================
    # LR SCHEDULER
    # =====================================================

    warmup_epochs = 10

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - warmup_epochs),
        eta_min=1e-6,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # =====================================================
    # AMP
    # =====================================================

    scaler = GradScaler()

    accumulation_steps = 4

    start_epoch = 0

    # =====================================================
    # RESUME  (checkpoint do *novo* treino com ImageEncoder)
    # =====================================================

    if args.resume_ckpt is not None and os.path.isfile(args.resume_ckpt):

        if is_master:
            print(f"\nCarregando checkpoint: {args.resume_ckpt}")

        checkpoint = torch.load(args.resume_ckpt, map_location=device)

        model.module.load_state_dict(checkpoint["model_state_dict"])
        attribute_embedder.module.load_state_dict(checkpoint["attribute_embedder_state_dict"])
        image_encoder.module.load_state_dict(checkpoint["image_encoder_state_dict"])

        ema_model.load_state_dict(checkpoint["ema_state_dict"])
        ema_embedder.load_state_dict(checkpoint["ema_embedder_state_dict"])
        ema_image_encoder.load_state_dict(checkpoint["ema_image_encoder_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = (
            args.start_epoch
            if args.start_epoch is not None
            else checkpoint["epoch"] + 1
        )

        if "scheduler_state_dict" in checkpoint and args.start_epoch is None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if is_master:
            if "global_step" in checkpoint:
                global_step = checkpoint["global_step"]
            print(f"Retomando da epoch {start_epoch} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # =====================================================
    # WARM START  (checkpoint antigo com ArcFace/IdentityAdapter)
    # Carrega UNet + AttributeEmbedder.
    # ImageConditionEncoder começa do zero.
    # =====================================================

    elif args.warmstart_ckpt is not None and os.path.isfile(args.warmstart_ckpt):

        if is_master:
            print(f"\nWarm start de: {args.warmstart_ckpt}")

        checkpoint = torch.load(args.warmstart_ckpt, map_location=device)

        model.module.load_state_dict(checkpoint["model_state_dict"])
        attribute_embedder.module.load_state_dict(checkpoint["attribute_embedder_state_dict"])

        ema_model.load_state_dict(
            checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
        )
        ema_embedder.load_state_dict(
            checkpoint.get("ema_embedder_state_dict", checkpoint["attribute_embedder_state_dict"])
        )
        # ImageConditionEncoder não existe no checkpoint antigo → inicia do zero

        if args.start_epoch is not None:
            start_epoch = args.start_epoch
            for _ in range(start_epoch):
                scheduler.step()

        if is_master:
            print(
                f"UNet e AttributeEmbedder carregados. "
                f"ImageConditionEncoder inicializado do zero. "
                f"Epoch {start_epoch} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    # =====================================================
    # DIRS
    # =====================================================

    save_dir    = os.path.join("models",  args.run_name)
    results_dir = os.path.join("results", args.run_name)

    os.makedirs(save_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    fixed_attrs    = None
    fixed_ref_imgs = None

    # =====================================================
    # FIXED SAMPLES + ORIGINAIS  (master, uma vez)
    # =====================================================

    if is_master:
        from PIL import Image as PILImage

        train_base = train_loader.dataset.dataset    # CelebALatentIdentityDataset

        samples_fixed = []
        for _i in range(min(16, len(train_loader.dataset))):
            real_idx       = train_loader.dataset.indices[_i]
            _, _attrs, _ref = train_base[real_idx]

            _fname = train_base.samples[real_idx][0]
            _img   = PILImage.open(
                os.path.join(train_base.image_dir, _fname)
            ).convert("RGB")
            _orig = train_base.transform(_img)

            samples_fixed.append((_attrs, _ref, _orig))

        fixed_attrs    = torch.stack([s[0] for s in samples_fixed]).to(device)
        fixed_ref_imgs = torch.stack([s[1] for s in samples_fixed]).to(device)
        orig_imgs      = torch.stack([s[2] for s in samples_fixed])

        save_images(
            orig_imgs,
            os.path.join(results_dir, "originals.jpg"),
            nrow=4,
        )
        print("Originais salvas em originals.jpg")

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)

        model.train()
        attribute_embedder.train()
        image_encoder.train()

        pbar = (
            tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs - 1}")
            if is_master
            else train_loader
        )

        epoch_losses = []

        optimizer.zero_grad(set_to_none=True)

        for i, (latents, attrs, ref_img) in enumerate(pbar):

            # latents : [B, 4, H/8, W/8]
            # attrs   : [B, 40]
            # ref_img : [B, 3, 256, 256]  em [-1, 1]

            latents = latents.to(device, non_blocking=True)
            attrs   = attrs.to(device,   non_blocking=True)
            ref_img = ref_img.to(device,  non_blocking=True)

            # =============================================
            # TIMESTEPS
            # =============================================

            t = diffusion.sample_timesteps(latents.shape[0]).to(device)

            # =============================================
            # GRADIENT ACCUMULATION CONTEXT
            # =============================================

            is_accumulating = (
                (i + 1) % accumulation_steps != 0
                and (i + 1) != len(train_loader)
            )

            model_ctx = (
                model.no_sync() if is_accumulating else contextlib.nullcontext()
            )
            emb_ctx = (
                attribute_embedder.no_sync() if is_accumulating else contextlib.nullcontext()
            )
            enc_ctx = (
                image_encoder.no_sync() if is_accumulating else contextlib.nullcontext()
            )

            with model_ctx, emb_ctx, enc_ctx:

                with autocast():

                    # =====================================
                    # CONTEXT
                    #   attr_tokens  [B, 512, 40]
                    #   img_tokens   [B, 512, num_img_tokens]
                    #   context      [B, 512, 40+num_img_tokens]
                    # =====================================

                    context = build_context(
                        attribute_embedder,
                        image_encoder,
                        attrs,
                        ref_img,
                        cfg_dropout_attr=args.cfg_dropout_attr,
                        cfg_dropout_img=args.cfg_dropout_img,
                        training=True,
                    )

                    # =====================================
                    # FORWARD
                    # =====================================

                    z_t, noise = diffusion.noise_images(latents, t)

                    predicted_noise = model(z_t, t, context=context)

                    loss = nn.functional.mse_loss(predicted_noise, noise)

                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

            # =============================================
            # OPTIMIZER STEP
            # =============================================

            if not is_accumulating:

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if is_master:

                    update_ema(ema_model,         model.module)
                    update_ema(ema_embedder,       attribute_embedder.module)
                    update_ema(ema_image_encoder,  image_encoder.module)

                    loss_display = loss.item() * accumulation_steps

                    pbar.set_postfix(MSE=loss_display)

                    board.log_scalar("Loss/Batch", loss_display, global_step)

                    epoch_losses.append(loss_display)

                    global_step += 1

        # =================================================
        # SCHEDULER
        # =================================================

        scheduler.step()

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()
        attribute_embedder.eval()
        image_encoder.eval()

        val_losses = []

        with torch.no_grad():

            for latents, attrs, ref_img in val_loader:

                latents = latents.to(device, non_blocking=True)
                attrs   = attrs.to(device,   non_blocking=True)
                ref_img = ref_img.to(device,  non_blocking=True)

                with autocast():

                    context = build_context(
                        attribute_embedder,
                        image_encoder,
                        attrs,
                        ref_img,
                        training=False,
                    )

                    t = diffusion.sample_timesteps(latents.shape[0]).to(device)

                    z_t, noise = diffusion.noise_images(latents, t)

                    predicted_noise = model(z_t, t, context=context)

                    val_loss = nn.functional.mse_loss(predicted_noise, noise)

                val_losses.append(val_loss.item())

        val_loss_tensor = torch.tensor(
            sum(val_losses) / len(val_losses), device=device
        )
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        avg_val_loss = val_loss_tensor.item()

        # =================================================
        # MASTER LOGGING
        # =================================================

        if is_master:

            avg_loss   = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch}"
                f" | Train Loss: {avg_loss:.6f}"
                f" | Val Loss: {avg_val_loss:.6f}"
                f" | LR: {current_lr:.6f}"
            )

            board.log_scalars("Loss/Epoch", {"train": avg_loss, "val": avg_val_loss}, epoch)
            board.log_scalar("Metrics/Learning_Rate", current_lr, epoch)

        # =================================================
        # CHECKPOINT
        # =================================================

        if is_master and (epoch % 10 == 0 or epoch == args.epochs - 1):

            checkpoint = {
                "epoch":                        epoch,
                "global_step":                  global_step,
                "model_state_dict":             model.module.state_dict(),
                "attribute_embedder_state_dict": attribute_embedder.module.state_dict(),
                "image_encoder_state_dict":     image_encoder.module.state_dict(),
                "ema_state_dict":               ema_model.state_dict(),
                "ema_embedder_state_dict":      ema_embedder.state_dict(),
                "ema_image_encoder_state_dict": ema_image_encoder.state_dict(),
                "optimizer_state_dict":         optimizer.state_dict(),
                "scheduler_state_dict":         scheduler.state_dict(),
                "scaler_state_dict":            scaler.state_dict(),
                "val_loss":                     avg_val_loss,
                "num_img_tokens":               num_img_tokens,
            }

            torch.save(checkpoint, os.path.join(save_dir, "ckpt.pt"))

            print("Checkpoint salvo.")

        # =================================================
        # SAMPLE
        # =================================================

        if is_master and (epoch % 10 == 0 or epoch == args.epochs - 1):

            print("Gerando imagens...")

            ema_model.eval()
            ema_embedder.eval()
            ema_image_encoder.eval()

            with torch.no_grad():

                # context para os 16 exemplos fixos (sem dropout)
                context_test = build_context(
                    ema_embedder,
                    ema_image_encoder,
                    fixed_attrs.to(device),
                    fixed_ref_imgs.to(device),
                    training=False,
                )

                sampled_latents = diffusion.sample(
                    ema_model,
                    n=16,
                    context=context_test,
                    channels=latent_dim,
                )

                sampled_latents = sampled_latents / 0.18215

                sampled_images = vae.decode(sampled_latents)

            save_images(
                sampled_images,
                os.path.join(results_dir, f"{epoch}.jpg"),
                nrow=4,
            )

            grid = make_grid(
                sampled_images,
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )

            board.log_image("Samples/Generated", grid, epoch)

    # =====================================================
    # FINALIZE
    # =====================================================

    if is_master:
        board.close()

    dist.destroy_process_group()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name",    type=str,   default="LDM_ImageCond")
    parser.add_argument("--epochs",      type=int,   default=2000)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--image_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=3e-4)

    parser.add_argument(
        "--img_tokens",
        type=int,
        default=16,
        help="Número de tokens extraídos da imagem de referência (deve ser quadrado perfeito: 4, 9, 16, 25).",
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Congela o backbone ResNet-18 (treina só a projeção). Recomendado.",
    )

    parser.add_argument(
        "--no_freeze_backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Treina o backbone end-to-end (mais lento, pode ajudar em datasets pequenos).",
    )

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Resume treino de condicionamento por imagem (ckpt gerado por este script).",
    )

    parser.add_argument(
        "--warmstart_ckpt",
        type=str,
        default=None,
        help="Warm start de checkpoint antigo (ArcFace/IdentityAdapter). "
             "Carrega UNet + AttributeEmbedder; ImageConditionEncoder inicia do zero.",
    )

    parser.add_argument(
        "--start_epoch",
        type=int,
        default=None,
        help="Força início na epoch N, posicionando o scheduler corretamente.",
    )

    parser.add_argument(
        "--cfg_dropout_attr",
        type=float,
        default=0.1,
        help="Prob de zerar tokens de atributo (CFG dropout).",
    )

    parser.add_argument(
        "--cfg_dropout_img",
        type=float,
        default=0.1,
        help="Prob de zerar tokens de imagem de referência (CFG dropout).",
    )

    args = parser.parse_args()

    train(args)
