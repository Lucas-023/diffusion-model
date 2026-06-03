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
from utils.utils_celeba import get_data_identity, save_images, setup_logging
from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
from models.modules import AttributeEmbedder, IdentityAdapter
from models.encoders import ArcFaceEncoder
from vae.modules import VAE

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR
)

import contextlib


# =========================================================
# DDP
# =========================================================

def setup_ddp():

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
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
            model.parameters()
        ):

            ema_param.data.mul_(decay).add_(
                param.data,
                alpha=(1 - decay)
            )


# =========================================================
# BUILD CONTEXT
# Combina tokens de atributo + tokens de identidade
# =========================================================

def build_context(
    attribute_embedder,
    identity_adapter,
    attrs,
    identity_emb,
    cfg_dropout_attr=0.05,
    cfg_dropout_id=0.05,
    training=True
):
    """
    Retorna context [B, context_dim, 40 + num_id_tokens].

    Durante treino aplica dropout independente nos dois componentes
    para suportar Classifier-Free Guidance multicondicional:
      - só atributos
      - só identidade
      - ambos
      - nenhum (totalmente incondicional)
    """

    attr_ctx = attribute_embedder(attrs)          # [B, 512, 40]
    id_tokens = identity_adapter(identity_emb)    # [B, 512, num_tokens]

    if training:

        if torch.rand(1).item() < cfg_dropout_attr:
            attr_ctx = torch.zeros_like(attr_ctx)

        if torch.rand(1).item() < cfg_dropout_id:
            id_tokens = torch.zeros_like(id_tokens)

    return torch.cat([attr_ctx, id_tokens], dim=-1)   # [B, 512, 44]


# =========================================================
# TRAIN
# =========================================================

def train(args):

    local_rank, global_rank = setup_ddp()

    is_master = (global_rank == 0)

    device = f"cuda:{local_rank}"

    if is_master:

        setup_logging(args.run_name)

        board = Board(run_name=args.run_name, enabled=True)

        global_step = 0

    # =====================================================
    # DATA
    # =====================================================

    train_loader, val_loader, _, train_sampler = get_data_identity(
        args,
        is_distributed=True
    )

    # =====================================================
    # CONFIG
    # =====================================================

    latent_dim = 4
    context_dim = 512
    identity_dim = 512
    num_id_tokens = 4

    # context total = 40 atributos + 4 tokens de identidade
    # UNet CrossAttention aceita seq length variável → sem mudanças na UNet

    # =====================================================
    # VAE (frozen)
    # =====================================================

    vae = VAE(
        in_channels=3,
        latent_dim=latent_dim
    ).to(device)

    vae.load_state_dict(
        torch.load(
            "vae/vae_epoch_62.pt",
            map_location=device
        )
    )

    vae.eval()

    for param in vae.parameters():
        param.requires_grad = False

    # =====================================================
    # IDENTITY ENCODER — ArcFace (fully frozen)
    # =====================================================

    identity_encoder = ArcFaceEncoder().to(device)

    identity_encoder.eval()

    # =====================================================
    # UNET
    # =====================================================

    model = UNet_cond(
        in_channels=latent_dim,
        out_channels=latent_dim,
        context_dim=context_dim
    ).to(device)

    # =====================================================
    # ATTRIBUTE EMBEDDER
    # =====================================================

    attribute_embedder = AttributeEmbedder(
        num_attributes=40,
        context_dim=context_dim
    ).to(device)

    # =====================================================
    # IDENTITY ADAPTER
    # =====================================================

    identity_adapter = IdentityAdapter(
        identity_dim=identity_dim,
        context_dim=context_dim,
        num_tokens=num_id_tokens
    ).to(device)

    # =====================================================
    # DIFFUSION
    # =====================================================

    diffusion = Diffusion_conditional(
        img_size=args.image_size // 8,
        device=device
    )

    # =====================================================
    # EMA
    # =====================================================

    ema_model = deepcopy(model).eval()
    ema_embedder = deepcopy(attribute_embedder).eval()
    ema_id_adapter = deepcopy(identity_adapter).eval()

    for p in ema_model.parameters():
        p.requires_grad = False

    for p in ema_embedder.parameters():
        p.requires_grad = False

    for p in ema_id_adapter.parameters():
        p.requires_grad = False

    # =====================================================
    # DDP
    # =====================================================

    model = DDP(
        model,
        device_ids=[local_rank],
        gradient_as_bucket_view=True
    )

    attribute_embedder = DDP(
        attribute_embedder,
        device_ids=[local_rank]
    )

    identity_adapter = DDP(
        identity_adapter,
        device_ids=[local_rank]
    )

    # =====================================================
    # OPTIMIZER
    # Treina: UNet + AttributeEmbedder + IdentityAdapter
    # NÃO treina: VAE, ArcFaceEncoder (ONNX frozen)
    # =====================================================

    optimizer = optim.AdamW(
        list(model.parameters()) +
        list(attribute_embedder.parameters()) +
        list(identity_adapter.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # =====================================================
    # LR SCHEDULER
    # =====================================================

    warmup_epochs = 10

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - warmup_epochs),
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # =====================================================
    # AMP
    # =====================================================

    scaler = GradScaler()

    accumulation_steps = 4

    start_epoch = 0

    # =====================================================
    # RESUME
    # =====================================================

    if (
        args.resume_ckpt is not None
        and os.path.isfile(args.resume_ckpt)
    ):

        if is_master:
            print(f"\nCarregando checkpoint: {args.resume_ckpt}")

        checkpoint = torch.load(
            args.resume_ckpt,
            map_location=device
        )

        model.module.load_state_dict(
            checkpoint["model_state_dict"]
        )

        attribute_embedder.module.load_state_dict(
            checkpoint["attribute_embedder_state_dict"]
        )

        identity_adapter.module.load_state_dict(
            checkpoint["identity_adapter_state_dict"]
        )

        ema_model.load_state_dict(
            checkpoint["ema_state_dict"]
        )

        ema_embedder.load_state_dict(
            checkpoint["ema_embedder_state_dict"]
        )

        ema_id_adapter.load_state_dict(
            checkpoint["ema_id_adapter_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        if is_master:
            print(f"Retomando da epoca {start_epoch}")

    # =====================================================
    # WARM START (checkpoint de atributos sem identidade)
    # Carrega UNet + AttributeEmbedder; IdentityAdapter
    # começa do zero. Optimizer e epoch reiniciam.
    # =====================================================

    elif (
        args.warmstart_ckpt is not None
        and os.path.isfile(args.warmstart_ckpt)
    ):

        if is_master:
            print(f"\nWarm start de: {args.warmstart_ckpt}")

        checkpoint = torch.load(
            args.warmstart_ckpt,
            map_location=device
        )

        model.module.load_state_dict(
            checkpoint["model_state_dict"]
        )

        attribute_embedder.module.load_state_dict(
            checkpoint["attribute_embedder_state_dict"]
        )

        ema_model.load_state_dict(
            checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
        )

        ema_embedder.load_state_dict(
            checkpoint.get("ema_embedder_state_dict", checkpoint["attribute_embedder_state_dict"])
        )

        if is_master:
            print("UNet e AttributeEmbedder carregados. IdentityAdapter inicializado do zero.")

    # =====================================================
    # DIRS
    # =====================================================

    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    fixed_attrs = None
    fixed_imgs = None

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)

        model.train()
        attribute_embedder.train()
        identity_adapter.train()

        pbar = tqdm(train_loader) if is_master else train_loader

        epoch_losses = []

        optimizer.zero_grad(set_to_none=True)

        for i, (latents, attrs, images) in enumerate(pbar):

            latents = latents.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)

            if is_master and fixed_attrs is None:
                fixed_attrs = attrs[:16].clone()
                fixed_imgs = images[:16].clone()

            # =============================================
            # TIMESTEPS
            # =============================================

            t = diffusion.sample_timesteps(
                latents.shape[0]
            ).to(device)

            # =============================================
            # ACCUMULATION CONTEXT
            # =============================================

            is_accumulating = (
                (i + 1) % accumulation_steps != 0
                and (i + 1) != len(train_loader)
            )

            model_ctx = (
                model.no_sync()
                if is_accumulating
                else contextlib.nullcontext()
            )

            emb_ctx = (
                attribute_embedder.no_sync()
                if is_accumulating
                else contextlib.nullcontext()
            )

            id_ctx = (
                identity_adapter.no_sync()
                if is_accumulating
                else contextlib.nullcontext()
            )

            with model_ctx, emb_ctx, id_ctx:

                with autocast():

                    # =====================================
                    # IDENTITY EMBEDDING (sem gradiente)
                    # =====================================

                    with torch.no_grad():
                        identity_emb = identity_encoder(images)

                    # =====================================
                    # COMBINED CONTEXT
                    # =====================================

                    context = build_context(
                        attribute_embedder,
                        identity_adapter,
                        attrs,
                        identity_emb,
                        cfg_dropout_attr=args.cfg_dropout_attr,
                        cfg_dropout_id=args.cfg_dropout_id,
                        training=True
                    )

                    # =====================================
                    # NOISE
                    # =====================================

                    z_t, noise = diffusion.noise_images(latents, t)

                    predicted_noise = model(
                        z_t,
                        t,
                        context=context
                    )

                    loss = nn.functional.mse_loss(
                        predicted_noise,
                        noise
                    )

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

                    update_ema(ema_model, model.module)
                    update_ema(ema_embedder, attribute_embedder.module)
                    update_ema(ema_id_adapter, identity_adapter.module)

                    loss_display = loss.item() * accumulation_steps

                    pbar.set_postfix(MSE=loss_display)

                    board.log_scalar(
                        "Loss/Batch",
                        loss_display,
                        global_step
                    )

                    epoch_losses.append(loss_display)

                    global_step += 1

        # =================================================
        # SCHEDULER STEP
        # =================================================

        scheduler.step()

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()
        attribute_embedder.eval()
        identity_adapter.eval()

        val_losses = []

        with torch.no_grad():

            for latents, attrs, images in val_loader:

                latents = latents.to(device, non_blocking=True)
                attrs = attrs.to(device, non_blocking=True)
                images = images.to(device, non_blocking=True)

                with autocast():

                    identity_emb = identity_encoder(images)

                    context = build_context(
                        attribute_embedder,
                        identity_adapter,
                        attrs,
                        identity_emb,
                        training=False
                    )

                    t = diffusion.sample_timesteps(
                        latents.shape[0]
                    ).to(device)

                    z_t, noise = diffusion.noise_images(latents, t)

                    predicted_noise = model(
                        z_t,
                        t,
                        context=context
                    )

                    val_loss = nn.functional.mse_loss(
                        predicted_noise,
                        noise
                    )

                val_losses.append(val_loss.item())

        avg_val_loss_local = sum(val_losses) / len(val_losses)

        val_loss_tensor = torch.tensor(
            avg_val_loss_local,
            device=device
        )

        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)

        avg_val_loss = val_loss_tensor.item()

        # =================================================
        # MASTER LOGGING
        # =================================================

        if is_master:

            avg_loss = (
                sum(epoch_losses) / len(epoch_losses)
                if epoch_losses else 0.0
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch}"
                f" | Train Loss: {avg_loss:.6f}"
                f" | Val Loss: {avg_val_loss:.6f}"
                f" | LR: {current_lr:.6f}"
            )

            board.log_scalar("Metrics/Loss_Epoch", avg_loss, epoch)
            board.log_scalar("Metrics/Val_Loss", avg_val_loss, epoch)
            board.log_scalar("Metrics/Learning_Rate", current_lr, epoch)

        # =================================================
        # CHECKPOINT
        # =================================================

        if is_master and (
            epoch % 10 == 0
            or epoch == args.epochs - 1
        ):

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "attribute_embedder_state_dict": attribute_embedder.module.state_dict(),
                "identity_adapter_state_dict": identity_adapter.module.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "ema_embedder_state_dict": ema_embedder.state_dict(),
                "ema_id_adapter_state_dict": ema_id_adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }

            torch.save(
                checkpoint,
                os.path.join(save_dir, "ckpt.pt")
            )

            print("Checkpoint salvo.")

        # =================================================
        # SAMPLE
        # =================================================

        if is_master and (
            epoch % 10 == 0
            or epoch == args.epochs - 1
        ):

            print("Gerando imagens...")

            ema_model.eval()
            ema_embedder.eval()
            ema_id_adapter.eval()

            with torch.no_grad():

                identity_emb = identity_encoder(
                    fixed_imgs.to(device)
                )

                context_test = build_context(
                    ema_embedder,
                    ema_id_adapter,
                    fixed_attrs.to(device),
                    identity_emb,
                    training=False
                )

                sampled_latents = diffusion.sample(
                    ema_model,
                    n=16,
                    context=context_test,
                    channels=latent_dim
                )

                sampled_latents = sampled_latents / 0.18215

                sampled_images = vae.decode(sampled_latents)

            save_images(
                sampled_images,
                os.path.join(results_dir, f"{epoch}.jpg"),
                nrow=4
            )

            grid = make_grid(
                sampled_images,
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
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

    parser.add_argument(
        "--run_name",
        type=str,
        default="LDM_Identity_Conditional"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Retoma treino de identidade completo (UNet + AttributeEmbedder + IdentityAdapter)."
    )

    parser.add_argument(
        "--warmstart_ckpt",
        type=str,
        default=None,
        help="Inicia de checkpoint de atributos sem identidade. Carrega UNet + AttributeEmbedder; IdentityAdapter começa do zero."
    )

    parser.add_argument(
        "--cfg_dropout_attr",
        type=float,
        default=0.05,
        help="Probabilidade de zerar contexto de atributos (CFG dropout)"
    )

    parser.add_argument(
        "--cfg_dropout_id",
        type=float,
        default=0.05,
        help="Probabilidade de zerar tokens de identidade (CFG dropout)"
    )

    args = parser.parse_args()

    train(args)
