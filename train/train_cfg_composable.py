"""
train_cfg_composable.py
========================
Treino com Composable CFG (Liu et al., ECCV 2022) — identidade e atributos
como dois condicionantes INDEPENDENTES.

Diferenças vs train_mixed_guidance_identity_only.py:
  • Sem Classifier Guidance e sem NoisyLatentAttrClassifier — atributos
    entram via cross-attention com seu próprio dropout de CFG.
  • Dropout INDEPENDENTE para id e attr (10% só-id, 10% só-attr, 10% ambos),
    cobrindo os 4 combos da fórmula composable na inferência:
        eps = eps(∅,∅)
            + s_id   · [eps(id,∅)    − eps(∅,∅)]
            + s_attr · [eps(id,attr) − eps(id,∅)]
  • Encoder de identidade selecionável: `clip_arcface` (default) ou
    `arcface_only`. Com CFG composable, `s_attr` consegue sobrepor a
    expressão capturada pelo CLIP, então o motivo histórico de jogar CLIP
    fora deixa de existir.

Contexto final passado à UNet:
    context = concat([id_tokens, attr_tokens], dim=tokens)
             = [B, 512, T_id + 40]
Sem mudar a UNet — o cross-attention atende a todos os tokens.
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
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import argparse
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid

from board import Board
from utils.utils_celeba import get_data_imagecond, save_images, setup_logging
from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
from models.encoders import ImageConditionEncoder, ArcFaceOnlyEncoder
from models.modules import AttributeEmbedder
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
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=(1 - decay))


# =========================================================
# CFG DROPOUT INDEPENDENTE
# =========================================================
#
# Sorteia, por amostra do batch, qual condicionante zerar.
# Probabilidades (defaults Liu et al. 2022):
#   p_id_only_drop   = 0.1  → zera id,   mantém attr
#   p_attr_only_drop = 0.1  → mantém id, zera attr
#   p_both_drop      = 0.1  → zera os dois
#   resto (0.7)              → mantém os dois
#
# Crucial: o sorteio é POR AMOSTRA, não por batch — assim cada batch vê
# todos os combos e o modelo aprende as 4 distribuições marginais.
# =========================================================

def cfg_masks(batch_size, p_id_only, p_attr_only, p_both, device):

    u = torch.rand(batch_size, device=device)

    drop_both     = u < p_both
    drop_id_only  = (u >= p_both)            & (u < p_both + p_id_only)
    drop_attr_only = (u >= p_both + p_id_only) & (u < p_both + p_id_only + p_attr_only)

    keep_id   = ~(drop_both | drop_id_only)
    keep_attr = ~(drop_both | drop_attr_only)

    # shape [B, 1, 1] para broadcast em [B, C, T]
    return keep_id.float().view(-1, 1, 1), keep_attr.float().view(-1, 1, 1)


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    image_encoder,
    attribute_embedder,
    attrs,
    ref_img=None,
    id_emb=None,
    clip_tokens_raw=None,
    encoder_type="clip_arcface",
    keep_id_mask=None,
    keep_attr_mask=None,
):
    """
    Monta context = concat([id_tokens, attr_tokens], dim=tokens).

    keep_id_mask, keep_attr_mask : [B, 1, 1] em {0, 1}
        Se 0, os tokens correspondentes são zerados (modo "uncond" daquele ramo).
        Se None, mantém tudo.
    """

    if encoder_type == "arcface_only":
        id_tokens = image_encoder(ref_img=ref_img, id_emb=id_emb)
    else:
        id_tokens = image_encoder(
            ref_img=ref_img,
            id_emb=id_emb,
            clip_tokens_raw=clip_tokens_raw,
        )
    # id_tokens: [B, C, T_id]

    attr_tokens = attribute_embedder(attrs)
    # attr_tokens: [B, C, 40]

    if keep_id_mask is not None:
        id_tokens = id_tokens * keep_id_mask
    if keep_attr_mask is not None:
        attr_tokens = attr_tokens * keep_attr_mask

    context = torch.cat([id_tokens, attr_tokens], dim=2)
    return context


# =========================================================
# COMPOSABLE CFG SAMPLING
# =========================================================
#
# eps = eps(∅,∅)
#     + s_id   * [eps(id,∅)    − eps(∅,∅)]
#     + s_attr * [eps(id,attr) − eps(id,∅)]
#
# A ordem importa: o termo de attr usa (id,∅) como baseline, não (∅,∅).
# Isso significa "dada a identidade, o quanto o atributo desloca eps".
# =========================================================

@torch.no_grad()
def _sample_composable_cfg(
    unet,
    diffusion,
    id_tokens_full,        # [N, C, T_id]
    attr_tokens_full,      # [N, C, 40]
    n,
    channels,
    device,
    s_id=3.0,
    s_attr=5.0,
):
    unet.eval()
    img_size = diffusion.img_size
    z_t      = torch.randn(n, channels, img_size, img_size, device=device)

    zeros_id   = torch.zeros_like(id_tokens_full)
    zeros_attr = torch.zeros_like(attr_tokens_full)

    ctx_uu = torch.cat([zeros_id,       zeros_attr],       dim=2)  # ∅, ∅
    ctx_iu = torch.cat([id_tokens_full, zeros_attr],       dim=2)  # id, ∅
    ctx_ia = torch.cat([id_tokens_full, attr_tokens_full], dim=2)  # id, attr

    for i in reversed(range(1, diffusion.noise_steps)):

        t_vec = torch.full((n,), i, device=device, dtype=torch.long)

        alpha_hat_t = diffusion.alpha_hat[t_vec][:, None, None, None]
        alpha_t     = diffusion.alpha[t_vec][:, None, None, None]
        beta_t      = diffusion.beta[t_vec][:, None, None, None]

        eps_uu = unet(z_t, t_vec, context=ctx_uu)
        eps_iu = unet(z_t, t_vec, context=ctx_iu)
        eps_ia = unet(z_t, t_vec, context=ctx_ia)

        eps = eps_uu + s_id * (eps_iu - eps_uu) + s_attr * (eps_ia - eps_iu)

        noise = torch.randn_like(z_t) if i > 1 else torch.zeros_like(z_t)
        z_t = (
            (1.0 / torch.sqrt(alpha_t))
            * (z_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t) * eps)
        ) + torch.sqrt(beta_t) * noise

    return z_t


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
    # =====================================================

    train_loader, val_loader, _, train_sampler, use_cache = get_data_imagecond(
        args,
        is_distributed=True,
    )

    if is_master:
        print(
            f"[Encoder] modo {'CACHE (rápido)' if use_cache else 'LIVE (lento)'}"
        )

    # =====================================================
    # CONFIG
    # =====================================================

    latent_dim     = 4
    context_dim    = 512
    num_img_tokens = args.img_tokens
    num_attrs      = 40

    # =====================================================
    # VAE (frozen)
    # =====================================================

    vae = VAE(in_channels=3, latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("vae/vae_epoch_62.pt", map_location=device))
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
    # IDENTITY ENCODER
    # =====================================================

    if args.encoder == "clip_arcface":
        image_encoder = ImageConditionEncoder(
            context_dim=context_dim,
            num_tokens=num_img_tokens,
            freeze_backbone=True,
        ).to(device)
        # tokens reais: 2 * num_img_tokens (clip + arcface concatenados)
    elif args.encoder == "arcface_only":
        image_encoder = ArcFaceOnlyEncoder(
            context_dim=context_dim,
            num_tokens=num_img_tokens,
        ).to(device)
    else:
        raise ValueError(f"--encoder inválido: {args.encoder}")

    # =====================================================
    # ATTRIBUTE EMBEDDER
    # =====================================================

    attribute_embedder = AttributeEmbedder(
        num_attributes=num_attrs,
        context_dim=context_dim,
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

    ema_model              = deepcopy(model).eval()
    ema_image_encoder      = deepcopy(image_encoder).eval()
    ema_attribute_embedder = deepcopy(attribute_embedder).eval()

    for p in ema_model.parameters():
        p.requires_grad = False
    for p in ema_image_encoder.parameters():
        p.requires_grad = False
    for p in ema_attribute_embedder.parameters():
        p.requires_grad = False

    # =====================================================
    # DDP
    # =====================================================

    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    image_encoder = DDP(
        image_encoder,
        device_ids=[local_rank],
        find_unused_parameters=False,
    )

    attribute_embedder = DDP(attribute_embedder, device_ids=[local_rank])

    # =====================================================
    # OPTIMIZER  (apenas um — sem classifier)
    # =====================================================

    optimizer = optim.AdamW(
        list(model.parameters())
        + list(image_encoder.parameters())
        + list(attribute_embedder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # =====================================================
    # LR SCHEDULER
    # =====================================================

    warmup_epochs = 10

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=(args.epochs - warmup_epochs), eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )

    # =====================================================
    # AMP
    # =====================================================

    scaler = GradScaler()
    accumulation_steps = 4

    start_epoch   = 0
    best_val_loss = float("inf")

    # =====================================================
    # RESUME
    # =====================================================

    if args.resume_ckpt is not None and os.path.isfile(args.resume_ckpt):

        if is_master:
            print(f"\nCarregando checkpoint: {args.resume_ckpt}")

        ckpt = torch.load(args.resume_ckpt, map_location=device)

        model.module.load_state_dict(ckpt["model_state_dict"])
        image_encoder.module.load_state_dict(ckpt["image_encoder_state_dict"])
        attribute_embedder.module.load_state_dict(ckpt["attribute_embedder_state_dict"])

        ema_model.load_state_dict(ckpt["ema_state_dict"])
        ema_image_encoder.load_state_dict(ckpt["ema_image_encoder_state_dict"])
        ema_attribute_embedder.load_state_dict(ckpt["ema_attribute_embedder_state_dict"])

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = (
            args.start_epoch
            if args.start_epoch is not None
            else ckpt["epoch"] + 1
        )

        if "scheduler_state_dict" in ckpt and args.start_epoch is None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()

        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        if is_master:
            if "global_step" in ckpt:
                global_step = ckpt["global_step"]
            if "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]
            print(f"Retomando epoch {start_epoch}")

    # =====================================================
    # WARM START — carrega só a UNet de outro checkpoint
    # =====================================================

    elif args.warmstart_ckpt is not None and os.path.isfile(args.warmstart_ckpt):

        if is_master:
            print(f"\nWarm start de: {args.warmstart_ckpt}")

        ckpt = torch.load(args.warmstart_ckpt, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(
            ckpt.get("ema_state_dict", ckpt["model_state_dict"])
        )

        if args.start_epoch is not None:
            start_epoch = args.start_epoch
            for _ in range(start_epoch):
                scheduler.step()

        if is_master:
            print("UNet carregada. Encoder e AttributeEmbedder do zero.")

    # =====================================================
    # DIRS
    # =====================================================

    save_dir    = os.path.join("models",  args.run_name)
    results_dir = os.path.join("results", args.run_name)

    os.makedirs(save_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    fixed_ref_imgs = None

    # =====================================================
    # FIXED SAMPLES
    # =====================================================

    if is_master:
        from PIL import Image as PILImage

        val_base = val_loader.dataset.dataset

        samples_fixed = []
        for _i in range(min(16, len(val_loader.dataset))):
            real_idx = val_loader.dataset.indices[_i]
            sample   = val_base[real_idx]

            if use_cache:
                _, _attrs, _ref, _idemb, _clip = sample
            else:
                _, _attrs, _ref = sample
                _idemb = None
                _clip  = None

            _fname = val_base.samples[real_idx][0]
            _img   = PILImage.open(
                os.path.join(val_base.image_dir, _fname)
            ).convert("RGB")
            _orig = val_base.transform(_img)

            samples_fixed.append((_ref, _orig, _attrs, _idemb, _clip))

        fixed_ref_imgs = torch.stack([s[0] for s in samples_fixed]).to(device)
        fixed_attrs    = torch.stack([s[2] for s in samples_fixed]).to(device)
        orig_imgs      = torch.stack([s[1] for s in samples_fixed])

        fixed_id_emb = (
            torch.stack([s[3] for s in samples_fixed]).to(device)
            if use_cache else None
        )

        fixed_clip = None
        if use_cache and args.encoder == "clip_arcface" and samples_fixed[0][4] is not None:
            fixed_clip = torch.stack([s[4] for s in samples_fixed]).to(device)

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
        image_encoder.train()
        attribute_embedder.train()

        pbar = (
            tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs - 1}")
            if is_master
            else train_loader
        )

        epoch_losses = []

        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):

            if use_cache:
                latents, attrs, ref_img, id_emb, clip_tokens = batch
                id_emb      = id_emb.to(device, non_blocking=True)
                clip_tokens = clip_tokens.to(device, non_blocking=True) if args.encoder == "clip_arcface" else None
            else:
                latents, attrs, ref_img = batch
                id_emb      = None
                clip_tokens = None

            latents = latents.to(device, non_blocking=True)
            attrs   = attrs.to(device,   non_blocking=True)
            ref_img = ref_img.to(device,  non_blocking=True)

            B = latents.shape[0]

            t = diffusion.sample_timesteps(B).to(device)

            with torch.no_grad():
                z_t, noise = diffusion.noise_images(latents, t)

            # ---- máscaras de dropout independentes por amostra ----
            keep_id, keep_attr = cfg_masks(
                B,
                p_id_only=args.cfg_dropout_id_only,
                p_attr_only=args.cfg_dropout_attr_only,
                p_both=args.cfg_dropout_both,
                device=device,
            )

            is_accumulating = (
                (i + 1) % accumulation_steps != 0
                and (i + 1) != len(train_loader)
            )

            ctx_managers = [
                model.no_sync() if is_accumulating else contextlib.nullcontext(),
                image_encoder.no_sync() if is_accumulating else contextlib.nullcontext(),
                attribute_embedder.no_sync() if is_accumulating else contextlib.nullcontext(),
            ]

            with ctx_managers[0], ctx_managers[1], ctx_managers[2]:
                with autocast():
                    context = build_context(
                        image_encoder=image_encoder,
                        attribute_embedder=attribute_embedder,
                        attrs=attrs,
                        ref_img=ref_img,
                        id_emb=id_emb,
                        clip_tokens_raw=clip_tokens,
                        encoder_type=args.encoder,
                        keep_id_mask=keep_id,
                        keep_attr_mask=keep_attr,
                    )

                    predicted_noise = model(z_t, t, context=context)
                    loss = F.mse_loss(predicted_noise, noise)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

            if not is_accumulating:

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if is_master:

                    update_ema(ema_model,              model.module)
                    update_ema(ema_image_encoder,      image_encoder.module)
                    update_ema(ema_attribute_embedder, attribute_embedder.module)

                    loss_display = loss.item() * accumulation_steps
                    pbar.set_postfix(MSE=loss_display)
                    board.log_scalar("Loss/Batch", loss_display, global_step)
                    epoch_losses.append(loss_display)
                    global_step += 1

        scheduler.step()

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()
        image_encoder.eval()
        attribute_embedder.eval()

        val_losses = []

        with torch.no_grad():

            for batch in val_loader:

                if use_cache:
                    latents, attrs, ref_img, id_emb, clip_tokens = batch
                    id_emb      = id_emb.to(device, non_blocking=True)
                    clip_tokens = clip_tokens.to(device, non_blocking=True) if args.encoder == "clip_arcface" else None
                else:
                    latents, attrs, ref_img = batch
                    id_emb      = None
                    clip_tokens = None

                latents = latents.to(device, non_blocking=True)
                attrs   = attrs.to(device,   non_blocking=True)
                ref_img = ref_img.to(device,  non_blocking=True)

                with autocast():

                    t = diffusion.sample_timesteps(latents.shape[0]).to(device)
                    z_t, noise = diffusion.noise_images(latents, t)

                    # validação SEM dropout — mede a likelihood condicionada
                    context = build_context(
                        image_encoder=image_encoder,
                        attribute_embedder=attribute_embedder,
                        attrs=attrs,
                        ref_img=ref_img,
                        id_emb=id_emb,
                        clip_tokens_raw=clip_tokens,
                        encoder_type=args.encoder,
                    )

                    predicted_noise = model(z_t, t, context=context)
                    v_loss = F.mse_loss(predicted_noise, noise)

                val_losses.append(v_loss.item())

        def reduce_mean(values):
            t = torch.tensor(sum(values) / len(values), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            return t.item()

        avg_val = reduce_mean(val_losses)

        # =================================================
        # MASTER LOGGING
        # =================================================

        if is_master:

            avg_train = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            lr_cur    = optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch}"
                f" | train: {avg_train:.6f}  val: {avg_val:.6f}"
                f" | LR: {lr_cur:.2e}"
            )

            board.log_scalars("Loss/Epoch", {"train": avg_train, "val": avg_val}, epoch)
            board.log_scalar("Metrics/LR", lr_cur, epoch)

        # =================================================
        # CHECKPOINT
        # =================================================

        if is_master:

            save_periodic = (epoch % 10 == 0 or epoch == args.epochs - 1)
            save_best     = (avg_val < best_val_loss)

            if save_periodic or save_best:

                ckpt = {
                    "epoch":                              epoch,
                    "global_step":                        global_step,
                    "model_state_dict":                   model.module.state_dict(),
                    "image_encoder_state_dict":           image_encoder.module.state_dict(),
                    "attribute_embedder_state_dict":      attribute_embedder.module.state_dict(),
                    "ema_state_dict":                     ema_model.state_dict(),
                    "ema_image_encoder_state_dict":       ema_image_encoder.state_dict(),
                    "ema_attribute_embedder_state_dict":  ema_attribute_embedder.state_dict(),
                    "optimizer_state_dict":               optimizer.state_dict(),
                    "scheduler_state_dict":               scheduler.state_dict(),
                    "scaler_state_dict":                  scaler.state_dict(),
                    "val_loss":                           avg_val,
                    "best_val_loss":                      best_val_loss,
                    "num_img_tokens":                     num_img_tokens,
                    "encoder":                            args.encoder,
                }

                if save_periodic:
                    torch.save(ckpt, os.path.join(save_dir, "ckpt.pt"))
                    print("Checkpoint salvo.")

                if save_best:
                    best_val_loss = avg_val
                    ckpt["best_val_loss"] = best_val_loss
                    torch.save(ckpt, os.path.join(save_dir, "ckpt_best.pt"))
                    print(f"Novo melhor val loss: {best_val_loss:.6f} → ckpt_best.pt")

        # =================================================
        # SAMPLE  —  composable CFG
        # =================================================

        if is_master and (epoch % 10 == 0 or epoch == args.epochs - 1):

            ema_model.eval()
            ema_image_encoder.eval()
            ema_attribute_embedder.eval()

            with torch.no_grad():

                if args.encoder == "arcface_only":
                    id_tok = ema_image_encoder(
                        ref_img=fixed_ref_imgs,
                        id_emb=fixed_id_emb,
                    )
                else:
                    id_tok = ema_image_encoder(
                        ref_img=fixed_ref_imgs,
                        id_emb=fixed_id_emb,
                        clip_tokens_raw=fixed_clip,
                    )

                attr_tok = ema_attribute_embedder(fixed_attrs)

                # ---- atributos originais ----
                print("Sampling (composable CFG — atributos originais)...")
                latents_orig = _sample_composable_cfg(
                    unet=ema_model,
                    diffusion=diffusion,
                    id_tokens_full=id_tok,
                    attr_tokens_full=attr_tok,
                    n=fixed_ref_imgs.shape[0],
                    channels=latent_dim,
                    device=device,
                    s_id=args.s_id_val,
                    s_attr=args.s_attr_val,
                )
                images_orig = vae.decode(latents_orig / 0.18215)

                save_images(images_orig, os.path.join(results_dir, f"{epoch}_attr_orig.jpg"), nrow=4)
                board.log_image(
                    "Samples/AttrOrig",
                    make_grid(images_orig, nrow=4, normalize=True, value_range=(-1, 1)),
                    epoch,
                )

                # ---- mesmo id, com Smiling forçado a 1 ----
                #   teste qualitativo: o modelo consegue fazer a pessoa sorrir?
                SMILING_IDX = 31
                attrs_smile = fixed_attrs.clone()
                attrs_smile[:, SMILING_IDX] = 1.0
                attr_tok_smile = ema_attribute_embedder(attrs_smile)

                print("Sampling (composable CFG — Smiling=1 forçado)...")
                latents_smile = _sample_composable_cfg(
                    unet=ema_model,
                    diffusion=diffusion,
                    id_tokens_full=id_tok,
                    attr_tokens_full=attr_tok_smile,
                    n=fixed_ref_imgs.shape[0],
                    channels=latent_dim,
                    device=device,
                    s_id=args.s_id_val,
                    s_attr=args.s_attr_val,
                )
                images_smile = vae.decode(latents_smile / 0.18215)

                save_images(images_smile, os.path.join(results_dir, f"{epoch}_attr_smile.jpg"), nrow=4)
                board.log_image(
                    "Samples/AttrSmile",
                    make_grid(images_smile, nrow=4, normalize=True, value_range=(-1, 1)),
                    epoch,
                )

    if is_master:
        board.close()

    dist.destroy_process_group()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name",   type=str,   default="LDM_CFGComposable")
    parser.add_argument("--epochs",     type=int,   default=2000)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--image_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)

    parser.add_argument(
        "--encoder",
        type=str,
        default="clip_arcface",
        choices=["clip_arcface", "arcface_only"],
        help="Encoder de identidade. clip_arcface dá mais fidelidade visual; "
             "arcface_only é puro identidade. Com CFG composable, clip_arcface "
             "é seguro porque s_attr sobrepõe a expressão da referência.",
    )

    parser.add_argument(
        "--img_tokens",
        type=int,
        default=16,
        help="Tokens por ramo do encoder (clip e arcface usam este número cada).",
    )

    parser.add_argument(
        "--encoder_cache_dir",
        type=str,
        default="./cache_encoder",
    )

    # ---- dropouts INDEPENDENTES de CFG ----
    parser.add_argument("--cfg_dropout_id_only",   type=float, default=0.1)
    parser.add_argument("--cfg_dropout_attr_only", type=float, default=0.1)
    parser.add_argument("--cfg_dropout_both",      type=float, default=0.1)

    # ---- escalas usadas só nos samples de validação ----
    parser.add_argument("--s_id_val",   type=float, default=3.0)
    parser.add_argument("--s_attr_val", type=float, default=5.0)

    parser.add_argument("--resume_ckpt",    type=str, default=None)
    parser.add_argument("--warmstart_ckpt", type=str, default=None)
    parser.add_argument("--start_epoch",    type=int, default=None)

    args = parser.parse_args()

    train(args)
