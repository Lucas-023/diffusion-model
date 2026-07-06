"""
train_mixed_guidance.py
=======================
Treino combinado de dois modelos para guidance híbrida na inferência:

  1. LDM condicionado só na IMAGEM de referência (Classifier-Free Guidance)
       - ImageConditionEncoder (CLIP-ViT-B/32 + ArcFace buffalo_l) extrai
         tokens da imagem de referência → [B, 512, 2*num_img_tokens]
       - CFG dropout na imagem com prob cfg_dropout_img
       - Na inferência: eps = eps_uncond + s_img*(eps_cond - eps_uncond)

  2. Classificador de atributos nos latentes RUIDOSOS (Classifier Guidance)
       - NoisyLatentAttrClassifier(z_t, t) → 40 logits de atributos
       - Treinado com BCEWithLogitsLoss usando z_t.detach() (sem impactar UNet)
       - Na inferência: eps += -s_attr * sqrt(1-alpha_hat_t) * grad_zt(log p(y|z_t))

Os dois modelos compartilham o z_t amostrado de cada batch, mas têm
optimizadores independentes (lr separado via --lr_cls).

-----------------------------------------------------------------
ACELERAÇÃO POR CACHE DE ENCODER (importante)
-----------------------------------------------------------------
O `ImageConditionEncoder` tem duas branches caras:
  - CLIP-ViT-B/32 forward (GPU, ~768-d patch tokens)
  - ArcFace ONNX (CPU↔GPU + rebind dinâmico do ONNX Runtime — foi
    a causa do warning `Expected shape from model of {1,512} does
    not match actual shape of {64,512}` e de 2+ s/it)

Como os dois encoders são FROZEN e DETERMINÍSTICOS, seus outputs
são iguais a cada epoch para a mesma imagem. Roda-se UMA VEZ:

    python cache_arcface.py

que salva em ./cache_encoder/{stem}.pt um dict:
    "arcface" : [512]          embedding ArcFace L2-normalizado
    "clip"    : [seq_len, 768] tokens CLIP (CLS removido, pré-projeção)

`get_data_imagecond` detecta o diretório automaticamente:
  - cache presente → dataset retorna (latent, attrs, ref_img,
    id_emb, clip_tokens_raw). O encoder pula CLIP+ArcFace e roda
    apenas as projeções treináveis (clip_proj / id_proj / id_norm).
  - cache ausente  → fallback (latent, attrs, ref_img) e o encoder
    roda CLIP/ArcFace live como antes (mais lento).

`ref_img` continua sendo retornado em ambos os modos porque é
usado para visualização (fixed samples / originals).
-----------------------------------------------------------------
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

from utils.board import Board
from utils.utils_celeba import get_data_imagecond, save_images, setup_logging
from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
from models.encoders import ImageConditionEncoder
from models.modules import NoisyLatentAttrClassifier
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
# BUILD IMG CONTEXT
# Só tokens de imagem — atributos ficam para o classificador.
# =========================================================

def build_img_context(
    image_encoder,
    ref_img=None,
    id_emb=None,
    clip_tokens_raw=None,
    cfg_dropout_img: float = 0.1,
    training: bool = True,
):
    """
    Modo live    : passar ref_img → roda CLIP + ArcFace.
    Modo cache   : passar id_emb e clip_tokens_raw (ref_img é ignorado pelo encoder).
    retorna      : [B, 512, 2*num_img_tokens]
    """
    img_tokens = image_encoder(
        ref_img=ref_img,
        id_emb=id_emb,
        clip_tokens_raw=clip_tokens_raw,
    )

    if training and torch.rand(1).item() < cfg_dropout_img:
        img_tokens = torch.zeros_like(img_tokens)

    return img_tokens


# =========================================================
# HYBRID SAMPLING  (image CFG + attribute classifier guidance)
# =========================================================

def _sample_hybrid(
    unet,
    classifier,
    diffusion,
    img_context,
    fixed_attrs,
    n,
    channels,
    device,
    cfg_scale_img=3.0,
    cg_scale_attr=1.0,
):
    unet.eval()
    classifier.eval()

    img_size = diffusion.img_size
    z_t = torch.randn(n, channels, img_size, img_size, device=device)
    zeros_ctx = torch.zeros_like(img_context)

    for i in reversed(range(1, diffusion.noise_steps)):

        t_vec = torch.full((n,), i, device=device, dtype=torch.long)

        alpha_hat_t = diffusion.alpha_hat[t_vec][:, None, None, None]
        alpha_t     = diffusion.alpha[t_vec][:, None, None, None]
        beta_t      = diffusion.beta[t_vec][:, None, None, None]

        with torch.no_grad():
            eps_cond   = unet(z_t, t_vec, context=img_context)
            eps_uncond = unet(z_t, t_vec, context=zeros_ctx)

        eps = eps_uncond + cfg_scale_img * (eps_cond - eps_uncond)

        z_for_cls = z_t.detach().requires_grad_(True)
        logits = classifier(z_for_cls, t_vec)
        log_p  = (
            F.logsigmoid(logits) * fixed_attrs
            + F.logsigmoid(-logits) * (1.0 - fixed_attrs)
        )
        grad = torch.autograd.grad(log_p.sum(), z_for_cls)[0]
        eps = eps - cg_scale_attr * torch.sqrt(1.0 - alpha_hat_t) * grad.detach()

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
    # Retorna (latent, attrs, ref_img) por sample.
    # =====================================================

    train_loader, val_loader, _, train_sampler, use_cache = get_data_imagecond(
        args,
        is_distributed=True,
    )

    if is_master:
        print(
            f"[Encoder] modo {'CACHE (rápido)' if use_cache else 'LIVE (lento — rode cache_arcface.py)'}"
        )

    # =====================================================
    # CONFIG
    # =====================================================

    latent_dim     = 4
    context_dim    = 512
    num_img_tokens = args.img_tokens

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
    # UNET  (condicionada só na imagem)
    # =====================================================

    model = UNet_cond(
        in_channels=latent_dim,
        out_channels=latent_dim,
        context_dim=context_dim,
    ).to(device)

    # =====================================================
    # IMAGE CONDITION ENCODER
    # =====================================================

    image_encoder = ImageConditionEncoder(
        context_dim=context_dim,
        num_tokens=num_img_tokens,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # =====================================================
    # NOISY LATENT ATTRIBUTE CLASSIFIER
    # =====================================================

    classifier = NoisyLatentAttrClassifier(
        latent_dim=latent_dim,
        time_emb_dim=256,
        num_attrs=40,
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
    ema_image_encoder = deepcopy(image_encoder).eval()
    ema_classifier    = deepcopy(classifier).eval()

    for p in ema_model.parameters():
        p.requires_grad = False
    for p in ema_image_encoder.parameters():
        p.requires_grad = False
    for p in ema_classifier.parameters():
        p.requires_grad = False

    # =====================================================
    # DDP
    # =====================================================

    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    image_encoder = DDP(
        image_encoder,
        device_ids=[local_rank],
        find_unused_parameters=True,
    )

    classifier = DDP(classifier, device_ids=[local_rank])

    # =====================================================
    # OPTIMIZERS
    # Dois optimizadores independentes:
    #   optimizer_diff  → UNet + proj do ImageEncoder
    #   optimizer_cls   → NoisyLatentAttrClassifier
    # =====================================================

    trainable_img_params = [
        p for p in image_encoder.parameters() if p.requires_grad
    ]

    optimizer_diff = optim.AdamW(
        list(model.parameters()) + trainable_img_params,
        lr=args.lr,
        weight_decay=1e-4,
    )

    optimizer_cls = optim.AdamW(
        classifier.parameters(),
        lr=args.lr_cls,
        weight_decay=1e-4,
    )

    # =====================================================
    # LR SCHEDULERS
    # =====================================================

    warmup_epochs = 10

    def make_scheduler(opt):
        return SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(opt, T_max=(args.epochs - warmup_epochs), eta_min=1e-6),
            ],
            milestones=[warmup_epochs],
        )

    scheduler_diff = make_scheduler(optimizer_diff)
    scheduler_cls  = make_scheduler(optimizer_cls)

    # =====================================================
    # AMP
    # =====================================================

    scaler_diff = GradScaler()
    scaler_cls  = GradScaler()

    accumulation_steps = 4

    start_epoch  = 0
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
        classifier.module.load_state_dict(ckpt["classifier_state_dict"])

        ema_model.load_state_dict(ckpt["ema_state_dict"])
        ema_image_encoder.load_state_dict(ckpt["ema_image_encoder_state_dict"])
        ema_classifier.load_state_dict(ckpt["ema_classifier_state_dict"])

        optimizer_diff.load_state_dict(ckpt["optimizer_diff_state_dict"])
        optimizer_cls.load_state_dict(ckpt["optimizer_cls_state_dict"])

        start_epoch = (
            args.start_epoch
            if args.start_epoch is not None
            else ckpt["epoch"] + 1
        )

        if "scheduler_diff_state_dict" in ckpt and args.start_epoch is None:
            scheduler_diff.load_state_dict(ckpt["scheduler_diff_state_dict"])
            scheduler_cls.load_state_dict(ckpt["scheduler_cls_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler_diff.step()
                scheduler_cls.step()

        if "scaler_diff_state_dict" in ckpt:
            scaler_diff.load_state_dict(ckpt["scaler_diff_state_dict"])
            scaler_cls.load_state_dict(ckpt["scaler_cls_state_dict"])

        if is_master:
            if "global_step" in ckpt:
                global_step = ckpt["global_step"]
            if "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]
            print(f"Retomando epoch {start_epoch} | LR diff: {optimizer_diff.param_groups[0]['lr']:.2e} | LR cls: {optimizer_cls.param_groups[0]['lr']:.2e}")

    # =====================================================
    # WARM START (checkpoint de image_cond sem classificador)
    # =====================================================

    elif args.warmstart_ckpt is not None and os.path.isfile(args.warmstart_ckpt):

        if is_master:
            print(f"\nWarm start de: {args.warmstart_ckpt}")

        ckpt = torch.load(args.warmstart_ckpt, map_location=device)

        model.module.load_state_dict(ckpt["model_state_dict"])
        image_encoder.module.load_state_dict(ckpt["image_encoder_state_dict"])

        ema_model.load_state_dict(
            ckpt.get("ema_state_dict", ckpt["model_state_dict"])
        )
        ema_image_encoder.load_state_dict(
            ckpt.get("ema_image_encoder_state_dict", ckpt["image_encoder_state_dict"])
        )

        if args.start_epoch is not None:
            start_epoch = args.start_epoch
            for _ in range(start_epoch):
                scheduler_diff.step()
                scheduler_cls.step()

        if is_master:
            print(
                f"UNet e ImageEncoder carregados. "
                f"Classifier inicializado do zero. "
                f"Epoch {start_epoch} | LR diff: {optimizer_diff.param_groups[0]['lr']:.2e}"
            )

    # =====================================================
    # DIRS
    # =====================================================

    save_dir    = os.path.join("models",  args.run_name)
    results_dir = os.path.join("results", args.run_name)

    os.makedirs(save_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    fixed_ref_imgs = None

    # =====================================================
    # FIXED SAMPLES + ORIGINAIS  (master, uma vez)
    # =====================================================

    if is_master:
        from PIL import Image as PILImage

        train_base = train_loader.dataset.dataset

        samples_fixed = []
        for _i in range(min(16, len(train_loader.dataset))):
            real_idx = train_loader.dataset.indices[_i]
            sample   = train_base[real_idx]

            if use_cache:
                _, _attrs, _ref, _idemb, _cliptok = sample
            else:
                _, _attrs, _ref = sample
                _idemb, _cliptok = None, None

            _fname = train_base.samples[real_idx][0]
            _img   = PILImage.open(
                os.path.join(train_base.image_dir, _fname)
            ).convert("RGB")
            _orig = train_base.transform(_img)

            samples_fixed.append((_ref, _orig, _attrs, _idemb, _cliptok))

        fixed_ref_imgs = torch.stack([s[0] for s in samples_fixed]).to(device)
        fixed_attrs    = torch.stack([s[2] for s in samples_fixed]).to(device)
        orig_imgs      = torch.stack([s[1] for s in samples_fixed])

        if use_cache:
            fixed_id_emb    = torch.stack([s[3] for s in samples_fixed]).to(device)
            fixed_clip_raw  = torch.stack([s[4] for s in samples_fixed]).to(device)
        else:
            fixed_id_emb   = None
            fixed_clip_raw = None

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
        classifier.train()

        pbar = (
            tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs - 1}")
            if is_master
            else train_loader
        )

        epoch_losses_diff = []
        epoch_losses_cls  = []

        optimizer_diff.zero_grad(set_to_none=True)
        optimizer_cls.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):

            if use_cache:
                latents, attrs, ref_img, id_emb, clip_tokens_raw = batch
                id_emb          = id_emb.to(device, non_blocking=True)
                clip_tokens_raw = clip_tokens_raw.to(device, non_blocking=True)
            else:
                latents, attrs, ref_img = batch
                id_emb          = None
                clip_tokens_raw = None

            latents = latents.to(device, non_blocking=True)
            attrs   = attrs.to(device,   non_blocking=True)
            ref_img = ref_img.to(device,  non_blocking=True)

            # =============================================
            # TIMESTEPS  (compartilhados pelos dois modelos)
            # =============================================

            t = diffusion.sample_timesteps(latents.shape[0]).to(device)

            # =============================================
            # NOISY LATENTS  (compartilhados)
            # =============================================

            with torch.no_grad():
                z_t, noise = diffusion.noise_images(latents, t)

            # =============================================
            # ACCUMULATION CONTEXT  (diffusion)
            # =============================================

            is_accumulating = (
                (i + 1) % accumulation_steps != 0
                and (i + 1) != len(train_loader)
            )

            model_ctx = (
                model.no_sync() if is_accumulating else contextlib.nullcontext()
            )
            enc_ctx = (
                image_encoder.no_sync() if is_accumulating else contextlib.nullcontext()
            )

            # =============================================
            # DIFFUSION STEP
            # =============================================

            with model_ctx, enc_ctx:

                with autocast():

                    context = build_img_context(
                        image_encoder,
                        ref_img=ref_img,
                        id_emb=id_emb,
                        clip_tokens_raw=clip_tokens_raw,
                        cfg_dropout_img=args.cfg_dropout_img,
                        training=True,
                    )

                    # z_t requer grad para o UNet, não precisamos recomputar
                    predicted_noise = model(z_t, t, context=context)

                    diff_loss = nn.functional.mse_loss(predicted_noise, noise)
                    diff_loss = diff_loss / accumulation_steps

                scaler_diff.scale(diff_loss).backward()

            # =============================================
            # CLASSIFIER STEP  (sem accumulation — mais leve)
            # z_t detachado para não vazar gradiente para o UNet
            # =============================================

            with autocast():

                pred_logits = classifier(z_t.detach(), t)
                cls_loss    = nn.functional.binary_cross_entropy_with_logits(
                    pred_logits,
                    attrs,
                )

            scaler_cls.scale(cls_loss).backward()
            scaler_cls.step(optimizer_cls)
            scaler_cls.update()
            optimizer_cls.zero_grad(set_to_none=True)

            # =============================================
            # OPTIMIZER STEP (diffusion)
            # =============================================

            if not is_accumulating:

                scaler_diff.step(optimizer_diff)
                scaler_diff.update()
                optimizer_diff.zero_grad(set_to_none=True)

                if is_master:

                    update_ema(ema_model,         model.module)
                    update_ema(ema_image_encoder,  image_encoder.module)
                    update_ema(ema_classifier,     classifier.module)

                    diff_display = diff_loss.item() * accumulation_steps
                    cls_display  = cls_loss.item()

                    pbar.set_postfix(MSE=diff_display, BCE=cls_display)

                    board.log_scalar("Loss/Batch/Diffusion",  diff_display, global_step)
                    board.log_scalar("Loss/Batch/Classifier", cls_display,  global_step)

                    epoch_losses_diff.append(diff_display)
                    epoch_losses_cls.append(cls_display)

                    global_step += 1

        # =================================================
        # SCHEDULERS
        # =================================================

        scheduler_diff.step()
        scheduler_cls.step()

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()
        image_encoder.eval()
        classifier.eval()

        val_diff_losses = []
        val_cls_losses  = []
        val_correct     = 0
        val_total       = 0

        with torch.no_grad():

            for batch in val_loader:

                if use_cache:
                    latents, attrs, ref_img, id_emb, clip_tokens_raw = batch
                    id_emb          = id_emb.to(device, non_blocking=True)
                    clip_tokens_raw = clip_tokens_raw.to(device, non_blocking=True)
                else:
                    latents, attrs, ref_img = batch
                    id_emb          = None
                    clip_tokens_raw = None

                latents = latents.to(device, non_blocking=True)
                attrs   = attrs.to(device,   non_blocking=True)
                ref_img = ref_img.to(device,  non_blocking=True)

                with autocast():

                    t = diffusion.sample_timesteps(latents.shape[0]).to(device)

                    z_t, noise = diffusion.noise_images(latents, t)

                    context = build_img_context(
                        image_encoder,
                        ref_img=ref_img,
                        id_emb=id_emb,
                        clip_tokens_raw=clip_tokens_raw,
                        training=False,
                    )

                    predicted_noise = model(z_t, t, context=context)
                    diff_loss = nn.functional.mse_loss(predicted_noise, noise)

                    pred_logits = classifier(z_t, t)
                    cls_loss    = nn.functional.binary_cross_entropy_with_logits(
                        pred_logits, attrs
                    )

                val_diff_losses.append(diff_loss.item())
                val_cls_losses.append(cls_loss.item())

                preds   = (pred_logits.sigmoid() > 0.5).float()
                val_correct += (preds == attrs).sum().item()
                val_total   += attrs.numel()

        def reduce_mean(values):
            t = torch.tensor(sum(values) / len(values), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            return t.item()

        avg_val_diff = reduce_mean(val_diff_losses)
        avg_val_cls  = reduce_mean(val_cls_losses)

        acc_tensor = torch.tensor(val_correct / val_total, device=device)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
        avg_val_acc = acc_tensor.item()

        # =================================================
        # MASTER LOGGING
        # =================================================

        if is_master:

            avg_diff = sum(epoch_losses_diff) / len(epoch_losses_diff) if epoch_losses_diff else 0.0
            avg_cls  = sum(epoch_losses_cls)  / len(epoch_losses_cls)  if epoch_losses_cls  else 0.0
            lr_diff  = optimizer_diff.param_groups[0]["lr"]
            lr_cls   = optimizer_cls.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch}"
                f" | Diff train: {avg_diff:.6f}  val: {avg_val_diff:.6f}"
                f" | Cls train: {avg_cls:.6f}  val: {avg_val_cls:.6f}"
                f" | Acc: {avg_val_acc:.4f}"
                f" | LR diff: {lr_diff:.2e}  cls: {lr_cls:.2e}"
            )

            board.log_scalars(
                "Loss/Epoch/Diffusion",
                {"train": avg_diff, "val": avg_val_diff},
                epoch,
            )
            board.log_scalars(
                "Loss/Epoch/Classifier",
                {"train": avg_cls, "val": avg_val_cls},
                epoch,
            )
            board.log_scalar("Metrics/Classifier_Accuracy", avg_val_acc, epoch)
            board.log_scalar("Metrics/LR_Diff", lr_diff, epoch)
            board.log_scalar("Metrics/LR_Cls",  lr_cls,  epoch)

        # =================================================
        # CHECKPOINT
        # =================================================

        if is_master:

            save_periodic = (epoch % 10 == 0 or epoch == args.epochs - 1)
            save_best     = (avg_val_diff < best_val_loss)

            if save_periodic or save_best:

                ckpt = {
                    "epoch":                        epoch,
                    "global_step":                  global_step,
                    "model_state_dict":             model.module.state_dict(),
                    "image_encoder_state_dict":     image_encoder.module.state_dict(),
                    "classifier_state_dict":        classifier.module.state_dict(),
                    "ema_state_dict":               ema_model.state_dict(),
                    "ema_image_encoder_state_dict": ema_image_encoder.state_dict(),
                    "ema_classifier_state_dict":    ema_classifier.state_dict(),
                    "optimizer_diff_state_dict":    optimizer_diff.state_dict(),
                    "optimizer_cls_state_dict":     optimizer_cls.state_dict(),
                    "scheduler_diff_state_dict":    scheduler_diff.state_dict(),
                    "scheduler_cls_state_dict":     scheduler_cls.state_dict(),
                    "scaler_diff_state_dict":       scaler_diff.state_dict(),
                    "scaler_cls_state_dict":        scaler_cls.state_dict(),
                    "val_loss_diff":                avg_val_diff,
                    "val_loss_cls":                 avg_val_cls,
                    "val_acc":                      avg_val_acc,
                    "best_val_loss":                best_val_loss,
                    "num_img_tokens":               num_img_tokens,
                }

                if save_periodic:
                    torch.save(ckpt, os.path.join(save_dir, "ckpt.pt"))
                    print("Checkpoint salvo.")

                if save_best:
                    best_val_loss = avg_val_diff
                    ckpt["best_val_loss"] = best_val_loss
                    torch.save(ckpt, os.path.join(save_dir, "ckpt_best.pt"))
                    print(f"Novo melhor val diff loss: {best_val_loss:.6f} → ckpt_best.pt")

        # =================================================
        # SAMPLE
        # =================================================

        if is_master and (epoch % 10 == 0 or epoch == args.epochs - 1):

            ema_model.eval()
            ema_image_encoder.eval()
            ema_classifier.eval()

            with torch.no_grad():
                context_test = build_img_context(
                    ema_image_encoder,
                    ref_img=fixed_ref_imgs.to(device),
                    id_emb=fixed_id_emb,
                    clip_tokens_raw=fixed_clip_raw,
                    training=False,
                )

            # -----------------------------------------------
            # 1. Image CFG only
            # -----------------------------------------------
            print("Gerando imagens (image CFG)...")

            with torch.no_grad():
                latents_img = diffusion.sample(
                    ema_model,
                    n=16,
                    context=context_test,
                    channels=latent_dim,
                )
                images_img = vae.decode(latents_img / 0.18215)

            save_images(
                images_img,
                os.path.join(results_dir, f"{epoch}_img.jpg"),
                nrow=4,
            )
            board.log_image(
                "Samples/ImageCFG",
                make_grid(images_img, nrow=4, normalize=True, value_range=(-1, 1)),
                epoch,
            )

            # -----------------------------------------------
            # 2. Full conditioning: image CFG + attribute CG
            # -----------------------------------------------
            print("Gerando imagens (image CFG + attribute CG)...")

            latents_hybrid = _sample_hybrid(
                unet=ema_model,
                classifier=ema_classifier,
                diffusion=diffusion,
                img_context=context_test,
                fixed_attrs=fixed_attrs,
                n=16,
                channels=latent_dim,
                device=device,
            )

            with torch.no_grad():
                images_hybrid = vae.decode(latents_hybrid / 0.18215)

            save_images(
                images_hybrid,
                os.path.join(results_dir, f"{epoch}_hybrid.jpg"),
                nrow=4,
            )
            board.log_image(
                "Samples/Hybrid",
                make_grid(images_hybrid, nrow=4, normalize=True, value_range=(-1, 1)),
                epoch,
            )

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

    parser.add_argument("--run_name",    type=str,   default="LDM_MixedGuidance")
    parser.add_argument("--epochs",      type=int,   default=2000)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--image_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=3e-4,
                        help="LR do diffusion model (UNet + ImageEncoder.proj).")
    parser.add_argument("--lr_cls",      type=float, default=1e-4,
                        help="LR do NoisyLatentAttrClassifier.")

    parser.add_argument(
        "--img_tokens",
        type=int,
        default=16,
        help="Tokens extraídos da imagem de referência (quadrado perfeito: 4, 9, 16, 25).",
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Congela backbone CLIP do ImageEncoder (ArcFace é sempre frozen; "
             "treina só as projeções clip_proj / id_proj).",
    )

    parser.add_argument(
        "--no_freeze_backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Treina o backbone CLIP end-to-end (ArcFace continua frozen).",
    )

    parser.add_argument(
        "--encoder_cache_dir",
        type=str,
        default="./cache_encoder",
        help="Diretório com embeddings ArcFace+CLIP pré-computados por cache_arcface.py. "
             "Se existir e contiver arquivos, o treino pula CLIP/ArcFace live "
             "(grande aceleração). Caso contrário roda live como fallback.",
    )

    parser.add_argument(
        "--cfg_dropout_img",
        type=float,
        default=0.1,
        help="Prob de zerar tokens de imagem no diffusion (CFG dropout).",
    )

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Resume deste treino (ckpt gerado por este script).",
    )

    parser.add_argument(
        "--warmstart_ckpt",
        type=str,
        default=None,
        help="Warm start de checkpoint de train_image_cond.py "
             "(carrega UNet + ImageEncoder; Classifier começa do zero).",
    )

    parser.add_argument(
        "--start_epoch",
        type=int,
        default=None,
        help="Força início na epoch N (posiciona scheduler corretamente).",
    )

    args = parser.parse_args()

    train(args)
