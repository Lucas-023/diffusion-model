import os
import sys

# Força o Python a enxergar a pasta principal do projeto
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
from utils.utils_celeba import get_data, save_images, setup_logging
from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
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

    dist.init_process_group(
        backend="nccl"
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank, global_rank


# =========================================================
# EMA
# =========================================================

def update_ema(
    ema_model,
    model,
    decay=0.9999
):

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
# TRAIN
# =========================================================

def train(args):

    local_rank, global_rank = setup_ddp()

    is_master = (global_rank == 0)

    device = f"cuda:{local_rank}"

    if is_master:

        setup_logging(args.run_name)

        print("\n🚀 LDM Condicional por Atributos - MultiGPU")

        board = Board(
            run_name=args.run_name,
            enabled=True
        )

        global_step = 0

    # =====================================================
    # DATA
    # =====================================================

    dataloader, sampler = get_data(
        args,
        is_distributed=True
    )

    # =====================================================
    # CONFIG
    # =====================================================

    latent_dim = 4
    context_dim = 512

    # =====================================================
    # VAE
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
    # MODEL
    # =====================================================

    model = UNet_cond(
        in_channels=latent_dim,
        out_channels=latent_dim,
        context_dim=context_dim
    ).to(device)

    diffusion = Diffusion_conditional(
        img_size=args.image_size // 8,
        device=device
    )

    ema_model = deepcopy(model).eval()

    # =====================================================
    # DDP
    # =====================================================

    model = DDP(
        model,
        device_ids=[local_rank],
        gradient_as_bucket_view=True
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = optim.AdamW(
        model.parameters(),
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
        schedulers=[
            warmup_scheduler,
            cosine_scheduler
        ],
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

            print(f"\n🔄 Carregando checkpoint:")
            print(args.resume_ckpt)

        checkpoint = torch.load(
            args.resume_ckpt,
            map_location=device
        )

        model.module.load_state_dict(
            checkpoint["model_state_dict"]
        )

        ema_model.load_state_dict(
            checkpoint["ema_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        if is_master:

            print(
                f"✅ Retomando da época {start_epoch}"
            )

    # =====================================================
    # DIRS
    # =====================================================

    save_dir = os.path.join(
        "models",
        args.run_name
    )

    results_dir = os.path.join(
        "results",
        args.run_name
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(
        start_epoch,
        args.epochs
    ):

        sampler.set_epoch(epoch)

        pbar = tqdm(dataloader) if is_master else dataloader

        epoch_losses = []

        for i, (images, attrs) in enumerate(pbar):

            images = images.to(device)
            attrs = attrs.to(device)

            # =============================================
            # VAE ENCODE
            # =============================================

            with torch.no_grad():

                posterior = vae.encode(images)

                z = posterior.sample()

                z = z * 0.18215

            # =============================================
            # TIMESTEPS
            # =============================================

            t = diffusion.sample_timesteps(
                z.shape[0]
            ).to(device)

            # =============================================
            # DUMMY CONTEXT
            # TEMPORÁRIO
            # =============================================

            context = torch.randn(
                z.shape[0],
                context_dim,
                1,
                device=device
            )

            # =============================================
            # ACCUMULATION
            # =============================================

            is_accumulating = (
                (i + 1) % accumulation_steps != 0
                and
                (i + 1) != len(dataloader)
            )

            sync_context = (
                model.no_sync()
                if is_accumulating
                else contextlib.nullcontext()
            )

            with sync_context:

                with autocast():

                    z_t, noise = diffusion.noise_images(
                        z,
                        t
                    )

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
            # STEP
            # =============================================

            if not is_accumulating:

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                if is_master:

                    update_ema(
                        ema_model,
                        model.module
                    )

                    loss_display = (
                        loss.item()
                        * accumulation_steps
                    )

                    pbar.set_postfix(
                        MSE=loss_display
                    )

                    board.log_scalar(
                        "Loss/Batch",
                        loss_display,
                        global_step
                    )

                    epoch_losses.append(
                        loss_display
                    )

                    global_step += 1

        # =================================================
        # SCHEDULER STEP
        # =================================================

        scheduler.step()

        # =================================================
        # MASTER
        # =================================================

        if is_master:

            avg_loss = (
                sum(epoch_losses)
                / len(epoch_losses)
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\n📊 Epoch {epoch}"
                f" | Loss: {avg_loss:.6f}"
                f" | LR: {current_lr:.6f}"
            )

            board.log_scalar(
                "Metrics/Loss_Epoch",
                avg_loss,
                epoch
            )

            board.log_scalar(
                "Metrics/Learning_Rate",
                current_lr,
                epoch
            )

            # =============================================
            # SAVE
            # =============================================

            if (
                epoch % 10 == 0
                or
                epoch == args.epochs - 1
            ):

                checkpoint = {

                    "epoch": epoch,

                    "model_state_dict":
                        model.module.state_dict(),

                    "ema_state_dict":
                        ema_model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),
                }

                torch.save(
                    checkpoint,
                    os.path.join(
                        save_dir,
                        "ckpt.pt"
                    )
                )

                print("💾 Checkpoint salvo.")

            # =============================================
            # SAMPLE
            # =============================================

            if (
                epoch % 10 == 0
                or
                epoch == args.epochs - 1
            ):

                print(
                    "🎨 Gerando imagens..."
                )

                with torch.no_grad():

                    context_test = torch.randn(
                        16,
                        context_dim,
                        1,
                        device=device
                    )

                    sampled_latents = diffusion.sample(
                        ema_model,
                        n=16,
                        context=context_test,
                        channels=latent_dim
                    )

                    sampled_latents = (
                        sampled_latents / 0.18215
                    )

                    sampled_images = vae.decode(
                        sampled_latents
                    )

                save_images(
                    sampled_images,
                    os.path.join(
                        results_dir,
                        f"{epoch}.jpg"
                    )
                )

                grid = make_grid(
                    sampled_images,
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1)
                )

                board.log_image(
                    "Samples/Generated",
                    grid,
                    epoch
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

    parser.add_argument(
        "--run_name",
        type=str,
        default="LDM_Conditional_Attributes"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
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
        default=None
    )

    args = parser.parse_args()

    train(args)