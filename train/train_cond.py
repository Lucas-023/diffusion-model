# ============================================================
# TRAIN SINGLE GPU - LATENT DIFFUSION CONDICIONAL
# ============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid

from board import Board

from utils.utils_celeba import (
    get_data,
    save_images,
    setup_logging
)

from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
from models.modules import AttributeEmbedder

from vae.modules import VAE


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


def train(args):

    setup_logging(args.run_name)

    device = args.device

    dataloader, _ = get_data(
        args,
        is_distributed=False
    )

    latent_dim = 4
    context_dim = 512

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

    # ========================================================
    # VAE (APENAS DECODER)
    # ========================================================

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

    # ========================================================
    # UNET
    # ========================================================

    model = UNet_cond(
        in_channels=latent_dim,
        out_channels=latent_dim,
        context_dim=context_dim
    ).to(device)

    # ========================================================
    # ATTRIBUTE EMBEDDER
    # ========================================================

    attribute_embedder = AttributeEmbedder(
        num_attributes=40,
        context_dim=context_dim
    ).to(device)

    # ========================================================
    # EMA
    # ========================================================

    ema_model = deepcopy(model)

    ema_attribute_embedder = deepcopy(
        attribute_embedder
    )

    ema_model.eval()
    ema_attribute_embedder.eval()

    for p in ema_model.parameters():
        p.requires_grad = False

    for p in ema_attribute_embedder.parameters():
        p.requires_grad = False

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.AdamW(

        list(model.parameters()) +

        list(attribute_embedder.parameters()),

        lr=args.lr
    )

    mse = nn.MSELoss()

    diffusion = Diffusion_conditional(
        img_size=32,
        device=device
    )

    scaler = GradScaler()

    board = Board(
        run_name=args.run_name,
        enabled=True
    )

    global_step = 0
    start_epoch = 0

    ckpt_path = os.path.join(
        save_dir,
        "ckpt.pt"
    )

    # ========================================================
    # RESUME
    # ========================================================

    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):

        print(f"\n🔄 Carregando checkpoint:")
        print(args.resume_ckpt)

        checkpoint = torch.load(
            args.resume_ckpt,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        attribute_embedder.load_state_dict(
            checkpoint["attribute_embedder_state_dict"]
        )

        ema_model.load_state_dict(
            checkpoint["ema_model_state_dict"]
        )

        ema_attribute_embedder.load_state_dict(
            checkpoint["ema_attribute_embedder_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        print(f"✅ Retomando treino da época {start_epoch}")

    else:

        print("\nNenhum checkpoint encontrado.")
        print("Treino começando do zero.")

    # ========================================================
    # FIXED LATENTS
    # ========================================================

    fixed_latents = None
    fixed_attributes = None

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(start_epoch, args.epochs):

        logging.info(f"A iniciar época {epoch}")

        pbar = tqdm(
            dataloader,
            desc=f"Época {epoch}/{args.epochs}"
        )

        epoch_losses = []

        for i, batch in enumerate(pbar):

            if isinstance(batch, dict):

                latents = batch["latent"].to(device)
                attributes = batch["attrs"].to(device)

            else:

                latents = batch[0].to(device)
                attributes = batch[1].to(device)
            if fixed_latents is None:

                fixed_latents = latents[:16]

                fixed_attributes = attributes[:16]

            optimizer.zero_grad(set_to_none=True)

            with autocast():

                # =============================================
                # CONDITIONING
                # =============================================

                context = attribute_embedder(
                    attributes
                )

                # classifier free guidance

                if torch.rand(1).item() < 0.1:

                    context = torch.zeros_like(context)

                # =============================================
                # DIFFUSION
                # =============================================

                t = diffusion.sample_timesteps(
                    latents.shape[0]
                ).to(device)

                z_t, noise = diffusion.noise_images(
                    latents,
                    t
                )

                predicted_noise = model(
                    z_t,
                    t,
                    context=context
                )

                loss = mse(
                    predicted_noise,
                    noise
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            # =================================================
            # EMA
            # =================================================

            update_ema(
                ema_model,
                model
            )

            update_ema(
                ema_attribute_embedder,
                attribute_embedder
            )

            epoch_losses.append(loss.item())

            pbar.set_postfix(
                MSE=loss.item()
            )

            board.log_scalar(
                "Loss/Batch",
                loss.item(),
                global_step
            )

            global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        print(f"\n📊 Época {epoch}")
        print(f"Loss Médio: {avg_loss:.6f}")

        board.log_scalar(
            "Loss/Epoch",
            avg_loss,
            epoch
        )

        # ====================================================
        # CHECKPOINT
        # ====================================================

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "attribute_embedder_state_dict":
                attribute_embedder.state_dict(),

            "ema_model_state_dict":
                ema_model.state_dict(),

            "ema_attribute_embedder_state_dict":
                ema_attribute_embedder.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),
        }

        torch.save(
            checkpoint,
            ckpt_path
        )

        # ====================================================
        # INFERENCE
        # ====================================================

        if epoch % 25 == 0 or epoch == args.epochs - 1:

            print("🎨 Gerando imagens...")

            with torch.no_grad():

                context_teste = ema_attribute_embedder(
                    fixed_attributes
                )

                sampled_latents = diffusion.sample(
                    ema_model,
                    n=16,
                    context=context_teste
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
                "Geracao/Teste",
                grid,
                epoch
            )

    board.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--run_name',
        type=str,
        default="LDM_Conditional_Attributes"
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=256
    )

    parser.add_argument(
        '--device',
        type=str,
        default="cuda"
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4
    )

    parser.add_argument(
        '--resume_ckpt',
        type=str,
        default=None
    )

    args = parser.parse_args()

    train(args)