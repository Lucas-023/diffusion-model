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
from utils.utils_celeba import save_images, setup_logging, get_data

from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
from vae.modules import VAE


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
# TREINO
# =========================================================

def train(args):

    setup_logging(args.run_name)

    device = args.device

    board = Board(
        run_name=args.run_name,
        enabled=True
    )

    global_step = 0

    # =====================================================
    # DATASET
    # =====================================================

    dataloader, _ = get_data(
        args,
        is_distributed=False
    )

    # =====================================================
    # MODELOS
    # =====================================================

    latent_dim = 4

    # -----------------------------------------------------
    # VAE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # UNET
    # -----------------------------------------------------

    model = UNet_cond(
        in_channels=latent_dim,
        out_channels=latent_dim,
        context_dim=args.context_dim,
        num_classes=args.num_classes
    ).to(device)

    ema_model = deepcopy(model).eval()

    for p in ema_model.parameters():
        p.requires_grad = False

    # =====================================================
    # DIFUSÃO
    # =====================================================

    diffusion = Diffusion_conditional(
        img_size=args.image_size // 8,
        device=device
    )

    # =====================================================
    # OTIMIZADOR
    # =====================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scaler = GradScaler()

    # =====================================================
    # CHECKPOINT
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

    ckpt_path = os.path.join(
        save_dir,
        "ckpt.pt"
    )

    start_epoch = 0
    best_loss = float("inf")

    if os.path.exists(ckpt_path):

        print(f"\n🔄 Carregando checkpoint:")
        print(ckpt_path)

        checkpoint = torch.load(
            ckpt_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        ema_model.load_state_dict(
            checkpoint["ema_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = checkpoint["epoch"] + 1

        best_loss = checkpoint.get(
            "loss",
            float("inf")
        )

        print(f"✅ Retomando da época {start_epoch}")

    else:

        print("\nNenhum checkpoint encontrado.")
        print("Treino começando do zero.")

    # =====================================================
    # FIXED BATCH
    # =====================================================

    fixed_latents, fixed_attrs = next(iter(dataloader))

    fixed_latents = fixed_latents[:16].to(device)
    fixed_attrs = fixed_attrs[:16].to(device)

    # =====================================================
    # LOOP
    # =====================================================

    for epoch in range(start_epoch, args.epochs):

        logging.info(f"Iniciando época {epoch}")

        pbar = tqdm(
            dataloader,
            desc=f"Época {epoch}/{args.epochs}"
        )

        epoch_losses = []

        for latents, attrs in pbar:

            latents = latents.to(device)
            attrs = attrs.to(device)

            # ---------------------------------------------
            # SCALE LATENT
            # ---------------------------------------------

            z_target = latents * 0.18215

            # ---------------------------------------------
            # TIMESTEPS
            # ---------------------------------------------

            t = diffusion.sample_timesteps(
                latents.shape[0]
            ).to(device)

            # ---------------------------------------------
            # ADD NOISE
            # ---------------------------------------------

            z_t, noise = diffusion.noise_images(
                z_target,
                t
            )

            optimizer.zero_grad(set_to_none=True)

            # ---------------------------------------------
            # FORWARD
            # ---------------------------------------------

            with autocast():

                predicted_noise = model(
                    z_t,
                    t,
                    attrs
                )

                loss = nn.functional.mse_loss(
                    predicted_noise,
                    noise
                )

            # ---------------------------------------------
            # BACKWARD
            # ---------------------------------------------

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            # ---------------------------------------------
            # EMA
            # ---------------------------------------------

            update_ema(
                ema_model,
                model
            )

            # ---------------------------------------------
            # LOG
            # ---------------------------------------------

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

        # =================================================
        # FIM ÉPOCA
        # =================================================

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        print(f"\n📊 Época {epoch}")
        print(f"Loss médio: {avg_loss:.6f}")

        board.log_scalar(
            "Loss/Epoch",
            avg_loss,
            epoch
        )

        # =================================================
        # CHECKPOINT
        # =================================================

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": avg_loss,
        }

        torch.save(
            checkpoint,
            ckpt_path
        )

        # -------------------------------------------------
        # BEST
        # -------------------------------------------------

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                checkpoint,
                os.path.join(
                    save_dir,
                    "best_ckpt.pt"
                )
            )

            print(
                f"🏆 Novo melhor modelo salvo!"
            )

        # -------------------------------------------------
        # PERIODIC
        # -------------------------------------------------

        if epoch % 25 == 0 and epoch > 0:

            torch.save(
                checkpoint,
                os.path.join(
                    save_dir,
                    f"ckpt_epoch_{epoch}.pt"
                )
            )

            print(
                f"📦 Checkpoint periódico salvo!"
            )

        # =================================================
        # SAMPLE
        # =================================================

        if epoch % 25 == 0 or epoch == args.epochs - 1:

            print("\n🎨 Gerando imagens...")

            with torch.no_grad():

                sampled_latents = diffusion.sample(
                    ema_model,
                    n=16,
                    labels=fixed_attrs,
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
                "Geracao/Teste",
                grid,
                epoch
            )

    board.close()

    print("\n✅ Treinamento finalizado!")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        type=str,
        default="LDM_CelebA_Attributes"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000
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
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--context_dim",
        type=int,
        default=512
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=40
    )

    args = parser.parse_args()

    train(args)