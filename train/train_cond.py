# ============================================================
# TRAIN SINGLE GPU - LATENT DIFFUSION CONDICIONAL
# ============================================================

import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp

from tqdm import tqdm
from copy import deepcopy

from torchvision.utils import make_grid

from board import Board

from utils.utils_celeba import (
    get_data,
    save_images,s
    setup_logging
)

from diffusion.conditional_ddpm import (
    Diffusion_conditional
)

from models.unet_conditional import (
    UNet_cond
)

from models.modules import (
    AttributeEmbedder
)

from vae.modules import VAE


# ============================================================
# CUDA
# ============================================================

torch.backends.cudnn.benchmark = True


# ============================================================
# EMA
# ============================================================

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


# ============================================================
# TRAIN
# ============================================================

def train(args):

    setup_logging(args.run_name)

    device = args.device

    # ========================================================
    # DATA
    # ========================================================

    train_loader, val_loader, test_loader, _ = get_data(
        args,
        is_distributed=False
    )

    # ========================================================
    # CONFIG
    # ========================================================

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
    # VAE
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

    # ========================================================
    # LOSS
    # ========================================================

    mse = nn.MSELoss()

    # ========================================================
    # DIFFUSION
    # ========================================================

    diffusion = Diffusion_conditional(
        img_size=32,
        device=device
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = torch.amp.GradScaler("cuda")

    # ========================================================
    # TENSORBOARD
    # ========================================================

    board = Board(
        run_name=args.run_name,
        enabled=True
    )

    # ========================================================
    # STATE
    # ========================================================

    global_step = 0
    start_epoch = 0

    best_val_loss = float("inf")

    ckpt_path = os.path.join(
        save_dir,
        "ckpt.pt"
    )

    best_ckpt_path = os.path.join(
        save_dir,
        "best_ckpt.pt"
    )

    # ========================================================
    # RESUME
    # ========================================================

    if args.resume_ckpt and os.path.isfile(
        args.resume_ckpt
    ):

        print("\n🔄 Carregando checkpoint:")
        print(args.resume_ckpt)

        checkpoint = torch.load(
            args.resume_ckpt,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        attribute_embedder.load_state_dict(
            checkpoint[
                "attribute_embedder_state_dict"
            ]
        )

        ema_model.load_state_dict(
            checkpoint["ema_model_state_dict"]
        )

        ema_attribute_embedder.load_state_dict(
            checkpoint[
                "ema_attribute_embedder_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        if "val_loss" in checkpoint:

            best_val_loss = checkpoint[
                "val_loss"
            ]

        print(
            f"✅ Retomando treino "
            f"da época {start_epoch}"
        )

    else:

        print("\nNenhum checkpoint encontrado.")
        print("Treino começando do zero.")

    # ========================================================
    # FIXED ATTRIBUTES
    # ========================================================

    fixed_attributes = None

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs
    ):

        logging.info(
            f"A iniciar época {epoch}"
        )

        model.train()
        attribute_embedder.train()

        pbar = tqdm(
            train_loader,
            desc=f"Treino {epoch}/{args.epochs}"
        )

        epoch_losses = []

        # ====================================================
        # TRAIN STEP
        # ====================================================

        for batch in pbar:

            latents = batch[0].to(
                device,
                non_blocking=True
            )

            attributes = batch[1].to(
                device,
                non_blocking=True
            )

            if fixed_attributes is None:

                fixed_attributes = (
                    attributes[:16].clone()
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast("cuda"):

                # ============================================
                # CONDITIONING
                # ============================================

                context = attribute_embedder(
                    attributes
                )

                # ============================================
                # CLASSIFIER FREE GUIDANCE
                # ============================================

                if torch.rand(1).item() < 0.1:

                    context = torch.zeros_like(
                        context
                    )

                # ============================================
                # TIMESTEPS
                # ============================================

                t = diffusion.sample_timesteps(
                    latents.shape[0]
                ).to(device)

                # ============================================
                # NOISE
                # ============================================

                z_t, noise = diffusion.noise_images(
                    latents,
                    t
                )

                # ============================================
                # PREDICTION
                # ============================================

                predicted_noise = model(
                    z_t,
                    t,
                    context=context
                )

                # ============================================
                # LOSS
                # ============================================

                loss = mse(
                    predicted_noise,
                    noise
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            # ================================================
            # EMA
            # ================================================

            update_ema(
                ema_model,
                model
            )

            update_ema(
                ema_attribute_embedder,
                attribute_embedder
            )

            epoch_losses.append(
                loss.item()
            )

            pbar.set_postfix(
                MSE=loss.item()
            )

            board.log_scalar(
                "Loss/Batch",
                loss.item(),
                global_step
            )

            global_step += 1

        avg_loss = (
            sum(epoch_losses)
            / len(epoch_losses)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()
        attribute_embedder.eval()

        val_losses = []

        with torch.no_grad():

            val_pbar = tqdm(
                val_loader,
                desc=(
                    f"Validação "
                    f"{epoch}/{args.epochs}"
                ),
                leave=False
            )

            for batch in val_pbar:

                latents = batch[0].to(
                    device,
                    non_blocking=True
                )

                attributes = batch[1].to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast("cuda"):

                    context = (
                        attribute_embedder(
                            attributes
                        )
                    )

                    t = (
                        diffusion
                        .sample_timesteps(
                            latents.shape[0]
                        )
                        .to(device)
                    )

                    z_t, noise = (
                        diffusion.noise_images(
                            latents,
                            t
                        )
                    )

                    predicted_noise = model(
                        z_t,
                        t,
                        context=context
                    )

                    val_loss = mse(
                        predicted_noise,
                        noise
                    )

                val_losses.append(
                    val_loss.item()
                )

        avg_val_loss = (
            sum(val_losses)
            / len(val_losses)
        )

        print(f"\n📊 Época {epoch}")

        print(
            f"Train Loss: "
            f"{avg_loss:.6f}"
        )

        print(
            f"Val Loss:   "
            f"{avg_val_loss:.6f}"
        )

        # ====================================================
        # TENSORBOARD
        # ====================================================

        board.log_scalar(
            "Loss/Epoch",
            avg_loss,
            epoch
        )

        board.log_scalar(
            "Loss/Validation",
            avg_val_loss,
            epoch
        )

        # ====================================================
        # CHECKPOINT
        # ====================================================

        checkpoint = {

            "epoch":
                epoch,

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

            "val_loss":
                avg_val_loss
        }

        torch.save(
            checkpoint,
            ckpt_path
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                checkpoint,
                best_ckpt_path
            )

            print(
                f"✅ Novo melhor modelo salvo "
                f"(Val Loss: "
                f"{avg_val_loss:.6f})"
            )

        # ====================================================
        # INFERENCE
        # ====================================================

        if (
            epoch % 25 == 0
            or epoch == args.epochs - 1
        ):

            print("\n🎨 Gerando imagens...")

            ema_model.eval()
            ema_attribute_embedder.eval()

            with torch.no_grad():

                context_teste = (
                    ema_attribute_embedder(
                        fixed_attributes.to(
                            device
                        )
                    )
                )

                sampled_latents = (
                    diffusion.sample(
                        ema_model,
                        n=16,
                        context=context_teste
                    )
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
                ),

                nrow=4
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

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    print("\n🧪 Avaliando no conjunto de teste...")

    model.eval()
    attribute_embedder.eval()

    test_losses = []

    with torch.no_grad():

        for batch in tqdm(test_loader):

            latents = batch[0].to(
                device,
                non_blocking=True
            )

            attributes = batch[1].to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast("cuda"):

                context = attribute_embedder(
                    attributes
                )

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

                test_loss = mse(
                    predicted_noise,
                    noise
                )

            test_losses.append(
                test_loss.item()
            )

    avg_test_loss = (
        sum(test_losses)
        / len(test_losses)
    )

    print(
        f"\n🧪 Test Loss Final: "
        f"{avg_test_loss:.6f}"
    )

    board.log_scalar(
        "Loss/Test",
        avg_test_loss,
        0
    )

    board.close()


# ============================================================
# MAIN
# ============================================================

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