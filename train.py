import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import argparse
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast 

from utils.utils import get_data, save_images, setup_logging
from models.unet import UNet
from diffusion.ddpm import Diffusion

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # 1. Enable cuDNN Benchmark (Huge speedup for fixed size images)
    torch.backends.cudnn.benchmark = True 
    
    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "ckpt.pt")

    model = UNet().to(device)
    ema_model = deepcopy(model)
    ema_decay = 0.9999
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # 2. Initialize GradScaler for Mixed Precision
    scaler = GradScaler()
    
    start_epoch = 0

    if os.path.exists(ckpt_path):
        print(f"🔄 Checkpoint encontrado em: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'ema_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Treino retomado da época {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_losses = []

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            
            optimizer.zero_grad()

            with autocast():
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            epoch_losses.append(loss.item())
            pbar.set_postfix(MSE=loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n📊 Época {epoch} - Loss Médio: {avg_loss:.6f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), 
            "loss": avg_loss,
        }
        torch.save(checkpoint, ckpt_path)

        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"🎨 Gerando imagens de teste...")
            sampled_images = diffusion.sample(ema_model, n=16)
            save_images(sampled_images, os.path.join(results_dir, f"{epoch}.jpg"))
            torch.save(checkpoint, os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDPM_CIFAR10", help="Nome do run")  
    parser.add_argument('--epochs', type=int, default=1700, help="Total de épocas")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size") 
    parser.add_argument('--image_size', type=int, default=32, help="Resolução") 
    parser.add_argument('--device', type=str, default="cuda", help="Device")
    parser.add_argument('--lr', type=float, default=2e-4, help="LR")
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':

    main()
