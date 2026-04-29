import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import argparse
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast 
from torchvision.utils import make_grid 
from torch.utils.data import Dataset, DataLoader # <-- Adicionado aqui

from board import Board
# Removemos o get_data daqui, mantemos só os utilitários de log e imagem
from utils.utils_celeba import save_images, setup_logging 
from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
from models.modules import LatentConditionProjector
from vae.modules import VAE 

# ==========================================
# DATASET LOCAL (Sem mexer nos outros arquivos)
# ==========================================
class LocalLatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = torch.load(latent_path, map_location='cpu')
        return latent, 0 

# ==========================================
# LOOP DE TREINO
# ==========================================
def train(args):
    setup_logging(args.run_name)
    device = args.device
    
    # 1. Configurando os dados diretamente aqui
    # Ele vai procurar a pasta 'latent_cache' no mesmo local onde você roda o comando no terminal
    latent_dir = "./cache_latent" 
    if not os.path.exists(latent_dir):
        raise FileNotFoundError(f"Não encontrei a pasta de cache em: {os.path.abspath(latent_dir)}")
        
    dataset = LocalLatentDataset(latent_dir)
    # Dataloader simples para 1 GPU:
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    torch.backends.cudnn.benchmark = True 
    
    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "ckpt.pt")

    # 2. Inicialização dos Modelos
    latent_dim = 4
    context_dim = 512

    # A. Carrega VAE (Apenas para gerar as imagens de teste no final)
    vae = VAE(in_channels=3, latent_dim=latent_dim).to(device)
    # Lembre-se de colocar o caminho real da sua VAE aqui:
    vae.load_state_dict(torch.load("vae/vae_epoch_62.pt", map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False 

    # B. U-Net Condicional e Projetor
    model = UNet_cond(in_channels=latent_dim, out_channels=latent_dim, context_dim=context_dim).to(device)
    projector = LatentConditionProjector(latent_dim=latent_dim, context_dim=context_dim).to(device)
    
    ema_model = deepcopy(model)
    ema_projector = deepcopy(projector)
    ema_model.eval()
    ema_projector.eval()

    for p in ema_model.parameters():
        p.requires_grad = False

    for p in ema_projector.parameters():
        p.requires_grad = False
    ema_decay = 0.9999
    
    optimizer = optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), 
        lr=args.lr
    )
    
    mse = nn.MSELoss()
    diffusion = Diffusion_conditional(img_size=args.image_size // 8, device=device) 
    
    scaler = GradScaler()
    start_epoch = 0

    board = Board(run_name=args.run_name, enabled=True)
    global_step = 0 

    fixed_latents, _ = next(iter(dataloader))
    fixed_latents = fixed_latents[:16].to(device) * 0.18215

    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"A iniciar época {epoch}:")
        pbar = tqdm(dataloader, desc=f"Época {epoch}/{args.epochs}")
        epoch_losses = []

        for i, (latents, _) in enumerate(pbar):
            latents = latents.to(device)
            t = diffusion.sample_timesteps(latents.shape[0]).to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # Prepara o alvo (Já está encodado, apenas escalamos)
                z_target = latents * 0.18215
                
                # Projetor cria o Contexto
                context = projector(z_target) 
                assert context.ndim == 3
                assert context.shape[1] == context_dim

                if torch.rand(1, device=device).item() < 0.1:
                    context = torch.zeros_like(context)

                # Adiciona ruído
                z_t, noise = diffusion.noise_images(z_target, t)
                
                # UNet prevê o ruído
                predicted_noise = model(z_t, t, context=context)
                
                loss = mse(noise, predicted_noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            
            # EMA
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
                for ema_param, param in zip(ema_projector.parameters(), projector.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            epoch_losses.append(loss.item())
            pbar.set_postfix(MSE=loss.item())
            
            board.log_scalar("Loss/Batch", loss.item(), global_step)
            global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n📊 Época {epoch} - Loss Médio: {avg_loss:.6f}")
        board.log_scalar("Loss/Epoca", avg_loss, epoch)

        # 3. Salvamento e Imagens de Teste
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "ema_projector_state_dict": ema_projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), 
            "loss": avg_loss,
        }
        torch.save(checkpoint, ckpt_path)

        # Salva último checkpoint (resume)
        torch.save(checkpoint, ckpt_path)

        # Salva melhor checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt_path = os.path.join(save_dir, "best_ckpt.pt")
            torch.save(checkpoint, best_ckpt_path)
            print(f"🏆 Novo melhor checkpoint salvo! Loss: {avg_loss:.6f}")

        # Salva snapshots periódicos
        if epoch % 25 == 0 and epoch > 0:
            periodic_ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save(checkpoint, periodic_ckpt_path)
            print(f"📦 Checkpoint periódico salvo na época {epoch}")

        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"🎨 A gerar imagens de teste a partir do latente...")
            condicoes_latentes = fixed_latents
            
            with torch.no_grad():
                context_teste = ema_projector(condicoes_latentes)
                sampled_latents = diffusion.sample(ema_model, n=16, context=context_teste)
                sampled_latents = sampled_latents / 0.18215
                sampled_images = vae.decode(sampled_latents)
            
            save_images(sampled_images, os.path.join(results_dir, f"{epoch}.jpg"))
            grid = make_grid(sampled_images, nrow=4, normalize=True, value_range=(-1, 1))
            board.log_image("Geracao/Teste", grid, epoch)

    board.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="Latent_Diffusion_CelebA", help="Nome do run")  
    parser.add_argument('--epochs', type=int, default=1000, help="Total de épocas")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size") 
    parser.add_argument('--image_size', type=int, default=256, help="Resolução Original") 
    parser.add_argument('--device', type=str, default="cuda", help="Device")
    parser.add_argument('--lr', type=float, default=2e-4, help="LR")
    args = parser.parse_args()
    train(args)