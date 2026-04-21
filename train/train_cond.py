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

from board import Board
from utils.utils_celeba import get_data, save_images, setup_logging
from diffusion.conditional_ddpm import Diffusion_conditional

# --- NOVOS IMPORTS ---
from models.unet_conditional import UNet_cond
from models.modules import LatentConditionProjector
from vae.modules import VAE # Assumindo que guardou a sua VAE aqui

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    torch.backends.cudnn.benchmark = True 
    
    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "ckpt.pt")

    # ==========================================
    # 1. INICIALIZAÇÃO DOS MODELOS (VAE, UNet, Projetor)
    # ==========================================
    latent_dim = 4
    context_dim = 512

    # A. Carregar VAE e CONGELAR
    vae = VAE(in_channels=3, latent_dim=latent_dim).to(device)
    # Aqui deve carregar os pesos da VAE que já treinou
    vae.load_state_dict(torch.load("caminho/para/vae_treinada.pt"))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False # Congela a VAE

    # B. Inicializar U-Net Condicional e Projetor
    model = UNet_cond(in_channels=latent_dim, out_channels=latent_dim, context_dim=context_dim).to(device)
    projector = LatentConditionProjector(latent_dim=latent_dim, context_dim=context_dim).to(device)
    
    ema_model = deepcopy(model)
    ema_decay = 0.9999
    
    # C. O otimizador agora precisa treinar a UNet E o Projetor ao mesmo tempo!
    optimizer = optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), 
        lr=args.lr
    )
    
    mse = nn.MSELoss()
    # A resolução agora é a do espaço latente (ex: 256/8 = 32)
    diffusion = Diffusion_conditional(img_size=args.image_size // 8, device=device) 
    
    scaler = GradScaler()
    start_epoch = 0

    # Lógica de retoma de checkpoint omitida para brevidade (pode manter a sua original, 
    # apenas certifique-se de salvar/carregar também o state_dict do 'projector')

    # --- INICIALIZAÇÃO DO TENSORBOARD ---
    board = Board(run_name=args.run_name, enabled=True)
    global_step = 0 

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"A iniciar época {epoch}:")
        pbar = tqdm(dataloader, desc=f"Época {epoch}/{args.epochs}")
        epoch_losses = []

        # ==========================================
        # 2. LOOP DE TREINO (Fluxo Latente)
        # ==========================================
        for i, (images, _) in enumerate(pbar):
            images = images.to(device) # Imagem real (ex: 3x256x256)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            
            optimizer.zero_grad()

            with autocast():
                # A. Passa pela VAE (Sem gradientes para poupar memória)
                with torch.no_grad():
                    mu, _ = vae.encode(images)
                    # Usamos apenas a média (mu) para uma condição limpa e determinística
                    z_target = mu * 0.18215 # Fator de escala padrão do Stable Diffusion para estabilizar a variância
                
                # B. Gera o Contexto (A condição)
                # Como combinámos, vamos fazer auto-condicionamento usando a própria imagem
                context = projector(z_target) 

                # C. Processo de Difusão no Espaço Latente
                # Adicionamos ruído ao latente (z_target) e não aos píxeis
                z_t, noise = diffusion.noise_images(z_target, t)
                
                # A UNet agora recebe o latente ruidoso (z_t) e a condição (context)
                predicted_noise = model(z_t, t, context=context)
                
                loss = mse(noise, predicted_noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            
            # Atualização EMA
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            epoch_losses.append(loss.item())
            pbar.set_postfix(MSE=loss.item())
            
            board.log_scalar("Loss/Batch", loss.item(), global_step)
            global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n📊 Época {epoch} - Loss Médio: {avg_loss:.6f}")
        board.log_scalar("Loss/Epoca", avg_loss, epoch)

        # ==========================================
        # 3. GERAÇÃO DE IMAGENS DE TESTE E GUARDA
        # ==========================================
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(), # Salvar o projetor!
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), 
            "loss": avg_loss,
        }
        torch.save(checkpoint, ckpt_path)

        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"🎨 A gerar imagens de teste no espaço latente...")
            
            # --- NOTA IMPORTANTE PARA A INFERÊNCIA ---
            # Aqui pegamos em algumas imagens reais do batch para servirem de "condição"
            condicoes_reais = images[:16] 
            with torch.no_grad():
                mu_cond, _ = vae.encode(condicoes_reais)
                context_teste = projector(mu_cond * 0.18215)
                
                # A função sample precisará de aceitar o 'context_teste' (ver nota abaixo)
                sampled_latents = diffusion.sample(ema_model, n=16, context=context_teste)
                
                # Descomprime os latentes gerados de volta para píxeis
                sampled_latents = sampled_latents / 0.18215
                sampled_images = vae.decode(sampled_latents)
            
            save_images(sampled_images, os.path.join(results_dir, f"{epoch}.jpg"))
            
            grid = make_grid(sampled_images, nrow=4, normalize=True, value_range=(-1, 1))
            board.log_image("Geracao/Teste", grid, epoch)

    board.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="Latent_Diffusion_CelebA", help="Nome do run")  
    parser.add_argument('--epochs', type=int, default=1700, help="Total de épocas")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size") 
    parser.add_argument('--image_size', type=int, default=256, help="Resolução Original (em pixeis)") 
    parser.add_argument('--device', type=str, default="cuda", help="Device")
    parser.add_argument('--lr', type=float, default=1e-4, help="LR") # LR costuma ser menor em LDM
    args = parser.parse_args()
    train(args)