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
from utils.utils import get_data, save_images, setup_logging
from diffusion.conditional_ddpm import Diffusion_conditional

# --- NOVOS IMPORTS ---
from models.unet_conditional import UNet_cond
from models.modules import LatentConditionProjector
from models.vae import VAE # Assumindo que guardou a sua VAE aqui

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp():
    """Inicializa o comunicador Multi-Node e Multi-GPU"""
    # NCCL é o backend padrão e mais rápido para GPUs NVIDIA
    dist.init_process_group(backend="nccl")
    
    # LOCAL_RANK: ID da GPU no computador atual (será 0 nos dois PCs)
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # RANK: ID global na rede (PC 1 será 0, PC 2 será 1)
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank


def update_ema(ema_model, model, decay=0.9995):
    """Atualiza os pesos do EMA de forma suave"""
    ema_model.eval()
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1 - decay))

def train(args):
    local_rank, global_rank = setup_ddp()
    is_master = (global_rank == 0)
    
    if is_master:
        setup_logging(args.run_name)
        print("\n🚀 LDM Condicional - Modo Multi-GPU (DDP)")

    dataloader, sampler = get_data(args, is_distributed=True)
    
    # --- 1. CONFIGURAÇÃO DO ESPAÇO LATENTE ---
    latent_dim = 4
    context_dim = 512
    
    # Inicia VAE (Congelada e SEM DDP)
    vae = VAE(in_channels=3, latent_dim=latent_dim).to(local_rank)
    # vae.load_state_dict(torch.load("sua_vae.pt", map_location=f"cuda:{local_rank}"))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Inicia UNet e Projetor
    model = UNet_cond(in_channels=latent_dim, out_channels=latent_dim, context_dim=context_dim).to(local_rank)
    projector = LatentConditionProjector(latent_dim=latent_dim, context_dim=context_dim).to(local_rank)
    
    diffusion = Diffusion_conditional(img_size=args.image_size // 8, device=local_rank)
    
    ema_model = deepcopy(model).eval() if is_master else None

    # --- 2. ENVOLVER TREINÁVEIS NO DDP ---
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    projector = DDP(projector, device_ids=[local_rank], gradient_as_bucket_view=True)

    # --- 3. OTIMIZADOR DUPLO ---
    optimizer = optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), 
        lr=args.lr
    )
    
    start_epoch = 0
    # (Lógica de resume_ckpt mantida como a sua, mas lembre de carregar o projector_state_dict também)

    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader) if is_master else dataloader
        
        for images, _ in pbar:
            images = images.to(local_rank)
            
            # --- 4. FLUXO DO LATENT DIFFUSION ---
            with torch.no_grad():
                mu, _ = vae.encode(images)
                z_target = mu * 0.18215 # Fator de escala
            
            # Gera a condição
            context = projector(z_target)
            
            t = torch.randint(low=1, high=diffusion.noise_steps, size=(images.shape[0],)).to(local_rank)
            
            # Adiciona ruído ao LATENTE
            z_t, noise = diffusion.noise_images(z_target, t)
            
            # Predição passando o CONTEXTO
            predicted_noise = model(z_t, t, context=context)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if is_master:
                update_ema(ema_model, model.module)
                pbar.set_postfix(MSE=loss.item())
                
        if is_master and epoch % 10 == 0:
            # (Lógica de salvar ckpt igual à sua, adicionando 'projector_state_dict': projector.module.state_dict())
            
            # Lógica de inferência para imagens de teste precisará usar a VAE para decodificar, 
            # exatamente como fizemos no script de treino de uma GPU!
            pass

    dist.destroy_process_group()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDPM_MultiNode", help="Nome da pasta")  
    parser.add_argument('--epochs', type=int, default=2500, help="Total de épocas")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size POR GPU") 
    parser.add_argument('--image_size', type=int, default=32, help="Resolução da imagem") 
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning Rate")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Caminho do .pt antigo")

    args = parser.parse_args()
    train(args)