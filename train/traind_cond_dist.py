import os
import sys

# Força o Python a enxergar a pasta principal do projeto (diffusion-model)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from models.modules import LatentConditionProjector
from vae.modules import VAE 

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import contextlib

def setup_ddp():
    """Inicializa o comunicador Multi-Node e Multi-GPU"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank

def update_ema(ema_model, model, decay=0.9999): 
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
        print("\n🚀 LDM Condicional (Image-to-Image) - Modo Multi-GPU")
        board = Board(run_name=args.run_name, enabled=True)
        global_step = 0

    # AVISO: A partir de agora, o get_data deve retornar os LATENTES (.pt) e não as imagens (.jpg)
    dataloader, sampler = get_data(args, is_distributed=True)
    
    # --- CONFIGURAÇÃO DO ESPAÇO LATENTE ---
    latent_dim = 4
    context_dim = 512
    
    # Inicia VAE (Apenas para descompressão no final da época pelo PC Mestre)
    vae = VAE(in_channels=3, latent_dim=latent_dim).to(local_rank)
    # vae.load_state_dict(torch.load("sua_vae.pt", map_location=f"cuda:{local_rank}"))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Inicia UNet e Projetor (O Projetor agora precisa aceitar entrada espacial 4x32x32)
    model = UNet_cond(in_channels=latent_dim, out_channels=latent_dim, context_dim=context_dim).to(local_rank)
    projector = LatentConditionProjector(latent_dim=latent_dim, context_dim=context_dim).to(local_rank)
    
    diffusion = Diffusion_conditional(img_size=args.image_size // 8, device=local_rank)
    ema_model = deepcopy(model).eval() if is_master else None

    # ENVOLVER TREINÁVEIS NO DDP
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    projector = DDP(projector, device_ids=[local_rank], gradient_as_bucket_view=True)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), 
        lr=args.lr,
        weight_decay=1e-4 
    )
    
    # --- SCHEDULERS ---
    epocas_warmup = 10 
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=epocas_warmup)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - epocas_warmup), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[epocas_warmup])    

    scaler = GradScaler()
    accumulation_steps = 4
    start_epoch = 0

    # --- LÓGICA DE RESUME ---
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        if is_master:
            print(f"🔄 Carregando checkpoint: {args.resume_ckpt}")
            
        checkpoint = torch.load(args.resume_ckpt, map_location=f"cuda:{local_rank}")
        model.module.load_state_dict(checkpoint['model_state_dict'])
        projector.module.load_state_dict(checkpoint['projector_state_dict'])
        
        if is_master and ema_model is not None:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        if is_master:
            print(f"✅ Treino retomado a partir da época {start_epoch}!")
            
    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader) if is_master else dataloader
        
        epoch_losses = [] 
        
        # MUDANÇA 1: O loop agora recebe 'latents' direto do dataloader
        for i, (latents, _) in enumerate(pbar):
            latents = latents.to(local_rank)
            
            is_accumulating = (i + 1) % accumulation_steps != 0 and (i + 1) != len(dataloader)
            
            sync_model = model.no_sync() if is_accumulating else contextlib.nullcontext()
            sync_proj = projector.no_sync() if is_accumulating else contextlib.nullcontext()
            
            with sync_model, sync_proj:
                with autocast():
                    # MUDANÇA 2: A VAE foi removida daqui. Escalonamos o latente diretamente.
                    z_target = latents * 0.18215 
                    
                    # MUDANÇA 3: Self-Conditioning. O projetor lê a própria imagem alvo (latente)
                    context = projector(z_target)
                    
                    t = torch.randint(low=1, high=diffusion.noise_steps, size=(latents.shape[0],)).to(local_rank)
                    
                    z_t, noise = diffusion.noise_images(z_target, t)
                    predicted_noise = model(z_t, t, context=context)
                    
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
            
            if not is_accumulating:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if is_master:
                    update_ema(ema_model, model.module)
                    loss_display = loss.item() * accumulation_steps 
                    pbar.set_postfix(MSE=loss_display)
                    board.log_scalar("Loss/Batch", loss_display, global_step)
                    epoch_losses.append(loss_display)
                    global_step += 1     
                           
        scheduler.step()
                
        # 3. SALVAR CHECKPOINT E INFERÊNCIA
        if is_master:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            lr_atual = optimizer.param_groups[0]['lr']
            
            print(f"\n📊 Época {epoch} - Loss Médio: {avg_loss:.6f} | LR Atual: {lr_atual:.6f}")
            
            board.log_scalar("Metricas/Loss_Epoca", avg_loss, epoch)
            board.log_scalar("Metricas/Learning_Rate", lr_atual, epoch)
            
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(results_dir, exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'projector_state_dict': projector.module.state_dict(), 
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_dir, "ckpt.pt"))
                
                # --- INFERÊNCIA COM LATENT CACHE ---
                print(f"🎨 A gerar imagens de teste no TensorBoard...")
                latentes_reais = latents[:16] # Pega 16 latentes do batch
                
                with torch.no_grad():
                    # 1. Decodifica os latentes reais para mostrar a "Condição" no TensorBoard
                    imagens_reais = vae.decode(latentes_reais / 0.18215)
                    grid_condicao = make_grid(imagens_reais, nrow=4, normalize=True, value_range=(-1, 1))
                    board.log_image("Visualizacao/Condicao_Real", grid_condicao, epoch)
                    
                    # 2. Gera os contextos baseados nos latentes e pede para a UNet gerar do zero
                    context_teste = projector.module(latentes_reais * 0.18215) 
                    sampled_latents = diffusion.sample(ema_model, n=16, context=context_teste)
                    
                    # 3. Decodifica as imagens geradas pela UNet
                    sampled_images = vae.decode(sampled_latents / 0.18215)
                
                save_images(sampled_images, os.path.join(results_dir, f"{epoch}.jpg"))
                grid_gerada = make_grid(sampled_images, nrow=4, normalize=True, value_range=(-1, 1))
                board.log_image("Visualizacao/Imagem_Gerada", grid_gerada, epoch)

    if is_master:
        board.close() 
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDPM_MultiNode", help="Nome da pasta")  
    parser.add_argument('--epochs', type=int, default=2500, help="Total de épocas")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size POR GPU") 
    parser.add_argument('--image_size', type=int, default=256, help="Resolução Original (256)") 
    parser.add_argument('--lr', type=float, default=8e-4, help="Learning Rate")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Caminho do .pt antigo")

    args = parser.parse_args()
    train(args)