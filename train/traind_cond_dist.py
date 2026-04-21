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

# --- NOVOS IMPORTS ---
from models.unet_conditional import UNet_cond
from models.modules import LatentConditionProjector
from vae.modules import VAE # Assumindo que guardou a sua VAE aqui

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


def update_ema(ema_model, model, decay=0.9999): # 0.9999 é melhor para LDM
    """Atualiza os pesos do EMA de forma suave"""
    ema_model.eval()
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1 - decay))

def train(args):
    local_rank, global_rank = setup_ddp()
    is_master = (global_rank == 0)
    
    # 1. INICIALIZAÇÃO TENSORBOARD E LOGS (SÓ NO MESTRE)
    if is_master:
        setup_logging(args.run_name)
        print("\n🚀 LDM Condicional - Modo Multi-GPU (DDP)")
        board = Board(run_name=args.run_name, enabled=True)
        global_step = 0

    dataloader, sampler = get_data(args, is_distributed=True)
    
    # --- CONFIGURAÇÃO DO ESPAÇO LATENTE ---
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

    # ENVOLVER TREINÁVEIS NO DDP
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    projector = DDP(projector, device_ids=[local_rank], gradient_as_bucket_view=True)

    # OTIMIZADOR DUPLO
    optimizer = optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), 
        lr=args.lr,
        weight_decay=1e-4 # Um pouco de regularização ajuda no LDM
    )
    
    # --- SCHEDULER: WARMUP + COSINE ANNEALING ---
    epocas_warmup = 10 # O LR vai subir suavemente durante as primeiras 10 épocas
    
    # 1. Sobe de 1% (0.01) do LR até 100% nas primeiras 10 épocas
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=epocas_warmup)
    
    # 2. Depois das 10 épocas, desce em formato de curva de cosseno até um valor mínimo (ex: 1e-6)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - epocas_warmup), eta_min=1e-6)
    
    # 3. Junta os dois sequencialmente
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[epocas_warmup])    

    scaler = GradScaler()
    accumulation_steps = 4
    
    start_epoch = 0
    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader) if is_master else dataloader
        
        epoch_losses = [] # Para calcular a média no final da época
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(local_rank)
            
            # --- LÓGICA DE ACÚMULO DE GRADIENTES ---
            # Só sincroniza as placas de rede se for o último passo do acúmulo ou o último batch do dataloader
            is_accumulating = (i + 1) % accumulation_steps != 0 and (i + 1) != len(dataloader)
            
            # Gerenciadores de contexto para bloquear a sincronização de rede
            sync_model = model.no_sync() if is_accumulating else contextlib.nullcontext()
            sync_proj = projector.no_sync() if is_accumulating else contextlib.nullcontext()
            
            # O processamento pesado entra aqui
            with sync_model, sync_proj:
                # AUTOCAST: Converte operações pesadas (FP32) para leves (FP16)
                with autocast():
                    with torch.no_grad():
                        mu, _ = vae.encode(images)
                        z_target = mu * 0.18215 # Fator de escala
                    
                    context = projector(z_target)
                    t = torch.randint(low=1, high=diffusion.noise_steps, size=(images.shape[0],)).to(local_rank)
                    
                    z_t, noise = diffusion.noise_images(z_target, t)
                    predicted_noise = model(z_t, t, context=context)
                    
                    # Calcula o erro
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                    # Divide o loss para compensar as somas futuras
                    loss = loss / accumulation_steps
                
                # Backward pass escalonado em FP16 (Ainda bloqueado para não ir à rede)
                scaler.scale(loss).backward()
            
            # --- ATUALIZAÇÃO DOS PESOS (AGORA SIM VAI PELA REDE) ---
            if not is_accumulating:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 2. LOGS DURANTE O BATCH (SÓ NO MESTRE)
                if is_master:
                    update_ema(ema_model, model.module)
                    # Multiplicamos o loss de volta só para o gráfico do Tensorboard ficar legível
                    loss_display = loss.item() * accumulation_steps 
                    pbar.set_postfix(MSE=loss_display)
                    board.log_scalar("Loss/Batch", loss_display, global_step)
                    epoch_losses.append(loss_display)
                    global_step += 1                
        # --- ATUALIZA O LEARNING RATE (Em todas as GPUs para manter o sincronismo) ---
        scheduler.step()
                
        # 3. SALVAR CHECKPOINT E INFERÊNCIA (SÓ NO MESTRE)
        if is_master:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            lr_atual = optimizer.param_groups[0]['lr']
            
            print(f"\n📊 Época {epoch} - Loss Médio: {avg_loss:.6f} | LR Atual: {lr_atual:.6f}")
            
            # Mais logs de acompanhamento geral
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
                
                # --- INFERÊNCIA NO ESPAÇO LATENTE ---
                print(f"🎨 A gerar imagens de teste no TensorBoard...")
                condicoes_reais = images[:16] # Pega 16 imagens do batch para servir de condição
                
                # Regista as imagens reais no TensorBoard para compararmos com as geradas
                grid_condicao = make_grid(condicoes_reais, nrow=4, normalize=True, value_range=(-1, 1))
                board.log_image("Visualizacao/Condicao_Real", grid_condicao, epoch)
                
                with torch.no_grad():
                    mu_cond, _ = vae.encode(condicoes_reais)
                    context_teste = projector.module(mu_cond * 0.18215) 
                    
                    sampled_latents = diffusion.sample(ema_model, n=16, context=context_teste)
                    
                    sampled_latents = sampled_latents / 0.18215
                    sampled_images = vae.decode(sampled_latents)
                
                # Salva no disco e regista as imagens geradas no TensorBoard
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
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning Rate")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Caminho do .pt antigo")

    args = parser.parse_args()
    train(args)