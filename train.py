import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from copy import deepcopy

# Importando seus módulos
from utils.utils import get_data, save_images, setup_logging
from models.unet import UNet
from diffusion.ddpm import Diffusion

def train(args):
    # 1. Configuração Inicial e Diretórios
    setup_logging(args.run_name)
    device = args.device
    print("\n" + "="*60)
    print("🔍 DEBUG: Verificando dataset no treino")
    print("="*60)
    # =========================
    
    dataloader = get_data(args)
    
    # ===== ADICIONE AQUI =====
    print(f"📊 Total de batches no dataloader: {len(dataloader)}")
    print(f"📊 Tamanho do dataset: {len(dataloader.dataset):,}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Cálculo: {len(dataloader.dataset)} / {args.batch_size} = {len(dataloader.dataset)/args.batch_size:.2f}")
    print("="*60 + "\n")    
    # Define caminhos universais (independente do PC)
    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Caminho do checkpoint principal
    ckpt_path = os.path.join(save_dir, "ckpt.pt")

    # 2. Instanciação do Modelo Principal e EMA
    model = UNet().to(device)
    
    # ✅ EMA: Cria versão suavizada do modelo
    ema_model = deepcopy(model)
    ema_decay = 0.9999
    print(f"✅ EMA inicializado com decay={ema_decay}")
    
    # Otimizador
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    start_epoch = 0

    # 3. Lógica de RESUME Inteligente
    if os.path.exists(ckpt_path):
        print(f"🔄 Checkpoint encontrado em: {ckpt_path}")
        print("⏳ Carregando estado completo do treino...")
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Restaura Modelo (Pesos)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restaura EMA (se existir no checkpoint)
        if 'ema_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            print("✅ EMA carregado do checkpoint")
        else:
            # Checkpoint antigo sem EMA - inicializa do modelo atual
            ema_model = deepcopy(model)
            print("⚠️ Checkpoint antigo sem EMA - inicializando do modelo atual")
        
        # Restaura Otimizador
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restaura Scheduler
        #if 'scheduler_state_dict' in checkpoint:
           #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Define a próxima época
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"✅ Treino retomado da época {start_epoch}")
    else:
        print(f"🚀 Nenhum checkpoint encontrado em '{save_dir}'. Iniciando do ZERO.")

    print(f"Treinando no device: {device}")
    print(f"🛡️ Gradient clipping ativado (max_norm=1.0)")

    # 4. Loop de Treino
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        # ✅ Tracking de loss médio da época
        epoch_losses = []

        # Loop dos Batches
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            
            # ✅ GRADIENT CLIPPING - Previne explosão de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # ✅ ATUALIZA EMA - Versão suavizada do modelo
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            # Tracking
            epoch_losses.append(loss.item())

            # Mostra o Loss e o LR atual na barra de progresso
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(MSE=loss.item(), LR=f"{current_lr:.6f}")

        # FIM DA ÉPOCA
        
        # ✅ Calcula e mostra loss médio da época
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        last_batch_loss = loss.item()
        print(f"\n📊 Época {epoch} - Loss Médio: {avg_loss:.6f} | Último Batch: {last_batch_loss:.6f}")
        
        # Atualiza o Scheduler
        #scheduler.step()

        # Salva o Checkpoint Completo
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),  # ✅ Salva EMA
            "optimizer_state_dict": optimizer.state_dict(),
            #"scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,  # ✅ Salva loss médio, não último batch
            "last_batch_loss": last_batch_loss,  # Info adicional
        }
        torch.save(checkpoint, ckpt_path)

        # Salva Imagens e Backup a cada 10 épocas
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"🎨 Gerando imagens de teste da época {epoch}...")
            
            # ✅ USA EMA MODEL para gerar imagens (muito melhor!)
            sampled_images = diffusion.sample(ema_model, n=images.shape[0])
            save_images(sampled_images, os.path.join(results_dir, f"{epoch}.jpg"))
            
            # Salva uma cópia "eterna" dessa época específica
            torch.save(checkpoint, os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt"))
            
            print(f"✅ Checkpoint e imagens da época {epoch} salvos!")

def main():
    parser = argparse.ArgumentParser()
    
    # Hyperparâmetros
    parser.add_argument('--run_name', type=str, default="DDPM_CIFAR10", help="Nome da pasta do run")  
    parser.add_argument('--epochs', type=int, default=1700, help="Total de épocas")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size") 
    parser.add_argument('--image_size', type=int, default=32, help="Resolução da imagem") 
    parser.add_argument('--device', type=str, default="cuda", help="cuda ou cpu")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning Rate Inicial")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 DDPM Training - Versão com EMA + Gradient Clipping")
    print("="*60)
    print(f"📦 Run: {args.run_name}")
    print(f"📊 Épocas: {args.epochs}")
    print(f"🎯 Batch Size: {args.batch_size}")
    print(f"🖼️  Image Size: {args.image_size}x{args.image_size}")
    print(f"💻 Device: {args.device}")
    print(f"📈 Learning Rate: {args.lr}")
    print("="*60 + "\n")
    
    train(args)

if __name__ == '__main__':
    main()
