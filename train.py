import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import torch.optim.lr_scheduler as lr_scheduler

# Importando seus módulos
from utils.utils import get_data, save_images, setup_logging
from models.unet import unet  # Certifique-se que o nome da classe no arquivo é 'unet' ou 'UNet'
from diffusion.ddpm import Diffusion

def train(args):
    # 1. Configuração Inicial e Diretórios
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # Define caminhos universais (independente do PC)
    # Estrutura: ./models/DDPM_CIFAR10/
    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Caminho do checkpoint principal (o "save game" mais recente)
    ckpt_path = os.path.join(save_dir, "ckpt.pt")

    # 2. Instanciação
    model = unet().to(device)
    
    # Otimizador: Começa com o LR definido nos argumentos (padrão 3e-4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler: Redução suave do LR ao longo das épocas
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    start_epoch = 0

    # ---------------------------------------------------------
    # 3. Lógica de RESUME Inteligente
    # ---------------------------------------------------------
    if os.path.exists(ckpt_path):
        print(f"🔄 Checkpoint encontrado em: {ckpt_path}")
        print("⏳ Carregando estado completo do treino...")
        
        # Carrega tudo para o device correto (GPU/CPU)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Restaura Modelo (Pesos)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restaura Otimizador (Momentum e correções de gradiente)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restaura Scheduler (Sabe exatamente em qual passo do cosseno parou)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Define a próxima época
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"✅ Treino retomado da época {start_epoch}. O Momentum foi preservado!")
    else:
        print(f"🚀 Nenhum checkpoint encontrado em '{save_dir}'. Iniciando do ZERO.")

    print(f"Treinando no device: {device}")

    # ---------------------------------------------------------
    # 4. Loop de Treino
    # ---------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)

        # --- Loop dos Batches ---
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Mostra o Loss e o LR atual na barra de progresso
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(MSE=loss.item(), LR=f"{current_lr:.6f}")

        # --- FIM DA ÉPOCA ---
        
        # 1. Atualiza o Scheduler (CRUCIAL: Tem que ser uma vez por época)
        scheduler.step()

        # 2. Salva o Checkpoint Completo (Sobrescreve o arquivo ckpt.pt)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss.item(),
            "args": args # Opcional: salva os hyperparametros usados
        }
        torch.save(checkpoint, ckpt_path)

        # 3. Salva Imagens e Backup a cada 10 épocas
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # Gera imagens
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join(results_dir, f"{epoch}.jpg"))
            
            # Salva uma cópia "eterna" dessa época específica
            torch.save(checkpoint, os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt"))

def main():
    parser = argparse.ArgumentParser()
    
    # Hyperparâmetros
    parser.add_argument('--run_name', type=str, default="DDPM_CIFAR10", help="Nome da pasta do run")  
    parser.add_argument('--epochs', type=int, default=500, help="Total de épocas")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size") 
    parser.add_argument('--image_size', type=int, default=64, help="Resolução da imagem") 
    parser.add_argument('--device', type=str, default="cuda", help="cuda ou cpu")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning Rate Inicial")
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
