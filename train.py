import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

# Importando nossos módulos criados anteriormente
from utils.utils import get_data, save_images, setup_logging
from models.unet import unet
from diffusion.ddpm import Diffusion

def train(args):
    # 1. Configuração Inicial
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # Instancia o Modelo e o Gerente
    model = unet().to(device) # Certifique-se que sua classe chama 'unet' ou 'UNet'
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)

    # ---------------------------------------------------------
    # 2. Lógica de RESUME (Carregar Checkpoint se existir)
    # ---------------------------------------------------------
    start_epoch = 0
    # Define o caminho onde o arquivo .pt vai ficar
    ckpt_path = os.path.join("models", args.run_name, "ckpt.pt")

    if os.path.exists(ckpt_path):
        print(f"🔄 Checkpoint encontrado em: {ckpt_path}")
        print("⏳ Carregando pesos e estado do otimizador...")
        
        # Carrega o arquivo para o device correto
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Restaura o modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restaura o otimizador (CRUCIAL para manter o aprendizado suave)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Define a época de início como a próxima após a salva
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"✅ Treino retomado da época {start_epoch}!")
    else:
        print("🚀 Iniciando treino do zero.")

    print(f"Treinando no device: {device}")

    # ---------------------------------------------------------
    # 3. Loop de Treino (Agora começa de start_epoch)
    # ---------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}:")
        pbar = tqdm(dataloader)

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        # ---------------------------------------------------------
        # 4. Salvar Checkpoint Completo (Sobrescreve o anterior)
        # ---------------------------------------------------------
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }
        torch.save(checkpoint, ckpt_path)
        # ---------------------------------------------------------

        # Salva imagens de teste e um backup do modelo a cada 10 épocas
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # Salva imagem
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            
            # (Opcional) Salva uma cópia permanente desse peso específico
            torch.save(checkpoint, os.path.join("models", args.run_name, f"ckpt_epoch_{epoch}.pt"))

def main():
    parser = argparse.ArgumentParser()
    
    # Mudei o nome padrão para refletir que é CIFAR
    parser.add_argument('--run_name', type=str, default="DDPM_CIFAR10", help="Nome da pasta para salvar")  
    parser.add_argument('--epochs', type=int, default=500, help="Quantas vezes passar pelo dataset")
    parser.add_argument('--batch_size', type=int, default=32, help="Quantas imagens processar por vez") 
    parser.add_argument('--image_size', type=int, default=64, help="Tamanho da imagem") 
    parser.add_argument('--device', type=str, default="cuda", help="cuda ou cpu")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning Rate")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == '__main__':
    main()
  
