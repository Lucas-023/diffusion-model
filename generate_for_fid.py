import os
import torch
import argparse
from tqdm import tqdm
from torchvision.utils import save_image

# Suas importações
from models.unet import UNet
from diffusion.ddpm import Diffusion

def generate_images(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando geração no dispositivo: {device}")

    # 1. Configura as Pastas
    output_dir = os.path.join("fid_samples", args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Inicializa o Modelo e a Difusão
    model = UNet(image_size=args.image_size).to(device)
    diffusion = Diffusion(img_size=args.image_size, device=device)

    # 3. Carrega OBRIGATORIAMENTE o modelo EMA do Checkpoint
    ckpt_path = os.path.join("models", args.run_name, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Carrega os pesos do EMA (crucial para bater a qualidade do paper)
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("✅ Pesos do modelo EMA carregados com sucesso!")
    else:
        print("⚠️ AVISO: ema_state_dict não encontrado. Usando pesos normais.")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    # 4. Loop de Geração em Batches
    total_generated = 0
    batches = args.num_images // args.batch_size
    remainder = args.num_images % args.batch_size

    print(f"Gerando {args.num_images} imagens usando DDPM padrão (1000 passos)...")
    print(f"Isso vai demorar. Pegue um café! ☕\n")

    with torch.no_grad():
        for i in tqdm(range(batches)):
            # Usando a função sample original (DDPM)
            sampled_images = diffusion.sample(model, n=args.batch_size)
            
            # Salva cada imagem do batch individualmente
            for j in range(args.batch_size):
                img_idx = total_generated + j
                save_image(sampled_images[j], os.path.join(output_dir, f"img_{img_idx:05d}.png"))
            
            total_generated += args.batch_size

        # Gera o restinho se a divisão não for exata
        if remainder > 0:
            sampled_images = diffusion.sample(model, n=remainder)
            for j in range(remainder):
                img_idx = total_generated + j
                save_image(sampled_images[j], os.path.join(output_dir, f"img_{img_idx:05d}.png"))

    print(f"\n🎉 Geração concluída! Imagens salvas em: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True, help="Nome da pasta do modelo treinado")
    parser.add_argument('--image_size', type=int, default=32, help="Tamanho da imagem (32 para CIFAR-10)")
    parser.add_argument('--batch_size', type=int, default=128, help="Imagens geradas por vez na GPU")
    parser.add_argument('--num_images', type=int, default=10000, help="Total de imagens para gerar (Paper usou 50000)")
    args = parser.parse_args()
    
    generate_images(args)