import os
import torch
import argparse
from models.unet import UNet
from diffusion.ddpm import Diffusion
from utils.utils import save_images

def generate(args):
    device = args.device
    print(f"🚀 Iniciando geração de imagens no device: {device}")
    
    # 1. Configura modelo e difusão
    model = UNet(image_size=args.image_size).to(device)
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # 2. Carrega o checkpoint
    # IMPORTANTE: Coloque o caminho exato do seu modelo da época 1525 aqui!
    ckpt_path = os.path.join("models", args.run_name, "ckpt.pt") 
    
    if not os.path.exists(ckpt_path):
        # Tenta procurar com o nome de backup da época se o principal não existir
        ckpt_path = os.path.join("models", args.run_name, f"ckpt_epoch_1520.pt")
        
    if os.path.exists(ckpt_path):
        print(f"✅ Carregando pesos do arquivo: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Usa os pesos do EMA se existirem (são muito melhores para geração)
        if 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict'])
            print("🌟 Usando os pesos suavizados do EMA (Qualidade Máxima)!")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("⚠️ EMA não encontrado, usando pesos normais do modelo.")
    else:
        print(f"❌ ERRO: Checkpoint não encontrado em {ckpt_path}")
        return

    # 3. Gera as imagens
    model.eval()
    print(f"🎨 Gerando {args.num_images} imagens. Isso pode levar alguns segundos...")
    sampled_images = diffusion.sample(model, n=args.num_images)
    
    # 4. Salva o resultado
    os.makedirs("results/generated", exist_ok=True)
    save_path = os.path.join("results/generated", "generated_images_fixed.jpg")
    save_images(sampled_images, save_path)
    print(f"✅ Imagens salvas com sucesso em: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDPM_CIFAR10", help="Nome da pasta do run")
    parser.add_argument('--num_images', type=int, default=36, help="Quantidade de imagens para gerar (ex: 16, 36, 64)")
    parser.add_argument('--image_size', type=int, default=32, help="Resolução da imagem")
    parser.add_argument('--device', type=str, default="cuda", help="cuda ou cpu")
    args = parser.parse_args()
    
    generate(args)