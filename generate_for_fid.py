import os
import torch
import argparse
from tqdm import tqdm
from torchvision.utils import save_image

# Ajuste as importações conforme a estrutura do seu projeto
from models.unet import UNet
from diffusion.ddpm import Diffusion

def generate_images(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando geração no dispositivo: {device}")

    # 1. Configura a Pasta de Saída
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
    
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("✅ Pesos do modelo EMA carregados com sucesso!")
    else:
        print("⚠️ AVISO: ema_state_dict não encontrado. Usando pesos normais.")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    # 4. Loop de Geração Inteligente
    images_left = args.num_images
    total_generated = 0

    print(f"\nGerando um total de {args.num_images} imagens...")
    
    with torch.no_grad():
        with tqdm(total=args.num_images, desc="Gerando") as pbar:
            while images_left > 0:
                # Garante que o último batch não gere imagens a mais
                current_batch_size = min(args.batch_size, images_left)

                # ==============================================================
                # 🔄 ESCOLHA O MÉTODO DE GERAÇÃO (Comente/Descomente aqui)
                # ==============================================================

                # --- OPÇÃO 1: DDPM (Artigo Original, 1000 passos, Lento) ---
                # sampled_images = diffusion.sample(model, n=current_batch_size)
                # # Correção obrigatória: desnormaliza de [-1, 1] para [0, 1]
                # sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2.0 
                
                # --- OPÇÃO 2: DDIM (Rápido, 50 passos) ---
                sampled_images = diffusion.sample_ddim(model, n=current_batch_size, ddim_timesteps=50, ddim_eta=0.0)
                # Nota: Não precisa de correção aqui, pois nosso sample_ddim já devolve [0, 1]!

                # ==============================================================

                # Salva cada imagem do batch individualmente
                for j in range(current_batch_size):
                    img_idx = total_generated + j
                    save_image(sampled_images[j], os.path.join(output_dir, f"img_{img_idx:05d}.png"))
                
                total_generated += current_batch_size
                images_left -= current_batch_size
                pbar.update(current_batch_size)

    print(f"\n🎉 Geração concluída! {total_generated} imagens salvas em: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True, help="Nome da pasta do modelo treinado")
    parser.add_argument('--image_size', type=int, default=32, help="Tamanho da imagem")
    parser.add_argument('--batch_size', type=int, default=128, help="Imagens geradas por vez na GPU")
    parser.add_argument('--num_images', type=int, default=50000, help="Total de imagens para gerar")
    args = parser.parse_args()
    
    generate_images(args)