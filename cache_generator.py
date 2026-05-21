# Arquivo: create_cache.py
import os
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from vae.modules import VAE # Sua VAE

def make_cache():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Carrega a VAE pré-treinada
    vae = VAE(in_channels=3, latent_dim=4).to(device)
    vae.load_state_dict(torch.load("vae/vae_epoch_62.pt"))
    vae.eval()
    
    pasta_imagens = "CelebA_data/img_align_celeba/img_align_celeba"
    pasta_cache = "cache_latent"
    os.makedirs(pasta_cache, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    imagens = os.listdir(pasta_imagens)
    print(f"Iniciando compressão de {len(imagens)} imagens...")
    
    with torch.no_grad():
        for img_name in tqdm(imagens):
            if not img_name.endswith(('.jpg', '.png')): continue
            
            # Lê e transforma a imagem
            img_path = os.path.join(pasta_imagens, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device) # Adiciona batch dimension
            
            # Passa pela VAE para virar Latente
            mu, _ = vae.encode(img_tensor)
            latent = mu.squeeze(0).cpu() # Tira do batch e manda pra RAM
            
            # Salva como .pt (ex: 000001.jpg vira 000001.pt)
            nome_arquivo_pt = img_name.split('.')[0] + ".pt"
            torch.save(latent, os.path.join(pasta_cache, nome_arquivo_pt))
            
    print("✅ Cache Latente concluído com sucesso!")

if __name__ == "__main__":
    make_cache()