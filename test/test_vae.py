import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from vae.modules import VAE # Ajuste o import conforme seu projeto

def test_vae(checkpoint_path, data_path, device="cuda"):
    # 1. Configurações
    image_size = 256
    latent_dim = 4
    num_samples = 8 # Quantas imagens comparar
    output_dir = "vae_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Carregar Modelo
    model = VAE(in_channels=3, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✅ Modelo carregado de: {checkpoint_path}")

    # 3. Preparar Dados (Mesmo transform do treino)
    transform = transforms.Compose([
        transforms.CenterCrop(178), 
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)

    with torch.no_grad():
        # --- TESTE 1: RECONSTRUÇÃO ---
        # Passa imagens reais pela VAE
        recon_images, _ = model(real_images)
        
        # Desnormaliza para salvar (volta de [-1, 1] para [0, 1])
        comparison = torch.cat([real_images, recon_images], dim=0)
        comparison = (comparison + 1) / 2
        
        grid_recon = torchvision.utils.make_grid(comparison, nrow=num_samples)
        torchvision.utils.save_image(grid_recon, f"{output_dir}/1_reconstrucao.png")
        print("📸 Teste de reconstrução salvo.")

        # --- TESTE 2: GERAÇÃO (SAMPLING) ---
        # Gera 16 faces totalmente novas a partir de ruído Gaussiano
        # Note: Seu encoder tem 3 downsamples, então 256 -> 32x32.
        latent_h_w = image_size // (2**3) 
        z = torch.randn(16, latent_dim, latent_h_w, latent_h_w).to(device)
        generated = model.decode(z)
        generated = (generated + 1) / 2
        
        grid_gen = torchvision.utils.make_grid(generated, nrow=4)
        torchvision.utils.save_image(grid_gen, f"{output_dir}/2_geracao_aleatoria.png")
        print("🎲 Teste de geração aleatória salvo.")

        # --- TESTE 3: INTERPOLAÇÃO (O "PULO DO GATO") ---
        # Pega a face A e a face B e cria uma transição entre elas
        mu, _ = model.encode(real_images)
        z_a = mu[0:1] # Primeira face do batch
        z_b = mu[1:2] # Segunda face do batch
        
        steps = 8
        alpha = torch.linspace(0, 1, steps).to(device)
        interpolated_latents = []
        for a in alpha:
            # Mistura linear entre os dois pontos no espaço latente
            z_interp = z_a * (1 - a) + z_b * a
            interpolated_latents.append(z_interp)
        
        z_interp_batch = torch.cat(interpolated_latents, dim=0)
        interp_images = model.decode(z_interp_batch)
        interp_images = (interp_images + 1) / 2
        
        grid_interp = torchvision.utils.make_grid(interp_images, nrow=steps)
        torchvision.utils.save_image(grid_interp, f"{output_dir}/3_interpolacao.png")
        print("🎢 Teste de interpolação salvo.")

if __name__ == "__main__":
    # COLOQUE OS CAMINHOS AQUI
    CHECKPOINT = "vae/vae_epoch_62.pt" 
    DATA_DIR = "CelebA_data/img_align_celeba/" # Onde está a pasta img_align_celeba
    
    test_vae(CHECKPOINT, DATA_DIR)