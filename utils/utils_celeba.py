import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from matplotlib import pyplot as plt

# ==========================================
# 1. FUNÇÕES MANTIDAS (NÃO APAGUE!)
# O train.py precisa delas para salvar logs e imagens
# ==========================================
def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

# ==========================================
# 2. NOVA LÓGICA: LATENT CACHE
# ==========================================
class CelebALatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        # map_location='cpu' evita estourar a memória da GPU 0 acidentalmente
        latent = torch.load(latent_path, map_location='cpu')
        return latent, 0 

def get_data(args, is_distributed=True):
    # ATENÇÃO: Defina aqui o caminho onde a pasta 'latents_cache' vai ficar
    latent_dir = "/home/al.lucas.barcelos/Modelos/diffusion-model/CelebA_data/latents_cache"
    
    dataset = CelebALatentDataset(latent_dir)
    
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=sampler, 
            shuffle=False,  # DDP requer shuffle False aqui (o Sampler já embaralha)
            num_workers=8,  # Manti os 8 workers que você já usava!
            pin_memory=True,
            drop_last=True  # Crucial para evitar Deadlock no Multi-GPU
        )
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        return dataloader, None