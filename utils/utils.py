import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    # 1. Traz os valores de [-1, 1] para [0, 1]
    # O .clamp garante que nada escape do intervalo se o modelo "exagerar"
    images = (images.clamp(-1, 1) + 1) / 2
    
    # 2. Transforma de [0, 1] (float) para [0, 255] (inteiro)
    images = (images * 255).type(torch.uint8)
    
    # 3. Cria o grid e salva (igual você já fazia)
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # --- AQUI ESTÁ A MUDANÇA ---
    # 1. root="./cifar10_data": Aponta direto para sua pasta
    # 2. download=False: Não tenta conectar na internet, confia que os arquivos estão lá
    dataset = torchvision.datasets.CIFAR10(
        root="./cifar10_data", 
        train=True, 
        download=False, 
        transform=transforms
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
