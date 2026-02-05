import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    # 1. Traz os valores de [-1, 1] para [0, 1]
    images = (images.clamp(-1, 1) + 1) / 2
    
    # 2. Transforma de [0, 1] (float) para [0, 255] (inteiro)
    images = (images * 255).type(torch.uint8)
    
    # 3. Cria o grid e salva
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    """
    Carrega CIFAR-10 COMPLETO (train + test = 60.000 imagens)
    Para modelos generativos não-supervisionados, usamos todo o dataset disponível.
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Carrega conjunto de treino (50.000 imagens)
    dataset_train = torchvision.datasets.CIFAR10(
        root="./cifar10_data", 
        train=True, 
        download=True,
        transform=transforms
    )
    
    # Carrega conjunto de teste (10.000 imagens)
    dataset_test = torchvision.datasets.CIFAR10(
        root="./cifar10_data", 
        train=False, 
        download=True,
        transform=transforms
    )
    
    # Concatena train + test = 60.000 imagens
    dataset_full = ConcatDataset([dataset_train, dataset_test])
    
    # Logging
    print(f"\n{'='*60}")
    print(f"📊 DATASET CARREGADO:")
    print(f"{'='*60}")
    print(f"   Train set:  {len(dataset_train):>6,} imagens")
    print(f"   Test set:   {len(dataset_test):>6,} imagens")
    print(f"   TOTAL:      {len(dataset_full):>6,} imagens")
    print(f"{'='*60}")
    
    # Calcula batches
    num_batches = (len(dataset_full) + args.batch_size - 1) // args.batch_size
    print(f"   Batch size: {args.batch_size:>6}")
    print(f"   Batches:    {num_batches:>6} por época")
    print(f"{'='*60}\n")
    
    dataloader = DataLoader(
        dataset_full, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
