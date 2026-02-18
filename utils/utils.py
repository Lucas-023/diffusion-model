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
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset_train = torchvision.datasets.CIFAR10(root="./cifar10_data", train=True, download=True, transform=transforms)
    dataset_test = torchvision.datasets.CIFAR10(root="./cifar10_data", train=False, download=True, transform=transforms)
    dataset_full = ConcatDataset([dataset_train, dataset_test])
    
    dataloader = DataLoader(
        dataset_full, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,      # Keep at 4 (or try 8 if you have a strong CPU)
        pin_memory=True,    # Crucial for GPU
        persistent_workers=True # <--- NEW: Keeps workers alive between epochs
    )
    
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)