import os
import torch
import torchvision
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler # <-- Importante para o Multi-GPU


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


# Atualize a sua função get_data para esta:
def get_data(args, is_distributed=False):
    transforms = T.Compose([
        T.CenterCrop(178),
        T.Resize((args.image_size, args.image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Substituímos a classe chata do PyTorch pela nossa:
    dataset = MeuCelebA(
        root="/home/al.lucas.barcelos/Modelos/diffusion-model/CelebA_data", 
        transform=transforms
    )
    
    if is_distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=8, pin_memory=True)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        return dataloader
    

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


class MeuCelebA(Dataset):
    def __init__(self, root, transform=None):
        # Adicione mais um "img_align_celeba" no final da lista do os.path.join:
        self.img_dir = os.path.join(root, "celeba", "img_align_celeba", "img_align_celeba") 
        
        self.transform = transform        
        # Lê o arquivo que renomeamos para .txt (mas que por dentro tem as vírgulas do Kaggle)
        attr_path = os.path.join(root, "celeba", "list_attr_celeba.txt")
        self.df = pd.read_csv(attr_path) 
        
        self.img_names = self.df['image_id'].values
        # Pega as labels dos atributos e converte para formato PyTorch
        self.labels = (self.df.drop('image_id', axis=1).values > 0).astype(int)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label