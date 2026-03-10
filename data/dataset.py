import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

def normalize_to_neg_one_to_one(t):
    return (t * 2) - 1

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset_train = torchvision.datasets.CIFAR10(
        root="./cifar10_data", 
        train=True,  
        download=True, 
        transform=transforms
    )
    
    dataset_test = torchvision.datasets.CIFAR10(
        root="./cifar10_data", 
        train=False,  
        download=True, 
        transform=transforms
    )
    
    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset([dataset_train, dataset_test])
    
    print(f"✅ Dataset completo: {len(dataset)} imagens (train + test)")
    
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    class Args:
        batch_size = 32
        image_size = 32
        dataset_path = "./cifar10_data"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testando no sistema: {device.upper()}")
    
    loader = get_data(Args())
    
    images, labels = next(iter(loader))
    print(f"Batch carregado com sucesso. Shape: {images.shape}")
    print(f"Range de pixel: {images.min():.2f} a {images.max():.2f}")
