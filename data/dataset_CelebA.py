import torch
import torchvision
from torch.utils.data import DataLoader

def get_data(args):
    # Transformação ideal para rostos: Corta o centro (evitando fundo) e redimensiona
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(178),  # Corta o retângulo original (178x218) para quadrado
        torchvision.transforms.Resize(args.image_size), # Sobe para 256x256
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Deixa no range [-1, 1]
    ])
    
    # Usando ImageFolder no lugar de CelebA
    # O caminho aponta para a pasta PAI da pasta que contém as imagens
    dataset = torchvision.datasets.ImageFolder(
        root=args.dataset_path, 
        transform=transforms
    )
    
    print(f"✅ Dataset carregado: {len(dataset)} imagens")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return dataloader

if __name__ == "__main__":
    class Args:
        batch_size = 32
        image_size = 256
        # Caminho relativo baseado em onde você está rodando o script no terminal
        dataset_path = "./celeba_data/celeba/archive/img_align_celeba"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testando no sistema: {device.upper()}")
    
    loader = get_data(Args())
    
    images, labels = next(iter(loader))
    print(f"Batch carregado com sucesso. Shape: {images.shape}")
    print(f"Range de pixel: {images.min():.2f} a {images.max():.2f}")