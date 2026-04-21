import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler # <-- Importante para Multi-GPU

def get_data(args, is_distributed=False):
    transforms = T.Compose([
        T.CenterCrop(178),
        # 2. Redimensiona para o tamanho desejado (usando TUPLA!)
        T.Resize((args.image_size, args.image_size)),
        # 3. Espelhamento
        T.RandomHorizontalFlip(p=0.5),
        # 4. Transforma em Tensor (0 a 1)
        T.ToTensor(),
        # 5. Normaliza para (-1 a 1)
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # ⚠️ Aviso: download=True no CelebA costuma falhar por cota do Google Drive.
    dataset = torchvision.datasets.CelebA(
        root="./CelebA_data", 
        split='all',  
        download=True, 
        transform=transforms
    )
    
    print(f"✅ Dataset completo: {len(dataset)} imagens")
    
    if is_distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        return dataloader

if __name__ == "__main__":
    class Args:
        batch_size = 32
        image_size = 256 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Testando no sistema: {device.upper()}")
    
    # Testando no modo Single-GPU (is_distributed=False)
    loader = get_data(Args(), is_distributed=False)
    
    images, labels = next(iter(loader))
    print(f"📦 Batch carregado com sucesso. Shape: {images.shape}")
    print(f"🎨 Range de pixel: {images.min():.2f} a {images.max():.2f}")