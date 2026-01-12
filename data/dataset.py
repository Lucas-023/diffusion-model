import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# --- CORREÇÃO TÉCNICA (Universal) ---
# Definir esta função no escopo global resolve o erro de multiprocessing no Windows
# e continua funcionando normalmente no Linux.
def normalize_to_neg_one_to_one(t):
    return (t * 2) - 1

def get_data(args):
    """
    Carrega o dataset CIFAR-10 de forma compatível com Windows/Linux.
    """
    transforms = T.Compose([
        T.Resize(args.image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # Referência à função global em vez de lambda local
        T.Lambda(normalize_to_neg_one_to_one) 
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_path, 
        train=True, 
        download=True, 
        transform=transforms
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        # No Windows, workers > 0 exige que a função seja "pickleable" (global)
        num_workers=4,      
        pin_memory=True,    
        drop_last=True
    )

    return dataloader

# --- Bloco de Teste ---
if __name__ == "__main__":
    # Classe simples para simular argumentos sem precisar de argparse agora
    class Args:
        batch_size = 32
        image_size = 32
        dataset_path = "./cifar10_data"
    
    # Configuração de dispositivo para teste
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testando no sistema: {device.upper()}")
    
    loader = get_data(Args())
    
    # Se isso rodar sem travar, o fix funcionou
    images, labels = next(iter(loader))
    print(f"Batch carregado com sucesso. Shape: {images.shape}")
    print(f"Range de pixel: {images.min():.2f} a {images.max():.2f}")