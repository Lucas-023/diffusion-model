import os
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

def extract_cifar():
    output_dir = "cifar_real_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("📦 Baixando/Carregando CIFAR-10 real...")
    # Carrega as imagens originais SEM normalizar (para salvar como PNG normal)
    dataset = torchvision.datasets.CIFAR10(
        root="./data", 
        train=True, 
        download=True, 
        transform=torchvision.transforms.ToTensor()
    )
    
    print(f"Salvo as {len(dataset)} imagens originais em '{output_dir}'...")
    for i, (img, _) in enumerate(tqdm(dataset)):
        save_image(img, os.path.join(output_dir, f"real_{i:05d}.png"))
        
    print("✅ Extração concluída!")

if __name__ == "__main__":
    extract_cifar()