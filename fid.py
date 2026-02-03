"""
FID Score Calculator para DDPM
Calcula o Fréchet Inception Distance entre imagens geradas e dataset real

Uso:
    python calculate_fid.py --checkpoint models/DDPM_CIFAR10/ckpt_epoch_100.pt
    python calculate_fid.py --checkpoint models/DDPM_CIFAR10/ckpt.pt --num_samples 5000
"""

import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from scipy import linalg
from PIL import Image
import os

# Importa seus módulos
from models.unet import unet
from diffusion.ddpm import Diffusion

class InceptionV3FeatureExtractor:
    """Extrai features usando InceptionV3 para cálculo de FID"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Carrega InceptionV3 pré-treinado
        print("📥 Carregando InceptionV3...")
        inception = torchvision.models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()  # Remove última camada
        inception.eval()
        self.model = inception.to(device)
        
        # Freeze todos os parâmetros
        for param in self.model.parameters():
            param.requires_grad = False
            
        print("✅ InceptionV3 carregado!")
    
    def preprocess(self, images):
        """
        Preprocessa imagens para InceptionV3
        Input: Tensor (B, 3, H, W) com valores em [-1, 1]
        Output: Tensor (B, 3, 299, 299) com valores normalizados para ImageNet
        """
        # De [-1, 1] para [0, 1]
        images = (images + 1) / 2
        
        # Clamp para garantir
        images = torch.clamp(images, 0, 1)
        
        # Resize para 299x299 (tamanho esperado pelo Inception)
        if images.shape[-1] != 299:
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # Normaliza com stats do ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        return images
    
    def extract_features(self, images):
        """Extrai features de um batch de imagens"""
        with torch.no_grad():
            images = self.preprocess(images)
            features = self.model(images)
        return features.cpu().numpy()

def calculate_fid_statistics(features):
    """
    Calcula média e covariância das features
    
    Args:
        features: array (N, 2048) com features extraídas
    Returns:
        mu: média (2048,)
        sigma: matriz de covariância (2048, 2048)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calcula a distância de Fréchet entre duas gaussianas multivariadas
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print("⚠️ Covariância contém valores não-finitos. Adicionando epsilon à diagonal.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Parte imaginária muito grande: {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return fid

def get_real_dataset_features(extractor, num_samples=10000, batch_size=64, dataset_path='./cifar10_data'):
    """Extrai features do dataset real (CIFAR-10)"""
    
    print(f"\n📊 Extraindo features do dataset real ({num_samples} imagens)...")
    
    # Transforma para [-1, 1] (mesmo range que modelo gera)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=True,
        download=False,
        transform=transform
    )
    
    # Limita ao número de samples
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    
    for images, _ in tqdm(dataloader, desc="Extraindo features reais"):
        images = images.to(extractor.device)
        features = extractor.extract_features(images)
        all_features.append(features)
    
    all_features = np.concatenate(all_features, axis=0)
    
    print(f"✅ Features reais extraídas: {all_features.shape}")
    
    return all_features

def generate_and_extract_features(model, diffusion, extractor, num_samples=10000, batch_size=64):
    """Gera imagens com o modelo e extrai features"""
    
    print(f"\n🎨 Gerando {num_samples} imagens e extraindo features...")
    
    all_features = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Gerando e extraindo"):
        # Último batch pode ser menor
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Gera imagens
        with torch.no_grad():
            generated_images = diffusion.sample(model, n=current_batch_size)
        
        # Extrai features
        features = extractor.extract_features(generated_images)
        all_features.append(features)
    
    all_features = np.concatenate(all_features, axis=0)
    
    print(f"✅ Features geradas extraídas: {all_features.shape}")
    
    return all_features

def calculate_fid(args):
    """Função principal para calcular FID"""
    
    device = args.device
    
    print("\n" + "="*70)
    print("📊 CALCULANDO FID SCORE")
    print("="*70)
    print(f"📦 Checkpoint: {args.checkpoint}")
    print(f"🎯 Amostras: {args.num_samples}")
    print(f"📏 Batch Size: {args.batch_size}")
    print(f"💻 Device: {device}")
    print("="*70 + "\n")
    
    # 1. Carrega o modelo
    print("🔧 Carregando modelo...")
    model = unet().to(device)
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Carrega pesos do EMA se disponível, senão usa modelo normal
    if args.use_ema and 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("✅ Modelo EMA carregado!")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Modelo carregado!")
    else:
        # Checkpoint pode ser só os pesos direto
        model.load_state_dict(checkpoint)
        print("✅ Modelo carregado (checkpoint direto)!")
    
    model.eval()
    
    epoch_info = checkpoint.get('epoch', 'desconhecida')
    print(f"📅 Época do checkpoint: {epoch_info}")
    
    # 2. Inicializa extrator de features
    extractor = InceptionV3FeatureExtractor(device=device)
    
    # 3. Extrai features do dataset real
    if args.real_features_cache and os.path.exists(args.real_features_cache):
        print(f"\n📥 Carregando features reais do cache: {args.real_features_cache}")
        cache = np.load(args.real_features_cache)
        real_features = cache['features']
        print(f"✅ Features reais carregadas: {real_features.shape}")
    else:
        real_features = get_real_dataset_features(
            extractor, 
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            dataset_path=args.dataset_path
        )
        
        # Salva cache se solicitado
        if args.real_features_cache:
            print(f"💾 Salvando cache de features reais: {args.real_features_cache}")
            np.savez_compressed(args.real_features_cache, features=real_features)
    
    # 4. Gera imagens e extrai features
    generated_features = generate_and_extract_features(
        model, diffusion, extractor,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    # 5. Calcula estatísticas
    print("\n📈 Calculando estatísticas...")
    mu_real, sigma_real = calculate_fid_statistics(real_features)
    mu_gen, sigma_gen = calculate_fid_statistics(generated_features)
    
    # 6. Calcula FID
    print("🧮 Calculando FID...")
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    # 7. Resultados
    print("\n" + "="*70)
    print("🎉 RESULTADO")
    print("="*70)
    print(f"📊 FID Score: {fid_score:.2f}")
    print("="*70)
    
    # Interpretação
    print("\n📖 Interpretação:")
    if fid_score < 10:
        print("   🌟 EXCELENTE! Qualidade state-of-the-art")
    elif fid_score < 30:
        print("   ✅ MUITO BOM! Qualidade alta")
    elif fid_score < 50:
        print("   👍 BOM! Modelo está aprendendo bem")
    elif fid_score < 100:
        print("   ⚠️  RAZOÁVEL. Precisa de mais treino")
    else:
        print("   ❌ RUIM. Modelo precisa de ajustes")
    
    print("\n📚 Referência CIFAR-10:")
    print("   • DDPM original (paper): ~3.17")
    print("   • Bons modelos: 5-15")
    print("   • Modelos em treino: 20-50")
    print("   • Random noise: ~300+")
    
    # Salva resultado
    if args.output:
        result = {
            'fid_score': float(fid_score),
            'checkpoint': args.checkpoint,
            'epoch': epoch_info,
            'num_samples': args.num_samples,
            'mu_real': mu_real,
            'sigma_real': sigma_real,
            'mu_gen': mu_gen,
            'sigma_gen': sigma_gen
        }
        np.savez_compressed(args.output, **result)
        print(f"\n💾 Resultado salvo em: {args.output}")
    
    return fid_score

def main():
    parser = argparse.ArgumentParser(description='Calcula FID Score para modelo DDPM')
    
    # Argumentos principais
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Caminho do checkpoint (.pt)')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Número de amostras para calcular FID (padrão: 10000)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size para geração (padrão: 64)')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Tamanho da imagem (padrão: 64)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda ou cpu)')
    
    # Argumentos opcionais
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Usar modelo EMA se disponível (padrão: True)')
    parser.add_argument('--dataset_path', type=str, default='./cifar10_data',
                       help='Caminho do dataset CIFAR-10')
    parser.add_argument('--real_features_cache', type=str, default='cifar10_real_features.npz',
                       help='Cache de features reais (acelera cálculos futuros)')
    parser.add_argument('--output', type=str, default=None,
                       help='Arquivo para salvar resultado (.npz)')
    
    args = parser.parse_args()
    
    # Validações
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.checkpoint}")
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("⚠️ CUDA não disponível, usando CPU (será MUITO mais lento)")
        args.device = 'cpu'
    
    # Calcula FID
    calculate_fid(args)

if __name__ == '__main__':
    main()