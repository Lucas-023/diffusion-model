import os
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from copy import deepcopy

# --- IMPORTAÇÕES DDP ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Suas importações
from utils.utils import get_data, save_images, setup_logging
from models.unet import UNet
from diffusion.ddpm import Diffusion

def setup_ddp():
    """Inicializa o comunicador Multi-Node e Multi-GPU"""
    # NCCL é o backend padrão e mais rápido para GPUs NVIDIA
    dist.init_process_group(backend="nccl")
    
    # LOCAL_RANK: ID da GPU no computador atual (será 0 nos dois PCs)
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # RANK: ID global na rede (PC 1 será 0, PC 2 será 1)
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank

def update_ema(ema_model, model, decay=0.9995):
    """Atualiza os pesos do EMA de forma suave"""
    ema_model.eval()
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1 - decay))

def train(args):
    # 1. Configura DDP e define os Ranks
    local_rank, global_rank = setup_ddp()
    
    # O Mestre absoluto é APENAS o computador com o RANK global 0
    is_master = (global_rank == 0)
    
    if is_master:
        setup_logging(args.run_name)
        print("\n" + "="*60)
        print("🚀 DDPM Training - Modo Multi-Node (Rede)")
        print("="*60)

    # 2. Carrega Dados (O Sampler distribuído cuida de dividir os dados pela rede)
    dataloader, sampler = get_data(args, is_distributed=True)
    
    # 3. Inicializa Modelos na GPU local de cada PC
    model = UNet(image_size=args.image_size).to(local_rank)
    diffusion = Diffusion(img_size=args.image_size, device=local_rank)
    
    # Otimizador: Adam padrão (sem weight decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # EMA só precisa existir no PC Mestre (ele que vai salvar os arquivos)
    ema_model = deepcopy(model).eval() if is_master else None

    # Variáveis de controle
    start_epoch = 0

    # 4. Retomar de um Checkpoint (Opcional)
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        if is_master:
            print(f"📦 Carregando checkpoint: {args.resume_ckpt}")
        
        # Mapeia para CPU primeiro para evitar pico de VRAM, e cada PC carrega sua cópia
        checkpoint = torch.load(args.resume_ckpt, map_location="cpu")
        
        # Restaura os pesos
        model.load_state_dict(checkpoint['model_state_dict'])
        if is_master and 'ema_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        if is_master:
            print(f"✅ Retomando da época {start_epoch}!")

    # 5. Envolve o modelo no DDP
    # Importante: O DDP sincroniza os pesos iniciais de todos os PCs automaticamente aqui
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    # 6. Loop de Treino
    save_dir = os.path.join("models", args.run_name)
    results_dir = os.path.join("results", args.run_name)

    for epoch in range(start_epoch, args.epochs):
        # Obriga o sampler a embaralhar os dados de forma sincronizada entre os PCs
        sampler.set_epoch(epoch)
        
        # Barra de progresso APENAS no PC Mestre
        pbar = tqdm(dataloader) if is_master else dataloader
        
        for images, _ in pbar:
            images = images.to(local_rank)
            
            # Sorteia os timesteps
            t = diffusion.noise_steps
            t = torch.randint(low=1, high=t, size=(images.shape[0],)).to(local_rank)
            
            # Adiciona ruído
            x_t, noise = diffusion.noise_images(images, t)
            
            # Predição e Loss
            predicted_noise = model(x_t, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            
            # Backpropagation (O DDP sincroniza os gradientes pela rede magicamente aqui!)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if is_master:
                # Atualiza o EMA (usando o model.module para ignorar a casca do DDP)
                update_ema(ema_model, model.module)
                pbar.set_postfix(MSE=loss.item())
                
        # 7. Salvar e Avaliar (SÓ NO PC MESTRE)
        if is_master and epoch % 10 == 0:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), 
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_dir, "ckpt.pt"))
            
            # Gera imagens com o EMA para acompanhar a qualidade
            sampled_images = diffusion.sample(ema_model, n=16)
            save_images(sampled_images, os.path.join(results_dir, f"{epoch}.jpg"))

    # Desliga o comunicador da rede no final
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDPM_MultiNode", help="Nome da pasta")  
    parser.add_argument('--epochs', type=int, default=2500, help="Total de épocas")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size POR GPU") 
    parser.add_argument('--image_size', type=int, default=32, help="Resolução da imagem") 
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning Rate")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Caminho do .pt antigo")
    
    args = parser.parse_args()
    train(args)