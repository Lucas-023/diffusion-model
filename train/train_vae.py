import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import lpips 
from tqdm import tqdm # Barra de progresso

# Importe o VAE do seu arquivo modules.py
from vae.modules import VAE

class Board:
    def __init__(self, run_name, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            self.writer = None
        else:
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join("runs", run_name, time_str)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        if self.writer: self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        if self.writer: self.writer.add_image(tag, image, step)

    def log_layer_gradients(self, model, epoch):
        if self.writer:
            for name, params in model.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", params.grad, epoch)

    def close(self):
        if self.writer: self.writer.close()

def train_vae():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- CONFIGURAÇÕES ---
    batch_size = 32
    image_size = 256
    epochs = 100
    learning_rate = 1e-4
    dataset_path = "./celeba_data/celeba/archive/img_align_celeba"
    
    # Pesos das perdas
    kl_weight = 1e-6          
    perceptual_weight = 0.5   

    # --- RETOMAR TREINAMENTO ---
    # Se quiser começar do zero, deixe como None. 
    # Se quiser continuar, coloque o caminho. Ex: "checkpoints/vae_epoch_2.pt"
    resume_checkpoint = None  
    start_epoch = 0
    global_step = 0

    # --- DATASET ---
    transform = transforms.Compose([
        transforms.CenterCrop(178), 
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True, persistent_workers=True)

    # Inicializa Modelo, LPIPS e Otimizador
    model = VAE(in_channels=3, latent_dim=4).to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- LÓGICA PARA CARREGAR O CHECKPOINT ---
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"🔄 Retomando treinamento do checkpoint: {resume_checkpoint}")
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        
        # Tenta extrair o número da época do nome do arquivo (ex: vae_epoch_2.pt -> 2)
        try:
            start_epoch = int(resume_checkpoint.split('_')[-1].split('.')[0])
            global_step = start_epoch * len(dataloader)
            print(f"Continuando a partir da época {start_epoch}...")
        except Exception as e:
            print("Não foi possível inferir a época pelo nome, começando a contagem do 0.")
    
    board = Board(run_name="VAE_CelebA_256x256")

    print(f"🚀 Iniciando treinamento no dispositivo: {device}")
    print(f"Total de lotes (batches) por época: {len(dataloader)}")
    
    # --- LOOP DE TREINAMENTO ---
# 1. Crie o Scaler antes de iniciar o loop de épocas
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch}/{epochs-1}]")
        
        for batch_idx, (real_images, _) in pbar:
            # pin_memory=True + non_blocking=True faz a transferência ser instantânea
            real_images = real_images.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 2. Ativa a meia precisão (FP16) automaticamente onde for seguro
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                recon_images, kl_loss = model(real_images)
                
                recon_loss = F.l1_loss(recon_images, real_images)
                p_loss = loss_fn_vgg(recon_images, real_images).mean()
                loss = recon_loss + (perceptual_weight * p_loss) + (kl_weight * kl_loss)

            # 3. O Scaler cuida de não deixar os gradientes FP16 explodirem ou sumirem
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            board.log_scalar("Loss/Total", loss.item(), global_step)
            # ... (o resto dos logs continua igual)            board.log_scalar("Loss/Reconstrucao_L1", recon_loss.item(), global_step)
            board.log_scalar("Loss/Perceptual", p_loss.item(), global_step)
            board.log_scalar("Loss/KL", kl_loss.item(), global_step)
            
            global_step += 1

            # Atualiza os valores na barra de progresso
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'L1': f"{recon_loss.item():.4f}",
                'LPIPS': f"{p_loss.item():.4f}"
            })

        board.log_layer_gradients(model, epoch)

        # Salva o modelo e loga imagens a cada 2 épocas
        if (epoch + 1) % 2 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/vae_epoch_{epoch+1}.pt")
            print(f"💾 Checkpoint salvo na época {epoch + 1}!")

            model.eval()
            with torch.no_grad():
                sample_real = real_images[:8]
                sample_recon, _ = model(sample_real)

                sample_real = (sample_real + 1) / 2
                sample_recon = (sample_recon + 1) / 2

                grid_real = torchvision.utils.make_grid(sample_real, nrow=4)
                grid_recon = torchvision.utils.make_grid(sample_recon, nrow=4)

                board.log_image("Visualizacao/1_Reais", grid_real, epoch)
                board.log_image("Visualizacao/2_Reconstruidas", grid_recon, epoch)
            model.train()

    board.close()
    print("✅ Treinamento finalizado!")

if __name__ == "__main__":
    train_vae()