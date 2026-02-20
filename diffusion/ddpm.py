import torch
import torch.nn as nn
from tqdm import tqdm 
import logging
import numpy as np

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        # --- PREPARANDO O CRONOGRAMA DE RUÍDO (The Schedule) ---
        # Definimos o "beta" (taxa de destruição da imagem)
        self.beta = self.prepare_noise_schedule().to(device)
        
        # Alpha é o oposto do beta (quanto da imagem original sobra)
        self.alpha = 1. - self.beta
        
        # Alpha Hat (alpha_cumprod) é o acumulado até o passo t
        # Isso nos permite pular direto para o passo t sem loop
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        # Cria uma escala linear de ruído
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        # Erro comum: Esquecer o sqrt aqui!
        # x_t = sqrt(alpha_hat) * x + sqrt(1 - alpha_hat) * eps
        
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        
        epsilon = torch.randn_like(x)
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            # MUDA AQUI: De range(1, ...) para range(0, ...)
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                
                predicted_noise = model(x, t)
                
                # Coeficientes
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                # MUDA AQUI: De i > 1 para i > 0
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # FÓRMULA CORRETA:
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        return x    
if __name__ == "__main__":
    # Teste rápido para ver se a matemática não quebra
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff = Diffusion(device=device)
    
    # Simula uma imagem (Batch 2)
    fake_img = torch.randn(2, 3, 64, 64).to(device)
    t = diff.sample_timesteps(2).to(device)
    
    noisy_img, noise = diff.noise_images(fake_img, t)
    
    print("Gerenciador de Difusão criado com sucesso!")
    print(f"Shape da imagem ruidosa: {noisy_img.shape}")
    print(f"Shape do ruído isolado: {noise.shape}")
