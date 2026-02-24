import torch
from tqdm import tqdm

class DDIM:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device
        
        # Recria o exato mesmo cronograma de ruído do DDPM para os pesos baterem
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    @torch.no_grad()
    def sample(self, model, n, ddim_timesteps=50, ddim_eta=0.0):
        """
        Gera imagens usando o método DDIM para pular passos.
        """
        print(f"🚀 Iniciando DDIM com {ddim_timesteps} passos e eta={ddim_eta}...")
        model.eval()

        # Define os passos que vamos de fato visitar (ex: pula de 20 em 20)
        step_size = self.noise_steps // ddim_timesteps
        times = torch.arange(0, self.noise_steps, step_size, device=self.device)
        times = torch.flip(times, [0]) # Inverte para [980, 960, 940...]

        # Começa com ruído puro
        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

        for i, t in enumerate(times):
            # Cria o tensor de tempo para o batch atual
            t_tensor = torch.full((n,), t, device=self.device, dtype=torch.long)
            
            # A UNet prevê o ruído
            predicted_noise = model(x, t_tensor)
            
            # Pega o alpha acumulado do passo atual
            alpha_bar_t = self.alpha_hat[t]
            
            # Pega o alpha acumulado do PRÓXIMO passo que vamos visitar (t_prev)
            if i < len(times) - 1:
                t_prev = times[i + 1]
                alpha_bar_t_prev = self.alpha_hat[t_prev]
            else:
                # No último passo, a imagem não tem ruído (alpha_bar = 1.0)
                alpha_bar_t_prev = torch.tensor(1.0, device=self.device)

            # --- A FÓRMULA DO DDIM ---
            # 1. Previsão da imagem perfeitamente limpa (x0)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            
            # 2. Desvio padrão da aleatoriedade (sigma)
            sigma = ddim_eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            
            # 3. Direção que aponta para o próximo passo no processo reverso
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * predicted_noise
            
            # 4. Ruído aleatório (se eta > 0)
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            
            # Atualiza a imagem com o grande salto!
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma * noise

        # Desnormaliza para salvar a imagem corretamente
        x = (x.clamp(-1, 1) + 1) / 2
        return x