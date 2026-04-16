import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """
    Bloco residual idêntico ao do Stable Diffusion.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, in_channels)
        # Q, K, V juntos em uma única projeção
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalização
        norm_x = self.group_norm(x)
        
        # Gera Q, K e V
        qkv = self.qkv(norm_x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # O Flash Attention do PyTorch espera o formato: 
        # [Batch, Num_Heads, Sequencia, Dimensão]
        # Como estamos usando 1 Head para simplificar, usamos:
        # [Batch, 1, H*W, Channels]
        
        q = q.view(b, c, -1).transpose(1, 2).unsqueeze(1) # [b, 1, h*w, c]
        k = k.view(b, c, -1).transpose(1, 2).unsqueeze(1)
        v = v.view(b, c, -1).transpose(1, 2).unsqueeze(1)
        
        # A MÁGICA ACONTECE AQUI:
        # Essa função escolhe automaticamente o Flash Attention se a sua GPU suportar
        out = F.scaled_dot_product_attention(q, k, v)
        
        # Remove a dimensão do Head e volta pro formato [Batch, Channels, H*W]
        out = out.squeeze(1).transpose(1, 2)
        
        # Volta pro formato de imagem [Batch, Channels, Altura, Largura]
        out = out.view(b, c, h, w)
        
        # Projeção final + Conexão Residual
        return x + self.proj_out(out)    
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # O SD geralmente usa padding=0 e adiciona padding manual antes, 
        # mas padding=1 funciona perfeitamente bem.
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_dim=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling blocks
        self.block1 = ResidualBlock(base_channels, base_channels)
        self.down1 = Downsample(base_channels)

        self.block2 = ResidualBlock(base_channels, base_channels * 2)
        self.down2 = Downsample(base_channels * 2)

        self.block3 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.down3 = Downsample(base_channels * 4)

        # "Mid Blocks" (Gargalo com Atenção)
        self.mid_block1 = ResidualBlock(base_channels * 4, base_channels * 4)
        self.mid_attn = SelfAttention(base_channels * 4)
        self.mid_block2 = ResidualBlock(base_channels * 4, base_channels * 4)

        # Saída normalizada antes da projeção
        self.norm_out = nn.GroupNorm(8, base_channels * 4)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.down3(x)
        
        # Passa pelo gargalo (bottleneck)
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=4, base_channels=32, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, base_channels * 4, kernel_size=3, padding=1)

        # "Mid Blocks" (Gargalo com Atenção logo ao receber o latente)
        self.mid_block1 = ResidualBlock(base_channels * 4, base_channels * 4)
        self.mid_attn = SelfAttention(base_channels * 4)
        self.mid_block2 = ResidualBlock(base_channels * 4, base_channels * 4)

        # Upsampling blocks
        self.block1 = ResidualBlock(base_channels * 4, base_channels * 4)
        self.up1 = Upsample(base_channels * 4)

        self.block2 = ResidualBlock(base_channels * 4, base_channels * 2)
        self.up2 = Upsample(base_channels * 2)

        self.block3 = ResidualBlock(base_channels * 2, base_channels)
        self.up3 = Upsample(base_channels)

        # Normalização final
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        
        # Passa pelo gargalo logo no início
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        x = self.block1(x)
        x = self.up1(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.block3(x)
        x = self.up3(x)
        
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        
        return x
    
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim, out_channels=in_channels)

        # Mantive a sua lógica de separação 1x1, é idêntica em matemática à versão oficial!
        self.conv_mu      = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        self.conv_log_var = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

    def encode(self, x):
        h       = self.encoder(x)    
        mu      = self.conv_mu(h)          
        log_var = self.conv_log_var(h)      
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decode(z)
        kl_loss     = self._kl_loss(mu, log_var)
        return x_recon, kl_loss

    def _kl_loss(self, mu, log_var):
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return kl

    @torch.no_grad()
    def sample(self, num_samples=1, device="cuda"):
        # Uma imagem 64x64 com 3 downsamples (2^3 = 8) vira 8x8 no latente
        grid_size = 8  
        z = torch.randn(num_samples, self.conv_mu.out_channels,
                        grid_size, grid_size, device=device)
        return self.decode(z)