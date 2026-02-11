import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, time_emb_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


        self.time_mlp =  nn.Linear(time_emb_dim, out_channels)
        

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)         
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

class Downsample(nn.Module):
    """Downsampling com conv strided (igual paper)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling com interpolação + conv (igual paper)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Self-attention (QKV attention)"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(B, C, H * W)                    # [B, C, HW]
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        
        out = self.proj_out(out)
        return x + out

# Em modules.py

import math # Não esqueça de importar math

class SinusoidalPosEmb(nn.Module):
    """Positional encoding para timestep (igual paper)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings
