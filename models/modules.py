import torch.nn as nn 
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)
    
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(downsample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)

        
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class attention_block(nn.Module):
    def __init__(self, in_channels):
        super(attention_block, self).__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.reshape(batch_size, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        k = k.reshape(batch_size, C, H * W)                  # (B, C, H*W)
        v = v.reshape(batch_size, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        attn = torch.bmm(q, k) / (C ** 0.5)                   # (B, H*W, H*W)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)                              # (B, H*W, C)
        out = out.permute(0, 2, 1).reshape(batch_size, C, H, W)  # (B, C, H, W)

        out = self.proj_out(out)
        return x + out