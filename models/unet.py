import torch
import torch.nn as nn
import math

from models.modules import ResidualBlock, Upsample, Downsample, AttentionBlock, SinusoidalPosEmb # Importe a nova classe

class UNet(nn.Module):
    """
    UNet do paper DDPM
    
    Configuração padrão para CIFAR-10 (32×32):
    - base_channels: 128
    - channel_mult: [1, 2, 2, 2]
    - num_res_blocks: 2
    - attention_resolutions: [16]
    - dropout: 0.1
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        
        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        input_block_chans = [ch]
        
        # Resoluções: 32 → 16 → 8 → 4
        resolutions = [32, 16, 8, 4]
        
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, out_ch, time_emb_dim, dropout)]
                ch = out_ch
                
                # Add attention se resolução correta
                if resolutions[level] in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            # Downsample (exceto último nível)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_block_chans.append(ch)
        
        # Bottleneck (meio da UNet)
        self.middle_block = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_emb_dim, dropout),
        ])
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = base_channels * mult
            
            for i in range(num_res_blocks + 1):
                # Skip connection channels
                skip_ch = input_block_chans.pop()
                layers = [ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)]
                ch = out_ch
                
                # Add attention se resolução correta
                if resolutions[level] in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                self.up_blocks.append(nn.ModuleList(layers))
            
            # Upsample (exceto primeiro nível do decoder = último do encoder)
            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, time):
        """
        Args:
            x: [B, 3, 32, 32] - Imagem com ruído
            time: [B] - Timesteps
        Returns:
            [B, 3, 32, 32] - Ruído predito
        """
        # Time embedding
        t = self.time_mlp(time)
        
        # Input
        h = self.conv_in(x)
        
        # Encoder (com skip connections)
        hs = [h]
        for layers in self.down_blocks:
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Downsample):
                    h = layer(h)
            hs.append(h)
        
        # Bottleneck
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t)
            else:
                h = layer(h)
        
        # Decoder (com skip connections)
        for layers in self.up_blocks:
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    # Concatena skip connection
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = layer(h, t)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Upsample):
                    h = layer(h)
        
        # Output
        return self.conv_out(h)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧪 Testando UNet DDPM no device: {device}\n")
    
    # Cria modelo (config CIFAR-10)
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
    ).to(device)
    
    # Conta parâmetros
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total de parâmetros: {num_params:,}")
    print(f"📊 Comparação:")
    print(f"   Sua UNet antiga: ~10-15M parâmetros")
    print(f"   Esta UNet:       {num_params/1e6:.1f}M parâmetros")
    print(f"   Paper DDPM:      ~35M parâmetros\n")
    
    # Teste forward pass
    print("🔄 Testando forward pass...")
    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"✅ Input shape:  {x.shape}")
    print(f"✅ Output shape: {out.shape}")
    print(f"✅ Shapes match: {x.shape == out.shape}")
    
    # Verifica range do output
    print(f"\n📈 Estatísticas do output:")
    print(f"   Min: {out.min():.4f}")
    print(f"   Max: {out.max():.4f}")
    print(f"   Mean: {out.mean():.4f}")
    print(f"   Std: {out.std():.4f}")
    
    print("\n✅ Teste completo! Modelo pronto para treinar!")