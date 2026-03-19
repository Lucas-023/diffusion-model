import torch
import torch.nn as nn
import math

from models.modules import ResidualBlock, Upsample, Downsample, AttentionBlock, SinusoidalPosEmb, SiLU # Importe a nova classe

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        dropout=0.1,
        attention_resolutions=(16,),
        num_res_blocks=2, 
        image_size=32 #usado para determinar onde aplicar o bloco de self attention
    ):
        super().__init__()

        
        #deinindo mlp para o tempo
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        #convolução inicial
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        #encoder
        self.downs = nn.ModuleList()
        curr_channels = base_channels
        
        self.down_block_channels = [curr_channels] 

        resolution = image_size
        
        for level, mult in enumerate(channel_mults):
            out_channels_level = base_channels * mult
            
            #adiciona os blocos resiudais para esse nível
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(curr_channels, out_channels_level, time_dim, dropout)]
                curr_channels = out_channels_level
                
                #confere se deve adicionar atenção e adiciona se for o caso
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(curr_channels))
                
                self.downs.append(nn.ModuleList(layers))
                self.down_block_channels.append(curr_channels)

            #caso não seja o último nível adiciona o downsample
            if level != len(channel_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(curr_channels)]))
                self.down_block_channels.append(curr_channels)
                resolution //= 2

        #bottleneck
        #ResBlock -> Attention -> ResBlock
        self.mid = nn.ModuleList([
            ResidualBlock(curr_channels, curr_channels, time_dim, dropout),
            AttentionBlock(curr_channels),
            ResidualBlock(curr_channels, curr_channels, time_dim, dropout),
        ])

        #decoder
        self.ups = nn.ModuleList()
        
        #itera ao contrário para o upsampling
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_channels_level = base_channels * mult
            
            #no decoder temos num_res_blocks + 1 blocos resiudais por nível
            #espelhamento padrão da unet
            for _ in range(num_res_blocks + 1):

                skip_channels = self.down_block_channels.pop()
                layers = [ResidualBlock(curr_channels + skip_channels, out_channels_level, time_dim, dropout)]
                curr_channels = out_channels_level
                
                #adiciona atenção se estiver na resolução correto
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(curr_channels))
                
                self.ups.append(nn.ModuleList(layers))
            
            #adiciona upsample se não estiver no ultimo nível
            if level != 0:
                self.ups.append(nn.ModuleList([Upsample(curr_channels)]))
                resolution *= 2

        #output
        self.out_norm = nn.GroupNorm(32, curr_channels)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(curr_channels, out_channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        h = self.conv_in(x)
        
        skips = [h]

        for layer_group in self.downs:
            for layer in layer_group:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t)
                else:
                    h = layer(h)
            skips.append(h)

        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t)
            else:
                h = layer(h)

        for layer_group in self.ups:
            is_upsample_group = isinstance(layer_group[0], Upsample)
            
            if is_upsample_group:
                h = layer_group[0](h)
            else:
                skip = skips.pop()
                h = torch.cat((h, skip), dim=1)
                
                for layer in layer_group:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, t)
                    else:
                        h = layer(h)
        
        #output
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧪 Testando UNet DDPM no device: {device}\n")
    
    # Cria modelo (config CIFAR-10)
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
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
