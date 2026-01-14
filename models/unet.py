import torch
import torch.nn as nn

from modules import DoubleConv, up, down, attention_block

class unet(nn.Module):
    def __init__(self, in_Ch=3, out_ch = 3, time_dim = 256, device="cuda"):
        super().__init__()
        self.time_dim = time_dim
        self.device = device

        self.inc = DoubleConv(3, 64)

        self.down1 = down(64, 128)
        self.sa1 = attention_block(128)
        self.down2 = down(128, 256)
        self.sa2 = attention_block(256)
        self.down3 = down(256, 256)
        self.sa3 = attention_block(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = up(512, 128)
        self.sa4 = attention_block(128)
        self.up2 = up(256, 64)
        self.sa5 = attention_block(64)
        self.up3 = up(128, 64)        
        self.sa6 = attention_block(64)

        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

    def pos_encoding(self, t, channels):
        
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        # Hack para garantir dimensões corretas no batch
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t) 
        x = self.sa4(x)
        
        x = self.up2(x, x2, t) 
        x = self.sa5(x)
        
        x = self.up3(x, x1, t) 
        x = self.sa6(x)
        
        output = self.outc(x)
        return output
    


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testando UNet no device: {device}")
    
    net = unet(device=device).to(device)
    
    # Simula um batch de 2 imagens, 3 canais, 32x32px
    x = torch.randn(2, 3, 32, 32).to(device)
    
    # Simula 2 tempos aleatórios (ex: passo 50 e passo 900)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    print("Passando pela rede...")
    try:
        y = net(x, t)
        print(f"Sucesso! Shape de entrada: {x.shape}")
        print(f"Sucesso! Shape de saída:  {y.shape}")
    except Exception as e:
        print(f"Erro no forward: {e}")
        # Dica de debug: Se der erro de tamanho no 'cat', verifique os canais no __init__
