import torch
import torch.nn as nn
from torch.nn import SiLU as SiLU

from models.modules import (
    ResidualBlock,
    Upsample,
    Downsample,
    SpatialTransformer,
    SinusoidalPosEmb
)

class UNet_cond(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4),
        dropout=0.1,
        attention_resolutions=(16, 8),
        context_dim=512,
        num_res_blocks=2,
        image_size=32
    ):
        super().__init__()

        # =====================================================
        # TIME EMBEDDING
        # =====================================================

        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # =====================================================
        # INPUT
        # =====================================================

        self.conv_in = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        # =====================================================
        # ENCODER
        # =====================================================

        self.downs = nn.ModuleList()

        curr_channels = base_channels

        self.down_block_channels = [curr_channels]

        resolution = image_size

        for level, mult in enumerate(channel_mults):

            out_channels_level = base_channels * mult

            for _ in range(num_res_blocks):

                layers = []

                layers.append(
                    ResidualBlock(
                        curr_channels,
                        out_channels_level,
                        time_dim,
                        dropout
                    )
                )

                curr_channels = out_channels_level

                if resolution in attention_resolutions:

                    layers.append(
                        SpatialTransformer(
                            curr_channels,
                            context_dim=context_dim
                        )
                    )

                self.downs.append(
                    nn.ModuleList(layers)
                )

                self.down_block_channels.append(
                    curr_channels
                )

            if level != len(channel_mults) - 1:

                self.downs.append(
                    nn.ModuleList([
                        Downsample(curr_channels)
                    ])
                )

                self.down_block_channels.append(
                    curr_channels
                )

                resolution //= 2

        # =====================================================
        # BOTTLENECK
        # =====================================================

        self.mid = nn.ModuleList([

            ResidualBlock(
                curr_channels,
                curr_channels,
                time_dim,
                dropout
            ),

            SpatialTransformer(
                curr_channels,
                context_dim=context_dim
            ),

            ResidualBlock(
                curr_channels,
                curr_channels,
                time_dim,
                dropout
            ),
        ])

        # =====================================================
        # DECODER
        # =====================================================

        self.ups = nn.ModuleList()

        for level, mult in reversed(
            list(enumerate(channel_mults))
        ):

            out_channels_level = base_channels * mult

            for _ in range(num_res_blocks + 1):

                skip_channels = self.down_block_channels.pop()

                layers = []

                layers.append(
                    ResidualBlock(
                        curr_channels + skip_channels,
                        out_channels_level,
                        time_dim,
                        dropout
                    )
                )

                curr_channels = out_channels_level

                if resolution in attention_resolutions:

                    layers.append(
                        SpatialTransformer(
                            curr_channels,
                            context_dim=context_dim
                        )
                    )

                self.ups.append(
                    nn.ModuleList(layers)
                )

            if level != 0:

                self.ups.append(
                    nn.ModuleList([
                        Upsample(curr_channels)
                    ])
                )

                resolution *= 2

        # =====================================================
        # OUTPUT
        # =====================================================

        self.out_norm = nn.GroupNorm(
            32,
            curr_channels
        )

        self.out_act = SiLU()

        self.out_conv = nn.Conv2d(
            curr_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(self, x, t, context):

        t = self.time_mlp(t)

        h = self.conv_in(x)

        skips = [h]

        # =====================================================
        # DOWN
        # =====================================================

        for layer_group in self.downs:

            for layer in layer_group:

                if isinstance(layer, ResidualBlock):

                    h = layer(h, t)

                elif isinstance(layer, SpatialTransformer):

                    h = layer(h, context)

                else:

                    h = layer(h)

            skips.append(h)

        # =====================================================
        # MID
        # =====================================================

        for layer in self.mid:

            if isinstance(layer, ResidualBlock):

                h = layer(h, t)

            elif isinstance(layer, SpatialTransformer):

                h = layer(h, context)

            else:

                h = layer(h)

        # =====================================================
        # UP
        # =====================================================

        for layer_group in self.ups:

            is_upsample_group = isinstance(
                layer_group[0],
                Upsample
            )

            if is_upsample_group:

                h = layer_group[0](h)

            else:

                skip = skips.pop()

                h = torch.cat(
                    (h, skip),
                    dim=1
                )

                for layer in layer_group:

                    if isinstance(layer, ResidualBlock):

                        h = layer(h, t)

                    elif isinstance(layer, SpatialTransformer):

                        h = layer(h, context)

                    else:

                        h = layer(h)

        # =====================================================
        # OUTPUT
        # =====================================================

        h = self.out_norm(h)

        h = self.out_act(h)

        h = self.out_conv(h)

        return h


# =============================================================
# TESTE
# =============================================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🧪 Testando UNet Condicional no device: {device}\n")

    model = UNet_cond(
        in_channels=4,
        out_channels=4,
        base_channels=128,
        context_dim=512,
    ).to(device)

    x = torch.randn(
        2,
        4,
        32,
        32
    ).to(device)

    t = torch.randint(
        0,
        1000,
        (2,)
    ).to(device)

    # =========================================================
    # AGORA:
    # 40 atributos do CelebA
    # cada atributo -> embedding 512
    # contexto final:
    # [B, 512, 40]
    # =========================================================

    context = torch.randn(
        2,
        512,
        40
    ).to(device)

    with torch.no_grad():

        out = model(
            x,
            t,
            context
        )

    print(f"✅ Input shape:   {x.shape}")
    print(f"✅ Context shape: {context.shape}")
    print(f"✅ Output shape:  {out.shape}")
    print(f"\n✅ Tudo funcionando perfeitamente.")