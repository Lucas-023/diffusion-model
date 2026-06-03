import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def normalization(channels):
    return nn.GroupNorm(32, channels)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)

    raise ValueError(f"unsupported dims: {dims}")

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()

    return module

# =========================================================
# QKV ATTENTION
# =========================================================

class QKVAttentionLegacy(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):

        bs, width, length = qkv.shape

        ch = width // (3 * self.n_heads)

        q, k, v = qkv.reshape(
            bs * self.n_heads,
            ch * 3,
            length
        ).split(ch, dim=1)

        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = torch.einsum(
            "bct,bcs->bts",
            q * scale,
            k * scale
        )

        weight = torch.softmax(
            weight.float(),
            dim=-1
        ).type(weight.dtype)

        a = torch.einsum(
            "bts,bcs->bct",
            weight,
            v
        )

        return a.reshape(bs, -1, length)

class QKVAttention(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):

        bs, width, length = qkv.shape

        ch = width // (3 * self.n_heads)

        q, k, v = qkv.chunk(3, dim=1)

        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = torch.einsum(
            "bct,bcs->bts",

            (q * scale).view(
                bs * self.n_heads,
                ch,
                length
            ),

            (k * scale).view(
                bs * self.n_heads,
                ch,
                length
            ),
        )

        weight = torch.softmax(
            weight.float(),
            dim=-1
        ).type(weight.dtype)

        a = torch.einsum(
            "bts,bcs->bct",
            weight,
            v.reshape(
                bs * self.n_heads,
                ch,
                length
            )
        )

        return a.reshape(bs, -1, length)

# =========================================================
# RESBLOCK
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            32,
            in_channels
        )

        self.act1 = SiLU()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.time_emb = nn.Linear(
            time_emb_dim,
            out_channels
        )

        self.time_act = SiLU()

        self.norm2 = nn.GroupNorm(
            32,
            out_channels
        )

        self.act2 = SiLU()

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1
        )

        if in_channels != out_channels:

            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                1
            )

        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        t_vec = self.time_act(t)
        t_vec = self.time_emb(t_vec)[:, :, None, None]

        h = h + t_vec

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

# =========================================================
# DOWNSAMPLE / UPSAMPLE
# =========================================================

class Downsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode='nearest'
        )

        return self.conv(x)

# =========================================================
# ATTENTION BLOCK
# =========================================================

class AttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()

        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)

        self.qkv = conv_nd(
            1,
            channels,
            channels * 3,
            1
        )

        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(
            conv_nd(
                1,
                channels,
                channels,
                1
            )
        )

    def forward(self, x):

        b, c, *spatial = x.shape

        x_in = x

        x = x.reshape(b, c, -1)

        qkv = self.qkv(
            self.norm(x)
        )

        h = self.attention(qkv)

        h = self.proj_out(h)

        return (x + h).reshape(
            b,
            c,
            *spatial
        )

# =========================================================
# POSITIONAL EMBEDDING
# =========================================================

class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):

        device = time.device

        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)

        embeddings = torch.exp(
            torch.arange(
                half_dim,
                device=device
            ) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat([
            torch.sin(embeddings),
            torch.cos(embeddings)
        ], dim=-1)

        return embeddings

# =========================================================
# CROSS ATTENTION
# =========================================================

class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        cond_dim,
        num_heads=1,
        num_head_channels=-1
    ):
        super().__init__()

        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)

        self.to_q = conv_nd(
            1,
            channels,
            channels,
            1
        )

        self.to_kv = conv_nd(
            1,
            cond_dim,
            channels * 2,
            1
        )

        self.proj_out = zero_module(
            conv_nd(
                1,
                channels,
                channels,
                1
            )
        )

    def forward(self, x, cond):

        b, c, *spatial = x.shape

        x_in = x

        x = x.reshape(b, c, -1)

        x = self.norm(x)

        q = self.to_q(x)

        cond = cond.reshape(
            b,
            cond.shape[1],
            -1
        )

        kv = self.to_kv(cond)

        k, v = kv.chunk(2, dim=1)

        hw_seq_len = q.shape[2]

        cond_seq_len = k.shape[2]

        ch_per_head = c // self.num_heads

        q = q.view(
            b * self.num_heads,
            ch_per_head,
            hw_seq_len
        )

        k = k.view(
            b * self.num_heads,
            ch_per_head,
            cond_seq_len
        )

        v = v.view(
            b * self.num_heads,
            ch_per_head,
            cond_seq_len
        )

        scale = 1 / math.sqrt(math.sqrt(ch_per_head))

        weight = torch.einsum(
            "bct,bcs->bts",
            q * scale,
            k * scale
        )

        weight = torch.softmax(
            weight.float(),
            dim=-1
        ).type(weight.dtype)

        a = torch.einsum(
            "bts,bcs->bct",
            weight,
            v
        )

        a = a.reshape(
            b,
            c,
            hw_seq_len
        )

        h = self.proj_out(a)

        return (
            x_in.reshape(b, c, -1) + h
        ).reshape(
            b,
            c,
            *spatial
        )

# =========================================================
# SPATIAL TRANSFORMER
# =========================================================

class SpatialTransformer(nn.Module):

    def __init__(
        self,
        channels,
        context_dim,
        num_heads=1,
        num_head_channels=-1
    ):
        super().__init__()

        self.attn1 = AttentionBlock(
            channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels
        )

        self.attn2 = CrossAttentionBlock(
            channels,
            context_dim,
            num_heads=num_heads,
            num_head_channels=num_head_channels
        )

        self.norm3 = normalization(channels)

        self.ff1 = conv_nd(
            2,
            channels,
            channels * 4,
            1
        )

        self.act = SiLU()

        self.ff2 = zero_module(
            conv_nd(
                2,
                channels * 4,
                channels,
                1
            )
        )

    def forward(self, x, context):

        x = self.attn1(x)

        x = self.attn2(
            x,
            cond=context
        )

        h = self.norm3(x)

        h = self.ff1(h)

        h = self.act(h)

        h = self.ff2(h)

        return x + h

# =========================================================
# ATTRIBUTE EMBEDDER
# =========================================================

class AttributeEmbedder(nn.Module):

    def __init__(
        self,
        num_attributes=40,
        context_dim=512
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            2,
            context_dim
        )

        self.attribute_tokens = nn.Parameter(
            torch.randn(
                num_attributes,
                context_dim
            )
        )

    def forward(self, attrs):

        attrs = attrs.long()

        x = self.embedding(attrs)

        x = x + self.attribute_tokens.unsqueeze(0)

        x = x.permute(0, 2, 1)

        return x


# =========================================================
# IDENTITY ADAPTER
# =========================================================

class IdentityAdapter(nn.Module):
    """
    Projects a per-image identity embedding into a sequence of
    context tokens that can be concatenated with the attribute
    context before being fed to the UNet cross-attention.

    Output shape [B, context_dim, num_tokens] matches the
    attribute context shape [B, context_dim, 40] so they can
    be concatenated along the last dimension.
    """

    def __init__(
        self,
        identity_dim=512,
        context_dim=512,
        num_tokens=4
    ):
        super().__init__()

        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(identity_dim, context_dim * num_tokens),
            nn.GELU(),
            nn.Linear(context_dim * num_tokens, context_dim * num_tokens),
        )

        self.norm = nn.LayerNorm(context_dim)

    def forward(self, identity_emb):
        # identity_emb : [B, identity_dim]
        # returns      : [B, context_dim, num_tokens]
        B = identity_emb.shape[0]

        tokens = self.proj(identity_emb)

        tokens = tokens.view(B, self.num_tokens, -1)

        tokens = self.norm(tokens)

        return tokens.permute(0, 2, 1)