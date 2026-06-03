import os
import torch
from torchvision.utils import save_image

from models.unet_conditional import UNet_cond
from models.modules import AttributeEmbedder
from vae.modules import VAE
from diffusion.conditional_ddpm import Diffusion_conditional


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "models/LDM_Conditional_Attributes/ckpt.pt"

SAVE_DIR = "results/testando"

LATENT_SCALE = 0.18215

NUM_IMAGES = 16

CFG_SCALE = 3.0

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# ATRIBUTOS CELEBA (40 atributos, índice conforme dataset)
# ============================================================
# 0:  5_o_Clock_Shadow    1:  Arched_Eyebrows   2:  Attractive
# 3:  Bags_Under_Eyes     4:  Bald               5:  Bangs
# 6:  Big_Lips            7:  Big_Nose           8:  Black_Hair
# 9:  Blond_Hair         10:  Blurry            11:  Brown_Hair
# 12: Bushy_Eyebrows     13:  Chubby            14:  Double_Chin
# 15: Eyeglasses         16:  Goatee            17:  Gray_Hair
# 18: Heavy_Makeup       19:  High_Cheekbones   20:  Male
# 21: Mouth_Slightly_Open 22: Mustache          23:  Narrow_Eyes
# 24: No_Beard           25:  Oval_Face         26:  Pale_Skin
# 27: Pointy_Nose        28:  Receding_Hairline 29:  Rosy_Cheeks
# 30: Sideburns          31:  Smiling           32:  Straight_Hair
# 33: Wavy_Hair          34:  Wearing_Earrings  35:  Wearing_Hat
# 36: Wearing_Lipstick   37:  Wearing_Necklace  38:  Wearing_Necktie
# 39: Young
# ============================================================

# Exemplo: mulher jovem sorrindo com maquiagem
ATTRS_ON = [2, 5, 18, 19, 24, 25, 31, 33, 34, 36, 39]

attrs = torch.zeros(40, dtype=torch.float32)
for idx in ATTRS_ON:
    attrs[idx] = 1.0

attrs = attrs.unsqueeze(0).repeat(NUM_IMAGES, 1).to(DEVICE)


# ============================================================
# LOAD MODELS
# ============================================================

print("Carregando modelos...")

vae = VAE(in_channels=3, latent_dim=4).to(DEVICE)

unet = UNet_cond(
    in_channels=4,
    out_channels=4,
    context_dim=512
).to(DEVICE)

attribute_embedder = AttributeEmbedder(
    num_attributes=40,
    context_dim=512
).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

# Carrega pesos EMA se disponíveis (melhor qualidade)
unet_weights = ckpt.get("ema_state_dict", ckpt.get("model_state_dict"))
embedder_weights = ckpt.get(
    "ema_embedder_state_dict",
    ckpt.get("attribute_embedder_state_dict")
)

unet.load_state_dict(unet_weights)
attribute_embedder.load_state_dict(embedder_weights)

vae_ckpt_path = "vae/vae_epoch_62.pt"
vae.load_state_dict(
    torch.load(vae_ckpt_path, map_location=DEVICE)
)

vae.eval()
unet.eval()
attribute_embedder.eval()

print(f"Modelos carregados. (época {ckpt.get('epoch', '?')})")


# ============================================================
# DIFFUSION
# ============================================================

diffusion = Diffusion_conditional(
    img_size=32,
    device=DEVICE
)


# ============================================================
# GERAÇÃO COM CFG
# ============================================================

print(f"Gerando {NUM_IMAGES} imagens com CFG={CFG_SCALE}...")

with torch.no_grad():

    context = attribute_embedder(attrs)

    sampled_latents = diffusion.sample(
        unet,
        n=NUM_IMAGES,
        context=context,
        channels=4,
        cfg_scale=CFG_SCALE
    )

    sampled_latents = sampled_latents / LATENT_SCALE

    images = vae.decode(sampled_latents)

images = (images.clamp(-1, 1) + 1) / 2

save_image(
    images,
    os.path.join(SAVE_DIR, "geradas.png"),
    nrow=4
)

print(f"Imagens salvas em: {SAVE_DIR}/geradas.png")
