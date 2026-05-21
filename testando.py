import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from models.unet_conditional import UNet_cond
from models.modules import LatentConditionProjector
from vae.modules import VAE
from diffusion.conditional_ddpm import Diffusion_conditional


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda"

IMAGE_PATH = "teste.jpg"

CKPT_PATH = "peso/ckpt.pt"

SAVE_DIR = "teste_resultado"

IMAGE_SIZE = 256

LATENT_SCALE = 0.18215

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# LOAD MODELS
# ============================================================

print("Carregando modelos...")

vae = VAE(
    in_channels=3,
    latent_dim=4
).to(DEVICE)

unet = UNet_cond(
    in_channels=4,
    out_channels=4,
    context_dim=512
).to(DEVICE)

projector = LatentConditionProjector(
    latent_dim=4,
    context_dim=512
).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

unet.load_state_dict(ckpt["ema_state_dict"])
projector.load_state_dict(ckpt["projector_state_dict"])

vae.eval()
unet.eval()
projector.eval()

print("Modelos carregados.")


# ============================================================
# DIFFUSION
# ============================================================

diffusion = Diffusion_conditional(
    img_size=32,
    device=DEVICE
)


# ============================================================
# LOAD IMAGE
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

img = Image.open(IMAGE_PATH).convert("RGB")

img_tensor = transform(img).unsqueeze(0).to(DEVICE)

save_image(
    (img_tensor * 0.5 + 0.5).clamp(0, 1),
    os.path.join(SAVE_DIR, "condicao.png")
)

print("Imagem carregada.")


# ============================================================
# ENCODE IMAGE -> LATENT
# ============================================================

with torch.no_grad():

    mu, log_var = vae.encode(img_tensor)

    z = vae.reparameterize(mu, log_var)

    z = z * LATENT_SCALE

print("Latente criado:", z.shape)


# ============================================================
# CREATE CONTEXT
# ============================================================

with torch.no_grad():

    context = projector(z)

print("Contexto:", context.shape)


# ============================================================
# START FROM PURE NOISE
# ============================================================

x = torch.randn((1, 4, 32, 32)).to(DEVICE)

print("Iniciando denoising...")


# ============================================================
# DDPM REVERSE PROCESS
# ============================================================

with torch.no_grad():

    for i in tqdm(reversed(range(1, diffusion.noise_steps)),
                  total=diffusion.noise_steps - 1):

        t = torch.tensor([i]).to(DEVICE)

        predicted_noise = unet(
            x,
            t,
            context=context
        )

        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]

        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            1 / torch.sqrt(alpha)
        ) * (
            x - (
                (1 - alpha)
                / torch.sqrt(1 - alpha_hat)
            ) * predicted_noise
        ) + torch.sqrt(beta) * noise

        # ====================================================
        # SAVE INTERMEDIATE STEPS
        # ====================================================

        if i % 100 == 0:

            latent_preview = x / LATENT_SCALE

            decoded = vae.decode(latent_preview)

            decoded = (decoded.clamp(-1, 1) + 1) / 2

            save_image(
                decoded,
                os.path.join(SAVE_DIR, f"step_{i}.png")
            )


print("Denoising completo.")


# ============================================================
# FINAL DECODE
# ============================================================

with torch.no_grad():

    x = x / LATENT_SCALE

    final_img = vae.decode(x)

    final_img = (final_img.clamp(-1, 1) + 1) / 2


save_image(
    final_img,
    os.path.join(SAVE_DIR, "resultado_final.png")
)

print("Imagem salva.")