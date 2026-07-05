"""
generate_cfg_composable.py
==========================
Inferência com Composable CFG (Liu et al., ECCV 2022) — identidade e
atributos como dois condicionantes INDEPENDENTES.

    eps = eps(∅,∅)
        + s_id   · [eps(id,∅)    − eps(∅,∅)]
        + s_attr · [eps(id,attr) − eps(id,∅)]

Carrega checkpoints produzidos por train_cfg_composable.py.

Uso básico:
    python generate_cfg_composable.py \\
        --ref_image foto.jpg \\
        --ckpt models/LDM_CFGComposable/ckpt_best.pt \\
        --manual_attrs Smiling Young \\
        --s_id 3.0 --s_attr 5.0

Modo --from_test (amostras aleatórias do split de teste, pares orig|gerada):
    python generate_cfg_composable.py \\
        --from_test --seed 0 --n 4 \\
        --ckpt models/LDM_CFGComposable/ckpt_best.pt \\
        --enable Smiling --disable Bald
"""

import os
import argparse

import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from models.unet_conditional import UNet_cond
from models.encoders import ImageConditionEncoder, ArcFaceOnlyEncoder, AttributePredictor
from models.modules import AttributeEmbedder
from diffusion.conditional_ddpm import Diffusion_conditional
from vae.modules import VAE
from utils.utils_celeba import CelebALatentIdentityDataset
from utils.edit_common import _fix_clip_state_dict


# ============================================================
# CELEBA
# ============================================================

CELEBA_ATTRS = [
    "5_o_Clock_Shadow",    "Arched_Eyebrows",    "Attractive",
    "Bags_Under_Eyes",     "Bald",                "Bangs",
    "Big_Lips",            "Big_Nose",            "Black_Hair",
    "Blond_Hair",          "Blurry",              "Brown_Hair",
    "Bushy_Eyebrows",      "Chubby",              "Double_Chin",
    "Eyeglasses",          "Goatee",              "Gray_Hair",
    "Heavy_Makeup",        "High_Cheekbones",     "Male",
    "Mouth_Slightly_Open", "Mustache",            "Narrow_Eyes",
    "No_Beard",            "Oval_Face",           "Pale_Skin",
    "Pointy_Nose",         "Receding_Hairline",   "Rosy_Cheeks",
    "Sideburns",           "Smiling",             "Straight_Hair",
    "Wavy_Hair",           "Wearing_Earrings",    "Wearing_Hat",
    "Wearing_Lipstick",    "Wearing_Necklace",    "Wearing_Necktie",
    "Young",
]

ATTR_TO_IDX = {n: i for i, n in enumerate(CELEBA_ATTRS)}

TRANSFORM = T.Compose([
    T.CenterCrop(178),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)


def _load_test_samples(n, seed, data_root="./CelebA_data", latent_dir="./cache_latent"):
    image_dir = os.path.join(data_root, "celeba", "img_align_celeba", "img_align_celeba")

    txt_path = os.path.join(data_root, "celeba", "list_attr_celeba.txt")
    csv_path = os.path.join(data_root, "celeba", "list_attr_celeba.csv")
    attr_path = txt_path if os.path.exists(txt_path) else csv_path

    dataset = CelebALatentIdentityDataset(
        latent_dir=latent_dir,
        attr_path=attr_path,
        image_dir=image_dir,
        arcface_dir=None,
    )

    total      = len(dataset)
    train_size = int(0.70 * total)
    val_size   = int(0.15 * total)
    test_size  = total - train_size - val_size

    _, _, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    rng     = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(test_dataset), generator=rng)[:n].tolist()

    samples = []
    for idx in indices:
        _, attrs, ref_img = test_dataset[idx]
        samples.append((ref_img, attrs))
    return samples


# ============================================================
# COMPOSABLE CFG SAMPLER
# ============================================================

@torch.no_grad()
def sample_composable_cfg(
    unet,
    diffusion,
    id_tokens,        # [N, C, T_id]
    attr_tokens,      # [N, C, 40]
    n,
    channels=4,
    s_id=3.0,
    s_attr=5.0,
    device="cuda",
):
    unet.eval()
    img_size = diffusion.img_size
    z_t = torch.randn(n, channels, img_size, img_size, device=device)

    zeros_id   = torch.zeros_like(id_tokens)
    zeros_attr = torch.zeros_like(attr_tokens)

    ctx_uu = torch.cat([zeros_id,   zeros_attr], dim=2)
    ctx_iu = torch.cat([id_tokens,  zeros_attr], dim=2)
    ctx_ia = torch.cat([id_tokens,  attr_tokens], dim=2)

    for i in tqdm(reversed(range(1, diffusion.noise_steps)),
                  desc="Sampling", total=diffusion.noise_steps - 1):

        t_vec = torch.full((n,), i, device=device, dtype=torch.long)

        alpha_hat_t = diffusion.alpha_hat[t_vec][:, None, None, None]
        alpha_t     = diffusion.alpha[t_vec][:, None, None, None]
        beta_t      = diffusion.beta[t_vec][:, None, None, None]

        eps_uu = unet(z_t, t_vec, context=ctx_uu)
        eps_iu = unet(z_t, t_vec, context=ctx_iu)
        eps_ia = unet(z_t, t_vec, context=ctx_ia)

        eps = eps_uu + s_id * (eps_iu - eps_uu) + s_attr * (eps_ia - eps_iu)

        noise = torch.randn_like(z_t) if i > 1 else torch.zeros_like(z_t)
        z_t = (
            (1.0 / torch.sqrt(alpha_t))
            * (z_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t) * eps)
        ) + torch.sqrt(beta_t) * noise

    return z_t


# ============================================================
# ENCODER DISPATCH
# ============================================================

def _build_id_tokens(image_encoder, encoder_type, ref_img):
    if encoder_type == "arcface_only":
        return image_encoder(ref_img=ref_img)
    return image_encoder(ref_img=ref_img)


# ============================================================
# MAIN
# ============================================================

def generate(args):

    device = args.device

    # ---------- VAE ----------
    print("Carregando VAE...")
    vae = VAE(in_channels=3, latent_dim=4).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ---------- CHECKPOINT ----------
    print(f"Carregando checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    num_img_tokens = ckpt.get("num_img_tokens", args.img_tokens)
    encoder_type   = ckpt.get("encoder", args.encoder)
    print(f"  época: {ckpt.get('epoch', '?')} | tokens: {num_img_tokens} | encoder: {encoder_type}")

    # ---------- UNET ----------
    unet = UNet_cond(in_channels=4, out_channels=4, context_dim=512).to(device)
    unet.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
    unet.eval()

    # ---------- IDENTITY ENCODER ----------
    if encoder_type == "arcface_only":
        image_encoder = ArcFaceOnlyEncoder(context_dim=512, num_tokens=num_img_tokens).to(device)
    elif encoder_type == "clip_arcface":
        image_encoder = ImageConditionEncoder(
            context_dim=512, num_tokens=num_img_tokens, freeze_backbone=True,
        ).to(device)
    else:
        raise ValueError(f"encoder desconhecido no ckpt: {encoder_type}")

    image_encoder_state = ckpt.get(
        "ema_image_encoder_state_dict", ckpt["image_encoder_state_dict"]
    )
    if encoder_type == "clip_arcface":
        image_encoder_state = _fix_clip_state_dict(image_encoder_state)
    image_encoder.load_state_dict(image_encoder_state)
    image_encoder.eval()

    # ---------- ATTRIBUTE EMBEDDER ----------
    attribute_embedder = AttributeEmbedder(num_attributes=40, context_dim=512).to(device)
    attribute_embedder.load_state_dict(
        ckpt.get("ema_attribute_embedder_state_dict", ckpt["attribute_embedder_state_dict"])
    )
    attribute_embedder.eval()

    # ---------- DIFFUSION ----------
    diffusion = Diffusion_conditional(
        img_size=32, device=device, noise_steps=args.noise_steps,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # =====================================================
    # MODO --from_test
    # =====================================================
    if args.from_test:
        print(f"\nModo --from_test  seed={args.seed}  n={args.n}")
        samples = _load_test_samples(
            n=args.n, seed=args.seed,
            data_root=args.data_root, latent_dir=args.latent_dir,
        )

        orig_imgs, gen_imgs = [], []

        for i, (ref_cpu, attrs_cpu) in enumerate(samples):
            print(f"\n--- Amostra {i+1}/{args.n} ---")
            ref_img = ref_cpu.unsqueeze(0).to(device)

            attrs_vec = attrs_cpu.clone()
            for name in (args.enable or []):
                attrs_vec[ATTR_TO_IDX[name]] = 1.0
                print(f"  + {name}")
            for name in (args.disable or []):
                attrs_vec[ATTR_TO_IDX[name]] = 0.0
                print(f"  - {name}")

            active = [CELEBA_ATTRS[j] for j in range(40) if attrs_vec[j] == 1.0]
            print(f"  atributos finais: {active}")

            attrs = attrs_vec.unsqueeze(0).to(device)

            with torch.no_grad():
                id_tok   = _build_id_tokens(image_encoder, encoder_type, ref_img)
                attr_tok = attribute_embedder(attrs)

            latents = sample_composable_cfg(
                unet=unet, diffusion=diffusion,
                id_tokens=id_tok, attr_tokens=attr_tok,
                n=1, channels=4,
                s_id=args.s_id, s_attr=args.s_attr,
                device=device,
            )

            with torch.no_grad():
                gen = vae.decode(latents / 0.18215)

            orig_imgs.append(((ref_cpu.clamp(-1, 1) + 1) / 2).cpu())
            gen_imgs.append(((gen.squeeze(0).clamp(-1, 1) + 1) / 2).cpu())

        grid = []
        for o, g in zip(orig_imgs, gen_imgs):
            grid.append(o); grid.append(g)
        grid = torch.stack(grid)

        out_path = os.path.join(args.save_dir, f"test_seed{args.seed}_sid{args.s_id}_sattr{args.s_attr}.png")
        save_image(grid, out_path, nrow=2)
        print(f"\nSalvo em: {out_path}")
        return

    # =====================================================
    # MODO --ref_image
    # =====================================================
    print(f"Carregando referência: {args.ref_image}")
    ref_img = load_image(args.ref_image).to(device)
    ref_img = ref_img.repeat(args.n, 1, 1, 1)

    if args.manual_attrs is not None:
        attrs_vec = torch.zeros(40, dtype=torch.float32)
        for name in args.manual_attrs:
            attrs_vec[ATTR_TO_IDX[name]] = 1.0
        print(f"Atributos manuais: {args.manual_attrs}")
    else:
        attr_predictor = AttributePredictor(
            num_attributes=40, pretrained_path=args.attr_predictor_ckpt,
        ).to(device)
        attr_predictor.eval()
        ref_single = load_image(args.ref_image).to(device)
        with torch.no_grad():
            attrs_vec = attr_predictor.predict(ref_single).squeeze(0).cpu()
        detected = [CELEBA_ATTRS[i] for i in range(40) if attrs_vec[i] == 1.0]
        print(f"Atributos detectados: {detected}")

    for name in (args.enable or []):
        attrs_vec[ATTR_TO_IDX[name]] = 1.0; print(f"  + {name}")
    for name in (args.disable or []):
        attrs_vec[ATTR_TO_IDX[name]] = 0.0; print(f"  - {name}")

    attrs = attrs_vec.unsqueeze(0).repeat(args.n, 1).to(device)

    with torch.no_grad():
        id_tok   = _build_id_tokens(image_encoder, encoder_type, ref_img)
        attr_tok = attribute_embedder(attrs)

    print(f"\nGerando {args.n} imagens com s_id={args.s_id}  s_attr={args.s_attr}")

    latents = sample_composable_cfg(
        unet=unet, diffusion=diffusion,
        id_tokens=id_tok, attr_tokens=attr_tok,
        n=args.n, channels=4,
        s_id=args.s_id, s_attr=args.s_attr,
        device=device,
    )

    with torch.no_grad():
        images = vae.decode(latents / 0.18215)

    images = (images.clamp(-1, 1) + 1) / 2

    out_path = os.path.join(args.save_dir, "generated_composable.png")
    save_image(images, out_path, nrow=args.nrow)
    print(f"\nSalvo em: {out_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Geração com Composable CFG (identidade + atributos independentes)"
    )

    parser.add_argument("--ref_image",   type=str, default=None)
    parser.add_argument("--from_test",   action="store_true")
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--data_root",   type=str, default="./CelebA_data")
    parser.add_argument("--latent_dir",  type=str, default="./cache_latent")

    parser.add_argument("--ckpt",        type=str,
                        default="models/LDM_CFGComposable/ckpt_best.pt")
    parser.add_argument("--vae_ckpt",    type=str, default="vae/vae_epoch_62.pt")
    parser.add_argument("--attr_predictor_ckpt", type=str, default=None)

    parser.add_argument("--manual_attrs", nargs="*", default=None,
                        help="Define atributos manualmente. Ex: --manual_attrs Smiling Young")
    parser.add_argument("--enable",  nargs="*", default=[])
    parser.add_argument("--disable", nargs="*", default=[])

    parser.add_argument("--n",     type=int, default=4)
    parser.add_argument("--nrow",  type=int, default=4)

    # Escalas Composable CFG
    parser.add_argument("--s_id",   type=float, default=3.0,
                        help="Escala do termo de identidade.")
    parser.add_argument("--s_attr", type=float, default=5.0,
                        help="Escala do termo de atributos (baseline = id-only).")

    # Fallbacks se o ckpt for antigo
    parser.add_argument("--encoder", type=str, default="clip_arcface",
                        choices=["clip_arcface", "arcface_only"])
    parser.add_argument("--img_tokens", type=int, default=16)

    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--save_dir",   type=str, default="results/cfg_composable")
    parser.add_argument("--device",     type=str, default="cuda")

    args = parser.parse_args()

    if not args.from_test and args.ref_image is None:
        parser.error("Informe --ref_image ou use --from_test.")

    if args.enable:
        for n in args.enable:
            if n not in ATTR_TO_IDX:
                parser.error(f"Atributo desconhecido em --enable: {n}")
    if args.disable:
        for n in args.disable:
            if n not in ATTR_TO_IDX:
                parser.error(f"Atributo desconhecido em --disable: {n}")
    if args.manual_attrs:
        for n in args.manual_attrs:
            if n not in ATTR_TO_IDX:
                parser.error(f"Atributo desconhecido em --manual_attrs: {n}")

    generate(args)
