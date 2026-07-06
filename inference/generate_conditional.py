import os
import argparse
import torch
from torchvision.utils import save_image

from models.unet_conditional import UNet_cond
from models.modules import AttributeEmbedder
from vae.modules import VAE
from diffusion.conditional_ddpm import Diffusion_conditional


# ============================================================
# ATRIBUTOS CELEBA
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

ATTR_TO_IDX = {name: i for i, name in enumerate(CELEBA_ATTRS)}


# ============================================================
# GERAÇÃO
# ============================================================

def generate(args):

    device = args.device

    # ========================================================
    # MODELOS
    # ========================================================

    print("Carregando modelos...")

    vae = VAE(in_channels=3, latent_dim=4).to(device)
    vae.load_state_dict(
        torch.load(args.vae_ckpt, map_location=device)
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    unet = UNet_cond(
        in_channels=4,
        out_channels=4,
        context_dim=512
    ).to(device)

    attribute_embedder = AttributeEmbedder(
        num_attributes=40,
        context_dim=512
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)

    unet_weights = ckpt.get("ema_state_dict", ckpt.get("model_state_dict"))
    embedder_weights = ckpt.get(
        "ema_embedder_state_dict",
        ckpt.get("attribute_embedder_state_dict")
    )

    unet.load_state_dict(unet_weights)
    attribute_embedder.load_state_dict(embedder_weights)

    unet.eval()
    attribute_embedder.eval()

    print(f"Checkpoint: época {ckpt.get('epoch', '?')}")

    # ========================================================
    # ATRIBUTOS
    # ========================================================

    attrs_vec = torch.zeros(40, dtype=torch.float32)

    if args.attrs:
        for name in args.attrs:
            if name not in ATTR_TO_IDX:
                raise ValueError(
                    f"Atributo desconhecido: '{name}'\n"
                    f"Disponiveis: {CELEBA_ATTRS}"
                )
            attrs_vec[ATTR_TO_IDX[name]] = 1.0

    active = [CELEBA_ATTRS[i] for i in range(40) if attrs_vec[i] == 1.0]
    print(f"Atributos ativos: {active if active else ['nenhum (geracao livre)']}")

    attrs = attrs_vec.unsqueeze(0).repeat(args.n, 1).to(device)

    # ========================================================
    # DIFUSÃO
    # ========================================================

    diffusion = Diffusion_conditional(
        img_size=32,
        device=device
    )

    # ========================================================
    # GERAR
    # ========================================================

    print(f"Gerando {args.n} imagens com CFG={args.cfg_scale}...")

    with torch.no_grad():

        context = attribute_embedder(attrs)

        sampled_latents = diffusion.sample(
            unet,
            n=args.n,
            context=context,
            channels=4,
            cfg_scale=args.cfg_scale
        )

        sampled_latents = sampled_latents / 0.18215

        images = vae.decode(sampled_latents)

    images = (images.clamp(-1, 1) + 1) / 2

    os.makedirs(args.save_dir, exist_ok=True)

    out_path = os.path.join(args.save_dir, "generated.png")

    save_image(images, out_path, nrow=args.nrow)

    print(f"Salvo em: {out_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gera imagens condicionadas em atributos do CelebA"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/LDM_Conditional_Attributes/ckpt.pt",
        help="Caminho para o checkpoint do modelo condicional"
    )

    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="vae/vae_epoch_62.pt",
        help="Caminho para o checkpoint da VAE"
    )

    parser.add_argument(
        "--attrs",
        nargs="*",
        default=[],
        help=(
            "Atributos CelebA a ativar. Ex: --attrs Smiling Young Blond_Hair\n"
            f"Disponiveis: {CELEBA_ATTRS}"
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=16,
        help="Numero de imagens a gerar"
    )

    parser.add_argument(
        "--nrow",
        type=int,
        default=4,
        help="Imagens por linha no grid"
    )

    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
        help="Escala de Classifier-Free Guidance"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/generated_conditional",
        help="Diretorio para salvar as imagens"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda ou cpu"
    )

    args = parser.parse_args()

    generate(args)
