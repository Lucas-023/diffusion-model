"""
Geração condicional a partir de uma foto de referência.

Fluxo:
  1. Carrega a foto de referência.
  2. Extrai embedding de identidade via ArcFaceEncoder.
  3. Obtém atributos: da referência via AttributePredictor ou manualmente.
  4. Edita atributos desejados (ativa / desativa).
  5. Gera imagem da mesma pessoa com os atributos editados.

Uso básico:
  python generate_from_ref.py \
      --ref_image foto.jpg \
      --ckpt models/LDM_Identity_Conditional/ckpt.pt \
      --vae_ckpt vae/vae_epoch_62.pt \
      --enable Smiling Eyeglasses \
      --disable Bald

Uso com atributos manuais (sem AttributePredictor):
  python generate_from_ref.py \
      --ref_image foto.jpg \
      --ckpt ... \
      --manual_attrs Young Smiling Black_Hair
"""

import os
import argparse

import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image

from models.unet_conditional import UNet_cond
from models.modules import AttributeEmbedder, IdentityAdapter
from models.encoders import ArcFaceEncoder, AttributePredictor
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
# TRANSFORM
# ============================================================

TRANSFORM = T.Compose([
    T.CenterCrop(178),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)   # [1, 3, 256, 256]


# ============================================================
# MAIN
# ============================================================

def generate(args):

    device = args.device

    # ========================================================
    # VAE
    # ========================================================

    print("Carregando VAE...")

    vae = VAE(in_channels=3, latent_dim=4).to(device)
    vae.load_state_dict(
        torch.load(args.vae_ckpt, map_location=device)
    )
    vae.eval()

    for p in vae.parameters():
        p.requires_grad = False

    # ========================================================
    # UNET + EMBEDDERS
    # ========================================================

    print("Carregando UNet e embedders...")

    unet = UNet_cond(
        in_channels=4,
        out_channels=4,
        context_dim=512,
    ).to(device)

    attribute_embedder = AttributeEmbedder(
        num_attributes=40,
        context_dim=512,
    ).to(device)

    identity_adapter = IdentityAdapter(
        identity_dim=512,
        context_dim=512,
        num_tokens=4,
    ).to(device)

    identity_encoder = ArcFaceEncoder().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)

    unet.load_state_dict(
        ckpt.get("ema_state_dict", ckpt["model_state_dict"])
    )

    attribute_embedder.load_state_dict(
        ckpt.get("ema_embedder_state_dict", ckpt["attribute_embedder_state_dict"])
    )

    identity_adapter.load_state_dict(
        ckpt.get("ema_id_adapter_state_dict", ckpt["identity_adapter_state_dict"])
    )

    unet.eval()
    attribute_embedder.eval()
    identity_adapter.eval()
    identity_encoder.eval()

    print(f"Checkpoint: época {ckpt.get('epoch', '?')}")

    # ========================================================
    # IMAGEM DE REFERÊNCIA
    # ========================================================

    print(f"Carregando referência: {args.ref_image}")

    ref_img = load_image(args.ref_image).to(device)

    # ========================================================
    # IDENTIDADE
    # ========================================================

    with torch.no_grad():
        identity_emb = identity_encoder(ref_img)          # [1, 512]
        identity_emb = identity_emb.repeat(args.n, 1)     # [N, 512]
        id_tokens = identity_adapter(identity_emb)         # [N, 512, 4]

    # ========================================================
    # ATRIBUTOS
    # ========================================================

    if args.manual_attrs:

        # Atributos definidos manualmente
        attrs_vec = torch.zeros(40, dtype=torch.float32)

        for name in args.manual_attrs:
            if name not in ATTR_TO_IDX:
                raise ValueError(
                    f"Atributo desconhecido: '{name}'\n"
                    f"Disponíveis: {CELEBA_ATTRS}"
                )
            attrs_vec[ATTR_TO_IDX[name]] = 1.0

        print("Atributos manuais definidos.")

    else:

        # Extrai atributos da foto de referência via AttributePredictor
        attr_predictor = AttributePredictor(
            num_attributes=40,
            pretrained_path=args.attr_predictor_ckpt,
        ).to(device)

        attr_predictor.eval()

        with torch.no_grad():
            attrs_vec = attr_predictor.predict(ref_img).squeeze(0).cpu()

        active = [
            CELEBA_ATTRS[i]
            for i in range(40)
            if attrs_vec[i] == 1.0
        ]

        print(f"Atributos detectados na referência: {active}")

    # ========================================================
    # EDIÇÃO DE ATRIBUTOS
    # ========================================================

    for name in (args.enable or []):
        if name not in ATTR_TO_IDX:
            raise ValueError(f"Atributo desconhecido: '{name}'")
        attrs_vec[ATTR_TO_IDX[name]] = 1.0
        print(f"  + Ativado: {name}")

    for name in (args.disable or []):
        if name not in ATTR_TO_IDX:
            raise ValueError(f"Atributo desconhecido: '{name}'")
        attrs_vec[ATTR_TO_IDX[name]] = 0.0
        print(f"  - Desativado: {name}")

    active_final = [
        CELEBA_ATTRS[i]
        for i in range(40)
        if attrs_vec[i] == 1.0
    ]

    print(f"Atributos finais: {active_final}")

    attrs = attrs_vec.unsqueeze(0).repeat(args.n, 1).to(device)

    # ========================================================
    # CONTEXTO COMBINADO
    # ========================================================

    with torch.no_grad():

        attr_ctx = attribute_embedder(attrs)                      # [N, 512, 40]
        context = torch.cat([attr_ctx, id_tokens], dim=-1)        # [N, 512, 44]

    # ========================================================
    # DIFUSÃO
    # ========================================================

    diffusion = Diffusion_conditional(
        img_size=32,
        device=device
    )

    print(f"Gerando {args.n} imagens com CFG={args.cfg_scale}...")

    with torch.no_grad():

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

    out_path = os.path.join(args.save_dir, "generated_from_ref.png")

    save_image(images, out_path, nrow=args.nrow)

    print(f"Salvo em: {out_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gera imagens a partir de uma foto de referência"
    )

    parser.add_argument(
        "--ref_image",
        type=str,
        required=True,
        help="Caminho para a foto de referência da pessoa"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/LDM_Identity_Conditional/ckpt.pt",
        help="Checkpoint do modelo (UNet + AttributeEmbedder + IdentityAdapter)"
    )

    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="vae/vae_epoch_62.pt",
        help="Checkpoint da VAE"
    )

    parser.add_argument(
        "--attr_predictor_ckpt",
        type=str,
        default=None,
        help="Checkpoint do AttributePredictor. None usa ImageNet (impreciso)."
    )

    parser.add_argument(
        "--manual_attrs",
        nargs="*",
        default=None,
        help=(
            "Define atributos manualmente em vez de usar o AttributePredictor. "
            "Ex: --manual_attrs Young Smiling Black_Hair"
        )
    )

    parser.add_argument(
        "--enable",
        nargs="*",
        default=[],
        help="Atributos a ativar na imagem gerada. Ex: --enable Smiling Eyeglasses"
    )

    parser.add_argument(
        "--disable",
        nargs="*",
        default=[],
        help="Atributos a desativar na imagem gerada. Ex: --disable Bald"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Número de imagens a gerar"
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
        default="results/from_ref",
        help="Diretório para salvar as imagens"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda ou cpu"
    )

    args = parser.parse_args()

    generate(args)
