import argparse
import os

import torch

from diffusion.conditional_ddpm import Diffusion_conditional
from models.modules import AttributeEmbedder
from models.unet_conditional import UNet_cond
from utils.utils_celeba import save_images
from vae.modules import VAE


CELEBA_ATTRS = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


def parse_attributes(text: str) -> torch.Tensor:
    """
    Aceita algo como:
    'Smiling=1,Young=1,Blond_Hair=0'
    e retorna um tensor [1, 40] com 0/1.
    """
    values = {name.strip(): 0 for name in CELEBA_ATTRS}

    if text.strip():
        for item in text.split(','):
            if '=' not in item:
                raise ValueError(
                    f"Atributo inválido: '{item}'. Use Nome=0 ou Nome=1."
                )
            name, raw_value = item.split('=', 1)
            name = name.strip()
            if name not in values:
                raise ValueError(
                    f"Atributo desconhecido: '{name}'. "
                    f"Use um dos nomes da lista CelebA."
                )
            try:
                values[name] = int(float(raw_value.strip()))
            except ValueError as exc:
                raise ValueError(f"Valor inválido para {name}: {raw_value}") from exc

    vector = torch.tensor([values[name] for name in CELEBA_ATTRS], dtype=torch.float32)
    return vector.unsqueeze(0)


def resolve_checkpoint(run_name: str, device: str) -> str:
    candidates = [
        os.path.join("models", run_name, "best_dif_cond_att_ckpt.pt"),
        os.path.join("models", run_name, "dif_cond_att_ckpt.pt"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            print(f"✅ Checkpoint encontrado: {path}")
            return path

    raise FileNotFoundError(
        "Nenhum checkpoint encontrado. "
        "Treine o modelo primeiro e gere os arquivos 'best_dif_cond_att_ckpt.pt' ou 'dif_cond_att_ckpt.pt' "
        "em 'models/<run_name>/'."
    )


def generate_image(args):
    device = args.device
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        print("⚠️ CUDA indisponível; usando CPU.")

    print("\n🚀 Gerando imagem condicional...")
    print(f"- Device: {device}")
    print(f"- Atributos: {args.attributes}")

    # -----------------------------
    # Carrega modelo e pesos
    # -----------------------------
    ckpt_path = resolve_checkpoint(args.run_name, device)
    vae_path = os.path.join("vae", "vae_epoch_62.pt")

    if not os.path.isfile(vae_path):
        raise FileNotFoundError(
            f"Arquivo de VAE não encontrado: {vae_path}. "
            "Coloque o peso em 'vae/vae_epoch_62.pt'."
        )

    latent_dim = 4
    context_dim = 512

    model = UNet_cond(
        in_channels=latent_dim,
        out_channels=latent_dim,
        context_dim=context_dim,
    ).to(device)

    attribute_embedder = AttributeEmbedder(
        num_attributes=40,
        context_dim=context_dim,
    ).to(device)

    vae = VAE(in_channels=3, latent_dim=latent_dim).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    attribute_embedder.load_state_dict(checkpoint["attribute_embedder_state_dict"])

    vae.load_state_dict(torch.load(vae_path, map_location=device))

    model.eval()
    attribute_embedder.eval()
    vae.eval()

    # -----------------------------
    # Atributos escolhidos
    # -----------------------------
    attrs = parse_attributes(args.attributes).to(device)
    context = attribute_embedder(attrs)

    # -----------------------------
    # Geração do latente
    # -----------------------------
    diffusion = Diffusion_conditional(img_size=32, device=device)

    with torch.no_grad():
        sampled_latents = diffusion.sample(
            model,
            n=1,
            context=context,
            cfg_scale=args.cfg_scale,
        )

        # Mesmo escalonamento usado no treino
        sampled_latents = sampled_latents / 0.18215
        sampled_images = vae.decode(sampled_latents)

    os.makedirs("results/generated_conditional", exist_ok=True)
    out_path = os.path.join("results/generated_conditional", args.output_name)

    save_images(sampled_images, out_path, nrow=1)

    print(f"\n✅ Imagem salva em: {out_path}")
    print("Atributos usados:")
    for name, value in zip(CELEBA_ATTRS, attrs[0].cpu().tolist()):
        if value > 0:
            print(f"  - {name}: 1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera uma imagem condicional no CelebA")
    parser.add_argument(
        "--run_name",
        type=str,
        default="LDM_Conditional_Attributes",
        help="Nome da pasta do checkpoint treinado em models/",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        default="Smiling=1,Young=1,Attractive=1",
        help="Atributos desejados no formato Nome=0/1,Nome=0/1",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
        help="Escala de classifier-free guidance (ex.: 1.5 a 5.0)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="generated_conditional.png",
        help="Nome do arquivo de saída",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda ou cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reprodutibilidade",
    )

    args = parser.parse_args()
    generate_image(args)
