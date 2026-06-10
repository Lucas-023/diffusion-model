"""
test_identity.py
================
Inferência do LDM condicional por identidade (traind_cond_identity.py).

Fluxo
-----
1. Carrega foto de referência.
2. Extrai embedding de identidade via ArcFaceEncoder.
3. Obtém atributos do arquivo de anotações CelebA (ou manualmente).
4. Aplica edições de atributos (--enable / --disable).
5. Gera N imagens com DDPM + CFG.
6. Salva grid com referência + imagens geradas.

Exemplos
--------
  # Lê atributos do CelebA e ativa sorriso:
  python test/test_identity.py \
      --ref  CelebA_data/celeba/img_align_celeba/img_align_celeba/000001.jpg \
      --ckpt models/LDM_Identity_Conditional/ckpt_best.pt \
      --enable Smiling

  # Define todos os atributos manualmente:
  python test/test_identity.py \
      --ref  foto.jpg \
      --ckpt models/LDM_Identity_Conditional/ckpt_best.pt \
      --manual_attrs Young Female Smiling Black_Hair

  # Usa ckpt_best.pt automaticamente se existir:
  python test/test_identity.py \
      --ref  foto.jpg \
      --run_name LDM_Identity_Conditional
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image

from models.unet_conditional import UNet_cond
from models.modules import AttributeEmbedder, IdentityAdapter
from models.encoders import ArcFaceEncoder
from vae.modules import VAE
from diffusion.conditional_ddpm import Diffusion_conditional


# ============================================================
# ATRIBUTOS CELEBA (ordem do arquivo list_attr_celeba.txt)
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

TRANSFORM = T.Compose([
    T.CenterCrop(178),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ============================================================
# HELPERS
# ============================================================

def load_image(path):
    img = Image.open(path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)   # [1, 3, 256, 256]


def load_celeba_attrs(image_path, attrs_file):
    """Lê os atributos de uma imagem CelebA do arquivo de anotações.

    Suporta o formato TXT oficial (list_attr_celeba.txt) e CSV do Kaggle.
    O nome da imagem é extraído do basename de image_path.
    """
    image_name = os.path.basename(image_path)

    if attrs_file.endswith(".txt"):
        with open(attrs_file, "r") as f:
            lines = f.readlines()
        # linha 0: num imagens; linha 1: nomes dos atributos; linha 2+: dados
        for line in lines[2:]:
            parts = line.strip().split()
            if parts[0] == image_name:
                # valores em {-1, 1} → {0, 1}
                vals = [(int(x) + 1) // 2 for x in parts[1:]]
                return torch.tensor(vals, dtype=torch.float32)
        raise ValueError(f"Imagem '{image_name}' não encontrada em {attrs_file}")

    else:  # CSV do Kaggle
        import pandas as pd
        df = pd.read_csv(attrs_file)
        row = df[df.iloc[:, 0] == image_name]
        if row.empty:
            raise ValueError(f"Imagem '{image_name}' não encontrada em {attrs_file}")
        vals = [(int(x) + 1) // 2 for x in row.iloc[0, 1:].tolist()]
        return torch.tensor(vals, dtype=torch.float32)


def resolve_ckpt(args):
    """Retorna o caminho do checkpoint a usar.
    Preferência: --ckpt explícito > ckpt_best.pt > ckpt.pt via --run_name.
    """
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"Checkpoint não encontrado: {args.ckpt}")
        return args.ckpt

    if not args.run_name:
        raise ValueError("Informe --ckpt ou --run_name.")

    base = os.path.join("models", args.run_name)
    best = os.path.join(base, "ckpt_best.pt")
    last = os.path.join(base, "ckpt.pt")

    if os.path.isfile(best):
        print(f"Usando ckpt_best.pt de '{args.run_name}'")
        return best
    if os.path.isfile(last):
        print(f"ckpt_best.pt não encontrado, usando ckpt.pt de '{args.run_name}'")
        return last

    raise FileNotFoundError(f"Nenhum checkpoint encontrado em {base}")


def load_models(ckpt_path, device):
    """Carrega UNet, AttributeEmbedder e IdentityAdapter dos pesos EMA."""

    ckpt = torch.load(ckpt_path, map_location=device)

    num_id_tokens = 4   # padrão do traind_cond_identity.py

    unet = UNet_cond(in_channels=4, out_channels=4, context_dim=512).to(device)
    attribute_embedder = AttributeEmbedder(num_attributes=40, context_dim=512).to(device)
    identity_adapter = IdentityAdapter(
        identity_dim=512, context_dim=512, num_tokens=num_id_tokens
    ).to(device)

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

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", None)
    info = f"época {epoch}"
    if val_loss is not None:
        info += f" | val loss {val_loss:.6f}"
    print(f"Checkpoint carregado: {info}")

    return unet, attribute_embedder, identity_adapter


# ============================================================
# INFERÊNCIA
# ============================================================

def generate(args):

    device = args.device

    ckpt_path = resolve_ckpt(args)

    # ----------------------------------------------------------
    # VAE
    # ----------------------------------------------------------

    vae = VAE(in_channels=3, latent_dim=4).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ----------------------------------------------------------
    # LDM
    # ----------------------------------------------------------

    unet, attribute_embedder, identity_adapter = load_models(ckpt_path, device)

    identity_encoder = ArcFaceEncoder().to(device)
    identity_encoder.eval()

    # ----------------------------------------------------------
    # IMAGEM DE REFERÊNCIA
    # ----------------------------------------------------------

    print(f"\nReferência: {args.ref}")
    ref_img = load_image(args.ref).to(device)   # [1, 3, 256, 256]

    # ----------------------------------------------------------
    # IDENTIDADE (ArcFace)
    # ----------------------------------------------------------

    with torch.no_grad():
        identity_emb = identity_encoder(ref_img)              # [1, 512]
        identity_emb = identity_emb.repeat(args.n, 1)         # [N, 512]
        with torch.amp.autocast("cuda", enabled=False):
            id_tokens = identity_adapter(identity_emb.float()) # [N, 512, 4]
        id_tokens = id_tokens.float()

    # ----------------------------------------------------------
    # ATRIBUTOS
    # ----------------------------------------------------------

    if args.manual_attrs is not None:
        attrs_vec = torch.zeros(40, dtype=torch.float32)
        for name in args.manual_attrs:
            if name not in ATTR_TO_IDX:
                raise ValueError(
                    f"Atributo desconhecido: '{name}'\n"
                    f"Disponíveis:\n  {CELEBA_ATTRS}"
                )
            attrs_vec[ATTR_TO_IDX[name]] = 1.0
        detected = None
        print("Modo manual: atributos definidos pelo usuário.")

    else:
        attrs_vec = load_celeba_attrs(args.ref, args.celeba_attrs_file)
        detected = [CELEBA_ATTRS[i] for i in range(40) if attrs_vec[i] == 1.0]
        print(f"Atributos CelebA: {detected}")

    # ----------------------------------------------------------
    # EDIÇÕES
    # ----------------------------------------------------------

    for name in (args.enable or []):
        if name not in ATTR_TO_IDX:
            raise ValueError(f"Atributo desconhecido: '{name}'")
        attrs_vec[ATTR_TO_IDX[name]] = 1.0

    for name in (args.disable or []):
        if name not in ATTR_TO_IDX:
            raise ValueError(f"Atributo desconhecido: '{name}'")
        attrs_vec[ATTR_TO_IDX[name]] = 0.0

    active = [CELEBA_ATTRS[i] for i in range(40) if attrs_vec[i] == 1.0]

    if args.enable or args.disable:
        enabled_str  = f"  + ativados:   {args.enable}"  if args.enable  else ""
        disabled_str = f"  - desativados: {args.disable}" if args.disable else ""
        print(f"Edições aplicadas:\n{enabled_str}\n{disabled_str}".strip())

    print(f"Atributos finais: {active}")

    # ----------------------------------------------------------
    # CONTEXTO
    # ----------------------------------------------------------

    attrs = attrs_vec.unsqueeze(0).repeat(args.n, 1).to(device)

    with torch.no_grad():
        attr_ctx = attribute_embedder(attrs)                       # [N, 512, 40]
        context  = torch.cat([attr_ctx, id_tokens], dim=-1)        # [N, 512, 44]

    # ----------------------------------------------------------
    # DIFUSÃO (DDPM)
    # ----------------------------------------------------------

    diffusion = Diffusion_conditional(img_size=32, device=device)

    print(f"\nGerando {args.n} imagens com CFG={args.cfg_scale}...")

    with torch.no_grad():
        sampled_latents = diffusion.sample(
            unet,
            n=args.n,
            context=context,
            channels=4,
            cfg_scale=args.cfg_scale,
        )
        sampled_latents = sampled_latents / 0.18215
        generated = vae.decode(sampled_latents)   # [N, 3, 256, 256]

    # ----------------------------------------------------------
    # SALVAR
    # ----------------------------------------------------------

    os.makedirs(args.save_dir, exist_ok=True)

    # Imagens geradas normalizadas para [0, 1]
    generated_01 = (generated.clamp(-1, 1) + 1) / 2
    ref_01        = (ref_img.clamp(-1, 1) + 1) / 2   # [1, 3, 256, 256]

    # Grid: referência na primeira posição, geradas a seguir
    all_images = torch.cat([ref_01.cpu(), generated_01.cpu()], dim=0)
    grid = make_grid(all_images, nrow=args.nrow, padding=4)

    grid_path = os.path.join(args.save_dir, "result.png")
    save_image(grid, grid_path)
    print(f"Grid salvo em: {grid_path}")

    # Imagens individuais
    if args.save_individual:
        for idx, img in enumerate(generated_01):
            save_image(img, os.path.join(args.save_dir, f"gen_{idx:02d}.png"))
        print(f"Imagens individuais salvas em: {args.save_dir}/gen_*.png")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inferência do LDM condicional por identidade (ArcFace + atributos CelebA)"
    )

    # Referência
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Foto de referência da pessoa",
    )

    # Checkpoint
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Caminho direto para o checkpoint (.pt)",
    )
    ckpt_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nome da run (usa ckpt_best.pt se existir, senão ckpt.pt)",
    )

    # VAE
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="vae/vae_epoch_62.pt",
    )

    # Atributos
    parser.add_argument(
        "--celeba_attrs_file",
        type=str,
        default="CelebA_data/celeba/list_attr_celeba.csv",
        help="Arquivo de anotações CelebA (.txt oficial ou .csv do Kaggle). "
             "Usado quando --manual_attrs não é fornecido.",
    )
    parser.add_argument(
        "--manual_attrs",
        nargs="*",
        default=None,
        help="Define todos os atributos manualmente (ignora AttributePredictor). "
             "Ex: --manual_attrs Young Smiling Black_Hair",
    )
    parser.add_argument(
        "--enable",
        nargs="*",
        default=[],
        help="Atributos a ativar sobre os detectados. Ex: --enable Smiling Eyeglasses",
    )
    parser.add_argument(
        "--disable",
        nargs="*",
        default=[],
        help="Atributos a desativar sobre os detectados. Ex: --disable Bald",
    )

    # Geração
    parser.add_argument("--n",         type=int,   default=4,    help="Número de imagens a gerar")
    parser.add_argument("--nrow",      type=int,   default=5,    help="Imagens por linha no grid (ref + geradas)")
    parser.add_argument("--cfg_scale", type=float, default=3.0,  help="Escala de CFG")
    parser.add_argument("--device",    type=str,   default="cuda")

    # Saída
    parser.add_argument("--save_dir",        type=str,  default="results/test_identity")
    parser.add_argument("--save_individual", action="store_true", help="Salva também cada imagem individualmente")

    args = parser.parse_args()

    generate(args)
