"""
generate_mixed_guidance.py
==========================
Inferência com guidance híbrida a partir de uma foto de referência:

  • CFG  (Classifier-Free Guidance) para a imagem de referência
      eps_cfg = eps_uncond + s_img * (eps_cond - eps_uncond)

  • CG   (Classifier Guidance) para os atributos faciais
      grad   = ∇_zt  log p(y | zt)
      eps    = eps_cfg − s_attr * √(1 − ᾱ_t) * grad

Os dois termos se somam na previsão de ruído antes do passo de denoising.

Uso básico:
    python generate_mixed_guidance.py \\
        --ref_image foto.jpg \\
        --ckpt models/LDM_MixedGuidance/ckpt.pt \\
        --enable Smiling Eyeglasses \\
        --disable Bald \\
        --cfg_scale_img 3.0 \\
        --cg_scale_attr 1.0

Atributos manuais (ignora a imagem para atributos):
    python generate_mixed_guidance.py \\
        --ref_image foto.jpg \\
        --ckpt ... \\
        --manual_attrs Young Smiling Black_Hair \\
        --cg_scale_attr 0.5
"""

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tv_models
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from models.unet_conditional import UNet_cond
from models.encoders import ImageConditionEncoder, AttributePredictor
from models.modules import NoisyLatentAttrClassifier
from diffusion.conditional_ddpm import Diffusion_conditional
from vae.modules import VAE


# ============================================================
# LEGACY ENCODER  (ResNet-18, checkpoints antes do CLIP+ArcFace)
# ============================================================

class _LegacyImageConditionEncoder(nn.Module):
    """ResNet-18 encoder used before the CLIP+ArcFace switch."""

    def __init__(self, context_dim=512, num_tokens=16, freeze_backbone=True):
        super().__init__()

        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"num_tokens={num_tokens} deve ser quadrado perfeito")

        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        feat_dim = 256

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((side, side))
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.num_tokens = num_tokens

        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, ref_img):
        x = ref_img.float() * 0.5 + 0.5
        x = (x - self.imagenet_mean) / self.imagenet_std
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(2).permute(0, 2, 1)
        return self.proj(feats).permute(0, 2, 1)   # [B, context_dim, num_tokens]


# ============================================================
# ENCODER TYPE DETECTION
# ============================================================

def _detect_encoder_type(enc_state_dict):
    """Returns 'clip_arcface' for new encoder, 'resnet' for legacy."""
    for k in enc_state_dict:
        if k.startswith("clip."):
            return "clip_arcface"
    return "resnet"


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
# LOOP DE AMOSTRAGEM HÍBRIDA
#
# Não usa diffusion.sample() — aquele método tem no_grad hardcoded.
# Aqui o grad de z_t é necessário para o classifier guidance.
# ============================================================

@torch.no_grad()
def _unet_forward(unet, z_t, t, context):
    return unet(z_t, t, context=context)


def sample_hybrid(
    unet,
    classifier,           # NoisyLatentAttrClassifier ou None (desativa CG)
    diffusion,
    img_context,          # [N, 512, num_img_tokens]  tokens da imagem condicionada
    attrs,                # [N, 40]  atributos alvo (float)
    n,
    channels=4,
    cfg_scale_img=3.0,
    cg_scale_attr=1.0,
    cg_t_thresh=None,     # aplica CG só para t <= cg_t_thresh (None = todos)
    device="cuda",
):
    """
    Loop DDPM com guidance híbrida.

    Retorna x [N, channels, img_size, img_size] no espaço de latentes.
    """

    unet.eval()
    if classifier is not None:
        classifier.eval()

    img_size = diffusion.img_size

    # Ruído inicial
    z_t = torch.randn(n, channels, img_size, img_size, device=device)

    zeros_ctx = torch.zeros_like(img_context)  # contexto vazio para CFG

    for i in tqdm(reversed(range(1, diffusion.noise_steps)), desc="Sampling", total=diffusion.noise_steps - 1):

        t_vec = torch.full((n,), i, device=device, dtype=torch.long)

        alpha_hat_t = diffusion.alpha_hat[t_vec][:, None, None, None]   # [N,1,1,1]
        alpha_t     = diffusion.alpha[t_vec][:, None, None, None]
        beta_t      = diffusion.beta[t_vec][:, None, None, None]

        # ------------------------------------------------
        # 1. CFG para imagem (sem grad na UNet)
        # ------------------------------------------------
        eps_cond   = _unet_forward(unet, z_t, t_vec, img_context)
        eps_uncond = _unet_forward(unet, z_t, t_vec, zeros_ctx)

        eps = eps_uncond + cfg_scale_img * (eps_cond - eps_uncond)

        # ------------------------------------------------
        # 2. Classifier Guidance para atributos
        #    Ativo quando classifier não é None, cg_scale_attr > 0 e i <= cg_t_thresh
        # ------------------------------------------------
        apply_cg = (
            classifier is not None
            and cg_scale_attr > 0.0
            and (cg_t_thresh is None or i <= cg_t_thresh)
        )

        if apply_cg:

            z_for_cls = z_t.detach().requires_grad_(True)

            logits = classifier(z_for_cls, t_vec)   # [N, 40]

            # log p(y|zt) = Σ_k [ y_k·log σ(l_k) + (1-y_k)·log σ(-l_k) ]
            log_p = (
                F.logsigmoid(logits) * attrs
                + F.logsigmoid(-logits) * (1.0 - attrs)
            )

            score = log_p.sum()

            grad = torch.autograd.grad(score, z_for_cls)[0]

            # Escala o gradiente de acordo com o nível de ruído
            eps = eps - cg_scale_attr * torch.sqrt(1.0 - alpha_hat_t) * grad.detach()

        # ------------------------------------------------
        # 3. Passo DDPM
        # ------------------------------------------------
        noise = torch.randn_like(z_t) if i > 1 else torch.zeros_like(z_t)

        z_t = (
            (1.0 / torch.sqrt(alpha_t))
            * (z_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t) * eps)
        ) + torch.sqrt(beta_t) * noise

    return z_t


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
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ========================================================
    # CHECKPOINT
    # ========================================================

    print(f"Carregando checkpoint: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)

    num_img_tokens = ckpt.get("num_img_tokens", args.img_tokens)

    print(f"  época: {ckpt.get('epoch', '?')} | tokens de imagem: {num_img_tokens}")

    # ========================================================
    # UNET
    # ========================================================

    unet = UNet_cond(
        in_channels=4,
        out_channels=4,
        context_dim=512,
    ).to(device)

    unet.load_state_dict(
        ckpt.get("ema_state_dict", ckpt["model_state_dict"])
    )
    unet.eval()

    # ========================================================
    # IMAGE CONDITION ENCODER  (auto-detects old vs new)
    # ========================================================

    enc_sd   = ckpt.get("ema_image_encoder_state_dict", ckpt["image_encoder_state_dict"])
    enc_type = _detect_encoder_type(enc_sd)

    if enc_type == "clip_arcface":
        print("  encoder: CLIP + ArcFace (novo)")
        image_encoder = ImageConditionEncoder(
            context_dim=512,
            num_tokens=num_img_tokens,
            freeze_backbone=True,
        ).to(device)
    else:
        print("  encoder: ResNet-18 (legado)")
        image_encoder = _LegacyImageConditionEncoder(
            context_dim=512,
            num_tokens=num_img_tokens,
            freeze_backbone=True,
        ).to(device)

    image_encoder.load_state_dict(enc_sd)
    image_encoder.eval()

    # ========================================================
    # NOISY LATENT CLASSIFIER  (opcional — ausente em ckpts antigos)
    # ========================================================

    has_classifier = (
        "classifier_state_dict" in ckpt
        or "ema_classifier_state_dict" in ckpt
    )

    if has_classifier:
        classifier = NoisyLatentAttrClassifier(
            latent_dim=4,
            time_emb_dim=256,
            num_attrs=40,
        ).to(device)
        classifier.load_state_dict(
            ckpt.get("ema_classifier_state_dict", ckpt["classifier_state_dict"])
        )
        classifier.eval()
        print("  classifier: carregado")
    else:
        classifier = None
        print("  classifier: ausente no checkpoint — CG de atributos desativado")
        if args.cg_scale_attr > 0:
            print("  [aviso] --cg_scale_attr ignorado (sem classificador no checkpoint)")

    # ========================================================
    # DIFFUSION
    # ========================================================

    diffusion = Diffusion_conditional(
        img_size=32,
        device=device,
        noise_steps=args.noise_steps,
    )

    # ========================================================
    # IMAGEM DE REFERÊNCIA → CONTEXT
    # ========================================================

    print(f"Carregando referência: {args.ref_image}")

    ref_img = load_image(args.ref_image).to(device)           # [1, 3, 256, 256]
    ref_img = ref_img.repeat(args.n, 1, 1, 1)                 # [N, 3, 256, 256]

    with torch.no_grad():
        img_context = image_encoder(ref_img)                  # [N, 512, num_img_tokens]

    # ========================================================
    # ATRIBUTOS
    # ========================================================

    if args.manual_attrs:

        attrs_vec = torch.zeros(40, dtype=torch.float32)
        for name in args.manual_attrs:
            if name not in ATTR_TO_IDX:
                raise ValueError(f"Atributo desconhecido: '{name}'\nDisponíveis: {CELEBA_ATTRS}")
            attrs_vec[ATTR_TO_IDX[name]] = 1.0
        print("Atributos manuais definidos.")

    else:

        attr_predictor = AttributePredictor(
            num_attributes=40,
            pretrained_path=args.attr_predictor_ckpt,
        ).to(device)
        attr_predictor.eval()

        ref_single = load_image(args.ref_image).to(device)

        with torch.no_grad():
            attrs_vec = attr_predictor.predict(ref_single).squeeze(0).cpu()

        detected = [CELEBA_ATTRS[i] for i in range(40) if attrs_vec[i] == 1.0]
        print(f"Atributos detectados na referência: {detected}")

    # -------------------------------------------------------
    # Edição de atributos
    # -------------------------------------------------------

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

    active_final = [CELEBA_ATTRS[i] for i in range(40) if attrs_vec[i] == 1.0]
    print(f"Atributos finais: {active_final}")

    attrs = attrs_vec.unsqueeze(0).repeat(args.n, 1).to(device)  # [N, 40]

    # ========================================================
    # AMOSTRAGEM HÍBRIDA
    # ========================================================

    print(
        f"\nGerando {args.n} imagens com "
        f"CFG img={args.cfg_scale_img}  CG attr={args.cg_scale_attr}"
        + (f"  (CG ativo para t≤{args.cg_t_thresh})" if args.cg_t_thresh else "")
    )

    sampled_latents = sample_hybrid(
        unet=unet,
        classifier=classifier,
        diffusion=diffusion,
        img_context=img_context,
        attrs=attrs,
        n=args.n,
        channels=4,
        cfg_scale_img=args.cfg_scale_img,
        cg_scale_attr=args.cg_scale_attr,
        cg_t_thresh=args.cg_t_thresh,
        device=device,
    )

    # ========================================================
    # DECODE + SALVAR
    # ========================================================

    with torch.no_grad():
        sampled_latents = sampled_latents / 0.18215
        images = vae.decode(sampled_latents)

    images = (images.clamp(-1, 1) + 1) / 2

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "generated_mixed.png")
    save_image(images, out_path, nrow=args.nrow)

    print(f"\nSalvo em: {out_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Geração com CFG (imagem) + Classifier Guidance (atributos)"
    )

    parser.add_argument(
        "--ref_image",
        type=str,
        required=True,
        help="Foto de referência da pessoa (identidade e atributos base).",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/LDM_MixedGuidance/ckpt.pt",
        help="Checkpoint gerado por train_mixed_guidance.py.",
    )

    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="vae/vae_epoch_62.pt",
        help="Checkpoint da VAE.",
    )

    parser.add_argument(
        "--attr_predictor_ckpt",
        type=str,
        default=None,
        help="Checkpoint do AttributePredictor para extrair atributos da referência. "
             "None usa ImageNet (impreciso). Ignorado se --manual_attrs for fornecido.",
    )

    parser.add_argument(
        "--manual_attrs",
        nargs="*",
        default=None,
        help="Define atributos manualmente (ignora AttributePredictor). "
             "Ex: --manual_attrs Young Smiling Black_Hair",
    )

    parser.add_argument(
        "--enable",
        nargs="*",
        default=[],
        help="Atributos a ativar na imagem gerada. Ex: --enable Smiling Eyeglasses",
    )

    parser.add_argument(
        "--disable",
        nargs="*",
        default=[],
        help="Atributos a desativar na imagem gerada. Ex: --disable Bald",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Número de imagens a gerar.",
    )

    parser.add_argument(
        "--nrow",
        type=int,
        default=4,
        help="Imagens por linha no grid de saída.",
    )

    parser.add_argument(
        "--cfg_scale_img",
        type=float,
        default=3.0,
        help="Escala de CFG para a imagem de referência (0 = sem CFG).",
    )

    parser.add_argument(
        "--cg_scale_attr",
        type=float,
        default=1.0,
        help="Escala de Classifier Guidance para atributos (0 = sem CG).",
    )

    parser.add_argument(
        "--cg_t_thresh",
        type=int,
        default=None,
        help="Aplica CG apenas para passos t ≤ cg_t_thresh. "
             "Útil para suavizar instabilidade em t alto. "
             "None aplica em todos os passos.",
    )

    parser.add_argument(
        "--img_tokens",
        type=int,
        default=16,
        help="Tokens de imagem (lido do checkpoint automaticamente; "
             "este valor é o fallback se não estiver no ckpt).",
    )

    parser.add_argument(
        "--noise_steps",
        type=int,
        default=1000,
        help="Número de passos de denoising.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mixed_guidance",
        help="Diretório para salvar as imagens geradas.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda ou cpu.",
    )

    args = parser.parse_args()

    generate(args)
