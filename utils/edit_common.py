"""
utils/edit_common.py
====================
Infraestrutura compartilhada pelos dois scripts de EDIÇÃO de foto real
(preservando estrutura — roupa, fundo, pose) com checkpoints de
train_cfg_composable*.py:

    edit_sdedit.py          — img2img estilo SDEdit (começa o DDIM de um
                              latente parcialmente ruidoso da própria foto)
    edit_ddim_inversion.py  — inversão DDIM determinística (recupera o z_T
                              exato que reconstrói a foto, depois reamostra
                              com os atributos editados)

Diferença para generate_cfg_composable.py: lá a geração parte de ruído puro
e a foto só entra via tokens globais (ArcFace/CLIP) — estrutura espacial da
foto não é preservada. Aqui a foto entra ESPACIALMENTE, via VAE.encode.

Preprocessamento de fotos pessoais:
    Os latentes de treino vêm de imagens do CelebA alinhado (178x218,
    CenterCrop(178) → Resize(256)). Uma foto qualquer precisa reproduzir
    esse enquadramento, senão o latente fica fora da distribuição de
    treino. prepare_photo() usa o CelebAAligner (data/correct_alignment.py)
    para alinhar a foto ao template NATIVO do CelebA (178x218, via ajuste
    de similaridade nos 5 pontos), e então aplica a MESMA CELEBA_TRANSFORM
    usada em imagens reais do dataset — em vez do template do ArcFace
    (FaceAligner), que recorta olhos-nariz-boca de forma mais apertada e
    produz um enquadramento diferente do que o VAE/ArcFaceEncoder/
    ImageConditionEncoder foram treinados para ver.
"""

import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from tqdm import tqdm

from models.unet_conditional import UNet_cond
from models.encoders import ImageConditionEncoder, ArcFaceOnlyEncoder
from models.modules import AttributeEmbedder
from diffusion.conditional_ddpm import Diffusion_conditional
from vae.modules import VAE


# ============================================================
# COMPATIBILIDADE DE CHECKPOINT
# ============================================================

def _fix_clip_state_dict(state):
    """Checkpoints antigos foram salvos com uma versão do `transformers`
    cujo CLIPVisionModel não tinha o submódulo `vision_model` (chaves
    "clip.embeddings...", "clip.encoder...", "clip.post_layernorm...").
    A versão atual sempre envolve a transformer em `self.vision_model`
    ("clip.vision_model.embeddings...", etc). Remapeia as chaves antigas
    para o layout atual, sem precisar fixar uma versão antiga do pacote."""
    fixed = {}
    for k, v in state.items():
        if k.startswith("clip.") and not k.startswith("clip.vision_model."):
            k = "clip.vision_model." + k[len("clip."):]
        fixed[k] = v
    return fixed


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

# Mesma transform do treino (cache_generator.py / cache_arcface.py) —
# válida apenas para imagens já no enquadramento do CelebA (178x218).
CELEBA_TRANSFORM = T.Compose([
    T.CenterCrop(178),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ============================================================
# PREPROCESSAMENTO DA FOTO
# ============================================================

def prepare_photo(path, device, aligner=None):
    """
    Carrega uma foto e a leva ao enquadramento 256x256 do treino.

    - Imagem 178x218 (CelebA alinhado): usa a transform exata do treino.
    - Qualquer outra: CelebAAligner alinha ao template nativo do CelebA
      (178x218, sem rosto detectado cai em center-crop respeitando a
      proporção 178:218 — com aviso, enquadramento pode ficar fora da
      distribuição de treino); em seguida aplica a mesma CELEBA_TRANSFORM
      usada em imagens reais do dataset.

    Retorna tensor [1, 3, 256, 256] em [-1, 1].
    """
    # exif_transpose aplica a rotação gravada pelo celular no EXIF — sem
    # isso, fotos tiradas em retrato chegam DEITADAS no aligner/encoders.
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

    if img.size == (178, 218):
        print(f"[foto] {path}: enquadramento CelebA — transform do treino.")
        return CELEBA_TRANSFORM(img).unsqueeze(0).to(device)

    if aligner is None:
        from data.correct_alignment import CelebAAligner
        aligner = CelebAAligner()

    aligned_pil, found = aligner.align_pil(img)
    if not found:
        print(f"[foto] AVISO: nenhum rosto detectado em {path} — "
              "center-crop CelebA como fallback (enquadramento pode "
              "ficar fora da distribuição de treino).")
    else:
        print(f"[foto] {path}: rosto detectado — alinhado ao template nativo do CelebA.")

    return CELEBA_TRANSFORM(aligned_pil).unsqueeze(0).to(device)


# ============================================================
# ATRIBUTOS ORIGINAIS + EDIÇÃO
# ============================================================

def resolve_original_attrs(args, photo_path, device):
    """
    Vetor [40] de atributos da FOTO ORIGINAL, por uma de duas vias:
      --orig_attrs Nome1 Nome2 ...  → manual (os listados = 1, resto = 0)
      --classifier_ckpt caminho.pt  → CLIPAttributeClassifier sobre o rosto
                                      alinhado (FaceAligner, como no treino
                                      do classificador)
    """
    if args.orig_attrs is not None:
        vec = torch.zeros(40)
        for name in args.orig_attrs:
            vec[ATTR_TO_IDX[name]] = 1.0
        print(f"[attrs] manuais: {args.orig_attrs}")
        return vec

    if args.classifier_ckpt is None:
        raise SystemExit(
            "Informe os atributos da foto original: --orig_attrs Nome1 ... "
            "ou --classifier_ckpt <checkpoint do CLIPAttributeClassifier>."
        )

    from data.face_align import FaceAligner
    from models.attribute_classifier import CLIPAttributeClassifier

    aligner = FaceAligner()
    photo = ImageOps.exif_transpose(Image.open(photo_path))
    aligned_pil, found = aligner.align_pil(photo)
    if not found:
        print("[attrs] AVISO: rosto não detectado para o classificador — "
              "usando fallback de center-crop (predição pode degradar).")

    aligned = torch.from_numpy(np.array(aligned_pil)).float()
    aligned = (aligned.permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0).to(device)

    classifier = CLIPAttributeClassifier(
        pretrained_path=args.classifier_ckpt
    ).to(device).eval()

    vec = classifier.predict(aligned).squeeze(0).cpu()
    detected = [CELEBA_ATTRS[i] for i in range(40) if vec[i] == 1.0]
    print(f"[attrs] detectados: {detected}")

    del classifier
    torch.cuda.empty_cache()
    return vec


def apply_edits(attrs_vec, enable, disable):
    """Aplica --enable / --disable sobre o vetor original (cópia)."""
    edited = attrs_vec.clone()
    for name in (enable or []):
        edited[ATTR_TO_IDX[name]] = 1.0
        print(f"[edit] + {name}")
    for name in (disable or []):
        edited[ATTR_TO_IDX[name]] = 0.0
        print(f"[edit] - {name}")
    return edited


def validate_attr_names(parser, args):
    for flag in ("orig_attrs", "enable", "disable"):
        for n in (getattr(args, flag, None) or []):
            if n not in ATTR_TO_IDX:
                parser.error(f"Atributo desconhecido em --{flag}: {n}")


# ============================================================
# PIPELINE (VAE + UNet + encoder + embedder + diffusion)
# ============================================================

class EditPipeline:

    def __init__(self, ckpt_path, vae_ckpt, device, noise_steps=1000):
        self.device = device

        print("Carregando VAE...")
        self.vae = VAE(in_channels=3, latent_dim=4).to(device)
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        print(f"Carregando checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        self.num_img_tokens = ckpt.get("num_img_tokens", 16)
        self.encoder_type   = ckpt.get("encoder", "clip_arcface")
        print(f"  época: {ckpt.get('epoch', '?')} | tokens: "
              f"{self.num_img_tokens} | encoder: {self.encoder_type}")

        self.unet = UNet_cond(in_channels=4, out_channels=4, context_dim=512).to(device)
        self.unet.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
        self.unet.eval()

        if self.encoder_type == "arcface_only":
            self.image_encoder = ArcFaceOnlyEncoder(
                context_dim=512, num_tokens=self.num_img_tokens,
            ).to(device)
        elif self.encoder_type in ("clip_arcface", "clip_arcface_split"):
            self.image_encoder = ImageConditionEncoder(
                context_dim=512, num_tokens=self.num_img_tokens,
                freeze_backbone=True,
            ).to(device)
        else:
            raise ValueError(f"encoder desconhecido no ckpt: {self.encoder_type}")

        image_encoder_state = ckpt.get(
            "ema_image_encoder_state_dict", ckpt["image_encoder_state_dict"]
        )
        if self.encoder_type in ("clip_arcface", "clip_arcface_split"):
            image_encoder_state = _fix_clip_state_dict(image_encoder_state)
        self.image_encoder.load_state_dict(image_encoder_state)
        self.image_encoder.eval()

        self.attribute_embedder = AttributeEmbedder(
            num_attributes=40, context_dim=512,
        ).to(device)
        self.attribute_embedder.load_state_dict(
            ckpt.get("ema_attribute_embedder_state_dict", ckpt["attribute_embedder_state_dict"])
        )
        self.attribute_embedder.eval()

        self.diffusion = Diffusion_conditional(
            img_size=32, device=device, noise_steps=noise_steps,
        )

    # --------------------------------------------------------
    # VAE — mesma convenção do cache de treino (mu, x0.18215)
    # --------------------------------------------------------

    @torch.no_grad()
    def encode(self, img):
        mu, _ = self.vae.encode(img)
        return mu * 0.18215

    @torch.no_grad()
    def decode(self, z):
        return self.vae.decode(z / 0.18215)

    # --------------------------------------------------------
    # CONTEXTOS — cadeia do composable CFG
    # --------------------------------------------------------

    @torch.no_grad()
    def build_context_chain(self, ref_img, attrs, s_id, s_attr, s_clip=None):
        """
        Monta a cadeia de contextos e escalas do composable CFG na MESMA
        ordem do treino:

          2 ramos (clip_arcface / arcface_only):
            [(∅,∅), (id,∅), (id,attr)]              escalas [s_id, s_attr]
          3 ramos (clip_arcface_split):
            [(∅,∅,∅), (id,∅,∅), (id,clip,∅),
             (id,clip,attr)]                escalas [s_id, s_clip, s_attr]

        Retorna (contexts, scales). contexts[-1] é o contexto totalmente
        condicionado — é o que a inversão DDIM usa sozinho (escala 1).
        """
        attrs = attrs.view(1, -1).to(self.device)

        if self.encoder_type == "clip_arcface_split":
            clip_tok, id_tok = self.image_encoder(
                ref_img=ref_img, return_separate=True,
            )
            attr_tok = self.attribute_embedder(attrs)

            z_id, z_clip, z_attr = (
                torch.zeros_like(id_tok),
                torch.zeros_like(clip_tok),
                torch.zeros_like(attr_tok),
            )
            contexts = [
                torch.cat([z_id,   z_clip,   z_attr],   dim=2),
                torch.cat([id_tok, z_clip,   z_attr],   dim=2),
                torch.cat([id_tok, clip_tok, z_attr],   dim=2),
                torch.cat([id_tok, clip_tok, attr_tok], dim=2),
            ]
            scales = [s_id, s_clip if s_clip is not None else s_id, s_attr]
            return contexts, scales

        id_tok   = self.image_encoder(ref_img=ref_img)
        attr_tok = self.attribute_embedder(attrs)

        z_id, z_attr = torch.zeros_like(id_tok), torch.zeros_like(attr_tok)
        contexts = [
            torch.cat([z_id,   z_attr],   dim=2),
            torch.cat([id_tok, z_attr],   dim=2),
            torch.cat([id_tok, attr_tok], dim=2),
        ]
        return contexts, [s_id, s_attr]


# ============================================================
# DDIM — amostragem guiada e inversão
# ============================================================

def ddim_times(diffusion, ddim_steps):
    """Timesteps do DDIM em ordem ASCENDENTE (mesmo sub-conjunto do treino)."""
    step = max(diffusion.noise_steps // ddim_steps, 1)
    return list(range(1, diffusion.noise_steps, step))


def _eps_guided(unet, z, t, contexts, scales, device):
    """
    eps composable:  eps(c0) + Σ_i s_i · [eps(c_i) − eps(c_{i−1})]
    Os K forwards são feitos em UM batch (contextos concatenados).
    """
    K = len(contexts)
    n = z.shape[0]

    z_b   = z.repeat(K, 1, 1, 1)
    t_b   = torch.full((K * n,), t, device=device, dtype=torch.long)
    ctx_b = torch.cat(contexts, dim=0)

    eps_all = unet(z_b, t_b, context=ctx_b).chunk(K, dim=0)

    eps = eps_all[0]
    for i, s in enumerate(scales):
        eps = eps + s * (eps_all[i + 1] - eps_all[i])
    return eps


@torch.no_grad()
def ddim_sample(pipe, z_start, times_desc, contexts, scales, desc="DDIM"):
    """
    Loop DDIM determinístico (eta=0), idêntico ao de
    train_cfg_composable_paired.py: o último salto vai direto a
    alpha_bar=1 usando a predição de x0 do último passo real.

    z_start é tratado como z no timestep times_desc[0] — ruído puro
    (geração), latente parcialmente ruidoso (SDEdit) ou z_T invertido.
    """
    diffusion = pipe.diffusion
    z = z_start

    for idx, t in enumerate(tqdm(times_desc, desc=desc)):

        eps = _eps_guided(pipe.unet, z, t, contexts, scales, pipe.device)

        alpha_bar_t = diffusion.alpha_hat[t]

        is_last = (idx == len(times_desc) - 1)
        alpha_bar_next = (
            torch.tensor(1.0, device=pipe.device) if is_last
            else diffusion.alpha_hat[times_desc[idx + 1]]
        )

        pred_x0 = (z - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        z = (
            torch.sqrt(alpha_bar_next) * pred_x0
            + torch.sqrt(torch.clamp(1.0 - alpha_bar_next, min=0.0)) * eps
        )

    return z


@torch.no_grad()
def ddim_invert(pipe, z0, times_asc, context, desc="Inversão"):
    """
    Inversão DDIM: percorre os mesmos timesteps em ordem ascendente,
    aplicando o update determinístico com os papéis de "atual" e
    "próximo" trocados. eps é avaliado no timestep de DESTINO com o
    latente atual — aproximação padrão (os timesteps adjacentes do DDIM
    são próximos o bastante para o campo de eps variar pouco entre eles);
    é o espelho exato do salto final do amostrador (alpha_bar=1 ↔ t_min).

    `context` deve ser o contexto TOTALMENTE condicionado com os atributos
    ORIGINAIS da foto (escala 1 — sem guidance: guidance alto amplifica o
    erro da aproximação e desloca z_T para fora da distribuição).
    """
    diffusion = pipe.diffusion
    z = z0
    n = z0.shape[0]

    prev_alpha_bar = torch.tensor(1.0, device=pipe.device)

    for t in tqdm(times_asc, desc=desc):

        t_vec = torch.full((n,), t, device=pipe.device, dtype=torch.long)
        eps = pipe.unet(z, t_vec, context=context)

        alpha_bar_t = diffusion.alpha_hat[t]

        pred_x0 = (
            z - torch.sqrt(1.0 - prev_alpha_bar) * eps
        ) / torch.sqrt(prev_alpha_bar)

        z = (
            torch.sqrt(alpha_bar_t) * pred_x0
            + torch.sqrt(1.0 - alpha_bar_t) * eps
        )
        prev_alpha_bar = alpha_bar_t

    return z


# ============================================================
# SAÍDA
# ============================================================

def save_row(images, path, labels=None):
    """
    Salva uma linha de imagens [-1,1] lado a lado. `labels` (se dado) é
    impresso no console na ordem das colunas — mais simples e legível do
    que desenhar texto na imagem.
    """
    from torchvision.utils import save_image

    row = torch.cat([img.cpu() for img in images], dim=0)
    row = (row.clamp(-1, 1) + 1) / 2
    save_image(row, path, nrow=len(images))

    if labels:
        print("Colunas (esq → dir): " + " | ".join(labels))
    print(f"Salvo em: {path}")
