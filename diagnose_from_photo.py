"""
diagnose_from_photo.py
======================
Diagnostica POR QUE generate_from_photo.py produz resultados piores que
generate_from_test_paired.py, isolando cada causa candidata com um número:

  [A] Caminho live vs cache — a mesma imagem do CelebA, processada pelo
      caminho live de generate_from_photo.py (CELEBA_TRANSFORM + ArcFace/CLIP
      on-the-fly), produz os MESMOS embeddings do cache_encoder/ usado no
      treino?  cos ≈ 1.0 esperado. Se < 0.99, há bug no caminho live.

  [B] Erro do CelebAAligner em domínio conhecido — a mesma imagem do CelebA,
      forçada a passar pelo CelebAAligner (detecção + warp, como uma foto
      nova passaria), ainda produz embeddings próximos do cache?
      Isso mede o teto de erro que o alinhamento sozinho introduz.
      cos > 0.95 esperado; se cair muito, o alinhamento é o problema.

  [C] A foto do usuário — rosto detectado? fração de borda preta após o
      warp (CelebA nunca tem borda preta; CLIP/ArcFace nunca viram isso
      no treino)? Com --photo2 (outra foto da MESMA pessoa), estabilidade
      do embedding de identidade entre as duas fotos.

Uso (na VM, com cache_encoder/ e CelebA_data/ presentes):
    python diagnose_from_photo.py --photo minha_foto.jpg
    python diagnose_from_photo.py --photo foto1.jpg --photo2 foto2.jpg \\
        --celeba_img 182638.jpg
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from utils.edit_common import CELEBA_TRANSFORM
from models.encoders import ArcFaceEncoder
from data.correct_alignment import CelebAAligner
from cache_arcface import clip_encode

IMAGE_DIR = "./CelebA_data/celeba/img_align_celeba/img_align_celeba"
CACHE_DIR = "./cache_encoder"


def _cos(a, b):
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def _token_cos(live_tokens, cache_tokens):
    """Média do cosseno por token — [S,768] vs [S,768]."""
    return F.cosine_similarity(live_tokens, cache_tokens, dim=-1).mean().item()


def _black_border_frac(pil_178x218):
    """Fração de pixels quase-pretos no crop 178x178 que o modelo vê
    (depois do CenterCrop(178) que remove 20px do topo e da base)."""
    arr = np.array(T.CenterCrop(178)(pil_178x218))
    mask = (arr < 12).all(axis=-1)
    return float(mask.mean())


def check_celeba_consistency(fname, arcface, clip_model, aligner, device):
    path = os.path.join(IMAGE_DIR, fname)
    stem = os.path.splitext(fname)[0]
    cache_path = os.path.join(CACHE_DIR, stem + ".pt")

    if not os.path.exists(path):
        print(f"[A/B] PULADO: {path} não existe.")
        return
    if not os.path.exists(cache_path):
        print(f"[A/B] PULADO: {cache_path} não existe (rode cache_arcface.py).")
        return

    cache = torch.load(cache_path, map_location="cpu")
    arc_cache, clip_cache = cache["arcface"], cache["clip"]

    img = Image.open(path).convert("RGB")

    # ---------- [A] caminho live (mesma transform do treino) ----------
    x = CELEBA_TRANSFORM(img).unsqueeze(0)
    arc_live = arcface(x).cpu()
    clip_live = clip_encode(clip_model, x.to(device), device).squeeze(0)

    cos_arc = _cos(arc_live, arc_cache)
    cos_clip = _token_cos(clip_live, clip_cache)
    print(f"\n[A] live vs cache ({fname}) — sem realinhar:")
    print(f"    ArcFace cos = {cos_arc:.4f}   (esperado > 0.99)")
    print(f"    CLIP    cos = {cos_clip:.4f}   (esperado > 0.99)")
    if cos_arc < 0.99 or cos_clip < 0.99:
        print("    >>> PROBLEMA: o caminho live NÃO reproduz o cache do treino.")
    else:
        print("    OK: o caminho live de generate_from_photo.py é fiel ao treino.")

    # ---------- [B] força o CelebAAligner (como uma foto nova) ----------
    realigned, found = aligner.align_pil(img)
    x2 = CELEBA_TRANSFORM(realigned).unsqueeze(0)
    arc_re = arcface(x2).cpu()
    clip_re = clip_encode(clip_model, x2.to(device), device).squeeze(0)

    cos_arc2 = _cos(arc_re, arc_cache)
    cos_clip2 = _token_cos(clip_re, clip_cache)
    print(f"\n[B] realinhado pelo CelebAAligner vs cache ({fname}):")
    print(f"    rosto detectado = {found}")
    print(f"    ArcFace cos = {cos_arc2:.4f}   (esperado > 0.95)")
    print(f"    CLIP    cos = {cos_clip2:.4f}   (esperado > 0.95)")
    if not found:
        print("    >>> PROBLEMA: detector falhou até numa imagem do CelebA.")
    elif cos_arc2 < 0.90:
        print("    >>> PROBLEMA: o realinhamento sozinho já degrada bastante a "
              "identidade — o template/warp do CelebAAligner é suspeito.")
    else:
        print("    OK: o alinhamento introduz pouco erro em domínio conhecido.")


def check_photo(photo_path, arcface, aligner, device, label="foto"):
    img = Image.open(photo_path).convert("RGB")
    aligned, found = aligner.align_pil(img)

    frac = _black_border_frac(aligned)
    print(f"\n[C] {label}: {photo_path}")
    print(f"    resolução original = {img.size}")
    print(f"    rosto detectado    = {found}")
    print(f"    borda preta no crop 178x178 = {frac * 100:.1f}%   (esperado 0%)")
    if not found:
        print("    >>> PROBLEMA: caiu no fallback de center-crop — o "
              "enquadramento está fora da distribuição de treino.")
    if frac > 0.02:
        print("    >>> PROBLEMA: borda preta vinda do warpAffine — a foto é "
              "mais fechada que o enquadramento do CelebA. CLIP/ArcFace nunca "
              "viram isso no treino. Use uma foto com mais margem ao redor do "
              "rosto (ombros/fundo visíveis) ou borderMode=BORDER_REPLICATE.")

    out = os.path.splitext(os.path.basename(photo_path))[0] + "_diag_aligned.png"
    aligned.save(out)
    print(f"    alinhada salva em: {out}  (compare lado a lado com uma imagem "
          "do img_align_celeba — rosto no mesmo lugar? mesmo tamanho?)")

    emb = arcface(CELEBA_TRANSFORM(aligned).unsqueeze(0)).cpu()
    return emb


def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    arcface = ArcFaceEncoder()
    aligner = CelebAAligner()

    from transformers import CLIPVisionModel
    clip_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device).eval()

    # ---------- [A] + [B]: imagem do CelebA como controle ----------
    check_celeba_consistency(args.celeba_img, arcface, clip_model, aligner, device)

    # ---------- [C]: a(s) foto(s) do usuário ----------
    emb1 = check_photo(args.photo, arcface, aligner, device, label="foto")

    if args.photo2:
        emb2 = check_photo(args.photo2, arcface, aligner, device, label="foto2")
        cos12 = _cos(emb1, emb2)
        print(f"\n[C] estabilidade de identidade entre as duas fotos:")
        print(f"    ArcFace cos(foto, foto2) = {cos12:.4f}")
        print("    Referência: mesma pessoa costuma dar > 0.5; pessoas "
              "diferentes ~0.1. Se duas fotos SUAS derem < 0.3, o embedding "
              "de identidade está instável para as suas fotos (enquadramento/"
              "qualidade), e o modelo recebe uma identidade 'borrada'.")

    print("\n" + "=" * 60)
    print("Como ler: [A] ruim → bug no caminho live do script. "
          "[A] ok e [B] ruim → CelebAAligner. [A] e [B] ok → o problema é a "
          "FOTO estar fora da distribuição (borda preta, enquadramento, "
          "qualidade) e/ou o vetor de atributos — ver testes de geração no "
          "checklist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnóstico do gap entre generate_from_test_paired.py "
                     "e generate_from_photo.py."
    )
    parser.add_argument("--photo", type=str, required=True,
                        help="A foto nova que está dando resultado ruim.")
    parser.add_argument("--photo2", type=str, default=None,
                        help="Opcional: OUTRA foto da mesma pessoa, para medir "
                             "estabilidade do embedding de identidade.")
    parser.add_argument("--celeba_img", type=str, default="182638.jpg",
                        help="Uma imagem do img_align_celeba usada como "
                             "controle (ideal: uma do split de teste, ex. um "
                             "'arquivo alvo' impresso por "
                             "generate_from_test_paired.py).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
