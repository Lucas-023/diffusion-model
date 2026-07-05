"""
generate_from_photo.py
=======================
Geração condicionada em uma foto QUALQUER (não precisa vir do CelebA nem
ter passado por cache_arcface.py) — a foto é alinhada ao MESMO template
nativo do CelebA (178x218, via CelebAAligner) e passa pela MESMA
transform usada nas imagens reais de treino, antes de virar condicionante
de identidade. Sem isso, uma foto com outro enquadramento cai fora da
distribuição que o VAE/ArcFaceEncoder/ImageConditionEncoder foram
treinados para ver (ver utils/edit_common.py:prepare_photo).

Diferença para edit_sdedit.py / edit_ddim_inversion.py: aqui a amostragem
começa de RUÍDO PURO — a foto entra só via tokens globais (ArcFace/CLIP),
a estrutura espacial (pose, fundo, roupa) NÃO é preservada. Para editar
preservando estrutura, use aqueles dois scripts.

Uso:
    python generate_from_photo.py \\
        --photo minha_foto.jpg \\
        --ckpt models/LDM_CFGComp_clipArc_paired/ckpt_best.pt \\
        --orig_attrs Smiling Young \\
        --s_id 3.0 --s_attr 5.0 --sampler ddim

    # atributos detectados automaticamente (precisa do classificador):
    python generate_from_photo.py \\
        --photo minha_foto.jpg \\
        --ckpt models/LDM_CFGComp_clipArc_paired/ckpt_best.pt \\
        --classifier_ckpt models/attribute_classifier/ckpt_best.pt \\
        --enable Smiling --disable Bald
"""

import os
import argparse

import torch
from torchvision.utils import save_image

from utils.edit_common import (
    EditPipeline,
    prepare_photo,
    resolve_original_attrs,
    apply_edits,
    validate_attr_names,
    CELEBA_ATTRS,
)
from utils.composable_cfg_sampling import (
    sample_composable_ddim,
    sample_split_ddim,
    sample_composable_ddpm,
    sample_split_ddpm,
)


def generate(args):
    device = args.device

    print(f"Alinhando foto ao template do CelebA: {args.photo}")
    ref_img = prepare_photo(args.photo, device)  # [1, 3, 256, 256] em [-1, 1]

    print(f"Carregando pipeline (VAE + UNet + encoders): {args.ckpt}")
    pipeline = EditPipeline(
        ckpt_path=args.ckpt, vae_ckpt=args.vae_ckpt,
        device=device, noise_steps=args.noise_steps,
    )
    print(f"  encoder: {pipeline.encoder_type}")

    attrs_vec = resolve_original_attrs(args, args.photo, device)
    attrs_vec = apply_edits(attrs_vec, args.enable, args.disable)
    active = [CELEBA_ATTRS[i] for i in range(40) if attrs_vec[i] == 1.0]
    print(f"Atributos finais: {active}")

    ref_img_n = ref_img.repeat(args.n, 1, 1, 1)
    attrs = attrs_vec.unsqueeze(0).repeat(args.n, 1).to(device)

    with torch.no_grad():
        attr_tok = pipeline.attribute_embedder(attrs)

        if pipeline.encoder_type == "clip_arcface_split":
            clip_tok, id_tok = pipeline.image_encoder(ref_img=ref_img_n, return_separate=True)
        else:
            id_tok = pipeline.image_encoder(ref_img=ref_img_n)

    print(f"\nGerando {args.n} imagens com sampler={args.sampler}  "
          f"s_id={args.s_id}  s_attr={args.s_attr}"
          + (f"  s_clip={args.s_clip}" if pipeline.encoder_type == "clip_arcface_split" else ""))

    if pipeline.encoder_type == "clip_arcface_split":
        sampler_fn = sample_split_ddpm if args.sampler == "ddpm" else sample_split_ddim
        kwargs = dict(
            unet=pipeline.unet, diffusion=pipeline.diffusion,
            id_tokens=id_tok, clip_tokens=clip_tok, attr_tokens=attr_tok,
            n=args.n, channels=4, device=device,
            s_id=args.s_id, s_clip=args.s_clip, s_attr=args.s_attr,
        )
    else:
        sampler_fn = sample_composable_ddpm if args.sampler == "ddpm" else sample_composable_ddim
        kwargs = dict(
            unet=pipeline.unet, diffusion=pipeline.diffusion,
            id_tokens=id_tok, attr_tokens=attr_tok,
            n=args.n, channels=4, device=device,
            s_id=args.s_id, s_attr=args.s_attr,
        )

    if args.sampler == "ddim":
        kwargs["ddim_steps"] = args.ddim_steps

    latents = sampler_fn(**kwargs)

    with torch.no_grad():
        images = pipeline.decode(latents)
    images = (images.clamp(-1, 1) + 1) / 2

    os.makedirs(args.save_dir, exist_ok=True)

    aligned_out = os.path.join(args.save_dir, "aligned_input.png")
    save_image((ref_img.clamp(-1, 1) + 1) / 2, aligned_out)
    print(f"Foto alinhada salva em: {aligned_out}")

    out_path = os.path.join(
        args.save_dir,
        f"generated_{args.sampler}_sid{args.s_id}_sattr{args.s_attr}.png",
    )
    save_image(images, out_path, nrow=args.nrow)
    print(f"Geradas salvas em: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geração com Composable CFG a partir de uma foto qualquer, "
                     "alinhada ao template nativo do CelebA."
    )

    parser.add_argument("--photo", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vae_ckpt", type=str, default="vae/vae_epoch_62.pt")

    parser.add_argument("--orig_attrs", nargs="*", default=None,
                         help="Define os atributos da foto ORIGINAL manualmente (Ex: Smiling "
                              "Young). Sem isso, precisa de --classifier_ckpt.")
    parser.add_argument("--classifier_ckpt", type=str, default=None,
                         help="Checkpoint do CLIPAttributeClassifier para detectar os "
                              "atributos originais da foto automaticamente.")

    parser.add_argument("--enable", nargs="*", default=[])
    parser.add_argument("--disable", nargs="*", default=[])

    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--nrow", type=int, default=4)

    parser.add_argument("--s_id", type=float, default=3.0)
    parser.add_argument("--s_clip", type=float, default=3.0,
                         help="Só usado com --encoder clip_arcface_split.")
    parser.add_argument("--s_attr", type=float, default=5.0)

    parser.add_argument(
        "--sampler", type=str, default="ddim", choices=["ddim", "ddpm"],
        help="ddim: determinístico, --ddim_steps passos (rápido). "
             "ddpm: ancestral completo, noise_steps passos (mais lento).",
    )
    parser.add_argument("--ddim_steps", type=int, default=50, help="Só usado com --sampler ddim.")
    parser.add_argument("--noise_steps", type=int, default=1000)

    parser.add_argument("--save_dir", type=str, default="results/from_photo")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    validate_attr_names(parser, args)

    if args.orig_attrs is None and args.classifier_ckpt is None:
        parser.error("Informe os atributos originais da foto: --orig_attrs ou --classifier_ckpt.")

    generate(args)
