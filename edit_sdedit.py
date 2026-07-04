"""
edit_sdedit.py
==============
Edição de atributo em FOTO REAL preservando estrutura (roupa, fundo, pose)
via img2img estilo SDEdit (Meng et al., 2022) + composable CFG.

Em vez de partir de ruído puro (generate_cfg_composable.py), o DDIM parte
da própria foto: o latente VAE é ruidificado até um timestep intermediário
(--strength) e o denoising segue dali com os atributos EDITADOS. Tudo que o
ruído não destruiu — layout, roupa, fundo — sobrevive na saída.

O único knob é --strength (fração do schedule de ruído):
    baixo  (~0.3) → preserva muito, mas a edição pode não "pegar";
    alto   (~0.8) → edição forte, mas volta a gerar quase do zero.
Por default testa 0.3/0.5/0.7 de uma vez e salva lado a lado — compare e
escolha por tipo de edição.

Não requer nenhum retreino: usa os checkpoints EMA existentes de
train_cfg_composable*.py (2 ramos ou clip_arcface_split).

Uso:
    python edit_sdedit.py \\
        --photo minha_foto.jpg \\
        --ckpt models/LDM_CFGComp_clipArc_paired/ckpt_best.pt \\
        --classifier_ckpt models/attr_classifier/ckpt_best.pt \\
        --enable Smiling \\
        --strengths 0.3 0.5 0.7

    (sem classificador: descreva a foto à mão com
     --orig_attrs Male Young Black_Hair ...)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch

from utils.edit_common import (
    EditPipeline,
    prepare_photo,
    resolve_original_attrs,
    apply_edits,
    validate_attr_names,
    ddim_times,
    ddim_sample,
    save_row,
)


def main(args):

    torch.manual_seed(args.seed)
    device = args.device

    pipe = EditPipeline(
        ckpt_path=args.ckpt,
        vae_ckpt=args.vae_ckpt,
        device=device,
        noise_steps=args.noise_steps,
    )

    img = prepare_photo(args.photo, device)

    attrs_orig = resolve_original_attrs(args, args.photo, device)
    attrs_edit = apply_edits(attrs_orig, args.enable, args.disable)

    # condicionamento vem da PRÓPRIA foto (ref = input) com os atributos
    # já editados — a estrutura espacial entra pelo latente, não pelos
    # tokens globais
    contexts, scales = pipe.build_context_chain(
        ref_img=img,
        attrs=attrs_edit,
        s_id=args.s_id,
        s_attr=args.s_attr,
        s_clip=args.s_clip,
    )

    z0 = pipe.encode(img)

    times_full_desc = ddim_times(pipe.diffusion, args.ddim_steps)[::-1]

    outputs = [img.squeeze(0)]
    labels  = ["original"]

    for strength in args.strengths:

        cutoff = int(strength * (pipe.diffusion.noise_steps - 1))
        times  = [t for t in times_full_desc if t <= cutoff]

        if not times:
            print(f"[strength={strength}] abaixo do primeiro timestep do "
                  "DDIM — pulando (aumente o valor).")
            continue

        t0 = times[0]
        alpha_bar_t0 = pipe.diffusion.alpha_hat[t0]

        # ruidifica o latente da foto até t0 (forward process fechado)
        noise = torch.randn_like(z0)
        z_t0 = (
            torch.sqrt(alpha_bar_t0) * z0
            + torch.sqrt(1.0 - alpha_bar_t0) * noise
        )

        print(f"\n[strength={strength}] começando o DDIM em t={t0} "
              f"({len(times)}/{len(times_full_desc)} passos)")

        z_out = ddim_sample(
            pipe, z_t0, times, contexts, scales,
            desc=f"SDEdit s={strength}",
        )

        outputs.append(pipe.decode(z_out).squeeze(0))
        labels.append(f"strength={strength}")

    edits = "_".join(
        [f"+{n}" for n in (args.enable or [])]
        + [f"-{n}" for n in (args.disable or [])]
    ) or "recon"

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(
        args.save_dir,
        f"sdedit_{edits}_sid{args.s_id}_sattr{args.s_attr}_seed{args.seed}.png",
    )
    save_row(outputs, out_path, labels=labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Edição de foto real via SDEdit (img2img) + composable CFG"
    )

    parser.add_argument("--photo", type=str, required=True,
                        help="Foto a editar (CelebA 178x218 ou foto pessoal "
                             "qualquer — alinhamento automático).")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint de train_cfg_composable*.py.")
    parser.add_argument("--vae_ckpt", type=str, default="vae/vae_epoch_62.pt")

    # ---- atributos da foto original (escolha UMA das vias) ----
    parser.add_argument("--orig_attrs", nargs="*", default=None,
                        help="Atributos positivos da foto, à mão. "
                             "Ex: --orig_attrs Male Young Black_Hair")
    parser.add_argument("--classifier_ckpt", type=str, default=None,
                        help="Checkpoint do CLIPAttributeClassifier para "
                             "prever os atributos automaticamente.")

    # ---- edição ----
    parser.add_argument("--enable",  nargs="*", default=[])
    parser.add_argument("--disable", nargs="*", default=[])

    # ---- SDEdit ----
    parser.add_argument("--strengths", type=float, nargs="*",
                        default=[0.3, 0.5, 0.7],
                        help="Frações do schedule de ruído a testar; cada "
                             "uma vira uma coluna na imagem de saída.")

    # ---- escalas do composable CFG ----
    parser.add_argument("--s_id",   type=float, default=3.0)
    parser.add_argument("--s_attr", type=float, default=5.0)
    parser.add_argument("--s_clip", type=float, default=3.0,
                        help="Só usado com checkpoints clip_arcface_split.")

    parser.add_argument("--ddim_steps",  type=int, default=50)
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--save_dir",    type=str, default="results/edit_sdedit")
    parser.add_argument("--device",      type=str, default="cuda")

    args = parser.parse_args()
    validate_attr_names(parser, args)

    main(args)
