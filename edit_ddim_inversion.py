"""
edit_ddim_inversion.py
======================
Edição de atributo em FOTO REAL preservando estrutura via INVERSÃO DDIM
(Song et al., 2020; uso para edição como em Prompt-to-Prompt, Hertz et al.,
2022) + composable CFG.

Com eta=0 o DDIM é um mapa determinístico ruído→imagem; rodá-lo ao
contrário recupera o z_T específico cuja trajetória reconstrói exatamente a
foto de entrada. Reamostrando desse z_T com os atributos EDITADOS, tudo que
a mudança de condicionamento não empurra — fundo, roupa, pose, iluminação —
segue a trajetória original quase inalterado. Preserva estrutura melhor que
o SDEdit (edit_sdedit.py) em força de edição equivalente, porque nenhuma
informação é destruída indiscriminadamente por ruído.

Detalhes que importam:
  - A inversão usa o contexto totalmente condicionado com os atributos
    ORIGINAIS da foto, em escala 1 (1 forward de UNet por passo) — guidance
    alto na inversão amplifica o erro de aproximação e desloca z_T para
    fora da distribuição.
  - A reamostragem usa guidance, mas com defaults mais baixos que os da
    geração do zero (s_id=1: a identidade já está codificada no próprio
    z_T; o orçamento de guidance vai para s_attr).
  - A saída inclui a coluna "recon" (reamostragem com os atributos
    ORIGINAIS e as mesmas escalas da edição): é o teto de fidelidade da
    inversão sob esse guidance — o que já difere da original ali é drift
    da inversão+guidance, não efeito da edição.

Não requer nenhum retreino: usa os checkpoints EMA existentes de
train_cfg_composable*.py (2 ramos ou clip_arcface_split).

Uso:
    python edit_ddim_inversion.py \\
        --photo minha_foto.jpg \\
        --ckpt models/LDM_CFGComp_clipArc_paired/ckpt_best.pt \\
        --classifier_ckpt models/attr_classifier/ckpt_best.pt \\
        --enable Smiling \\
        --s_id 1.0 --s_attr 3.0

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
    ddim_invert,
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

    # duas cadeias de contexto sobre o MESMO condicionamento de imagem
    # (ref = a própria foto): uma com os atributos originais (inversão +
    # coluna recon), outra com os editados (a edição em si)
    contexts_orig, scales = pipe.build_context_chain(
        ref_img=img, attrs=attrs_orig,
        s_id=args.s_id, s_attr=args.s_attr, s_clip=args.s_clip,
    )
    contexts_edit, _ = pipe.build_context_chain(
        ref_img=img, attrs=attrs_edit,
        s_id=args.s_id, s_attr=args.s_attr, s_clip=args.s_clip,
    )

    # ---- inversão: foto → z_T (atributos originais, escala 1) ----

    z0 = pipe.encode(img)

    times_asc = ddim_times(pipe.diffusion, args.ddim_steps)

    z_T = ddim_invert(pipe, z0, times_asc, context=contexts_orig[-1])

    # ---- reamostragem: recon (sanidade) e edição, ambas do mesmo z_T ----

    times_desc = times_asc[::-1]

    print("\nReconstrução (atributos originais — teto de fidelidade da inversão)...")
    z_recon = ddim_sample(pipe, z_T, times_desc, contexts_orig, scales, desc="Recon")

    print("\nEdição (atributos modificados)...")
    z_edit = ddim_sample(pipe, z_T, times_desc, contexts_edit, scales, desc="Edição")

    outputs = [
        img.squeeze(0),
        pipe.decode(z_recon).squeeze(0),
        pipe.decode(z_edit).squeeze(0),
    ]
    labels = ["original", "recon (attrs originais)", "editada"]

    edits = "_".join(
        [f"+{n}" for n in (args.enable or [])]
        + [f"-{n}" for n in (args.disable or [])]
    ) or "recon"

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(
        args.save_dir,
        f"inversion_{edits}_sid{args.s_id}_sattr{args.s_attr}.png",
    )
    save_row(outputs, out_path, labels=labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Edição de foto real via inversão DDIM + composable CFG"
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

    # ---- escalas do composable CFG (só na REAMOSTRAGEM; a inversão é
    # sempre em escala 1) — defaults mais baixos que na geração do zero:
    # guidance alto diverge da trajetória invertida e destrói a estrutura
    # que a inversão preservou ----
    parser.add_argument("--s_id",   type=float, default=1.0)
    parser.add_argument("--s_attr", type=float, default=3.0)
    parser.add_argument("--s_clip", type=float, default=1.0,
                        help="Só usado com checkpoints clip_arcface_split.")

    parser.add_argument("--ddim_steps",  type=int, default=50)
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--save_dir",    type=str, default="results/edit_inversion")
    parser.add_argument("--device",      type=str, default="cuda")

    args = parser.parse_args()
    validate_attr_names(parser, args)

    main(args)
