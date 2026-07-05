"""
generate_from_test_paired.py
=============================
Geração a partir do split de TESTE, usando exatamente o mesmo split
DISJUNTO POR IDENTIDADE que train_cfg_composable_paired.py usa
(utils.utils_celeba._split_indices_by_identity, seed=42).

Por que este script existe
---------------------------
generate_cfg_composable.py e generate_mixed_guidance.py recalculam o
split de teste com torch.utils.data.random_split, fatiando por IMAGEM,
não por identidade. train_cfg_composable_paired.py treina com um split
diferente: disjunto por PESSOA (nenhuma identidade aparece em mais de um
split — ver docstring de _split_indices_by_identity). Os dois splits não
batem, então usar --from_test dos outros scripts para avaliar um
checkpoint _paired pode sortear, como "teste", fotos de identidades que
na verdade estavam no TREINO desse checkpoint — vazamento de identidade,
justamente o que o pareamento por identidade foi desenhado para evitar.

Este script reconstrói o dataset pareado (CelebALatentImageCondPairedDataset)
e reaplica _split_indices_by_identity com o mesmo seed=42 do treino,
depois restringe a amostragem ao índice de TESTE resultante — garantindo
que as identidades geradas aqui nunca apareceram no treino do checkpoint
_paired carregado.

A referência de identidade usada é a mesma lógica de treino: outra foto
da mesma pessoa quando disponível (ref_img != alvo), não a própria foto
do alvo — ao contrário de generate_cfg_composable.py, que sempre usa
ref_img == alvo. Isso importa porque train_cfg_composable_paired.py foi
criado exatamente para parar de depender de ref==alvo (ver docstring do
próprio script de treino); gerar com ref==alvo aqui reintroduziria o
mesmo atalho (CLIP vendo a expressão exata do alvo) que o treino pareado
evita, tornando a avaliação inconsistente com o que o modelo aprendeu.

Suporta os três encoders de train_cfg_composable_paired.py:
    clip_arcface        -> composable CFG 2 termos (s_id, s_attr)
    arcface_only        -> composable CFG 2 termos (s_id, s_attr)
    clip_arcface_split  -> composable CFG 3 termos (s_id, s_clip, s_attr)

Uso:
    python generate_from_test_paired.py \\
        --ckpt models/LDM_CFGComp_clipArc_paired/ckpt_best.pt \\
        --n 8 --seed 0 \\
        --enable Smiling --disable Bald \\
        --s_id 3.0 --s_attr 5.0

    # checkpoint com --encoder clip_arcface_split:
    python generate_from_test_paired.py \\
        --ckpt models/LDM_CFGComp_split_paired/ckpt_best.pt \\
        --n 8 --s_id 3.0 --s_clip 3.0 --s_attr 5.0
"""

import os
import argparse

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.unet_conditional import UNet_cond
from models.encoders import ImageConditionEncoder, ArcFaceOnlyEncoder
from models.modules import AttributeEmbedder
from diffusion.conditional_ddpm import Diffusion_conditional
from vae.modules import VAE
from utils.utils_celeba import CelebALatentImageCondPairedDataset, _split_indices_by_identity
from utils.edit_common import _fix_clip_state_dict


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


# ============================================================
# SPLIT DE TESTE — idêntico ao usado por train_cfg_composable_paired.py
# ============================================================

def build_paired_test_split(latent_dir, attr_path, image_dir, encoder_cache_dir, identity_txt):
    dataset = CelebALatentImageCondPairedDataset(
        latent_dir=latent_dir,
        attr_path=attr_path,
        image_dir=image_dir,
        encoder_cache_dir=encoder_cache_dir,
        identity_txt=identity_txt,
    )

    identity_of_by_index = [
        dataset.identity_of.get(dataset.samples[i][0])
        for i in range(len(dataset))
    ]

    train_idx, val_idx, test_idx, n_train_id, n_val_id, n_test_id = \
        _split_indices_by_identity(identity_of_by_index, seed=42)

    print(
        f"[split identidade, seed=42] treino:{n_train_id} val:{n_val_id} teste:{n_test_id} "
        f"identidades | {len(test_idx)} imagens no split de teste"
    )

    overlap = set(dataset.samples[i][0] for i in train_idx) & \
              set(dataset.samples[i][0] for i in test_idx)
    assert not overlap, "BUG: split treino/teste sobrepõe imagens."

    return dataset, test_idx


# ============================================================
# COMPOSABLE CFG — DDIM, 2 ramos (id, attr)
# ============================================================

@torch.no_grad()
def sample_composable_ddim(
    unet, diffusion, id_tokens, attr_tokens,
    n, channels, device, s_id=3.0, s_attr=5.0, ddim_steps=50, eta=0.0,
):
    unet.eval()
    img_size = diffusion.img_size
    z_t = torch.randn(n, channels, img_size, img_size, device=device)

    zeros_id, zeros_attr = torch.zeros_like(id_tokens), torch.zeros_like(attr_tokens)
    ctx_uu = torch.cat([zeros_id,   zeros_attr], dim=2)
    ctx_iu = torch.cat([id_tokens,  zeros_attr], dim=2)
    ctx_ia = torch.cat([id_tokens,  attr_tokens], dim=2)

    step_size = max(diffusion.noise_steps // ddim_steps, 1)
    times = list(range(1, diffusion.noise_steps, step_size))[::-1]

    for idx, t in enumerate(tqdm(times, desc="Sampling (DDIM, id+attr)")):
        t_vec = torch.full((n,), t, device=device, dtype=torch.long)

        eps_uu = unet(z_t, t_vec, context=ctx_uu)
        eps_iu = unet(z_t, t_vec, context=ctx_iu)
        eps_ia = unet(z_t, t_vec, context=ctx_ia)
        eps = eps_uu + s_id * (eps_iu - eps_uu) + s_attr * (eps_ia - eps_iu)

        alpha_bar_t = diffusion.alpha_hat[t]
        is_last = (idx == len(times) - 1)
        alpha_bar_next = (
            torch.tensor(1.0, device=device) if is_last
            else diffusion.alpha_hat[times[idx + 1]]
        )

        pred_x0 = (z_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_next)
        )
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma ** 2, min=0.0)) * eps
        noise = torch.randn_like(z_t) if (eta > 0 and not is_last) else torch.zeros_like(z_t)

        z_t = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt + sigma * noise

    return z_t


# ============================================================
# COMPOSABLE CFG — DDIM, 3 ramos (id, clip, attr)
# ============================================================

@torch.no_grad()
def sample_split_ddim(
    unet, diffusion, id_tokens, clip_tokens, attr_tokens,
    n, channels, device, s_id=3.0, s_clip=3.0, s_attr=5.0, ddim_steps=50, eta=0.0,
):
    unet.eval()
    img_size = diffusion.img_size
    z_t = torch.randn(n, channels, img_size, img_size, device=device)

    zeros_id   = torch.zeros_like(id_tokens)
    zeros_clip = torch.zeros_like(clip_tokens)
    zeros_attr = torch.zeros_like(attr_tokens)

    ctx_000 = torch.cat([zeros_id,   zeros_clip,   zeros_attr], dim=2)
    ctx_i00 = torch.cat([id_tokens,  zeros_clip,   zeros_attr], dim=2)
    ctx_ic0 = torch.cat([id_tokens,  clip_tokens,  zeros_attr], dim=2)
    ctx_ica = torch.cat([id_tokens,  clip_tokens,  attr_tokens], dim=2)

    step_size = max(diffusion.noise_steps // ddim_steps, 1)
    times = list(range(1, diffusion.noise_steps, step_size))[::-1]

    for idx, t in enumerate(tqdm(times, desc="Sampling (DDIM, id+clip+attr)")):
        t_vec = torch.full((n,), t, device=device, dtype=torch.long)

        eps_000 = unet(z_t, t_vec, context=ctx_000)
        eps_i00 = unet(z_t, t_vec, context=ctx_i00)
        eps_ic0 = unet(z_t, t_vec, context=ctx_ic0)
        eps_ica = unet(z_t, t_vec, context=ctx_ica)

        eps = (
            eps_000
            + s_id   * (eps_i00 - eps_000)
            + s_clip * (eps_ic0 - eps_i00)
            + s_attr * (eps_ica - eps_ic0)
        )

        alpha_bar_t = diffusion.alpha_hat[t]
        is_last = (idx == len(times) - 1)
        alpha_bar_next = (
            torch.tensor(1.0, device=device) if is_last
            else diffusion.alpha_hat[times[idx + 1]]
        )

        pred_x0 = (z_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_next)
        )
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma ** 2, min=0.0)) * eps
        noise = torch.randn_like(z_t) if (eta > 0 and not is_last) else torch.zeros_like(z_t)

        z_t = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt + sigma * noise

    return z_t


# ============================================================
# MAIN
# ============================================================

def generate(args):
    device = args.device

    # ---------- attr / paths ----------
    image_dir = os.path.join(args.data_root, "celeba", "img_align_celeba", "img_align_celeba")
    txt_path  = os.path.join(args.data_root, "celeba", "list_attr_celeba.txt")
    csv_path  = os.path.join(args.data_root, "celeba", "list_attr_celeba.csv")
    attr_path = txt_path if os.path.exists(txt_path) else csv_path

    # ---------- split de teste pareado (idêntico ao treino) ----------
    dataset, test_idx = build_paired_test_split(
        latent_dir=args.latent_dir,
        attr_path=attr_path,
        image_dir=image_dir,
        encoder_cache_dir=args.encoder_cache_dir,
        identity_txt=args.identity_txt,
    )

    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(test_idx), generator=rng)[:args.n].tolist()
    chosen = [test_idx[i] for i in perm]

    # ---------- VAE ----------
    print("Carregando VAE...")
    vae = VAE(in_channels=3, latent_dim=4).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ---------- checkpoint ----------
    print(f"Carregando checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    num_img_tokens = ckpt.get("num_img_tokens", args.img_tokens)
    encoder_type   = ckpt.get("encoder", args.encoder)
    ref_pairing    = ckpt.get("ref_pairing", "?")
    print(f"  época: {ckpt.get('epoch', '?')} | tokens: {num_img_tokens} | "
          f"encoder: {encoder_type} | ref_pairing: {ref_pairing}")

    if ref_pairing != "identity":
        print(
            "AVISO: este checkpoint não tem ref_pairing == 'identity' — "
            "pode não ter sido treinado por train_cfg_composable_paired.py. "
            "Este script assume ref_img != alvo (pareado); resultados podem "
            "não corresponder ao que o checkpoint aprendeu."
        )

    # ---------- UNet ----------
    unet = UNet_cond(in_channels=4, out_channels=4, context_dim=512).to(device)
    unet.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
    unet.eval()

    # ---------- encoder de identidade ----------
    if encoder_type == "arcface_only":
        image_encoder = ArcFaceOnlyEncoder(context_dim=512, num_tokens=num_img_tokens).to(device)
    elif encoder_type in ("clip_arcface", "clip_arcface_split"):
        image_encoder = ImageConditionEncoder(
            context_dim=512, num_tokens=num_img_tokens, freeze_backbone=True,
        ).to(device)
    else:
        raise ValueError(f"encoder desconhecido no ckpt: {encoder_type}")

    image_encoder_state = ckpt.get(
        "ema_image_encoder_state_dict", ckpt["image_encoder_state_dict"]
    )
    if encoder_type in ("clip_arcface", "clip_arcface_split"):
        image_encoder_state = _fix_clip_state_dict(image_encoder_state)
    image_encoder.load_state_dict(image_encoder_state)
    image_encoder.eval()

    # ---------- attribute embedder ----------
    attribute_embedder = AttributeEmbedder(num_attributes=40, context_dim=512).to(device)
    attribute_embedder.load_state_dict(
        ckpt.get("ema_attribute_embedder_state_dict", ckpt["attribute_embedder_state_dict"])
    )
    attribute_embedder.eval()

    # ---------- diffusion ----------
    diffusion = Diffusion_conditional(img_size=32, device=device, noise_steps=args.noise_steps)

    os.makedirs(args.save_dir, exist_ok=True)

    orig_imgs, ref_imgs, gen_imgs = [], [], []

    for i, idx in enumerate(chosen):
        target_fname = dataset.samples[idx][0]
        print(f"\n--- Amostra {i + 1}/{len(chosen)}  (arquivo alvo: {target_fname}) ---")

        latent, attrs_cpu, ref_cpu, id_emb_cpu, clip_raw_cpu = dataset[idx]

        from PIL import Image as PILImage
        target_img = PILImage.open(os.path.join(dataset.image_dir, target_fname)).convert("RGB")
        target_tensor = dataset.transform(target_img)

        attrs_vec = attrs_cpu.clone()
        for name in (args.enable or []):
            attrs_vec[ATTR_TO_IDX[name]] = 1.0
            print(f"  + {name}")
        for name in (args.disable or []):
            attrs_vec[ATTR_TO_IDX[name]] = 0.0
            print(f"  - {name}")

        ref_img     = ref_cpu.unsqueeze(0).to(device)
        id_emb      = id_emb_cpu.unsqueeze(0).to(device)
        clip_raw    = clip_raw_cpu.unsqueeze(0).to(device)
        attrs       = attrs_vec.unsqueeze(0).to(device)

        with torch.no_grad():
            attr_tok = attribute_embedder(attrs)

            if encoder_type == "arcface_only":
                id_tok = image_encoder(ref_img=ref_img, id_emb=id_emb)
                latents = sample_composable_ddim(
                    unet=unet, diffusion=diffusion,
                    id_tokens=id_tok, attr_tokens=attr_tok,
                    n=1, channels=4, device=device,
                    s_id=args.s_id, s_attr=args.s_attr, ddim_steps=args.ddim_steps,
                )
            elif encoder_type == "clip_arcface_split":
                clip_tok, id_tok = image_encoder(
                    ref_img=ref_img, id_emb=id_emb, clip_tokens_raw=clip_raw,
                    return_separate=True,
                )
                latents = sample_split_ddim(
                    unet=unet, diffusion=diffusion,
                    id_tokens=id_tok, clip_tokens=clip_tok, attr_tokens=attr_tok,
                    n=1, channels=4, device=device,
                    s_id=args.s_id, s_clip=args.s_clip, s_attr=args.s_attr,
                    ddim_steps=args.ddim_steps,
                )
            else:  # clip_arcface
                id_tok = image_encoder(ref_img=ref_img, id_emb=id_emb, clip_tokens_raw=clip_raw)
                latents = sample_composable_ddim(
                    unet=unet, diffusion=diffusion,
                    id_tokens=id_tok, attr_tokens=attr_tok,
                    n=1, channels=4, device=device,
                    s_id=args.s_id, s_attr=args.s_attr, ddim_steps=args.ddim_steps,
                )

            gen = vae.decode(latents / 0.18215)

        orig_imgs.append(((target_tensor.clamp(-1, 1) + 1) / 2).cpu())
        ref_imgs.append(((ref_cpu.clamp(-1, 1) + 1) / 2).cpu())
        gen_imgs.append(((gen.squeeze(0).clamp(-1, 1) + 1) / 2).cpu())

    grid = []
    for o, r, g in zip(orig_imgs, ref_imgs, gen_imgs):
        grid.append(o)
        grid.append(r)
        grid.append(g)
    grid = torch.stack(grid)

    out_path = os.path.join(
        args.save_dir,
        f"test_paired_seed{args.seed}_sid{args.s_id}_sattr{args.s_attr}.png",
    )
    save_image(grid, out_path, nrow=3)
    print(f"\nSalvo em: {out_path}  (colunas: original | referência pareada | gerada)")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geração a partir do split de TESTE disjunto por identidade "
                     "(consistente com train_cfg_composable_paired.py)."
    )

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vae_ckpt", type=str, default="vae/vae_epoch_62.pt")

    parser.add_argument("--data_root", type=str, default="./CelebA_data")
    parser.add_argument("--latent_dir", type=str, default="./cache_latent")
    parser.add_argument("--encoder_cache_dir", type=str, default="./cache_encoder")
    parser.add_argument(
        "--identity_txt", type=str,
        default="./CelebA_data/celeba/identity_CelebA.txt",
    )

    parser.add_argument("--n", type=int, default=8, help="Nº de amostras de teste.")
    parser.add_argument("--seed", type=int, default=0, help="Seed para sortear quais amostras de teste usar.")

    parser.add_argument("--enable", nargs="*", default=[])
    parser.add_argument("--disable", nargs="*", default=[])

    parser.add_argument("--s_id", type=float, default=3.0)
    parser.add_argument("--s_clip", type=float, default=3.0, help="Só usado com --encoder clip_arcface_split.")
    parser.add_argument("--s_attr", type=float, default=5.0)

    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--noise_steps", type=int, default=1000)

    # fallbacks se o ckpt for antigo e não tiver essas chaves
    parser.add_argument("--encoder", type=str, default="clip_arcface",
                         choices=["clip_arcface", "arcface_only", "clip_arcface_split"])
    parser.add_argument("--img_tokens", type=int, default=16)

    parser.add_argument("--save_dir", type=str, default="results/from_test_paired")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    for group in (args.enable, args.disable):
        for name in group:
            if name not in ATTR_TO_IDX:
                parser.error(f"Atributo desconhecido: {name}")

    generate(args)
