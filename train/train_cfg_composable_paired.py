"""
train_cfg_composable_paired.py
================================
Variação de train_cfg_composable.py — a única mudança é a ORIGEM da imagem
de referência (identidade) usada para condicionar CLIP+ArcFace / ArcFace.

Em train_cfg_composable.py, ref_img é sempre a MESMA foto que é o alvo do
denoising. Isso deixa o CLIP ver a expressão exata que o modelo precisa
gerar — o modelo aprende a copiar a expressão da referência em vez de usar
o token de atributo, e nenhum valor de `s_attr` desfaz isso (diagnosticado
depois de rodar train_cfg_composable.py com --encoder clip_arcface: subir
`s_attr` não conseguia fazer quase ninguém sorrir).

Aqui, ref_img vem de uma foto DIFERENTE da mesma identidade quando
disponível (CelebALatentImageCondPairedDataset, ver utils/utils_celeba.py).
`attrs` (usado na loss) continua vindo do alvo, nunca da referência.

Checado com analyze_identity_variety.ipynb: no CelebA, identidades têm em
média ~20 fotos, e 91% das identidades com >=2 fotos variam em `Smiling` —
ou seja, a maioria dos batches vai conseguir sortear uma referência com
expressão diferente do alvo.

Todo o resto (fórmula composable, dropout independente, arquitetura,
checkpoint) é idêntico a train_cfg_composable.py — os checkpoints dos dois
scripts são intercambiáveis via --warmstart_ckpt / --resume_ckpt.

Requer, além do cache de encoder (./cache_encoder/, gerado por
cache_arcface.py):
    --identity_txt  identity_CelebA.txt oficial do CelebA (filename
                     identity_id, sem header). Nem todo mirror inclui esse
                     arquivo — ver analyze_identity_variety.ipynb.

Uso:
    python train/train_cfg_composable_paired.py \\
        --run_name LDM_CFGComp_clipArc_paired \\
        --encoder clip_arcface \\
        --batch_size 64 --epochs 200
"""

import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import argparse
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid

from board import Board
from utils.utils_celeba import get_data_imagecond_paired, save_images, setup_logging
from diffusion.conditional_ddpm import Diffusion_conditional

from models.unet_conditional import UNet_cond
from models.encoders import ImageConditionEncoder, ArcFaceOnlyEncoder
from models.modules import AttributeEmbedder
from vae.modules import VAE

from utils.fid import (
    InceptionFeatureExtractor,
    calculate_activation_statistics,
    calculate_frechet_distance,
    compute_real_activations_streamed,
    compute_fake_activations_batched,
    compute_fake_activations_batched_n,
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)

import contextlib


# =========================================================
# DDP
# =========================================================

def setup_ddp():

    if "RANK" not in os.environ:
        os.environ["RANK"]        = "0"
        os.environ["LOCAL_RANK"]  = "0"
        os.environ["WORLD_SIZE"]  = "1"
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="nccl")

    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank, global_rank


# =========================================================
# EMA
# =========================================================

def update_ema(ema_model, model, decay=0.9999):

    ema_model.eval()

    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=(1 - decay))


# =========================================================
# CFG DROPOUT INDEPENDENTE
# =========================================================
#
# Sorteia, por amostra do batch, qual condicionante zerar.
# Probabilidades (defaults Liu et al. 2022):
#   p_id_only_drop   = 0.1  → zera id,   mantém attr
#   p_attr_only_drop = 0.1  → mantém id, zera attr
#   p_both_drop      = 0.1  → zera os dois
#   resto (0.7)              → mantém os dois
#
# Crucial: o sorteio é POR AMOSTRA, não por batch — assim cada batch vê
# todos os combos e o modelo aprende as 4 distribuições marginais.
# =========================================================

def cfg_masks(batch_size, p_id_only, p_attr_only, p_both, device):

    u = torch.rand(batch_size, device=device)

    drop_both     = u < p_both
    drop_id_only  = (u >= p_both)            & (u < p_both + p_id_only)
    drop_attr_only = (u >= p_both + p_id_only) & (u < p_both + p_id_only + p_attr_only)

    keep_id   = ~(drop_both | drop_id_only)
    keep_attr = ~(drop_both | drop_attr_only)

    # shape [B, 1, 1] para broadcast em [B, C, T]
    return keep_id.float().view(-1, 1, 1), keep_attr.float().view(-1, 1, 1)


# =========================================================
# CFG DROPOUT INDEPENDENTE — modo split (3 ramos: ArcFace, CLIP, attr)
# =========================================================
#
# Diferente de cfg_masks (partição categórica sobre 2 ramos), aqui cada
# ramo é sorteado de forma independente (Bernoulli por ramo, por amostra).
# Com 3 ramos independentes uma partição categórica exigiria especificar
# 8 probabilidades à mão; sorteio independente é o padrão mais comum em
# composable diffusion multi-condicionante e ainda garante que todas as
# combinações aparecem no treino, só que sem frequências exatas impostas.
# =========================================================

def cfg_masks_split(batch_size, p_drop_id, p_drop_clip, p_drop_attr, device):

    keep_id   = (torch.rand(batch_size, device=device) >= p_drop_id).float().view(-1, 1, 1)
    keep_clip = (torch.rand(batch_size, device=device) >= p_drop_clip).float().view(-1, 1, 1)
    keep_attr = (torch.rand(batch_size, device=device) >= p_drop_attr).float().view(-1, 1, 1)

    return keep_id, keep_clip, keep_attr


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    image_encoder,
    attribute_embedder,
    attrs,
    ref_img=None,
    id_emb=None,
    clip_tokens_raw=None,
    encoder_type="clip_arcface",
    keep_id_mask=None,
    keep_attr_mask=None,
):
    """
    Monta context = concat([id_tokens, attr_tokens], dim=tokens).

    keep_id_mask, keep_attr_mask : [B, 1, 1] em {0, 1}
        Se 0, os tokens correspondentes são zerados (modo "uncond" daquele ramo).
        Se None, mantém tudo.
    """

    if encoder_type == "arcface_only":
        id_tokens = image_encoder(ref_img=ref_img, id_emb=id_emb)
    else:
        id_tokens = image_encoder(
            ref_img=ref_img,
            id_emb=id_emb,
            clip_tokens_raw=clip_tokens_raw,
        )
    # id_tokens: [B, C, T_id]

    attr_tokens = attribute_embedder(attrs)
    # attr_tokens: [B, C, 40]

    if keep_id_mask is not None:
        id_tokens = id_tokens * keep_id_mask
    if keep_attr_mask is not None:
        attr_tokens = attr_tokens * keep_attr_mask

    context = torch.cat([id_tokens, attr_tokens], dim=2)
    return context


# =========================================================
# BUILD CONTEXT — modo split (ArcFace, CLIP e attr como ramos
# independentes de CFG, em vez de ArcFace+CLIP fundidos num "id" só)
# =========================================================

def build_context_split(
    image_encoder,
    attribute_embedder,
    attrs,
    ref_img=None,
    id_emb=None,
    clip_tokens_raw=None,
    keep_id_mask=None,
    keep_clip_mask=None,
    keep_attr_mask=None,
):
    """
    Monta context = concat([id_tokens, clip_tokens, attr_tokens], dim=tokens).

    image_encoder deve ser um ImageConditionEncoder (clip_arcface) chamado
    com return_separate=True — ArcFace (identidade pura) e CLIP (aparência/
    estilo da referência) saem como tensores distintos, cada um com seu
    próprio dropout de CFG independente.
    """

    clip_tokens, id_tokens = image_encoder(
        ref_img=ref_img,
        id_emb=id_emb,
        clip_tokens_raw=clip_tokens_raw,
        return_separate=True,
    )
    # id_tokens, clip_tokens: [B, C, T_img]

    attr_tokens = attribute_embedder(attrs)
    # attr_tokens: [B, C, 40]

    if keep_id_mask is not None:
        id_tokens = id_tokens * keep_id_mask
    if keep_clip_mask is not None:
        clip_tokens = clip_tokens * keep_clip_mask
    if keep_attr_mask is not None:
        attr_tokens = attr_tokens * keep_attr_mask

    context = torch.cat([id_tokens, clip_tokens, attr_tokens], dim=2)
    return context


# =========================================================
# COMPOSABLE CFG SAMPLING
# =========================================================
#
# eps = eps(∅,∅)
#     + s_id   * [eps(id,∅)    − eps(∅,∅)]
#     + s_attr * [eps(id,attr) − eps(id,∅)]
#
# A ordem importa: o termo de attr usa (id,∅) como baseline, não (∅,∅).
# Isso significa "dada a identidade, o quanto o atributo desloca eps".
# =========================================================

@torch.no_grad()
def _sample_composable_ddim(
    unet,
    diffusion,
    id_tokens_full,        # [N, C, T_id]
    attr_tokens_full,      # [N, C, 40]
    n,
    channels,
    device,
    s_id=3.0,
    s_attr=5.0,
    ddim_steps=50,
    eta=0.0,
):
    """
    Composable CFG (Liu et al. 2022) + amostrador DDIM (Song et al. 2020).

    Substitui a amostragem ancestral (999 passos, 3 forwards de UNet cada)
    por um sub-conjunto de `ddim_steps` timesteps com update determinístico
    (eta=0) ou parcialmente estocástico (eta>0). Isso é usado só nas
    avaliações periódicas (samples de visualização e FID) — o treino em si
    (loss de denoising) não muda em nada.

    O modelo nunca é chamado em t=0 (sample_timesteps só sorteia t em
    [1, noise_steps) no treino); o último salto do DDIM vai direto para
    alpha_bar=1 (imagem limpa) usando a predição de x0 do último passo real.
    """
    unet.eval()
    img_size = diffusion.img_size
    z_t      = torch.randn(n, channels, img_size, img_size, device=device)

    zeros_id   = torch.zeros_like(id_tokens_full)
    zeros_attr = torch.zeros_like(attr_tokens_full)

    ctx_uu = torch.cat([zeros_id,       zeros_attr],       dim=2)  # ∅, ∅
    ctx_iu = torch.cat([id_tokens_full, zeros_attr],       dim=2)  # id, ∅
    ctx_ia = torch.cat([id_tokens_full, attr_tokens_full], dim=2)  # id, attr

    step_size = max(diffusion.noise_steps // ddim_steps, 1)
    times     = list(range(1, diffusion.noise_steps, step_size))[::-1]

    for idx, t in enumerate(times):

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
            (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_next)
        )
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma ** 2, min=0.0)) * eps
        noise  = torch.randn_like(z_t) if (eta > 0 and not is_last) else torch.zeros_like(z_t)

        z_t = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt + sigma * noise

    return z_t


# =========================================================
# COMPOSABLE CFG SAMPLING — modo split (ArcFace, CLIP, attr)
# =========================================================
#
# eps = eps(∅,∅,∅)
#     + s_id   * [eps(id,∅,∅)      − eps(∅,∅,∅)]
#     + s_clip * [eps(id,clip,∅)   − eps(id,∅,∅)]
#     + s_attr * [eps(id,clip,attr) − eps(id,clip,∅)]
#
# Cadeia: identidade (ArcFace) primeiro, depois aparência/estilo da
# referência (CLIP), depois atributos — cada termo mede "dado tudo antes
# dele, o quanto esse ramo desloca eps". s_clip=0 remove inteiramente a
# aparência/estilo da referência (a fonte do vazamento que atrapalha
# edições estruturais como Bald), mantendo identidade (ArcFace) + atributo.
# =========================================================

@torch.no_grad()
def _sample_split_ddim(
    unet,
    diffusion,
    id_tokens_full,        # [N, C, T_img]  (ArcFace)
    clip_tokens_full,      # [N, C, T_img]  (CLIP)
    attr_tokens_full,      # [N, C, 40]
    n,
    channels,
    device,
    s_id=3.0,
    s_clip=3.0,
    s_attr=5.0,
    ddim_steps=50,
    eta=0.0,
):
    unet.eval()
    img_size = diffusion.img_size
    z_t      = torch.randn(n, channels, img_size, img_size, device=device)

    zeros_id   = torch.zeros_like(id_tokens_full)
    zeros_clip = torch.zeros_like(clip_tokens_full)
    zeros_attr = torch.zeros_like(attr_tokens_full)

    ctx_000 = torch.cat([zeros_id,       zeros_clip,       zeros_attr],       dim=2)
    ctx_i00 = torch.cat([id_tokens_full, zeros_clip,       zeros_attr],       dim=2)
    ctx_ic0 = torch.cat([id_tokens_full, clip_tokens_full, zeros_attr],       dim=2)
    ctx_ica = torch.cat([id_tokens_full, clip_tokens_full, attr_tokens_full], dim=2)

    step_size = max(diffusion.noise_steps // ddim_steps, 1)
    times     = list(range(1, diffusion.noise_steps, step_size))[::-1]

    for idx, t in enumerate(times):

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
            (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_next)
        )
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma ** 2, min=0.0)) * eps
        noise  = torch.randn_like(z_t) if (eta > 0 and not is_last) else torch.zeros_like(z_t)

        z_t = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt + sigma * noise

    return z_t


# =========================================================
# TRAIN
# =========================================================

def train(args):

    local_rank, global_rank = setup_ddp()
    is_master = (global_rank == 0)
    device = f"cuda:{local_rank}"

    if is_master:
        setup_logging(args.run_name)
        board       = Board(run_name=args.run_name, enabled=True)
        global_step = 0

    # =====================================================
    # DATA  (referência = outra foto da mesma identidade)
    # =====================================================

    train_loader, val_loader, _, train_sampler, use_cache = get_data_imagecond_paired(
        args,
        is_distributed=True,
    )

    if is_master:
        print("[ref_pairing] identity (ref_img != alvo quando a identidade tem >=2 fotos)")

    # =====================================================
    # CONFIG
    # =====================================================

    latent_dim     = 4
    context_dim    = 512
    num_img_tokens = args.img_tokens
    num_attrs      = 40

    # =====================================================
    # VAE (frozen)
    # =====================================================

    vae = VAE(in_channels=3, latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("vae/vae_epoch_62.pt", map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # =====================================================
    # UNET
    # =====================================================

    model = UNet_cond(
        in_channels=latent_dim,
        out_channels=latent_dim,
        context_dim=context_dim,
    ).to(device)

    # =====================================================
    # IDENTITY ENCODER
    # =====================================================

    if args.encoder in ("clip_arcface", "clip_arcface_split"):
        image_encoder = ImageConditionEncoder(
            context_dim=context_dim,
            num_tokens=num_img_tokens,
            freeze_backbone=True,
        ).to(device)
        # tokens reais: 2 * num_img_tokens (clip + arcface); em
        # "clip_arcface_split" os dois ramos são chamados com
        # return_separate=True e viram condicionantes independentes de
        # CFG em vez de ficarem concatenados como um "id" só — mesma
        # classe/pesos de "clip_arcface", checkpoints intercambiáveis.
    elif args.encoder == "arcface_only":
        image_encoder = ArcFaceOnlyEncoder(
            context_dim=context_dim,
            num_tokens=num_img_tokens,
        ).to(device)
    else:
        raise ValueError(f"--encoder inválido: {args.encoder}")

    # =====================================================
    # ATTRIBUTE EMBEDDER
    # =====================================================

    attribute_embedder = AttributeEmbedder(
        num_attributes=num_attrs,
        context_dim=context_dim,
    ).to(device)

    # =====================================================
    # DIFFUSION
    # =====================================================

    diffusion = Diffusion_conditional(
        img_size=args.image_size // 8,
        device=device,
    )

    # =====================================================
    # EMA
    # =====================================================

    ema_model              = deepcopy(model).eval()
    ema_image_encoder      = deepcopy(image_encoder).eval()
    ema_attribute_embedder = deepcopy(attribute_embedder).eval()

    for p in ema_model.parameters():
        p.requires_grad = False
    for p in ema_image_encoder.parameters():
        p.requires_grad = False
    for p in ema_attribute_embedder.parameters():
        p.requires_grad = False

    # =====================================================
    # DDP
    # =====================================================

    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    image_encoder = DDP(
        image_encoder,
        device_ids=[local_rank],
        find_unused_parameters=False,
    )

    attribute_embedder = DDP(attribute_embedder, device_ids=[local_rank])

    # =====================================================
    # OPTIMIZER  (apenas um — sem classifier)
    # =====================================================

    optimizer = optim.AdamW(
        list(model.parameters())
        + list(image_encoder.parameters())
        + list(attribute_embedder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # =====================================================
    # LR SCHEDULER
    # =====================================================

    warmup_epochs = 10

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=(args.epochs - warmup_epochs), eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )

    # =====================================================
    # AMP
    # =====================================================

    scaler = GradScaler()
    accumulation_steps = 4

    start_epoch   = 0
    best_val_loss = float("inf")

    # =====================================================
    # RESUME
    # =====================================================

    if args.resume_ckpt is not None and os.path.isfile(args.resume_ckpt):

        if is_master:
            print(f"\nCarregando checkpoint: {args.resume_ckpt}")

        ckpt = torch.load(args.resume_ckpt, map_location=device)

        model.module.load_state_dict(ckpt["model_state_dict"])
        image_encoder.module.load_state_dict(ckpt["image_encoder_state_dict"])
        attribute_embedder.module.load_state_dict(ckpt["attribute_embedder_state_dict"])

        ema_model.load_state_dict(ckpt["ema_state_dict"])
        ema_image_encoder.load_state_dict(ckpt["ema_image_encoder_state_dict"])
        ema_attribute_embedder.load_state_dict(ckpt["ema_attribute_embedder_state_dict"])

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = (
            args.start_epoch
            if args.start_epoch is not None
            else ckpt["epoch"] + 1
        )

        if "scheduler_state_dict" in ckpt and args.start_epoch is None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()

        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        if is_master:
            if "global_step" in ckpt:
                global_step = ckpt["global_step"]
            if "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]
            print(f"Retomando epoch {start_epoch}")

    # =====================================================
    # WARM START — carrega só a UNet de outro checkpoint
    # (compatível com checkpoints de train_cfg_composable.py)
    # =====================================================

    elif args.warmstart_ckpt is not None and os.path.isfile(args.warmstart_ckpt):

        if is_master:
            print(f"\nWarm start de: {args.warmstart_ckpt}")

        ckpt = torch.load(args.warmstart_ckpt, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(
            ckpt.get("ema_state_dict", ckpt["model_state_dict"])
        )

        if args.start_epoch is not None:
            start_epoch = args.start_epoch
            for _ in range(start_epoch):
                scheduler.step()

        if is_master:
            print("UNet carregada. Encoder e AttributeEmbedder do zero.")

    # =====================================================
    # DIRS
    # =====================================================

    save_dir    = os.path.join("models",  args.run_name)
    results_dir = os.path.join("results", args.run_name)

    os.makedirs(save_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    fixed_ref_imgs = None

    # =====================================================
    # FIXED SAMPLES  (+ condicionamento do lado GERADO do FID)
    # =====================================================
    #
    # Reaproveita o mesmo laço de carregamento: os 16 primeiros elementos
    # viram as amostras fixas de visualização (originals.jpg / smile), até
    # --fid_num_samples elementos viram o condicionamento do lado GERADO
    # do FID. As estatísticas do lado REAL usam o dataset de validação
    # inteiro, calculadas à parte (streaming, ver bloco abaixo).
    # =====================================================

    fid_extractor  = None
    fid_mu_real    = None
    fid_sigma_real = None
    fid_ref_imgs   = None
    fid_attrs      = None
    fid_id_emb     = None
    fid_clip       = None
    last_fid_value = None

    if is_master:
        from PIL import Image as PILImage

        val_base = val_loader.dataset.dataset

        # ---- amostras fixas (visualização) + condicionamento do lado
        #      GERADO do FID: --fid_num_samples controla só isto — é o
        #      lado caro (amostragem por difusão completa, repetida a
        #      cada --fid_every épocas), então fica bem menor que o
        #      dataset de validação inteiro. ----
        n_fixed = min(16, len(val_loader.dataset))
        n_fake  = min(args.fid_num_samples, len(val_loader.dataset)) if args.fid_every > 0 else 0
        n_load  = max(n_fixed, n_fake)

        samples_fixed = []
        for _i in range(n_load):
            real_idx = val_loader.dataset.indices[_i]
            sample   = val_base[real_idx]

            # dataset pareado sempre retorna 5-tupla (cache é obrigatório)
            _, _attrs, _ref, _idemb, _clip = sample

            _fname = val_base.samples[real_idx][0]
            _img   = PILImage.open(
                os.path.join(val_base.image_dir, _fname)
            ).convert("RGB")
            _orig = val_base.transform(_img)

            samples_fixed.append((_ref, _orig, _attrs, _idemb, _clip))

        fixed_ref_imgs = torch.stack([s[0] for s in samples_fixed[:n_fixed]]).to(device)
        fixed_attrs    = torch.stack([s[2] for s in samples_fixed[:n_fixed]]).to(device)
        orig_imgs      = torch.stack([s[1] for s in samples_fixed[:n_fixed]])

        fixed_id_emb = torch.stack([s[3] for s in samples_fixed[:n_fixed]]).to(device)

        fixed_clip = None
        if args.encoder in ("clip_arcface", "clip_arcface_split") and samples_fixed[0][4] is not None:
            fixed_clip = torch.stack([s[4] for s in samples_fixed[:n_fixed]]).to(device)

        # nota: fixed_ref_imgs aqui já é a referência PAREADA (outra foto),
        # não a mesma foto de orig_imgs — é esperado que fiquem diferentes
        # (pose/expressão), é o comportamento que estamos testando.
        save_images(
            orig_imgs,
            os.path.join(results_dir, "originals.jpg"),
            nrow=4,
        )
        save_images(
            fixed_ref_imgs,
            os.path.join(results_dir, "references_paired.jpg"),
            nrow=4,
        )
        print("Originais salvas em originals.jpg, referências pareadas em references_paired.jpg")

        # -------------------------------------------------
        # FID: condicionamento para o lado GERADO (n_fake imagens,
        # recriadas pelo EMA a cada avaliação)
        # -------------------------------------------------

        if args.fid_every > 0:

            fid_ref_imgs = torch.stack([s[0] for s in samples_fixed[:n_fake]]).to(device)
            fid_attrs    = torch.stack([s[2] for s in samples_fixed[:n_fake]]).to(device)
            fid_id_emb   = torch.stack([s[3] for s in samples_fixed[:n_fake]]).to(device)

            fid_clip = None
            if args.encoder in ("clip_arcface", "clip_arcface_split") and samples_fixed[0][4] is not None:
                fid_clip = torch.stack([s[4] for s in samples_fixed[:n_fake]]).to(device)

    # =====================================================
    # FID: estatísticas REAIS — cálculo PREGUIÇOSO
    # =====================================================
    #
    # Passar todo o conjunto de validação pela InceptionV3 só compensa se
    # o treino de fato chegar a usar FID — com --fid_start_epoch, os
    # primeiros epochs (modelo ainda gerando ruído/rostos incoerentes,
    # FID não é informativo) são pulados, então adiamos esse custo para a
    # primeira vez que ele for realmente necessário, dentro do loop. Assim,
    # se o treino for interrompido antes de --fid_start_epoch, esse custo
    # nunca é pago.
    # =====================================================

    def _ensure_fid_real_stats():
        nonlocal fid_mu_real, fid_sigma_real, fid_extractor

        if fid_mu_real is not None:
            return

        n_real = len(val_loader.dataset)

        if fid_extractor is None:
            fid_extractor = InceptionFeatureExtractor(device=device)

        # Cache em disco: sem isso, todo --resume_ckpt repetiria esse
        # custo do zero.
        fid_cache_path = os.path.join(save_dir, "fid_real_stats.npz")

        if os.path.isfile(fid_cache_path):

            cached = np.load(fid_cache_path)

            if int(cached["n_real"]) == n_real:
                fid_mu_real    = cached["mu"]
                fid_sigma_real = cached["sigma"]
                print(f"[FID] Estatísticas reais carregadas do cache "
                      f"({fid_cache_path}, {n_real} imagens).")
                return
            else:
                print(f"[FID] Cache em {fid_cache_path} tem n_real="
                      f"{int(cached['n_real'])}, mas o split atual tem "
                      f"{n_real} — recalculando.")

        def _load_real_chunk(start, end):
            imgs = []
            for _i in range(start, end):
                real_idx = val_loader.dataset.indices[_i]
                fname    = val_base.samples[real_idx][0]
                img      = PILImage.open(
                    os.path.join(val_base.image_dir, fname)
                ).convert("RGB")
                imgs.append(val_base.transform(img))
            return torch.stack(imgs)

        print(f"[FID] Extraindo features InceptionV3 de {n_real} imagens reais "
              f"(todo o conjunto de validação, uma vez)...")
        real_acts = compute_real_activations_streamed(
            _load_real_chunk, n_real, fid_extractor, chunk_size=args.fid_real_batch_size
        )
        fid_mu_real, fid_sigma_real = calculate_activation_statistics(real_acts)

        np.savez(
            fid_cache_path,
            mu=fid_mu_real,
            sigma=fid_sigma_real,
            n_real=n_real,
        )
        print(f"[FID] Estatísticas reais salvas em {fid_cache_path}.")

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)

        model.train()
        image_encoder.train()
        attribute_embedder.train()

        pbar = (
            tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs - 1}")
            if is_master
            else train_loader
        )

        epoch_losses = []

        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):

            latents, attrs, ref_img, id_emb, clip_tokens = batch
            id_emb      = id_emb.to(device, non_blocking=True)
            clip_tokens = clip_tokens.to(device, non_blocking=True) if args.encoder in ("clip_arcface", "clip_arcface_split") else None

            latents = latents.to(device, non_blocking=True)
            attrs   = attrs.to(device,   non_blocking=True)
            ref_img = ref_img.to(device,  non_blocking=True)

            B = latents.shape[0]

            t = diffusion.sample_timesteps(B).to(device)

            with torch.no_grad():
                z_t, noise = diffusion.noise_images(latents, t)

            # ---- máscaras de dropout independentes por amostra ----
            if args.encoder == "clip_arcface_split":
                keep_id, keep_clip, keep_attr = cfg_masks_split(
                    B,
                    p_drop_id=args.split_dropout_id,
                    p_drop_clip=args.split_dropout_clip,
                    p_drop_attr=args.split_dropout_attr,
                    device=device,
                )
            else:
                keep_id, keep_attr = cfg_masks(
                    B,
                    p_id_only=args.cfg_dropout_id_only,
                    p_attr_only=args.cfg_dropout_attr_only,
                    p_both=args.cfg_dropout_both,
                    device=device,
                )

            is_accumulating = (
                (i + 1) % accumulation_steps != 0
                and (i + 1) != len(train_loader)
            )

            ctx_managers = [
                model.no_sync() if is_accumulating else contextlib.nullcontext(),
                image_encoder.no_sync() if is_accumulating else contextlib.nullcontext(),
                attribute_embedder.no_sync() if is_accumulating else contextlib.nullcontext(),
            ]

            with ctx_managers[0], ctx_managers[1], ctx_managers[2]:
                with autocast():
                    if args.encoder == "clip_arcface_split":
                        context = build_context_split(
                            image_encoder=image_encoder,
                            attribute_embedder=attribute_embedder,
                            attrs=attrs,
                            ref_img=ref_img,
                            id_emb=id_emb,
                            clip_tokens_raw=clip_tokens,
                            keep_id_mask=keep_id,
                            keep_clip_mask=keep_clip,
                            keep_attr_mask=keep_attr,
                        )
                    else:
                        context = build_context(
                            image_encoder=image_encoder,
                            attribute_embedder=attribute_embedder,
                            attrs=attrs,
                            ref_img=ref_img,
                            id_emb=id_emb,
                            clip_tokens_raw=clip_tokens,
                            encoder_type=args.encoder,
                            keep_id_mask=keep_id,
                            keep_attr_mask=keep_attr,
                        )

                    predicted_noise = model(z_t, t, context=context)
                    loss = F.mse_loss(predicted_noise, noise)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

            if not is_accumulating:

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if is_master:

                    update_ema(ema_model,              model.module)
                    update_ema(ema_image_encoder,      image_encoder.module)
                    update_ema(ema_attribute_embedder, attribute_embedder.module)

                    loss_display = loss.item() * accumulation_steps
                    pbar.set_postfix(MSE=loss_display)
                    board.log_scalar("Loss/Batch", loss_display, global_step)
                    epoch_losses.append(loss_display)
                    global_step += 1

        scheduler.step()

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()
        image_encoder.eval()
        attribute_embedder.eval()

        val_losses = []

        with torch.no_grad():

            for batch in val_loader:

                latents, attrs, ref_img, id_emb, clip_tokens = batch
                id_emb      = id_emb.to(device, non_blocking=True)
                clip_tokens = clip_tokens.to(device, non_blocking=True) if args.encoder in ("clip_arcface", "clip_arcface_split") else None

                latents = latents.to(device, non_blocking=True)
                attrs   = attrs.to(device,   non_blocking=True)
                ref_img = ref_img.to(device,  non_blocking=True)

                with autocast():

                    t = diffusion.sample_timesteps(latents.shape[0]).to(device)
                    z_t, noise = diffusion.noise_images(latents, t)

                    # validação SEM dropout — mede a likelihood condicionada
                    if args.encoder == "clip_arcface_split":
                        context = build_context_split(
                            image_encoder=image_encoder,
                            attribute_embedder=attribute_embedder,
                            attrs=attrs,
                            ref_img=ref_img,
                            id_emb=id_emb,
                            clip_tokens_raw=clip_tokens,
                        )
                    else:
                        context = build_context(
                            image_encoder=image_encoder,
                            attribute_embedder=attribute_embedder,
                            attrs=attrs,
                            ref_img=ref_img,
                            id_emb=id_emb,
                            clip_tokens_raw=clip_tokens,
                            encoder_type=args.encoder,
                        )

                    predicted_noise = model(z_t, t, context=context)
                    v_loss = F.mse_loss(predicted_noise, noise)

                val_losses.append(v_loss.item())

        def reduce_mean(values):
            t = torch.tensor(sum(values) / len(values), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            return t.item()

        avg_val = reduce_mean(val_losses)

        # =================================================
        # MASTER LOGGING
        # =================================================

        if is_master:

            avg_train = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            lr_cur    = optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch}"
                f" | train: {avg_train:.6f}  val: {avg_val:.6f}"
                f" | LR: {lr_cur:.2e}"
            )

            board.log_scalars("Loss/Epoch", {"train": avg_train, "val": avg_val}, epoch)
            board.log_scalar("Metrics/LR", lr_cur, epoch)

        # =================================================
        # CHECKPOINT
        # =================================================

        if is_master:

            save_periodic = (epoch % 10 == 0 or epoch == args.epochs - 1)
            save_best     = (avg_val < best_val_loss)

            if save_periodic or save_best:

                ckpt = {
                    "epoch":                              epoch,
                    "global_step":                        global_step,
                    "model_state_dict":                   model.module.state_dict(),
                    "image_encoder_state_dict":           image_encoder.module.state_dict(),
                    "attribute_embedder_state_dict":      attribute_embedder.module.state_dict(),
                    "ema_state_dict":                     ema_model.state_dict(),
                    "ema_image_encoder_state_dict":       ema_image_encoder.state_dict(),
                    "ema_attribute_embedder_state_dict":  ema_attribute_embedder.state_dict(),
                    "optimizer_state_dict":               optimizer.state_dict(),
                    "scheduler_state_dict":               scheduler.state_dict(),
                    "scaler_state_dict":                  scaler.state_dict(),
                    "val_loss":                           avg_val,
                    "best_val_loss":                      best_val_loss,
                    "num_img_tokens":                     num_img_tokens,
                    "encoder":                            args.encoder,
                    "ref_pairing":                         "identity",
                    "fid":                                last_fid_value,
                }

                if save_periodic:
                    torch.save(ckpt, os.path.join(save_dir, "ckpt.pt"))
                    print("Checkpoint salvo.")

                if save_best:
                    best_val_loss = avg_val
                    ckpt["best_val_loss"] = best_val_loss
                    torch.save(ckpt, os.path.join(save_dir, "ckpt_best.pt"))
                    print(f"Novo melhor val loss: {best_val_loss:.6f} → ckpt_best.pt")

        # =================================================
        # SAMPLE  —  composable CFG
        # =================================================

        if is_master and (epoch % 10 == 0 or epoch == args.epochs - 1):

            ema_model.eval()
            ema_image_encoder.eval()
            ema_attribute_embedder.eval()

            with torch.no_grad():

                clip_tok = None

                if args.encoder == "arcface_only":
                    id_tok = ema_image_encoder(
                        ref_img=fixed_ref_imgs,
                        id_emb=fixed_id_emb,
                    )
                elif args.encoder == "clip_arcface_split":
                    clip_tok, id_tok = ema_image_encoder(
                        ref_img=fixed_ref_imgs,
                        id_emb=fixed_id_emb,
                        clip_tokens_raw=fixed_clip,
                        return_separate=True,
                    )
                else:
                    id_tok = ema_image_encoder(
                        ref_img=fixed_ref_imgs,
                        id_emb=fixed_id_emb,
                        clip_tokens_raw=fixed_clip,
                    )

                attr_tok = ema_attribute_embedder(fixed_attrs)

                def _sample(attr_tokens):
                    if args.encoder == "clip_arcface_split":
                        return _sample_split_ddim(
                            unet=ema_model,
                            diffusion=diffusion,
                            id_tokens_full=id_tok,
                            clip_tokens_full=clip_tok,
                            attr_tokens_full=attr_tokens,
                            n=fixed_ref_imgs.shape[0],
                            channels=latent_dim,
                            device=device,
                            s_id=args.s_id_val,
                            s_clip=args.s_clip_val,
                            s_attr=args.s_attr_val,
                            ddim_steps=args.ddim_steps,
                        )
                    return _sample_composable_ddim(
                        unet=ema_model,
                        diffusion=diffusion,
                        id_tokens_full=id_tok,
                        attr_tokens_full=attr_tokens,
                        n=fixed_ref_imgs.shape[0],
                        channels=latent_dim,
                        device=device,
                        s_id=args.s_id_val,
                        s_attr=args.s_attr_val,
                        ddim_steps=args.ddim_steps,
                    )

                # ---- atributos originais ----
                print("Sampling (composable CFG — atributos originais)...")
                latents_orig = _sample(attr_tok)
                images_orig = vae.decode(latents_orig / 0.18215)

                save_images(images_orig, os.path.join(results_dir, f"{epoch}_attr_orig.jpg"), nrow=4)
                board.log_image(
                    "Samples/AttrOrig",
                    make_grid(images_orig, nrow=4, normalize=True, value_range=(-1, 1)),
                    epoch,
                )

                # ---- mesmo id, com Smiling forçado a 1 ----
                #   teste qualitativo: o modelo consegue fazer a pessoa sorrir?
                SMILING_IDX = 31
                attrs_smile = fixed_attrs.clone()
                attrs_smile[:, SMILING_IDX] = 1.0
                attr_tok_smile = ema_attribute_embedder(attrs_smile)

                print("Sampling (composable CFG — Smiling=1 forçado)...")
                latents_smile = _sample(attr_tok_smile)
                images_smile = vae.decode(latents_smile / 0.18215)

                save_images(images_smile, os.path.join(results_dir, f"{epoch}_attr_smile.jpg"), nrow=4)
                board.log_image(
                    "Samples/AttrSmile",
                    make_grid(images_smile, nrow=4, normalize=True, value_range=(-1, 1)),
                    epoch,
                )

        # =================================================
        # FID  —  custoso, só a cada --fid_every épocas e só depois de
        # --fid_start_epoch (nos primeiros epochs o modelo ainda gera
        # ruído/rostos incoerentes — FID não é informativo, só custa
        # tempo de treino que dá pra usar em steps de verdade).
        # =================================================
        #
        # As estatísticas REAIS são calculadas (ou carregadas do cache)
        # na primeira vez que este bloco roda de fato — ver
        # _ensure_fid_real_stats(). Aqui só recalculamos as estatísticas
        # do conjunto GERADO (o encoder/UNet EMA muda a cada época).
        # =================================================

        epochs_since_fid_start = epoch - args.fid_start_epoch

        if (
            is_master
            and args.fid_every > 0
            and epochs_since_fid_start >= 0
            and (epochs_since_fid_start % args.fid_every == 0 or epoch == args.epochs - 1)
        ):

            _ensure_fid_real_stats()

            ema_model.eval()
            ema_image_encoder.eval()
            ema_attribute_embedder.eval()

            with torch.no_grad():

                print(f"[FID] Gerando {fid_ref_imgs.shape[0]} amostras para avaliação...")

                clip_tok_fid = None

                if args.encoder == "arcface_only":
                    id_tok_fid = ema_image_encoder(
                        ref_img=fid_ref_imgs,
                        id_emb=fid_id_emb,
                    )
                elif args.encoder == "clip_arcface_split":
                    clip_tok_fid, id_tok_fid = ema_image_encoder(
                        ref_img=fid_ref_imgs,
                        id_emb=fid_id_emb,
                        clip_tokens_raw=fid_clip,
                        return_separate=True,
                    )
                else:
                    id_tok_fid = ema_image_encoder(
                        ref_img=fid_ref_imgs,
                        id_emb=fid_id_emb,
                        clip_tokens_raw=fid_clip,
                    )

                attr_tok_fid = ema_attribute_embedder(fid_attrs)

                if args.encoder == "clip_arcface_split":

                    def _gen_fid_chunk_split(id_chunk, clip_chunk, attr_chunk):
                        lat = _sample_split_ddim(
                            unet=ema_model,
                            diffusion=diffusion,
                            id_tokens_full=id_chunk,
                            clip_tokens_full=clip_chunk,
                            attr_tokens_full=attr_chunk,
                            n=id_chunk.shape[0],
                            channels=latent_dim,
                            device=device,
                            s_id=args.s_id_val,
                            s_clip=args.s_clip_val,
                            s_attr=args.s_attr_val,
                            ddim_steps=args.ddim_steps,
                        )
                        return vae.decode(lat / 0.18215)

                    fake_acts = compute_fake_activations_batched_n(
                        generate_fn=_gen_fid_chunk_split,
                        tokens_full=[id_tok_fid, clip_tok_fid, attr_tok_fid],
                        extractor=fid_extractor,
                        chunk_size=args.fid_batch_size,
                    )

                else:

                    def _gen_fid_chunk(id_chunk, attr_chunk):
                        lat = _sample_composable_ddim(
                            unet=ema_model,
                            diffusion=diffusion,
                            id_tokens_full=id_chunk,
                            attr_tokens_full=attr_chunk,
                            n=id_chunk.shape[0],
                            channels=latent_dim,
                            device=device,
                            s_id=args.s_id_val,
                            s_attr=args.s_attr_val,
                            ddim_steps=args.ddim_steps,
                        )
                        return vae.decode(lat / 0.18215)

                    fake_acts = compute_fake_activations_batched(
                        generate_fn=_gen_fid_chunk,
                        id_tokens_full=id_tok_fid,
                        attr_tokens_full=attr_tok_fid,
                        extractor=fid_extractor,
                        chunk_size=args.fid_batch_size,
                    )

                fid_mu_fake, fid_sigma_fake = calculate_activation_statistics(fake_acts)
                fid_value = calculate_frechet_distance(
                    fid_mu_real, fid_sigma_real, fid_mu_fake, fid_sigma_fake
                )
                last_fid_value = fid_value

                print(f"[FID] Epoch {epoch}: FID = {fid_value:.3f}")
                board.log_scalar("Metrics/FID", fid_value, epoch)

    if is_master:
        board.close()

    dist.destroy_process_group()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name",   type=str,   default="LDM_CFGComposable_Paired")
    parser.add_argument("--epochs",     type=int,   default=2000)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--image_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)

    parser.add_argument(
        "--encoder",
        type=str,
        default="clip_arcface",
        choices=["clip_arcface", "arcface_only", "clip_arcface_split"],
        help="Encoder de identidade. Com pareamento por identidade, "
             "clip_arcface é o caso que este script existe para consertar "
             "(CLIP não vê mais a expressão do alvo). clip_arcface_split "
             "usa a mesma classe/pesos de clip_arcface, mas trata ArcFace "
             "(identidade pura) e CLIP (aparência/estilo da referência) "
             "como ramos INDEPENDENTES de CFG composable (3 termos: s_id, "
             "s_clip, s_attr, em vez de 2) — permite zerar s_clip na "
             "inferência para eliminar o vazamento de aparência da "
             "referência em edições estruturais (ex.: Bald) sem perder "
             "identidade (ArcFace) nem controle de atributo.",
    )

    parser.add_argument(
        "--img_tokens",
        type=int,
        default=16,
        help="Tokens por ramo do encoder (clip e arcface usam este número cada).",
    )

    parser.add_argument(
        "--encoder_cache_dir",
        type=str,
        default="./cache_encoder",
    )

    parser.add_argument(
        "--identity_txt",
        type=str,
        default="./CelebA_data/celeba/identity_CelebA.txt",
        help="identity_CelebA.txt oficial do CelebA (filename identity_id, "
             "sem header). Ver analyze_identity_variety.ipynb.",
    )

    # ---- dropouts INDEPENDENTES de CFG (2 ramos: --encoder clip_arcface / arcface_only) ----
    parser.add_argument("--cfg_dropout_id_only",   type=float, default=0.1)
    parser.add_argument("--cfg_dropout_attr_only", type=float, default=0.1)
    parser.add_argument("--cfg_dropout_both",      type=float, default=0.1)

    # ---- dropouts INDEPENDENTES de CFG (3 ramos: --encoder clip_arcface_split) ----
    # Bernoulli por ramo, por amostra — não uma partição categórica como
    # acima (ver cfg_masks_split).
    parser.add_argument("--split_dropout_id",   type=float, default=0.1)
    parser.add_argument("--split_dropout_clip", type=float, default=0.1)
    parser.add_argument("--split_dropout_attr", type=float, default=0.1)

    # ---- escalas usadas só nos samples de validação ----
    parser.add_argument("--s_id_val",   type=float, default=3.0)
    parser.add_argument("--s_attr_val", type=float, default=5.0)
    parser.add_argument(
        "--s_clip_val", type=float, default=3.0,
        help="Escala de guidance para o ramo CLIP (aparência/estilo da "
             "referência) — só usado com --encoder clip_arcface_split. "
             "Baixe para perto de 0 para edições estruturais (ex.: Bald) "
             "onde a aparência da referência conflita com o atributo "
             "desejado; mantenha alto para reconstrução fiel.",
    )

    parser.add_argument(
        "--ddim_steps", type=int, default=50,
        help="Nº de passos do amostrador DDIM usado nas avaliações "
             "periódicas (samples de visualização e FID) — NÃO afeta o "
             "treino (que continua treinando em t contínuo via MSE de "
             "denoising). Substitui a amostragem ancestral de ~999 passos, "
             "cortando o custo de cada avaliação em ~noise_steps/ddim_steps "
             "vezes. 50 é o padrão comum na literatura (Song et al. 2020); "
             "suba se notar degradação visual perceptível nas amostras.",
    )

    # ---- FID (custoso — só roda a cada N épocas) ----
    parser.add_argument(
        "--fid_every", type=int, default=10,
        help="Calcula FID a cada N épocas a partir de --fid_start_epoch "
             "(0 desativa). As estatísticas REAIS usam o conjunto de "
             "validação inteiro e são calculadas (ou carregadas do cache "
             "em disco) na primeira vez que forem necessárias; a cada "
             "avaliação só as estatísticas do conjunto GERADO pelo EMA "
             "mudam.",
    )
    parser.add_argument(
        "--fid_start_epoch", type=int, default=0,
        help="Só começa a calcular FID a partir deste epoch. Nos primeiros "
             "epochs o modelo ainda gera ruído/rostos incoerentes e o FID "
             "não é informativo — pular esse trecho economiza tempo de "
             "amostragem (por difusão completa) que pode ir para steps de "
             "treino de verdade. Ajuste proporcional ao seu --epochs total, "
             "não a um valor absoluto.",
    )
    parser.add_argument(
        "--fid_num_samples", type=int, default=256,
        help="Nº de imagens GERADAS (via difusão completa) para estimar as "
             "estatísticas do lado 'fake' do FID a cada avaliação — não "
             "afeta o lado real, que sempre usa o split de validação "
             "inteiro. O ideal estatístico é >=2048 (dimensão das features "
             "da InceptionV3), mas cada unidade aqui é uma amostragem por "
             "difusão completa repetida a cada --fid_every épocas; 256 é um "
             "compromisso custo/estabilidade, não o ideal — suba se puder "
             "pagar o tempo de geração extra.",
    )
    parser.add_argument(
        "--fid_batch_size", type=int, default=16,
        help="Tamanho do chunk de amostragem/inferência para o lado GERADO "
             "do FID (controla o pico de uso de memória da difusão).",
    )
    parser.add_argument(
        "--fid_real_batch_size", type=int, default=128,
        help="Tamanho do chunk de leitura+InceptionV3 para o lado REAL do "
             "FID (só carrega imagem + forward, sem amostragem por difusão "
             "— pode ser bem maior que --fid_batch_size).",
    )

    parser.add_argument("--resume_ckpt",    type=str, default=None)
    parser.add_argument("--warmstart_ckpt", type=str, default=None)
    parser.add_argument("--start_epoch",    type=int, default=None)

    args = parser.parse_args()

    train(args)
