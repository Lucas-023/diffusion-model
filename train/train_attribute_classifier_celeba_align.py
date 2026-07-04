"""
Fine-tuning do classificador de atributos CelebA (CLIPAttributeClassifier)
— VARIANTE com alinhamento NATIVO do CelebA (CelebAAligner), em vez do
template ArcFace usado em train_attribute_classifier.py.

Por quê esta variante existe: várias das 40 classes do CelebA são atributos
de cabelo/chapéu/orelha/acessório (Bald, Bangs, Wearing_Hat,
Wearing_Earrings, Receding_Hairline, ...). O template ArcFace é otimizado
para reconhecimento de identidade (recorte apertado olhos-nariz-boca) e
pode cortar exatamente a região de cabeça/cabelo que esses atributos
precisam. O recorte nativo do CelebA (data/correct_alignment.py) mantém
mais dessa região E usa a MESMA convenção do resto do pipeline de difusão
(VAE, ArcFaceEncoder, ImageConditionEncoder) — ver pipeline.md seção 3.

Este script é um experimento para comparar contra o checkpoint treinado
com FaceAligner (train_attribute_classifier.py, val F1≈0.73) — usa
diretórios de cache e checkpoint DIFERENTES para não sobrescrever aquele
resultado.

Também inclui 3 melhorias na receita de treino (replicadas de volta em
train_attribute_classifier.py para manter a comparação de alinhamento
controlada — só o alinhamento difere entre os dois scripts):
  - Asymmetric Loss no lugar de BCEWithLogitsLoss(pos_weight=...).
  - Warmup linear + cosine decay de LR (o run anterior, com LR fixo,
    teve seu melhor epoch na época 2 e degradou depois).
  - Clipping de gradiente + EMA dos pesos (validação/checkpoint usam o
    modelo EMA, não o modelo cru) — mesmo padrão já usado em
    train/train_image_cond.py e train/train_dist.py.

Não faz parte do pipeline de treino da diffusion — é um script
independente para treinar o AttributePredictor "de verdade" (o
existente em models/encoders.py nunca foi treinado, é um placeholder
com pesos ImageNet aleatórios no fc).

O que este script faz, na ordem:
  1. Alinha cada foto do CelebA com CelebAAligner (data/correct_alignment.py)
     para o template NATIVO do CelebA, aplica CenterCrop(178) (mesmo recorte
     usado em todo o resto do pipeline, ver _IDENTITY_TRANSFORM em
     utils/utils_celeba.py e cache_generator.py) e cacheia o resultado em
     disco — alinhar por época seria caro (detecção de rosto via ONNX por
     imagem); alinhar uma vez e reusar o cache resolve isso.
  2. Split por IDENTIDADE (não por imagem) para não vazar a mesma
     pessoa entre train/val/test.
  3. Aumento de dados orientado a robustez de domínio (JPEG, blur,
     downsample/upsample, color jitter) — CelebA é mais "limpo" que
     fotos reais, isso ajuda a fechar esse gap.
  4. Asymmetric Loss por atributo (CelebA é muito desbalanceado — sem
     isso o modelo ignora atributos raros).
  5. Fine-tuning parcial do CLIP ViT-L/14 (só os últimos
     --unfreeze_last_n blocos + head treinam), com warmup+cosine LR,
     clipping de gradiente e EMA dos pesos.
  6. Calibração de limiar por atributo (F1-ótimo) na validação (usando
     o modelo EMA), salva junto do checkpoint.

Uso:
    python -m train.train_attribute_classifier_celeba_align \\
        --data_root ./CelebA_data/celeba \\
        --output ./checkpoints/attribute_classifier_celeba_align.pt \\
        --epochs 10 --batch_size 64

Requer: CelebA_data/celeba/{img_align_celeba/img_align_celeba,
list_attr_celeba.txt|csv} e, opcionalmente, identity_CelebA.txt para o
split por identidade (cai para split aleatório por imagem se ausente).
"""

import argparse
import io
import os
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data.correct_alignment import CelebAAligner
from models.attribute_classifier import CLIPAttributeClassifier


# ============================================================
# LOSS — Asymmetric Loss (Ben-Baruch et al., 2020)
#
# Substitui BCEWithLogitsLoss(pos_weight=...). pos_weight é um fator
# escalar único por atributo; ASL trata positivos e negativos com
# focusing/clipping assimétricos, o padrão atual para desbalanceamento
# severo em classificação multi-rótulo (exatamente o caso do CelebA —
# "Bald" ~2% positivo). Não precisa de pos_weight nem de calcular
# pos_frac do split de treino.
# ============================================================

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # fp32 sempre — sob autocast, logits pode vir em fp16, e o piso de
        # precisão do fp16 (~6e-8) é maior que self.eps (1e-8): um
        # sigmoid(logits)->0 (atributo raro, previsão confiante errada)
        # vira log(0)=-inf -> NaN na loss, silenciosamente pulando steps
        # via GradScaler em vez de estourar erro.
        logits = logits.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        pt = xs_pos * targets + xs_neg * (1 - targets)
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss *= torch.pow(1 - pt, one_sided_gamma)

        return -loss.sum(dim=-1).mean()


# ============================================================
# EMA — mesmo padrão usado em train/train_image_cond.py, train/train_dist.py
# ============================================================

def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1 - decay))


# ============================================================
# LEITURA DOS ATRIBUTOS (mesmo formato usado em utils/utils_celeba.py)
# ============================================================

def _read_attr_file(attr_path):
    if attr_path.endswith(".txt"):
        with open(attr_path, "r") as f:
            lines = f.readlines()
        attr_names = lines[1].split()
        samples = []
        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            attrs = [(int(x) + 1) // 2 for x in parts[1:]]  # {-1,1} -> {0,1}
            samples.append((filename, attrs))
        return samples, attr_names
    else:
        import pandas as pd
        df = pd.read_csv(attr_path)
        attr_names = list(df.columns[1:])
        samples = []
        for _, row in df.iterrows():
            filename = row.iloc[0]
            attrs = [1 if x == 1 else 0 for x in row.iloc[1:].tolist()]
            samples.append((filename, attrs))
        return samples, attr_names


def _find_attr_path(data_root):
    txt_path = os.path.join(data_root, "list_attr_celeba.txt")
    csv_path = os.path.join(data_root, "list_attr_celeba.csv")
    if os.path.exists(txt_path):
        return txt_path
    if os.path.exists(csv_path):
        return csv_path
    raise FileNotFoundError(f"Nenhum arquivo de atributos encontrado em {data_root}")


def _split_by_identity(filenames, identity_txt, seed=42, train_frac=0.70, val_frac=0.15):
    """Split disjunto por pessoa (evita vazamento). Cai para split aleatório
    por imagem se identity_CelebA.txt não existir."""
    if identity_txt and os.path.isfile(identity_txt):
        id_map = {}
        with open(identity_txt, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    id_map[parts[0]] = parts[1]
        groups = defaultdict(list)
        for i, fname in enumerate(filenames):
            groups[id_map.get(fname, f"__no_id_{i}")].append(i)
        keys = list(groups.keys())
        print(f"[split] {len(keys)} identidades encontradas (split disjunto por pessoa).")
    else:
        print("[split] identity_CelebA.txt não encontrado — split aleatório por imagem.")
        groups = {str(i): [i] for i in range(len(filenames))}
        keys = list(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    def flatten(ks):
        out = []
        for k in ks:
            out.extend(groups[k])
        return out

    return (
        flatten(keys[:n_train]),
        flatten(keys[n_train:n_train + n_val]),
        flatten(keys[n_train + n_val:]),
    )


# ============================================================
# PRÉ-CACHE DE ALINHAMENTO
# ============================================================

def build_align_cache(filenames, image_dir, cache_dir):
    """Alinha cada imagem uma única vez (template nativo do CelebA, ver
    data/correct_alignment.py) e salva em cache_dir já com CenterCrop(178)
    aplicado (mesmo recorte usado em todo o resto do pipeline — ver
    _IDENTITY_TRANSFORM em utils/utils_celeba.py e cache_generator.py).
    Idempotente (pula arquivos já processados) — pode ser interrompido e
    retomado."""
    os.makedirs(cache_dir, exist_ok=True)
    aligner = CelebAAligner()
    square_crop = T.CenterCrop(178)

    n_missing_face = 0
    todo = [f for f in filenames if not os.path.exists(os.path.join(cache_dir, f))]
    print(f"[align_cache] {len(filenames) - len(todo)} já em cache, {len(todo)} a processar.")

    for i, fname in enumerate(todo):
        src = os.path.join(image_dir, fname)
        img = Image.open(src).convert("RGB")
        aligned, found = aligner.align_pil(img)   # 178x218, convenção nativa do CelebA
        aligned = square_crop(aligned)            # 178x178, igual ao resto do pipeline
        if not found:
            n_missing_face += 1
        aligned.save(os.path.join(cache_dir, fname), quality=95)
        if (i + 1) % 2000 == 0:
            print(f"[align_cache] {i + 1}/{len(todo)} processadas ({n_missing_face} sem rosto detectado)")

    print(f"[align_cache] concluído. {n_missing_face} imagens sem rosto detectado (fallback center-crop).")


# ============================================================
# AUMENTO DE DADOS PARA ROBUSTEZ DE DOMÍNIO
# (aplicado só no treino, sobre o crop já alinhado)
# ============================================================

def _jpeg_recompress(img: Image.Image, min_q=30, max_q=90):
    q = random.randint(min_q, max_q)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _downsample_upsample(img: Image.Image, min_scale=0.3, max_scale=0.9):
    w, h = img.size
    scale = random.uniform(min_scale, max_scale)
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


class DomainRobustnessAugment:
    """Simula a diferença de qualidade entre fotos do CelebA (limpas) e
    fotos reais capturadas por câmeras de celular/webcam."""

    def __init__(self, p=0.5):
        self.p = p
        self.color_jitter = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)
        self.blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = self.color_jitter(img)
        if random.random() < self.p:
            img = self.blur(img)
        if random.random() < self.p:
            img = _downsample_upsample(img)
        if random.random() < self.p:
            img = _jpeg_recompress(img)
        return img


# ============================================================
# DATASET
# ============================================================

class AlignedCelebAAttrDataset(Dataset):
    def __init__(self, samples, attr_names, aligned_dir, indices, train=False, image_size=224):
        self.samples = samples
        self.attr_names = attr_names
        self.aligned_dir = aligned_dir
        self.indices = indices
        self.train = train
        self.image_size = image_size
        self.domain_aug = DomainRobustnessAugment(p=0.5) if train else None
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        filename, attrs = self.samples[idx]

        img = Image.open(os.path.join(self.aligned_dir, filename)).convert("RGB")

        if self.train:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.domain_aug is not None:
                img = self.domain_aug(img)

        img_t = self.to_tensor(img)
        attrs_t = torch.tensor(attrs, dtype=torch.float32)
        return img_t, attrs_t


# ============================================================
# CALIBRAÇÃO DE LIMIAR POR ATRIBUTO (F1-ótimo na validação)
# ============================================================

@torch.no_grad()
def calibrate_thresholds(probs, targets, n_steps=37):
    """probs, targets: [N, num_attrs] em CPU. Retorna thresholds [num_attrs]."""
    thresholds = torch.full((probs.shape[1],), 0.5)
    grid = torch.linspace(0.05, 0.95, n_steps)

    for a in range(probs.shape[1]):
        p = probs[:, a]
        y = targets[:, a]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (p > t).float()
            tp = (pred * y).sum()
            fp = (pred * (1 - y)).sum()
            fn = ((1 - pred) * y).sum()
            f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
            if f1 > best_f1:
                best_f1, best_t = f1, t.item()
        thresholds[a] = best_t

    return thresholds


# ============================================================
# TREINO
# ============================================================

def train(args):
    device = args.device

    attr_path = _find_attr_path(args.data_root)
    samples, attr_names = _read_attr_file(attr_path)
    filenames = [s[0] for s in samples]

    identity_txt = args.identity_txt or os.path.join(args.data_root, "identity_CelebA.txt")
    train_idx, val_idx, test_idx = _split_by_identity(filenames, identity_txt, seed=args.seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    image_dir = args.image_dir or os.path.join(
        args.data_root, "img_align_celeba", "img_align_celeba"
    )

    if not args.skip_align_cache:
        build_align_cache(filenames, image_dir, args.align_cache_dir)
    aligned_dir = args.align_cache_dir

    train_ds = AlignedCelebAAttrDataset(samples, attr_names, aligned_dir, train_idx, train=True)
    val_ds = AlignedCelebAAttrDataset(samples, attr_names, aligned_dir, val_idx, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = CLIPAttributeClassifier(
        num_attributes=len(attr_names),
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)

    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.clip.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr_head},
        {"params": backbone_params, "lr": args.lr_backbone},
    ], weight_decay=1e-4)

    # warmup linear curto + cosine decay — mesmo padrão de train_image_cond.py.
    # Sem isso, o run anterior (BCE+pos_weight, LR fixo) teve seu melhor
    # epoch na época 2 e degradou depois — LR constante é a causa provável.
    warmup_epochs = max(1, round(0.1 * args.epochs))
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    criterion = AsymmetricLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_f1 = -1.0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for img, attrs in train_loader:
            img, attrs = img.to(device, non_blocking=True), attrs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img)
                loss = criterion(logits, attrs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, decay=args.ema_decay)

            running_loss += loss.item() * img.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        # -------- validação (pesos EMA — mais estável que o modelo cru) --------
        all_probs, all_targets = [], []
        with torch.no_grad():
            for img, attrs in val_loader:
                img = img.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = ema_model(img)
                all_probs.append(torch.sigmoid(logits).float().cpu())
                all_targets.append(attrs)

        probs = torch.cat(all_probs)
        targets = torch.cat(all_targets)

        thresholds = calibrate_thresholds(probs, targets)
        preds = (probs > thresholds.unsqueeze(0)).float()
        tp = (preds * targets).sum(0)
        fp = (preds * (1 - targets)).sum(0)
        fn = ((1 - preds) * targets).sum(0)
        f1_per_attr = 2 * tp / (2 * tp + fp + fn + 1e-8)
        mean_f1 = f1_per_attr.mean().item()

        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} val_mean_f1={mean_f1:.4f}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            torch.save({
                "model": ema_model.state_dict(),
                "thresholds": thresholds,
                "attr_names": attr_names,
                "epoch": epoch,
                "val_mean_f1": mean_f1,
            }, args.output)
            print(f"[checkpoint] salvo em {args.output} (val_mean_f1={mean_f1:.4f})")

    print(f"Treino concluído. Melhor val_mean_f1={best_f1:.4f}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./CelebA_data/celeba")
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--identity_txt", type=str, default=None)
    p.add_argument("--align_cache_dir", type=str, default="./cache_aligned_celeba_native")
    p.add_argument("--skip_align_cache", action="store_true",
                    help="Pula a etapa de alinhamento (assume que --align_cache_dir já está populado).")
    p.add_argument("--output", type=str, default="./checkpoints/attribute_classifier_celeba_align.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--unfreeze_last_n", type=int, default=4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
