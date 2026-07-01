"""
utils/fid.py
=============
Fréchet Inception Distance (Heusel et al. 2017), usando a InceptionV3
pré-treinada do torchvision (pesos ImageNet padrão — não são os pesos
customizados do pytorch-fid, mas dispensam instalar esse pacote extra).

Custoso (forward de uma InceptionV3 em centenas de imagens + geração via
diffusão completa) — pensado para ser chamado a cada N épocas
(--fid_every nos scripts de treino), nunca por batch/step.

As estatísticas do conjunto REAL (mu_real, sigma_real) devem ser calculadas
uma única vez por treino (o conjunto de validação não muda); só as
estatísticas do conjunto GERADO precisam ser recalculadas a cada avaliação,
já que o modelo (EMA) muda.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionFeatureExtractor(nn.Module):
    """Extrai o vetor 2048-d do pool final da InceptionV3 (padrão do FID)."""

    def __init__(self, device):
        super().__init__()

        net = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
        )
        net.fc = nn.Identity()
        net.eval()

        for p in net.parameters():
            p.requires_grad = False

        self.net    = net.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, images):
        """
        images: [B, 3, H, W] em [-1, 1] (saída típica de vae.decode).
        Retorna features [B, 2048].
        """
        images = (images.clamp(-1, 1) + 1) / 2.0  # -> [0, 1]
        images = F.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return self.net(images)


@torch.no_grad()
def get_activations(images, extractor, batch_size=50):
    """
    images: [N, 3, H, W] em [-1, 1], pode estar na CPU — é movida para o
    device do extractor em chunks, para não estourar VRAM com N grande.
    """
    device = extractor.device
    acts = []

    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device, non_blocking=True)
        feat  = extractor(batch)
        acts.append(feat.cpu().numpy())

    return np.concatenate(acts, axis=0)


@torch.no_grad()
def compute_real_activations_streamed(load_chunk_fn, n_total, extractor, chunk_size=128):
    """
    Extrai features reais em chunks SEM nunca materializar todas as imagens
    na memória — necessário quando n_total é o dataset de validação inteiro
    (dezenas de milhares de imagens 256x256 não cabem de uma vez na RAM).

    load_chunk_fn(start, end) -> tensor [end-start, 3, H, W] em [-1, 1] (CPU).
    """
    acts = []

    for start in range(0, n_total, chunk_size):
        end    = min(start + chunk_size, n_total)
        images = load_chunk_fn(start, end).to(extractor.device, non_blocking=True)
        feat   = extractor(images)
        acts.append(feat.cpu().numpy())

    return np.concatenate(acts, axis=0)


@torch.no_grad()
def compute_fake_activations_batched(
    generate_fn,
    id_tokens_full,
    attr_tokens_full,
    extractor,
    chunk_size=16,
):
    """
    Gera imagens em chunks (para controlar memória durante a amostragem por
    difusão) e extrai as features da InceptionV3 de cada chunk.

    generate_fn(id_tok_chunk, attr_tok_chunk) -> imagens [-1, 1] no device.
    """
    acts = []
    n_total = id_tokens_full.shape[0]

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)

        images = generate_fn(id_tokens_full[start:end], attr_tokens_full[start:end])
        feat   = extractor(images)
        acts.append(feat.cpu().numpy())

    return np.concatenate(acts, axis=0)


@torch.no_grad()
def compute_fake_activations_batched_n(
    generate_fn,
    tokens_full,
    extractor,
    chunk_size=16,
):
    """
    Igual a compute_fake_activations_batched, mas para um número arbitrário
    de tensores condicionantes (ex.: modo split com ArcFace + CLIP + attr,
    em vez de só id + attr) — todos em tokens_full são fatiados nos mesmos
    índices e passados posicionalmente para generate_fn.

    tokens_full: lista/tupla de tensores [N, C, T_i] (mesmo N em todos).
    generate_fn(*chunks) -> imagens [-1, 1] no device.
    """
    acts = []
    n_total = tokens_full[0].shape[0]

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)

        chunks = [t[start:end] for t in tokens_full]
        images = generate_fn(*chunks)
        feat   = extractor(images)
        acts.append(feat.cpu().numpy())

    return np.concatenate(acts, axis=0)


def calculate_activation_statistics(activations):
    mu    = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Implementação padrão (Heusel et al. 2017 / pytorch-fid)."""

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset  = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covmean)
    )
