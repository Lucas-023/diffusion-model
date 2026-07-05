"""
utils/composable_cfg_sampling.py
=================================
Amostradores de Composable CFG (Liu et al., ECCV 2022) compartilhados
pelos scripts de geração que usam checkpoints de train_cfg_composable.py /
train_cfg_composable_paired.py:

    eps = eps(∅,∅)
        + s_id   · [eps(id,∅)    − eps(∅,∅)]
        + s_attr · [eps(id,attr) − eps(id,∅)]

e, para --encoder clip_arcface_split (ArcFace/CLIP como ramos
independentes):

    eps = eps(∅,∅,∅)
        + s_id   · [eps(id,∅,∅)      − eps(∅,∅,∅)]
        + s_clip · [eps(id,clip,∅)   − eps(id,∅,∅)]
        + s_attr · [eps(id,clip,attr) − eps(id,clip,∅)]

Cada combinação tem uma versão DDIM (determinística, ddim_steps << 1000,
usada nas avaliações periódicas do treino) e uma versão DDPM ancestral
(1000 passos, mesma família do treino em t contínuo, mais lenta).
"""

import torch
from tqdm import tqdm


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


@torch.no_grad()
def sample_composable_ddpm(
    unet, diffusion, id_tokens, attr_tokens,
    n, channels, device, s_id=3.0, s_attr=5.0,
):
    unet.eval()
    img_size = diffusion.img_size
    z_t = torch.randn(n, channels, img_size, img_size, device=device)

    zeros_id, zeros_attr = torch.zeros_like(id_tokens), torch.zeros_like(attr_tokens)
    ctx_uu = torch.cat([zeros_id,   zeros_attr], dim=2)
    ctx_iu = torch.cat([id_tokens,  zeros_attr], dim=2)
    ctx_ia = torch.cat([id_tokens,  attr_tokens], dim=2)

    for t in tqdm(reversed(range(1, diffusion.noise_steps)),
                  desc="Sampling (DDPM, id+attr)", total=diffusion.noise_steps - 1):

        t_vec = torch.full((n,), t, device=device, dtype=torch.long)

        alpha_hat_t = diffusion.alpha_hat[t_vec][:, None, None, None]
        alpha_t     = diffusion.alpha[t_vec][:, None, None, None]
        beta_t      = diffusion.beta[t_vec][:, None, None, None]

        eps_uu = unet(z_t, t_vec, context=ctx_uu)
        eps_iu = unet(z_t, t_vec, context=ctx_iu)
        eps_ia = unet(z_t, t_vec, context=ctx_ia)
        eps = eps_uu + s_id * (eps_iu - eps_uu) + s_attr * (eps_ia - eps_iu)

        noise = torch.randn_like(z_t) if t > 1 else torch.zeros_like(z_t)
        z_t = (
            (1.0 / torch.sqrt(alpha_t))
            * (z_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t) * eps)
        ) + torch.sqrt(beta_t) * noise

    return z_t


@torch.no_grad()
def sample_split_ddpm(
    unet, diffusion, id_tokens, clip_tokens, attr_tokens,
    n, channels, device, s_id=3.0, s_clip=3.0, s_attr=5.0,
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

    for t in tqdm(reversed(range(1, diffusion.noise_steps)),
                  desc="Sampling (DDPM, id+clip+attr)", total=diffusion.noise_steps - 1):

        t_vec = torch.full((n,), t, device=device, dtype=torch.long)

        alpha_hat_t = diffusion.alpha_hat[t_vec][:, None, None, None]
        alpha_t     = diffusion.alpha[t_vec][:, None, None, None]
        beta_t      = diffusion.beta[t_vec][:, None, None, None]

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

        noise = torch.randn_like(z_t) if t > 1 else torch.zeros_like(z_t)
        z_t = (
            (1.0 / torch.sqrt(alpha_t))
            * (z_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t) * eps)
        ) + torch.sqrt(beta_t) * noise

    return z_t
