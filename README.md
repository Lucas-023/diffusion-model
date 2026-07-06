# diffusion-model

Latent Diffusion Model condicional treinado no CelebA, com suporte a condicionamento por **atributos faciais** (40 atributos binários) e **identidade visual** (ArcFace).

## Instalação

```bash
# PyTorch com CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependências gerais
pip install tqdm matplotlib pillow tensorboard

# ArcFace (InsightFace)
pip install insightface onnxruntime-gpu
```

## Uso rápido

```bash
# Gerar a mesma pessoa com atributos editados
python inference/generate_from_ref.py \
    --ref_image foto.jpg \
    --ckpt models/LDM_Identity_v1/ckpt.pt \
    --enable Smiling Eyeglasses \
    --disable Bald

# Treino multi-GPU (atributos + identidade)
torchrun --nproc_per_node=4 train/traind_cond_identity.py \
    --run_name LDM_Identity_v1 \
    --epochs 2000 \
    --batch_size 64
```

## Documentação completa

Ver [docs/CFG_COMPOSABLE.md](docs/CFG_COMPOSABLE.md) e [docs/MIXED_GUIDANCE.md](docs/MIXED_GUIDANCE.md) para arquitetura detalhada, fluxos de dados e guia de uso completo.

## Resultados

DDPM 32×32 — FID Score: **3.35**
