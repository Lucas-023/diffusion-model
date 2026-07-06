# diffusion-model

Latent Diffusion Model (LDM) condicional treinado no CelebA, capaz de
editar atributos faciais de uma foto **preservando a identidade** da
pessoa. O treino final (`train_cfg_composable_paired.py`) usa apenas
**Classifier-Free Guidance (CFG) composicional** — sem Classifier
Guidance: identidade e os 40 atributos binários do CelebA são
condicionantes independentes, cada um injetado via cross-attention e
controlado por seu próprio peso de guidance na amostragem (`s_id`,
`s_attr`).

O condicionamento de identidade (encoder padrão `clip_arcface`) combina
dois sinais da foto de referência:
- **ArcFace** — embedding de identidade pura (frozen, ONNX).
- **CLIP** (`ImageConditionEncoder`, estilo IP-Adapter) — tokens visuais
  da imagem de referência (aparência/estilo), treináveis.

Há também `clip_arcface_split`, que trata ArcFace e CLIP como dois ramos
independentes de CFG (3 termos: `s_id`, `s_clip`, `s_attr`), permitindo
zerar `s_clip` para reduzir vazamento de aparência da referência em
edições estruturais (ex.: `Bald`) sem perder identidade.

Esse CLIP (ViT-B/32, dentro do modelo de difusão) é diferente do CLIP
usado no classificador de atributos (`CLIPAttributeClassifier`,
ViT-L/14): esse último só *lê* os atributos de uma foto para preencher a
edição, não participa da geração.

## Estrutura do repositório

```
inference/   scripts de geração e edição (linha de comando)
train/       scripts de treino (VAE, classificador, diffusion)
models/      arquitetura (UNet, encoders, classificador de atributos)
diffusion/   DDPM / DDIM
data/        datasets, alinhamento facial, cache de embeddings
utils/       amostragem, métricas (FID), logging (TensorBoard)
app/         editor web (FastAPI + frontend estático)
deploy/      Docker / systemd para subir o app em uma VM
notebooks/   análises e experimentos
test/        scripts de validação (FID, identidade, VAE)
```

## Instalação

```bash
# PyTorch com CUDA 12.1 (compatível com driver CUDA 12.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Resto das dependências (CLIP, ArcFace/insightface, FID, etc.)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Todos os comandos abaixo assumem que você está na raiz do repositório.

## Tutorial rápido

### 1. Editor web (jeito mais simples de testar)

Sobe um servidor local que recebe uma foto, detecta os atributos e
gera a versão editada — sem precisar mexer em scripts.

```bash
export DFM_CKPT=/caminho/para/ckpt_best.pt              # checkpoint do diffusion
export DFM_VAE_CKPT=/caminho/para/vae_epoch_62.pt        # checkpoint do VAE
export DFM_CLASSIFIER_CKPT=/caminho/para/attr_classifier.pt  # opcional

uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Abra `http://localhost:8000` no navegador. Para subir isso numa VM com
Docker ou systemd, veja os scripts em `deploy/`.

### 2. Gerar por linha de comando

```bash
python inference/generate_cfg_composable.py \
    --ref_image foto.jpg \
    --ckpt /caminho/para/ckpt_best.pt \
    --vae_ckpt vae/vae_epoch_62.pt \
    --enable Smiling Eyeglasses \
    --disable Bald \
    --s_id 3.0 --s_attr 5.0
```

- `--s_id` controla o quanto a identidade da referência é preservada.
- `--s_attr` controla a força da edição de atributos.
- `--enable` / `--disable` ligam/desligam atributos do CelebA (ex.:
  `Smiling`, `Eyeglasses`, `Bald`, `Young`, ...).

Edição de fotos reais (sem re-treinar) via `inference/edit_sdedit.py` e
`inference/edit_ddim_inversion.py`. `generate_mixed_guidance.py` é a
abordagem antiga com Classifier Guidance, mantida só como referência —
não é mais o pipeline usado.

### 3. Treinar do zero

```bash
# 1. VAE (necessário antes de qualquer treino de diffusion)
python train/train_vae.py --run_name VAE_v1

# 2. Classificador de atributos CelebA (opcional — só para prever os
#    atributos de uma foto de referência na hora da inferência com alinhamento baseado no celeba)
python train/train_attribute_classifier_celeba_align.py --run_name AttrClassifier_v1

# 3. Diffusion model (treino final, CFG composicional identidade + atributos), multi-GPU
torchrun --nproc_per_node=4 train/train_cfg_composable_paired.py \
    --run_name LDM_CFGComposable_v1 \
    --epochs 2000 \
    --batch_size 64
```

Progresso e imagens de amostra ficam disponíveis via TensorBoard
(`tensorboard --logdir runs/`).
