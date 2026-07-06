# Mixed Guidance — CFG (imagem) + Classifier Guidance (atributos)

Combina dois mecanismos de guidance distintos para controlar a geração:

| Condição | Tipo | Mecanismo |
|---|---|---|
| Imagem de referência | Classifier-Free Guidance | Dropout no treino, interpolação na inferência |
| Atributos faciais | Classifier Guidance | Classificador externo, gradiente em tempo de amostragem |

---

## Arquitetura

### Diffusion model — condicionado na imagem

A UNet recebe **apenas tokens da imagem** via cross-attention. Durante o treino, a imagem é zerada com probabilidade `cfg_dropout_img` para que o modelo aprenda o modo incondicional.

```
ref_img [B, 3, 256, 256]
    → ImageConditionEncoder
        ├─ CLIP-ViT-B/32 (frozen) → clip_proj  → [B, num_tokens, 512]
        └─ ArcFace buffalo_l (frozen) → id_proj → [B, num_tokens, 512]
        → concat tokens → [B, 2*num_tokens, 512] → permute
    → [B, 512, 2*num_tokens]   ← contexto da UNet (default num_tokens=16 → 32)
```

Na inferência, CFG interpola entre os dois modos:
```
eps = eps_uncond + s_img · (eps_cond − eps_uncond)
```

### Classifier de atributos — `NoisyLatentAttrClassifier`

Rede convolucional separada que recebe latentes **com ruído** e prediz os 40 atributos do CelebA. Treinada em paralelo com o diffusion model usando `BCEWithLogitsLoss`.

```
z_t [B, 4, 32, 32]  +  t [B]
    → conv blocks com GroupNorm
    → AdaptiveAvgPool → [B, 512]
    → concat com timestep embedding [B, 256]
    → head linear → logits [B, 40]
```

Durante o treino, `z_t` é **detachado** antes de entrar no classificador — os gradientes do BCE não afetam a UNet.

### Loop de amostragem híbrido

A cada passo `t` do DDPM:

```
1. CFG (sem grad):
   eps_cond   = UNet(z_t, t, img_context)
   eps_uncond = UNet(z_t, t, zeros)
   eps        = eps_uncond + s_img · (eps_cond − eps_uncond)

2. Classifier Guidance (com autograd em z_t):
   logits = Classifier(z_t, t)
   log_p  = Σ_k [ y_k·log σ(l_k) + (1−y_k)·log σ(−l_k) ]
   grad   = ∇_zt log_p
   eps    = eps − s_attr · √(1 − ᾱ_t) · grad

3. Passo DDPM:
   z_{t-1} = (1/√α_t) · (z_t − (1−α_t)/√(1−ᾱ_t) · eps) + √β_t · ruído
```

---

## Treino

```bash
torchrun --nproc_per_node=4 train/train_mixed_guidance.py \
    --run_name LDM_MixedGuidance \
    --epochs 2000 \
    --batch_size 64 \
    --lr 3e-4 \
    --lr_cls 1e-4 \
    --cfg_dropout_img 0.1
```

Warm start a partir de um checkpoint de `train_image_cond.py` (carrega UNet + ImageEncoder; classificador começa do zero):

```bash
torchrun --nproc_per_node=4 train/train_mixed_guidance.py \
    --run_name LDM_MixedGuidance \
    --warmstart_ckpt models/LDM_ImageCond/ckpt.pt
```

Retomar treino interrompido:

```bash
torchrun --nproc_per_node=4 train/train_mixed_guidance.py \
    --run_name LDM_MixedGuidance \
    --resume_ckpt models/LDM_MixedGuidance/ckpt.pt
```

---

## Inferência

### Atributos detectados automaticamente + edição

```bash
python generate_mixed_guidance.py \
    --ref_image foto.jpg \
    --ckpt models/LDM_MixedGuidance/ckpt.pt \
    --attr_predictor_ckpt models/attr_predictor.pt \
    --enable Smiling Eyeglasses \
    --disable Bald \
    --cfg_scale_img 3.0 \
    --cg_scale_attr 1.0
```

### Atributos definidos manualmente

```bash
python generate_mixed_guidance.py \
    --ref_image foto.jpg \
    --ckpt models/LDM_MixedGuidance/ckpt.pt \
    --manual_attrs Young Smiling Black_Hair \
    --cfg_scale_img 3.0 \
    --cg_scale_attr 1.0
```

### Só CFG (sem Classifier Guidance)

```bash
python generate_mixed_guidance.py \
    --ref_image foto.jpg \
    --ckpt models/LDM_MixedGuidance/ckpt.pt \
    --manual_attrs Smiling \
    --cfg_scale_img 3.0 \
    --cg_scale_attr 0.0
```

### CG aplicado só nos passos finais (mais estável)

```bash
python generate_mixed_guidance.py \
    --ref_image foto.jpg \
    --ckpt models/LDM_MixedGuidance/ckpt.pt \
    --enable Smiling \
    --cfg_scale_img 3.0 \
    --cg_scale_attr 1.5 \
    --cg_t_thresh 500
```

> **`--cg_t_thresh`**: Em timesteps altos o classificador opera sobre latentes com muito ruído, onde o gradiente pode ser instável. Limitar o CG aos últimos passos (ex: `t ≤ 500`) tende a produzir imagens mais coerentes quando `cg_scale_attr` é grande.

---

## Argumentos de inferência

| Argumento | Padrão | Descrição |
|---|---|---|
| `--ref_image` | — | Foto de referência (identidade visual) |
| `--ckpt` | `models/LDM_MixedGuidance/ckpt.pt` | Checkpoint do treino |
| `--vae_ckpt` | `vae/vae_epoch_62.pt` | Checkpoint da VAE |
| `--manual_attrs` | None | Atributos definidos manualmente |
| `--attr_predictor_ckpt` | None | Checkpoint do AttributePredictor |
| `--enable` | [] | Atributos a ativar |
| `--disable` | [] | Atributos a desativar |
| `--n` | 4 | Número de imagens geradas |
| `--cfg_scale_img` | 3.0 | Escala CFG da imagem |
| `--cg_scale_attr` | 1.0 | Escala Classifier Guidance dos atributos |
| `--cg_t_thresh` | None | Aplica CG só para `t ≤ N` |
| `--noise_steps` | 1000 | Passos de denoising |
| `--save_dir` | `results/mixed_guidance` | Diretório de saída |

---

## Atributos disponíveis

```
5_o_Clock_Shadow  Arched_Eyebrows   Attractive        Bags_Under_Eyes
Bald              Bangs             Big_Lips           Big_Nose
Black_Hair        Blond_Hair        Blurry             Brown_Hair
Bushy_Eyebrows    Chubby            Double_Chin        Eyeglasses
Goatee            Gray_Hair         Heavy_Makeup       High_Cheekbones
Male              Mouth_Slightly_Open  Mustache        Narrow_Eyes
No_Beard          Oval_Face         Pale_Skin          Pointy_Nose
Receding_Hairline Rosy_Cheeks       Sideburns          Smiling
Straight_Hair     Wavy_Hair         Wearing_Earrings   Wearing_Hat
Wearing_Lipstick  Wearing_Necklace  Wearing_Necktie    Young
```
