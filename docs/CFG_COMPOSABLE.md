# Composable CFG — Identidade + Atributos como condicionantes independentes

Branch: `cfgonnly`
Arquivo novo: `train/train_cfg_composable.py`

---

## Motivação

A abordagem anterior (`train_mixed_guidance_identity_only.py`) usava:

- **CFG** para a imagem de referência (identidade)
- **Classifier Guidance (CG)** para os 40 atributos do CelebA, via
  `NoisyLatentAttrClassifier` treinado em paralelo

O objetivo é "pegar uma foto e gerar a mesma pessoa com um atributo
mudado" (ex.: pessoa séria → sorrindo). Na prática o CG estava fraco
por três motivos estruturais:

1. **Sigmoid satura** no classificador binário multi-label — quando o
   classificador está confiante "not_smiling=1", o gradiente colapsa.
   Subir `cg_scale_attr` para compensar gera artefatos antes de gerar
   o sorriso.
2. O `log p(y|z_t)` somava os **40 atributos** — os 39 que não estão
   sendo editados dominam o gradiente. O sinal de "Smiling" fica
   diluído.
3. O classificador no latente é estruturalmente limitado: para `t`
   alto o latente é quase ruído puro e o classificador aprende o
   prior; para `t` baixo funciona, mas restam poucos passos pra
   editar a imagem. Janela útil estreita.

Historicamente foi por esses motivos que Ho & Salimans (2022)
propuseram CFG: o sinal é uma diferença de eps's, não satura, e
escala estavelmente.

---

## Solução: Composable CFG (Liu et al., ECCV 2022)

Identidade e atributos viram **dois condicionantes independentes**.
Na inferência, a previsão de ruído usa três forward passes:

```
eps = eps(∅, ∅)
    + s_id   · [eps(id, ∅)    − eps(∅, ∅)]
    + s_attr · [eps(id, attr) − eps(id, ∅)]
```

- `s_id` controla o quanto a identidade puxa a geração.
- `s_attr` controla o quanto os atributos puxam **dado** que a
  identidade já está aplicada.

Isso resolve o medo de "perder controle independente entre id e
atributos" — cada escala continua sendo ajustável separadamente,
exatamente como `cfg_scale_img` e `cg_scale_attr` eram antes.

A ordem do segundo termo importa: usa `eps(id, ∅)` como baseline (não
`eps(∅, ∅)`), o que significa "dada a identidade, quanto o atributo
desloca eps". Sem isso o termo de atributo ficaria competindo com o
de identidade em vez de ser ortogonal a ele.

---

## Mudanças concretas em relação ao identity_only

### Removido

- `NoisyLatentAttrClassifier` (não é mais usado).
- `optimizer_cls`, `scheduler_cls`, `scaler_cls` — agora há um único
  optimizer.
- Loss BCE de classificação.
- `_sample_hybrid` (CFG + CG) → substituído por
  `_sample_composable_cfg` (3 forwards de UNet por step).
- Amostragem enviesada de `t` (`u.sqrt() * noise_steps`). Sem CG, não
  faz sentido enviesar — volta a `sample_timesteps` uniforme.

### Adicionado

- **`cfg_masks(...)`** — sorteia dropout **por amostra** (não por
  batch), com as probabilidades:
  - 10% zera só identidade (mantém attr)
  - 10% zera só atributos (mantém id)
  - 10% zera os dois
  - 70% mantém ambos

  Crucial que seja por amostra: cada batch vê os quatro combos e o
  modelo aprende as quatro distribuições marginais que a fórmula
  composable exige na inferência.

- **`build_context(...)`** — monta o contexto da UNet como
  `concat([id_tokens, attr_tokens], dim=tokens)` aplicando as
  máscaras de dropout. Não requer mexer na UNet: o cross-attention
  atende a todos os tokens.

- **`_sample_composable_cfg(...)`** — implementa a fórmula acima.

- **Sampling de validação dobrado**: a cada checkpoint, salva
  `<epoch>_attr_orig.jpg` (com os atributos verdadeiros das
  referências) **e** `<epoch>_attr_smile.jpg` (mesma identidade, mas
  com `Smiling=1` forçado). Assim dá pra ver direto se o modelo está
  conseguindo editar.

- **Flag `--encoder {clip_arcface, arcface_only}`**, default
  `clip_arcface`. Com CFG composable, `s_attr` consegue sobrepor a
  expressão capturada pelo CLIP, então o motivo histórico de jogar
  CLIP fora (em `train_mixed_guidance_identity_only.py`) deixa de ser
  forte. Vale comparar os dois — ver seção "Como avaliar".

### Reutilizado

- `AttributeEmbedder` (já existia em `models/modules.py`) — 40 tokens,
  um por atributo, com binário modulando via `nn.Embedding(2, ...)`.
- `UNet_cond` — sem modificação. O contexto concat passa pelo
  cross-attention existente.

---

## Como rodar

### 1 GPU

```bash
python train/train_cfg_composable.py \
    --run_name LDM_CFGComp_clipArc \
    --batch_size 64 \
    --epochs 200
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 train/train_cfg_composable.py \
    --run_name LDM_CFGComp_clipArc \
    --batch_size 64
```

### Warm-start de um checkpoint anterior

Carrega só a UNet; o encoder de identidade e o `AttributeEmbedder`
começam do zero.

```bash
python train/train_cfg_composable.py \
    --warmstart_ckpt models/LDM_MixedGuidance/ckpt.pt \
    --run_name LDM_CFGComp_clipArc
```

### Resume

```bash
python train/train_cfg_composable.py \
    --resume_ckpt models/LDM_CFGComp_clipArc/ckpt.pt \
    --run_name LDM_CFGComp_clipArc
```

---

## Comparação CLIP+ArcFace vs ArcFace-only

Rodar os dois em paralelo (ou sequencial) e comparar:

```bash
python train/train_cfg_composable.py --run_name CFGcomp_clip    --encoder clip_arcface
python train/train_cfg_composable.py --run_name CFGcomp_arcface --encoder arcface_only
```

### Como avaliar

Na mesma época, abrir os pares `<epoch>_attr_orig.jpg` e
`<epoch>_attr_smile.jpg` dos dois runs e julgar:

| Critério                                                 | Quem deve ganhar           |
| -------------------------------------------------------- | -------------------------- |
| Fidelidade visual (cabelo, iluminação, pose, fundo)      | CLIP+ArcFace               |
| Edição convincente quando força `Smiling=1`              | ArcFace-only (a princípio) |
| Edição convincente após calibrar `s_attr`                | Empate esperado            |

Resultado prático esperado:

- Se CLIP+ArcFace mantiver edição razoável → fica com ele
  (vence em fidelidade).
- Se ArcFace-only continuar editando melhor mesmo subindo
  `s_attr` no CLIP+ArcFace → mantém ArcFace-only.

---

## Hiperparâmetros relevantes

| Flag                          | Default        | Comentário                              |
| ----------------------------- | -------------- | --------------------------------------- |
| `--encoder`                   | `clip_arcface` | ou `arcface_only`                       |
| `--img_tokens`                | `16`           | tokens por ramo do encoder              |
| `--cfg_dropout_id_only`       | `0.1`          |                                         |
| `--cfg_dropout_attr_only`     | `0.1`          |                                         |
| `--cfg_dropout_both`          | `0.1`          | necessário para o termo `eps(∅,∅)`      |
| `--s_id_val`                  | `3.0`          | escala usada nos samples de validação   |
| `--s_attr_val`                | `5.0`          | atributos binários precisam de mais     |
| `--lr`                        | `3e-4`         |                                         |
| `--batch_size`                | `64`           |                                         |
| `--epochs`                    | `2000`         |                                         |

---

## Sugestões de arquitetura para depois (não aplicadas)

Vale fazer só **depois** de ver o baseline CFG composable funcionar,
pra ter referência de comparação.

1. **Dual cross-attention (IP-Adapter style)** — em vez de concatenar
   `id_tokens` e `attr_tokens` no mesmo cross-attention, adicionar um
   segundo cross-attention separado em cada `SpatialTransformer`.
   Atualmente atributos competem com identidade pela atenção; com
   ramos separados cada condicionante tem sua banda dedicada.
   Custo: ~1 cross-attn extra por bloco da UNet.

2. **adaLN-Zero / FiLM para atributos** (estilo DiT, Peebles &
   Xie 2023) — atributos são 40 binários, formato ideal para
   modulação global (γ, β nos GroupNorms) em vez de tokens. Mais
   barato e mais estável. Vantagem: hoje a UNet só tem cross-attn em
   duas resoluções (`attention_resolutions=(16, 8)`); atributos via
   adaLN influenciariam **todos** os blocos.

Ambas exigem mexer em `models/unet_conditional.py` e/ou
`models/modules.py`.

---

## Próximo passo de código

`generate_cfg_composable.py` — espelho do `generate_mixed_guidance.py`
mas usando `_sample_composable_cfg`. Ainda não foi escrito; o
sampling de validação dentro do treino já cobre o teste qualitativo
durante o desenvolvimento.
