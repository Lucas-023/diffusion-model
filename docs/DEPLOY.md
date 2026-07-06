# Deploy — Editor web de atributos faciais

Aplicação web sobre o pipeline de edição de fotos reais
(`edit_sdedit.py` / `edit_ddim_inversion.py` via `utils/edit_common.py`):

- **Backend**: FastAPI (`app/backend/`) — carrega os modelos UMA vez e
  processa edições em uma fila de jobs (um por vez na GPU, com progresso).
- **Frontend**: página estática (`app/frontend/`) servida pelo próprio
  backend — sem Node, sem build.
- **Deploy**: Docker (recomendado) ou venv + systemd (`deploy/`).

Pensado para a VM ficar em outra máquina (a do professor): nenhum caminho
de peso é fixo no código — tudo entra por variável de ambiente.

---

## O que a VM precisa ter

- Driver NVIDIA funcionando (`nvidia-smi`).
- **Rota Docker**: Docker + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **Rota venv**: Python 3.10+ e as libs de sistema `libgl1 libglib2.0-0`
  (para o opencv do insightface).
- Acesso à internet **no primeiro boot**: o servidor baixa o CLIP
  (HuggingFace) e o detector `buffalo_l` (insightface) em runtime.
  Depois fica em cache.

## Pesos necessários

| Peso | Variável | Obrigatório |
| --- | --- | --- |
| Checkpoint de `train_cfg_composable*.py` (usa EMA se existir) | `DFM_CKPT` | sim |
| VAE (`vae_epoch_62.pt`) | `DFM_VAE_CKPT` | sim (default `vae/vae_epoch_62.pt`) |
| `CLIPAttributeClassifier` fine-tunado | `DFM_CLASSIFIER_CKPT` | não — sem ele o usuário marca os atributos da foto à mão |

O tipo de encoder (`clip_arcface`, `arcface_only`, `clip_arcface_split`)
é lido de dentro do checkpoint; o frontend se adapta (o slider `s_clip`
só aparece para `clip_arcface_split`).

---

## Rota 1 — Docker (recomendada)

```bash
git clone <repo> && cd diffusion-model/deploy

# diretório com os pesos na VM:
#   ckpt_best.pt  vae_epoch_62.pt  attr_classifier.pt
WEIGHTS_DIR=/home/prof/pesos docker compose up -d --build

docker compose logs -f     # acompanhar o load dos modelos
```

Nomes de arquivo diferentes? Sobrescreva:

```bash
WEIGHTS_DIR=/home/prof/pesos CKPT_NAME=meu_ckpt.pt docker compose up -d
```

App em `http://<ip-da-vm>:8000`.

## Rota 2 — venv + systemd

```bash
git clone <repo> && cd diffusion-model
bash deploy/setup_vm.sh

source .venv/bin/activate
export DFM_CKPT=/home/prof/pesos/ckpt_best.pt
export DFM_VAE_CKPT=/home/prof/pesos/vae_epoch_62.pt
export DFM_CLASSIFIER_CKPT=/home/prof/pesos/attr_classifier.pt
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Para rodar como serviço permanente, ajuste os caminhos em
`deploy/diffusion-editor.service` e:

```bash
sudo cp deploy/diffusion-editor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now diffusion-editor
```

---

## Variáveis de ambiente

Todas documentadas em `app/backend/config.py`. As principais:

| Variável | Default | Descrição |
| --- | --- | --- |
| `DFM_CKPT` | — | checkpoint do modelo de difusão |
| `DFM_VAE_CKPT` | `vae/vae_epoch_62.pt` | checkpoint do VAE |
| `DFM_CLASSIFIER_CKPT` | — | classificador de atributos (opcional) |
| `DFM_DEVICE` | auto | `cuda`, `cuda:1`, `cpu` |
| `DFM_RESULTS_DIR` | `results/webapp` | uploads + imagens geradas |
| `DFM_PRELOAD` | `1` | `0` = só carrega modelos no 1º request |
| `DFM_API_TOKEN` | — | se setado, a API exige header `X-API-Token` |

## Segurança

Não há autenticação por padrão — adequado para rede interna/VPN. Se a
porta for exposta, defina `DFM_API_TOKEN` (o frontend pede o token na
primeira chamada e o guarda no navegador). Para acesso remoto pontual sem
abrir porta, um túnel SSH resolve:

```bash
ssh -L 8000:localhost:8000 usuario@vm-do-professor
# e abrir http://localhost:8000 na máquina local
```

## API (para uso programático / notebooks)

```bash
# saúde / estado do load
curl http://vm:8000/api/health

# classificar atributos de uma foto
curl -F photo=@foto.jpg http://vm:8000/api/classify

# editar (assíncrono — devolve job_id)
curl -F photo=@foto.jpg \
     -F method=inversion \
     -F 'orig_attrs=["Male","Young","Black_Hair"]' \
     -F 'target_attrs=["Male","Young","Black_Hair","Smiling"]' \
     -F s_id=1.0 -F s_attr=3.0 \
     http://vm:8000/api/edit

# acompanhar
curl http://vm:8000/api/jobs/<job_id>
```

O resultado do job traz URLs das imagens em `/results/<job_id>/...`
(original alinhada, recon e editada na inversão; uma coluna por strength
no SDEdit).

## Teste rápido sem GPU

O servidor sobe e serve o frontend mesmo sem pesos/GPU (`/api/health`
reporta o problema de configuração). Com pesos e `DFM_DEVICE=cpu`
funciona de ponta a ponta, apenas lento — útil para validar a instalação.
