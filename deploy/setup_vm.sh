#!/usr/bin/env bash
# Setup do editor SEM Docker (venv direto na VM).
# Rodar da raiz do repositório:  bash deploy/setup_vm.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r app/requirements-app.txt

echo
echo "Setup concluído. Para subir o servidor:"
echo
echo "  source .venv/bin/activate"
echo "  export DFM_CKPT=/caminho/para/ckpt_best.pt"
echo "  export DFM_VAE_CKPT=/caminho/para/vae_epoch_62.pt"
echo "  export DFM_CLASSIFIER_CKPT=/caminho/para/attr_classifier.pt   # opcional"
echo "  uvicorn app.backend.main:app --host 0.0.0.0 --port 8000"
