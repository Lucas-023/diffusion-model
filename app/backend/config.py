"""
app/backend/config.py
=====================
Configuração do servidor via variáveis de ambiente (prefixo DFM_).

Nada de caminho hard-coded: os pesos moram onde a VM de deploy quiser.
Caminhos relativos são resolvidos contra a raiz do repositório.

| Variável               | Default                  | Descrição                                    |
| ---------------------- | ------------------------ | -------------------------------------------- |
| DFM_CKPT               | (obrigatória p/ editar)  | checkpoint de train_cfg_composable*.py       |
| DFM_VAE_CKPT           | vae/vae_epoch_62.pt      | checkpoint do VAE                            |
| DFM_CLASSIFIER_CKPT    | (opcional)               | checkpoint do CLIPAttributeClassifier        |
| DFM_DEVICE             | auto (cuda se houver)    | cuda / cuda:N / cpu                          |
| DFM_RESULTS_DIR        | results/webapp           | uploads e imagens geradas                    |
| DFM_NOISE_STEPS        | 1000                     | passos do schedule de treino                 |
| DFM_PRELOAD            | 1                        | 1 = carrega modelos no boot; 0 = 1º request  |
| DFM_API_TOKEN          | (vazio = sem auth)       | se setado, exige header X-API-Token          |
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _resolve(raw):
    if not raw:
        return None
    p = Path(raw)
    return str(p if p.is_absolute() else ROOT / p)


class Settings:
    def __init__(self):
        self.ckpt            = _resolve(os.getenv("DFM_CKPT"))
        self.vae_ckpt        = _resolve(os.getenv("DFM_VAE_CKPT", "vae/vae_epoch_62.pt"))
        self.classifier_ckpt = _resolve(os.getenv("DFM_CLASSIFIER_CKPT"))
        self.device          = os.getenv("DFM_DEVICE", "")  # vazio = auto
        self.results_dir     = _resolve(os.getenv("DFM_RESULTS_DIR", "results/webapp"))
        self.noise_steps     = int(os.getenv("DFM_NOISE_STEPS", "1000"))
        self.preload         = os.getenv("DFM_PRELOAD", "1") == "1"
        self.api_token       = os.getenv("DFM_API_TOKEN") or None

        self.frontend_dir = str(ROOT / "app" / "frontend")

    def missing(self):
        """Lista de problemas de configuração que impedem o load dos modelos."""
        problems = []
        if not self.ckpt:
            problems.append("DFM_CKPT não definida (checkpoint do modelo de difusão)")
        elif not Path(self.ckpt).is_file():
            problems.append(f"DFM_CKPT não encontrado: {self.ckpt}")
        if not self.vae_ckpt or not Path(self.vae_ckpt).is_file():
            problems.append(f"DFM_VAE_CKPT não encontrado: {self.vae_ckpt}")
        if self.classifier_ckpt and not Path(self.classifier_ckpt).is_file():
            problems.append(f"DFM_CLASSIFIER_CKPT não encontrado: {self.classifier_ckpt}")
        return problems
