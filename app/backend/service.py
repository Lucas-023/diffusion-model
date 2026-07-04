"""
app/backend/service.py
======================
Camada de modelo do servidor: carrega UMA vez o EditPipeline
(utils/edit_common.py), o FaceAligner e o CLIPAttributeClassifier, e expõe
duas operações:

    classify(pil)   — atributos CelebA da foto (com probabilidades)
    run_edit(job)   — SDEdit ou inversão DDIM, mesma lógica dos scripts
                      edit_sdedit.py / edit_ddim_inversion.py

Diferenças em relação aos scripts CLI (que recarregam tudo a cada chamada):
o classificador e o aligner ficam residentes, e o progresso é reportado no
Job por meio de um wrapper de iterável passado a ddim_sample/ddim_invert
(que iteram o que receberem via tqdm — nenhuma mudança em edit_common).

Um lock global de GPU serializa classify e edições: o worker de jobs é
único, mas /api/classify chega por request e não pode disputar VRAM.
"""

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from utils.edit_common import (
    CELEBA_ATTRS,
    ATTR_TO_IDX,
    EditPipeline,
    prepare_photo,
    ddim_times,
    ddim_sample,
    ddim_invert,
)


# ------------------------------------------------------------
# Progresso: ddim_sample/ddim_invert iteram `tqdm(times)`, e tqdm aceita
# qualquer iterável com __len__ — este wrapper atualiza o Job a cada passo.
# ------------------------------------------------------------

class _ProgressTracker:
    def __init__(self, job, total_steps):
        self.job = job
        self.total = max(total_steps, 1)
        self.done = 0

    def wrap(self, times, stage):
        self.job.stage = stage
        return _ProgressList(times, self)


class _ProgressList:
    def __init__(self, times, tracker):
        self.times = list(times)
        self.tracker = tracker

    def __len__(self):
        return len(self.times)

    def __iter__(self):
        for t in self.times:
            yield t
            self.tracker.done += 1
            self.tracker.job.progress = self.tracker.done / self.tracker.total


# ------------------------------------------------------------

def attr_vector(names):
    """Vetor [40] a partir de uma lista de nomes de atributos positivos."""
    vec = torch.zeros(40)
    for n in names:
        if n not in ATTR_TO_IDX:
            raise ValueError(f"Atributo desconhecido: {n}")
        vec[ATTR_TO_IDX[n]] = 1.0
    return vec


class ModelService:

    def __init__(self, settings):
        self.settings = settings
        self.gpu_lock = threading.Lock()
        self._load_lock = threading.Lock()

        self.state = "idle"          # idle | loading | ready | error
        self.load_error = None
        self.device = None
        self.pipe = None
        self.aligner = None
        self.classifier = None

    # --------------------------------------------------------
    # CARREGAMENTO
    # --------------------------------------------------------

    def load(self):
        with self._load_lock:
            if self.state in ("loading", "ready"):
                return
            self.state = "loading"
            self.load_error = None
        try:
            self._load()
            self.state = "ready"
            print("[service] modelos prontos.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.load_error = str(e)
            self.state = "error"

    def _load(self):
        problems = self.settings.missing()
        if problems:
            raise RuntimeError("; ".join(problems))

        if self.settings.device:
            device = self.settings.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[service] AVISO: {device} indisponível — usando CPU (lento).")
            device = "cpu"
        self.device = device
        print(f"[service] device: {device}")

        self.pipe = EditPipeline(
            ckpt_path=self.settings.ckpt,
            vae_ckpt=self.settings.vae_ckpt,
            device=device,
            noise_steps=self.settings.noise_steps,
        )

        from data.face_align import FaceAligner
        self.aligner = FaceAligner()

        if self.settings.classifier_ckpt:
            from models.attribute_classifier import CLIPAttributeClassifier
            print("[service] carregando CLIPAttributeClassifier...")
            self.classifier = CLIPAttributeClassifier(
                pretrained_path=self.settings.classifier_ckpt
            ).to(device).eval()
            for p in self.classifier.parameters():
                p.requires_grad = False

    def ensure_ready(self):
        if self.state == "idle":
            self.load()
        if self.state == "loading":
            raise RuntimeError("Modelos ainda carregando — tente novamente em instantes.")
        if self.state != "ready":
            raise RuntimeError(f"Modelos não carregados: {self.load_error}")

    def health(self):
        return {
            "state": self.state,
            "device": self.device,
            "ckpt": self.settings.ckpt,
            "vae_ckpt": self.settings.vae_ckpt,
            "classifier_available": bool(self.settings.classifier_ckpt),
            "encoder": getattr(self.pipe, "encoder_type", None),
            "config_problems": self.settings.missing(),
            "error": self.load_error,
        }

    # --------------------------------------------------------
    # CLASSIFICAÇÃO (mesmo fluxo de resolve_original_attrs, mas com
    # classificador residente e retornando probabilidades)
    # --------------------------------------------------------

    def classify(self, pil_img):
        self.ensure_ready()
        if self.classifier is None:
            raise RuntimeError(
                "Classificador não configurado (DFM_CLASSIFIER_CKPT) — "
                "marque os atributos da foto manualmente."
            )
        with self.gpu_lock:
            return self._classify_locked(pil_img)

    @torch.no_grad()
    def _classify_locked(self, pil_img):
        aligned_pil, found = self.aligner.align_pil(pil_img)
        x = torch.from_numpy(np.array(aligned_pil)).float()
        x = (x.permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0).to(self.device)

        logits = self.classifier(x)
        probs = torch.sigmoid(logits).squeeze(0).float().cpu()
        thr = self.classifier.thresholds.float().cpu()

        positive = [CELEBA_ATTRS[i] for i in range(40) if probs[i] > thr[i]]
        return {
            "face_found": bool(found),
            "attributes": positive,
            "probs": {CELEBA_ATTRS[i]: round(float(probs[i]), 4) for i in range(40)},
        }

    # --------------------------------------------------------
    # EDIÇÃO
    # --------------------------------------------------------

    def run_edit(self, job):
        self.ensure_ready()
        with self.gpu_lock:
            return self._run_edit_locked(job)

    @torch.no_grad()
    def _run_edit_locked(self, job):
        p = job.params
        device = self.device
        torch.manual_seed(p["seed"])

        job.stage = "preparando foto"
        img = prepare_photo(p["photo_path"], device, aligner=self.aligner)

        # atributos originais: enviados pelo front, ou classificados aqui
        if p.get("orig_attrs") is not None:
            orig_names = p["orig_attrs"]
        else:
            job.stage = "classificando atributos"
            if self.classifier is None:
                raise RuntimeError(
                    "orig_attrs não enviados e classificador não configurado."
                )
            orig_names = self._classify_locked(
                Image.open(p["photo_path"]).convert("RGB")
            )["attributes"]

        target_names = p["target_attrs"]
        attrs_orig = attr_vector(orig_names)
        attrs_edit = attr_vector(target_names)

        enable  = sorted(set(target_names) - set(orig_names))
        disable = sorted(set(orig_names) - set(target_names))

        out_dir = Path(self.settings.results_dir) / job.id
        out_dir.mkdir(parents=True, exist_ok=True)

        images = [("original", img.squeeze(0))]

        if p["method"] == "sdedit":
            images += self._sdedit(job, p, img, attrs_edit)
        elif p["method"] == "inversion":
            images += self._inversion(job, p, img, attrs_orig, attrs_edit)
        else:
            raise ValueError(f"Método desconhecido: {p['method']}")

        out_images = []
        for label, tensor in images:
            fname = label.replace(" ", "_").replace("=", "") + ".png"
            self._save_img(tensor, out_dir / fname)
            out_images.append({"label": label, "url": f"/results/{job.id}/{fname}"})

        return {
            "images": out_images,
            "orig_attrs": orig_names,
            "target_attrs": target_names,
            "enable": enable,
            "disable": disable,
            "method": p["method"],
        }

    # ---- SDEdit: espelho de edit_sdedit.main ----

    def _sdedit(self, job, p, img, attrs_edit):
        pipe = self.pipe
        contexts, scales = pipe.build_context_chain(
            ref_img=img, attrs=attrs_edit,
            s_id=p["s_id"], s_attr=p["s_attr"], s_clip=p["s_clip"],
        )
        z0 = pipe.encode(img)
        times_full_desc = ddim_times(pipe.diffusion, p["ddim_steps"])[::-1]

        per_strength = []
        for strength in p["strengths"]:
            cutoff = int(strength * (pipe.diffusion.noise_steps - 1))
            times = [t for t in times_full_desc if t <= cutoff]
            if times:
                per_strength.append((strength, times))

        if not per_strength:
            raise ValueError("Nenhum strength válido (valores baixos demais).")

        tracker = _ProgressTracker(job, sum(len(t) for _, t in per_strength))

        outputs = []
        for strength, times in per_strength:
            t0 = times[0]
            alpha_bar_t0 = pipe.diffusion.alpha_hat[t0]
            noise = torch.randn_like(z0)
            z_t0 = (
                torch.sqrt(alpha_bar_t0) * z0
                + torch.sqrt(1.0 - alpha_bar_t0) * noise
            )
            z_out = ddim_sample(
                pipe, z_t0,
                tracker.wrap(times, f"SDEdit strength={strength}"),
                contexts, scales, desc=f"SDEdit s={strength}",
            )
            outputs.append((f"strength={strength}", pipe.decode(z_out).squeeze(0)))
        return outputs

    # ---- Inversão DDIM: espelho de edit_ddim_inversion.main ----

    def _inversion(self, job, p, img, attrs_orig, attrs_edit):
        pipe = self.pipe
        contexts_orig, scales = pipe.build_context_chain(
            ref_img=img, attrs=attrs_orig,
            s_id=p["s_id"], s_attr=p["s_attr"], s_clip=p["s_clip"],
        )
        contexts_edit, _ = pipe.build_context_chain(
            ref_img=img, attrs=attrs_edit,
            s_id=p["s_id"], s_attr=p["s_attr"], s_clip=p["s_clip"],
        )

        z0 = pipe.encode(img)
        times_asc = ddim_times(pipe.diffusion, p["ddim_steps"])
        times_desc = times_asc[::-1]

        tracker = _ProgressTracker(job, 3 * len(times_asc))

        z_T = ddim_invert(
            pipe, z0, tracker.wrap(times_asc, "inversão DDIM"),
            context=contexts_orig[-1],
        )
        z_recon = ddim_sample(
            pipe, z_T, tracker.wrap(times_desc, "reconstrução"),
            contexts_orig, scales, desc="Recon",
        )
        z_edit = ddim_sample(
            pipe, z_T, tracker.wrap(times_desc, "edição"),
            contexts_edit, scales, desc="Edição",
        )

        return [
            ("recon", pipe.decode(z_recon).squeeze(0)),
            ("editada", pipe.decode(z_edit).squeeze(0)),
        ]

    # --------------------------------------------------------

    @staticmethod
    def _save_img(tensor, path):
        from torchvision.utils import save_image
        save_image((tensor.cpu().clamp(-1, 1) + 1) / 2, str(path))
