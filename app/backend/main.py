"""
app/backend/main.py
===================
API FastAPI do editor de atributos faciais + servidor do frontend estático.

Rodar (da raiz do repositório):

    export DFM_CKPT=models/LDM_CFGComp_clipArc_paired/ckpt_best.pt
    export DFM_CLASSIFIER_CKPT=models/attr_classifier/ckpt_best.pt
    uvicorn app.backend.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /api/health          estado dos modelos / problemas de config
    GET  /api/attributes      os 40 nomes de atributos do CelebA
    POST /api/classify        foto -> atributos detectados (+ probs)
    POST /api/edit            foto + parâmetros -> job_id (assíncrono)
    GET  /api/jobs/{id}       status / progresso / resultado do job
    GET  /results/...         imagens geradas
    GET  /                    frontend (app/frontend)
"""

import json
import shutil
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.backend.config import Settings
from app.backend.jobs import Job, JobQueue
from app.backend.service import CELEBA_ATTRS, ATTR_TO_IDX, ModelService

settings = Settings()
service = ModelService(settings)
jobs = JobQueue(worker_fn=service.run_edit)

UPLOAD_DIR = Path(settings.results_dir) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app):
    if settings.preload:
        threading.Thread(target=service.load, daemon=True).start()
    yield


app = FastAPI(title="Editor de atributos faciais — LDM composable CFG",
              lifespan=lifespan)


# ------------------------------------------------------------
# Autenticação opcional por token (DFM_API_TOKEN)
# ------------------------------------------------------------

def check_token(x_api_token: str | None = Header(default=None)):
    if settings.api_token and x_api_token != settings.api_token:
        raise HTTPException(status_code=401, detail="Token inválido ou ausente.")


auth = Depends(check_token)


# ------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------

@app.get("/api/health")
def health():
    info = service.health()
    info["auth_required"] = bool(settings.api_token)
    info["jobs_pending"] = jobs.pending()
    return info


@app.get("/api/attributes", dependencies=[auth])
def attributes():
    return {"attributes": CELEBA_ATTRS}


@app.post("/api/classify", dependencies=[auth])
def classify(photo: UploadFile = File(...)):
    try:
        img = Image.open(photo.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida.")
    try:
        return service.classify(img)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _parse_attr_list(raw, field):
    """JSON '["Smiling", ...]' -> lista validada de nomes de atributos."""
    if raw is None or raw == "":
        return None
    try:
        names = json.loads(raw)
        assert isinstance(names, list)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field}: esperado JSON de lista.")
    for n in names:
        if n not in ATTR_TO_IDX:
            raise HTTPException(status_code=400, detail=f"{field}: atributo desconhecido '{n}'.")
    return names


@app.post("/api/edit", dependencies=[auth])
def edit(
    photo: UploadFile = File(...),
    method: str = Form(...),                # "sdedit" | "inversion"
    target_attrs: str = Form(...),          # JSON: atributos positivos desejados
    orig_attrs: str = Form(default=""),     # JSON: positivos da foto ("" = classificar)
    s_id: float = Form(default=None),
    s_attr: float = Form(default=None),
    s_clip: float = Form(default=None),
    strengths: str = Form(default="0.3,0.5,0.7"),   # só sdedit
    ddim_steps: int = Form(default=50),
    seed: int = Form(default=0),
):
    if method not in ("sdedit", "inversion"):
        raise HTTPException(status_code=400, detail="method deve ser 'sdedit' ou 'inversion'.")

    target = _parse_attr_list(target_attrs, "target_attrs")
    if target is None:
        raise HTTPException(status_code=400, detail="target_attrs é obrigatório.")
    orig = _parse_attr_list(orig_attrs, "orig_attrs")

    # defaults por método — mesmos dos scripts CLI (a inversão usa guidance
    # mais baixo: escala alta diverge da trajetória invertida)
    if s_id is None:
        s_id = 1.0 if method == "inversion" else 3.0
    if s_attr is None:
        s_attr = 3.0 if method == "inversion" else 5.0
    if s_clip is None:
        s_clip = 1.0 if method == "inversion" else 3.0

    try:
        strengths_list = [float(s) for s in strengths.replace(" ", "").split(",") if s]
    except ValueError:
        raise HTTPException(status_code=400, detail="strengths: use números separados por vírgula.")
    if method == "sdedit" and not strengths_list:
        raise HTTPException(status_code=400, detail="sdedit requer ao menos um strength.")

    if not (1 <= ddim_steps <= 500):
        raise HTTPException(status_code=400, detail="ddim_steps fora do intervalo [1, 500].")

    suffix = Path(photo.filename or "foto.png").suffix.lower() or ".png"
    photo_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    with photo_path.open("wb") as f:
        shutil.copyfileobj(photo.file, f)
    try:
        Image.open(photo_path).verify()
    except Exception:
        photo_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida.")

    job = Job(params={
        "photo_path": str(photo_path),
        "method": method,
        "orig_attrs": orig,
        "target_attrs": target,
        "s_id": s_id,
        "s_attr": s_attr,
        "s_clip": s_clip,
        "strengths": strengths_list,
        "ddim_steps": ddim_steps,
        "seed": seed,
    })
    jobs.submit(job)
    return {"job_id": job.id, "queued_behind": max(jobs.pending() - 1, 0)}


@app.get("/api/jobs/{job_id}", dependencies=[auth])
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return job.to_dict()


# ------------------------------------------------------------
# ESTÁTICOS — resultados e frontend (o mount "/" vai por último)
# ------------------------------------------------------------

Path(settings.results_dir).mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=settings.results_dir), name="results")
app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
