# Editor de atributos faciais — imagem de deploy com GPU.
#
# Build (da RAIZ do repositório — o contexto precisa ver o código todo):
#   docker build -t diffusion-editor -f deploy/Dockerfile .
#
# Os PESOS NÃO são copiados para a imagem: monte-os em runtime (ver
# docker-compose.yml). Requer nvidia-container-toolkit no host.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# libgl1/libglib2.0-0: exigidas pelo opencv (dependência do insightface)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# camada de dependências separada do código (cache de build)
COPY requirements.txt ./requirements.txt
COPY app/requirements-app.txt ./requirements-app.txt
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install -r requirements-app.txt

COPY . /workspace

# pesos e resultados vivem fora da imagem
ENV DFM_RESULTS_DIR=/data/results

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
