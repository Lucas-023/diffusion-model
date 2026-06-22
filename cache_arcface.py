"""
cache_arcface.py
================
Pré-computa e salva embeddings do encoder de imagem para todas as imagens CelebA.

Salva em ./cache_encoder/{stem}.pt — dict com:
    "arcface" : tensor [512]           L2-normalizado (ArcFace / InsightFace)
    "clip"    : tensor [seq_len, 768]  tokens do patch CLIP (CLS removido, pré-projeção)

Pode ser interrompido e retomado: arquivos já existentes são pulados.

Uso:
    python cache_arcface.py
    python cache_arcface.py --batch_size 128 --num_workers 4
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel

from models.encoders import ArcFaceEncoder


IMAGE_DIR = "./CelebA_data/celeba/img_align_celeba/img_align_celeba"
CACHE_DIR = "./cache_encoder"

# Mesma transform usada no treino
_TRANSFORM = T.Compose([
    T.CenterCrop(178),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Normalização CLIP — espelha ImageConditionEncoder
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


# ============================================================
# DATASET SIMPLES
# ============================================================

class _CelebAImages(Dataset):

    def __init__(self, image_dir, filenames):
        self.image_dir = image_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        return _TRANSFORM(img), fname


# ============================================================
# CLIP → patch tokens  [B, seq_len, 768]
# ============================================================

@torch.no_grad()
def clip_encode(clip_model, imgs_01, device):
    """
    imgs_01 : [B, 3, H, W] em [-1, 1]
    Retorna : [B, seq_len, 768]  tokens de patch (CLS removido), em CPU
    """
    clip_mean = _CLIP_MEAN.to(device)
    clip_std  = _CLIP_STD.to(device)

    x = imgs_01.float() * 0.5 + 0.5          # [-1,1] -> [0,1]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - clip_mean) / clip_std

    out = clip_model(pixel_values=x)
    tokens = out.last_hidden_state[:, 1:, :]  # remove CLS → [B, seq_len, 768]
    return tokens.cpu()


# ============================================================
# MAIN
# ============================================================

def main(args):

    os.makedirs(CACHE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Pendentes
    # --------------------------------------------------------

    all_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    pending = [
        f for f in all_files
        if not os.path.exists(
            os.path.join(CACHE_DIR, os.path.splitext(f)[0] + ".pt")
        )
    ]

    total = len(all_files)
    done  = total - len(pending)
    print(f"Total: {total} | Já cacheados: {done} | Restantes: {len(pending)}")

    if not pending:
        print("Cache completo.")
        return

    # --------------------------------------------------------
    # Modelos
    # --------------------------------------------------------

    print("Carregando CLIP (openai/clip-vit-base-patch32)...")
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    print("Carregando ArcFace...")
    arcface = ArcFaceEncoder()   # roda internamente em CPU via ONNX

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataset = _CelebAImages(IMAGE_DIR, pending)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        shuffle=False,
    )

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    for imgs, fnames in tqdm(loader, desc="Cacheando encoder"):

        imgs_gpu = imgs.to(device)

        # CLIP — GPU
        clip_tokens = clip_encode(clip_model, imgs_gpu, device)  # [B, seq_len, 768]

        # ArcFace — CPU via ONNX (aceita imgs em [-1,1])
        arc_embs = arcface(imgs)                                  # [B, 512]

        for i, fname in enumerate(fnames):
            stem = os.path.splitext(fname)[0]
            torch.save(
                {
                    "arcface": arc_embs[i].cpu(),    # [512]
                    "clip":    clip_tokens[i],        # [seq_len, 768]
                },
                os.path.join(CACHE_DIR, stem + ".pt"),
            )

    print(f"\nCache salvo em {CACHE_DIR}/")
    print("Formato por arquivo: dict com chaves 'arcface' [512] e 'clip' [seq_len, 768]")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pré-computa embeddings ArcFace + CLIP para CelebA."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size para CLIP (ArcFace processa o mesmo batch).",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Número de workers do DataLoader.",
    )

    args = parser.parse_args()
    main(args)
