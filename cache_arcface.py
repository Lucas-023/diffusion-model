"""
Pré-computa embeddings ArcFace para todas as imagens CelebA.

Salva em ./cache_arcface/{stem}.pt — tensor [512] float32 L2-normalizado.
Pode ser interrompido e retomado: imagens já cacheadas são puladas.

Uso:
    python cache_arcface.py
"""

import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from models.encoders import ArcFaceEncoder

IMAGE_DIR = "./CelebA_data/celeba/img_align_celeba/img_align_celeba"
CACHE_DIR = "./cache_arcface"
BATCH_SIZE = 256
NUM_WORKERS = 8

TRANSFORM = T.Compose([
    T.CenterCrop(178),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class _CelebAImages(Dataset):

    def __init__(self, image_dir, filenames):
        self.image_dir = image_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        return TRANSFORM(img), fname


def main():

    os.makedirs(CACHE_DIR, exist_ok=True)

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
    done = total - len(pending)
    print(f"Total: {total} | Já cacheados: {done} | Restantes: {len(pending)}")

    if not pending:
        print("Cache completo.")
        return

    encoder = ArcFaceEncoder()

    dataset = _CelebAImages(IMAGE_DIR, pending)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        shuffle=False,
    )

    for imgs, fnames in tqdm(loader, desc="Cacheando ArcFace"):

        embs = encoder(imgs)   # [B, 512], L2-normalizado

        for emb, fname in zip(embs, fnames):
            stem = os.path.splitext(fname)[0]
            torch.save(emb.cpu(), os.path.join(CACHE_DIR, stem + ".pt"))

    print(f"Cache salvo em {CACHE_DIR}/")


if __name__ == "__main__":
    main()
