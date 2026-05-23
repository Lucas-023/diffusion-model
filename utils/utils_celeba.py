import os
import torch
import torchvision

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from matplotlib import pyplot as plt

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def plot_images(images):

    plt.figure(figsize=(32, 32))

    plt.imshow(
        torch.cat([
            torch.cat([i for i in images.cpu()], dim=-1),
        ], dim=-2).permute(1, 2, 0).cpu()
    )

    plt.show()


def save_images(images, path, **kwargs):

    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)

    grid = torchvision.utils.make_grid(
        images,
        **kwargs
    )

    ndarr = grid.permute(1, 2, 0).cpu().numpy()

    im = Image.fromarray(ndarr)

    im.save(path)


def setup_logging(run_name):

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    os.makedirs(
        os.path.join("models", run_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join("results", run_name),
        exist_ok=True
    )

# ==========================================
# DATASET:
# LATENTES + ATRIBUTOS
# ==========================================

class CelebALatentConditionalDataset(Dataset):

    def __init__(
        self,
        latent_dir,
        attr_path
    ):

        self.latent_dir = latent_dir

        # ======================================
        # SUPORTE PARA .TXT OU .CSV
        # ======================================

        if attr_path.endswith(".txt"):

            with open(attr_path, "r") as f:
                lines = f.readlines()

            self.num_images = int(lines[0])

            self.attr_names = lines[1].split()

            self.samples = []

            for line in lines[2:]:

                split = line.strip().split()

                filename = split[0]

                attrs = list(
                    map(int, split[1:])
                )

                # {-1,1} -> {0,1}
                attrs = [
                    (x + 1) // 2
                    for x in attrs
                ]

                attrs = torch.tensor(
                    attrs,
                    dtype=torch.float32
                )

                latent_filename = (
                    filename.split(".")[0]
                    + ".pt"
                )

                latent_path = os.path.join(
                    latent_dir,
                    latent_filename
                )

                if os.path.exists(latent_path):

                    self.samples.append(
                        (
                            latent_filename,
                            attrs
                        )
                    )

        else:

            # ==================================
            # CSV DO KAGGLE
            # ==================================

            import pandas as pd

            df = pd.read_csv(attr_path)

            self.attr_names = list(df.columns[1:])

            self.samples = []

            for _, row in df.iterrows():

                filename = row.iloc[0]

                attrs = row.iloc[1:].tolist()

                attrs = [
                    1 if x == 1 else 0
                    for x in attrs
                ]

                attrs = torch.tensor(
                    attrs,
                    dtype=torch.float32
                )

                latent_filename = (
                    filename.split(".")[0]
                    + ".pt"
                )

                latent_path = os.path.join(
                    latent_dir,
                    latent_filename
                )

                if os.path.exists(latent_path):

                    self.samples.append(
                        (
                            latent_filename,
                            attrs
                        )
                    )

        print("\n===================================")
        print(f"Dataset carregado:")
        print(f"Latentes encontrados: {len(self.samples)}")
        print(f"Número de atributos: {len(self.attr_names)}")
        print("===================================\n")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        latent_filename, attrs = self.samples[idx]

        latent_path = os.path.join(
            self.latent_dir,
            latent_filename
        )

        latent = torch.load(
            latent_path,
            map_location="cpu"
        )

        return latent, attrs

# ==========================================
# DATALOADER
# ==========================================

def get_data(
    args,
    is_distributed=True
):

    latent_dir = "./cache_latent"

    # ======================================
    # DETECTA AUTOMATICAMENTE
    # ======================================

    txt_path = (
        "./CelebA_data/celeba/"
        "list_attr_celeba.txt"
    )

    csv_path = (
        "./CelebA_data/celeba/"
        "list_attr_celeba.csv"
    )

    if os.path.exists(txt_path):

        attr_path = txt_path

    elif os.path.exists(csv_path):

        attr_path = csv_path

    else:

        raise FileNotFoundError(
            "Nenhum arquivo de atributos encontrado."
        )

    dataset = CelebALatentConditionalDataset(
        latent_dir=latent_dir,
        attr_path=attr_path
    )

    if is_distributed:

        sampler = DistributedSampler(
            dataset,
            shuffle=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        return dataloader, sampler

    else:

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        return dataloader, None