import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from matplotlib import pyplot as plt
from torchvision import transforms

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)

    grid = torchvision.utils.make_grid(images, **kwargs)

    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()

    im = Image.fromarray(ndarr)

    im.save(path)

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

# ==========================================
# DATASET CONDICIONAL CELEBA
# ==========================================

class CelebAConditionalDataset(Dataset):

    def __init__(
        self,
        image_dir,
        attr_path,
        image_size=256
    ):

        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

        with open(attr_path, "r") as f:
            lines = f.readlines()

        self.num_images = int(lines[0])

        self.attr_names = lines[1].split()

        self.samples = []

        for line in lines[2:]:

            split = line.strip().split()

            filename = split[0]

            attrs = list(map(int, split[1:]))

            # converte {-1,1} -> {0,1}
            attrs = [(x + 1) // 2 for x in attrs]

            attrs = torch.tensor(
                attrs,
                dtype=torch.float32
            )

            self.samples.append(
                (filename, attrs)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        filename, attrs = self.samples[idx]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, attrs

# ==========================================
# DATALOADER
# ==========================================

def get_data(args, is_distributed=True):

    image_dir = "./CelebA_data/celeba/img_align_celeba"

    attr_path = "./CelebA_data/celeba/list_attr_celeba.txt"

    dataset = CelebAConditionalDataset(
        image_dir=image_dir,
        attr_path=attr_path,
        image_size=args.image_size
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
            drop_last=True
        )

        return dataloader, sampler

    else:

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        return dataloader, None