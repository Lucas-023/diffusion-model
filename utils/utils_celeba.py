import os
import random
import torch
import torchvision
import torchvision.transforms as T

from collections import defaultdict

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

        latent_filename, _ = self.samples[idx]

        latent_path = os.path.join(
            self.latent_dir,
            latent_filename
        )

        data = torch.load(
            latent_path,
            map_location="cpu"
        )

        latent = data["latent"]
        attrs = data["attrs"]

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

    # ======================================
    # DATASET GLOBAL
    # ======================================

    dataset = CelebALatentConditionalDataset(
        latent_dir=latent_dir,
        attr_path=attr_path
    )

    # ======================================
    # SPLITS
    # ======================================

    train_size = int(0.70 * len(dataset))

    val_size = int(0.15 * len(dataset))

    test_size = (
        len(dataset)
        - train_size
        - val_size
    )

    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(

            dataset,

            [
                train_size,
                val_size,
                test_size
            ],

            generator=torch.Generator().manual_seed(42)
        )

    print("\n===================================")
    print("DATASET SPLITS")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")
    print("===================================\n")

    # ======================================
    # DATALOADERS
    # ======================================

    if is_distributed:

        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True
        )

        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False
        )

        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False
        )

        train_loader = DataLoader(

            train_dataset,

            batch_size=args.batch_size,

            sampler=train_sampler,

            shuffle=False,

            num_workers=8,

            pin_memory=True,

            drop_last=True,

            persistent_workers=True,

            prefetch_factor=4
        )

        val_loader = DataLoader(

            val_dataset,

            batch_size=args.batch_size,

            sampler=val_sampler,

            shuffle=False,

            num_workers=8,

            pin_memory=True,

            drop_last=False,

            persistent_workers=True,

            prefetch_factor=4
        )

        test_loader = DataLoader(

            test_dataset,

            batch_size=args.batch_size,

            sampler=test_sampler,

            shuffle=False,

            num_workers=8,

            pin_memory=True,

            drop_last=False,

            persistent_workers=True,

            prefetch_factor=4
        )

        return (
            train_loader,
            val_loader,
            test_loader,
            train_sampler
        )

    else:

        train_loader = DataLoader(

            train_dataset,

            batch_size=args.batch_size,

            shuffle=True,

            num_workers=8,

            pin_memory=True,

            drop_last=True,

            persistent_workers=True,

            prefetch_factor=4
        )

        val_loader = DataLoader(

            val_dataset,

            batch_size=args.batch_size,

            shuffle=False,

            num_workers=8,

            pin_memory=True,

            drop_last=False,

            persistent_workers=True,

            prefetch_factor=4
        )

        test_loader = DataLoader(

            test_dataset,

            batch_size=args.batch_size,

            shuffle=False,

            num_workers=8,

            pin_memory=True,

            drop_last=False,

            persistent_workers=True,

            prefetch_factor=4
        )

        return (
            train_loader,
            val_loader,
            test_loader,
            None
        )


# ==========================================
# DATASET: LATENTES + ATRIBUTOS + IMAGEM
# (para condicionamento de identidade)
# ==========================================

_IDENTITY_TRANSFORM = T.Compose([
    T.CenterCrop(178),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

class CelebALatentIdentityDataset(Dataset):
    """
    Retorna (latent, attrs, identity) onde identity é:
      - tensor [512]        se arcface_dir for fornecido (cache pré-computado)
      - tensor [3, 256, 256] caso contrário (imagem original para encoding em tempo real)
    """

    def __init__(
        self,
        latent_dir,
        attr_path,
        image_dir,
        arcface_dir=None,
        transform=None
    ):

        self.latent_dir = latent_dir
        self.image_dir = image_dir
        self.arcface_dir = arcface_dir
        self.transform = transform or _IDENTITY_TRANSFORM

        # ======================================
        # SUPORTE PARA .TXT OU .CSV
        # ======================================

        if attr_path.endswith(".txt"):

            with open(attr_path, "r") as f:
                lines = f.readlines()

            self.attr_names = lines[1].split()

            self.samples = []

            for line in lines[2:]:

                split = line.strip().split()

                filename = split[0]

                attrs = list(map(int, split[1:]))

                attrs = [(x + 1) // 2 for x in attrs]

                attrs = torch.tensor(attrs, dtype=torch.float32)

                latent_filename = (
                    filename.split(".")[0] + ".pt"
                )

                latent_path = os.path.join(
                    latent_dir, latent_filename
                )

                if os.path.exists(latent_path):
                    self.samples.append((filename, latent_filename, attrs))

        else:

            import pandas as pd

            df = pd.read_csv(attr_path)

            self.attr_names = list(df.columns[1:])

            self.samples = []

            for _, row in df.iterrows():

                filename = row.iloc[0]

                attrs = row.iloc[1:].tolist()

                attrs = [1 if x == 1 else 0 for x in attrs]

                attrs = torch.tensor(attrs, dtype=torch.float32)

                latent_filename = (
                    filename.split(".")[0] + ".pt"
                )

                latent_path = os.path.join(
                    latent_dir, latent_filename
                )

                if os.path.exists(latent_path):
                    self.samples.append((filename, latent_filename, attrs))

        print("\n===================================")
        print(f"Dataset (identity) carregado:")
        print(f"Amostras encontradas: {len(self.samples)}")
        print(f"Número de atributos: {len(self.attr_names)}")
        print("===================================\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_filename, latent_filename, _ = self.samples[idx]

        latent_path = os.path.join(
            self.latent_dir, latent_filename
        )

        data = torch.load(latent_path, map_location="cpu")

        latent = data["latent"]
        attrs = data["attrs"]

        if self.arcface_dir is not None:
            stem = os.path.splitext(img_filename)[0]
            emb_path = os.path.join(self.arcface_dir, stem + ".pt")
            identity = torch.load(emb_path, map_location="cpu")   # [512]
        else:
            img_path = os.path.join(self.image_dir, img_filename)
            img = Image.open(img_path).convert("RGB")
            identity = self.transform(img)                         # [3, 256, 256]

        return latent, attrs, identity


# ==========================================
# DATALOADER COM IDENTIDADE
# ==========================================

def get_data_identity(args, is_distributed=True):

    latent_dir = "./cache_latent"

    image_dir = (
        "./CelebA_data/celeba/"
        "img_align_celeba/img_align_celeba"
    )

    txt_path = (
        "./CelebA_data/celeba/list_attr_celeba.txt"
    )

    csv_path = (
        "./CelebA_data/celeba/list_attr_celeba.csv"
    )

    if os.path.exists(txt_path):
        attr_path = txt_path
    elif os.path.exists(csv_path):
        attr_path = csv_path
    else:
        raise FileNotFoundError(
            "Nenhum arquivo de atributos encontrado."
        )

    arcface_dir = "./cache_arcface"
    if not os.path.isdir(arcface_dir):
        arcface_dir = None
        print("Cache ArcFace nao encontrado — imagens carregadas em tempo real.")
    else:
        print(f"Cache ArcFace detectado: {arcface_dir}")

    dataset = CelebALatentIdentityDataset(
        latent_dir=latent_dir,
        attr_path=attr_path,
        image_dir=image_dir,
        arcface_dir=arcface_dir,
    )

    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    print("\n===================================")
    print("DATASET SPLITS")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")
    print("===================================\n")

    loader_kwargs = dict(
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if is_distributed:

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader, train_sampler, arcface_dir is not None

    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader, None, arcface_dir is not None


# ==========================================
# DATALOADER PARA CONDICIONAMENTO POR IMAGEM
# Sempre carrega imagens brutas (nunca cache ArcFace).
# ==========================================

class CelebALatentImageCondCachedDataset(CelebALatentIdentityDataset):
    """
    Variante do dataset de identidade que usa o cache pré-computado por
    `cache_arcface.py` (./cache_encoder/{stem}.pt → {"arcface":[512],
    "clip":[seq_len,768]}) e também carrega a imagem bruta — útil para
    visualização/originais.

    Cada sample retorna:
        latent          : [4, 32, 32]
        attrs           : [40]
        ref_img         : [3, 256, 256]  em [-1, 1]
        id_emb          : [512]          ArcFace L2-normalizado
        clip_tokens_raw : [49, 768]      tokens CLIP (CLS removido, pré-projeção)
    """

    def __init__(self, latent_dir, attr_path, image_dir, encoder_cache_dir, transform=None):
        super().__init__(
            latent_dir=latent_dir,
            attr_path=attr_path,
            image_dir=image_dir,
            arcface_dir=None,
            transform=transform,
        )
        self.encoder_cache_dir = encoder_cache_dir

    def __getitem__(self, idx):
        img_filename, latent_filename, _ = self.samples[idx]

        data = torch.load(
            os.path.join(self.latent_dir, latent_filename),
            map_location="cpu",
        )
        latent = data["latent"]
        attrs  = data["attrs"]

        img = Image.open(
            os.path.join(self.image_dir, img_filename)
        ).convert("RGB")
        ref_img = self.transform(img)

        stem = os.path.splitext(img_filename)[0]
        cache = torch.load(
            os.path.join(self.encoder_cache_dir, stem + ".pt"),
            map_location="cpu",
        )
        id_emb          = cache["arcface"]   # [512]
        clip_tokens_raw = cache["clip"]      # [seq_len, 768]

        return latent, attrs, ref_img, id_emb, clip_tokens_raw


class CelebALatentImageCondPairedDataset(CelebALatentImageCondCachedDataset):
    """
    Como CelebALatentImageCondCachedDataset, mas `ref_img`/`id_emb`/`clip_tokens_raw`
    vêm de uma foto DIFERENTE da mesma identidade quando disponível, em vez da
    mesma foto que é o alvo do denoising.

    Motivação: com ref == target, o CLIP vê a expressão exata que o modelo
    precisa gerar, então ele aprende a copiar a expressão da referência em vez
    de usar o token de atributo — nenhum valor de `s_attr` desfaz isso (ver
    docs/CFG_COMPOSABLE.md). Trocar a referência por outra foto da mesma pessoa
    quebra esse atalho: `attrs` (usado na loss) continua vindo do ALVO, não
    da referência.

    Checado com `analyze_identity_variety.ipynb`: no CelebA, 91% das
    identidades com >=2 fotos têm variação em `Smiling` (média ~20 fotos por
    identidade) — dá para sortear uma referência com expressão diferente na
    maioria dos casos.

    Identidades com 1 foto só (~0.4%) caem de volta em ref == target.

    Requer:
        identity_txt : caminho para identity_CelebA.txt (filename identity_id,
                        sem header — distribuição oficial do CelebA, nem
                        sempre incluída em mirrors de terceiros).
    """

    def __init__(
        self,
        latent_dir,
        attr_path,
        image_dir,
        encoder_cache_dir,
        identity_txt,
        transform=None,
    ):
        super().__init__(
            latent_dir=latent_dir,
            attr_path=attr_path,
            image_dir=image_dir,
            encoder_cache_dir=encoder_cache_dir,
            transform=transform,
        )
        self._build_identity_groups(identity_txt)

    def _build_identity_groups(self, identity_txt):
        if not os.path.isfile(identity_txt):
            raise FileNotFoundError(
                f"{identity_txt} não encontrado — necessário para "
                "--ref_pairing identity. É o identity_CelebA.txt oficial do "
                "CelebA (filename identity_id, sem header); nem todo mirror "
                "inclui esse arquivo. Rode analyze_identity_variety.ipynb "
                "para checar/gerar identity_groups_multi.csv antes."
            )

        id_map = {}
        with open(identity_txt, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                fname, ident = parts
                id_map[fname] = ident

        sample_filenames = {s[0] for s in self.samples}

        groups = defaultdict(list)
        self.identity_of = {}
        for fname in sample_filenames:
            ident = id_map.get(fname)
            self.identity_of[fname] = ident
            if ident is not None:
                groups[ident].append(fname)

        self.groups = groups

        n_multi = sum(1 for g in groups.values() if len(g) >= 2)
        print(
            f"[ref_pairing=identity] {len(groups)} identidades presentes no "
            f"cache | {n_multi} com >=2 fotos (referência sorteada != alvo)."
        )

    def __getitem__(self, idx):
        img_filename, latent_filename, _ = self.samples[idx]

        data = torch.load(
            os.path.join(self.latent_dir, latent_filename),
            map_location="cpu",
        )
        latent = data["latent"]
        attrs  = data["attrs"]          # sempre do ALVO, não da referência

        ident      = self.identity_of.get(img_filename)
        candidates = self.groups.get(ident, [img_filename])
        others     = [f for f in candidates if f != img_filename]
        ref_filename = random.choice(others) if others else img_filename

        img = Image.open(
            os.path.join(self.image_dir, ref_filename)
        ).convert("RGB")
        ref_img = self.transform(img)

        stem = os.path.splitext(ref_filename)[0]
        try:
            cache = torch.load(
                os.path.join(self.encoder_cache_dir, stem + ".pt"),
                map_location="cpu",
            )
        except FileNotFoundError:
            # referência sorteada não tem cache (ex.: caching interrompido) —
            # cai de volta pro alvo, que sabemos existir (é este __getitem__).
            stem = os.path.splitext(img_filename)[0]
            cache = torch.load(
                os.path.join(self.encoder_cache_dir, stem + ".pt"),
                map_location="cpu",
            )
            img = Image.open(
                os.path.join(self.image_dir, img_filename)
            ).convert("RGB")
            ref_img = self.transform(img)

        id_emb          = cache["arcface"]
        clip_tokens_raw = cache["clip"]

        return latent, attrs, ref_img, id_emb, clip_tokens_raw


def _split_indices_by_identity(identity_of_by_index, seed=42, train_frac=0.70, val_frac=0.15):
    """
    Split disjunto POR PESSOA: nenhuma identidade aparece em mais de um
    split, evitando o vazamento de random_split (que fatia por imagem e
    deixa a mesma pessoa em train/val/test simultaneamente).

    identity_of_by_index : lista onde o elemento i é o identity_id da
        amostra de índice i (aceita None — cada None vira seu próprio
        "grupo" de 1 imagem, não deveria ocorrer no CelebA mas evita
        travar caso identity_CelebA.txt não cubra 100% dos arquivos).

    Retorna (train_idx, val_idx, test_idx) — índices em `dataset.samples`.
    As proporções valem por IDENTIDADE, não por imagem — como o número de
    fotos por pessoa varia, o total de imagens por split fica próximo mas
    não exatamente 70/15/15.
    """

    groups = defaultdict(list)
    for idx, ident in enumerate(identity_of_by_index):
        key = ident if ident is not None else f"__no_identity_{idx}"
        groups[key].append(idx)

    identities = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(identities)

    n = len(identities)
    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)

    train_ids = identities[:n_train]
    val_ids   = identities[n_train:n_train + n_val]
    test_ids  = identities[n_train + n_val:]

    def flatten(id_list):
        idxs = []
        for k in id_list:
            idxs.extend(groups[k])
        return idxs

    return (
        flatten(train_ids), flatten(val_ids), flatten(test_ids),
        len(train_ids), len(val_ids), len(test_ids),
    )


def get_data_imagecond_paired(args, is_distributed=True):
    """
    Igual a get_data_imagecond, mas usa CelebALatentImageCondPairedDataset:
    a referência de identidade vem de OUTRA foto da mesma pessoa quando
    disponível, em vez da mesma foto que é o alvo do denoising (ver docstring
    da classe para o motivo). Requer cache de encoder (./cache_encoder/) —
    não tem fallback live, o custo de reabrir imagens ao vivo pra cada par
    trocado seria alto demais.
    """

    latent_dir = "./cache_latent"

    image_dir = (
        "./CelebA_data/celeba/"
        "img_align_celeba/img_align_celeba"
    )

    txt_path = "./CelebA_data/celeba/list_attr_celeba.txt"
    csv_path = "./CelebA_data/celeba/list_attr_celeba.csv"

    if os.path.exists(txt_path):
        attr_path = txt_path
    elif os.path.exists(csv_path):
        attr_path = csv_path
    else:
        raise FileNotFoundError("Nenhum arquivo de atributos encontrado.")

    identity_txt = getattr(
        args, "identity_txt", "./CelebA_data/celeba/identity_CelebA.txt"
    )

    encoder_cache_dir = getattr(args, "encoder_cache_dir", "./cache_encoder")
    if not (os.path.isdir(encoder_cache_dir) and len(os.listdir(encoder_cache_dir)) > 0):
        raise FileNotFoundError(
            f"--ref_pairing identity requer cache de encoder em "
            f"{encoder_cache_dir} (rode cache_arcface.py primeiro)."
        )

    dataset = CelebALatentImageCondPairedDataset(
        latent_dir=latent_dir,
        attr_path=attr_path,
        image_dir=image_dir,
        encoder_cache_dir=encoder_cache_dir,
        identity_txt=identity_txt,
    )

    identity_of_by_index = [
        dataset.identity_of.get(dataset.samples[i][0])
        for i in range(len(dataset))
    ]

    train_idx, val_idx, test_idx, n_train_id, n_val_id, n_test_id = \
        _split_indices_by_identity(identity_of_by_index, seed=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(dataset, val_idx)
    test_dataset  = torch.utils.data.Subset(dataset, test_idx)

    print("\n===================================")
    print("DATASET SPLITS (ImageCond, paired-identity, DISJUNTO POR PESSOA)")
    print(f"Identidades -> train: {n_train_id}  val: {n_val_id}  test: {n_test_id}")
    print(f"Imagens     -> train: {len(train_dataset)}  val: {len(val_dataset)}  test: {len(test_dataset)}")
    print("===================================\n")

    loader_kwargs = dict(
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    use_cache = True  # sempre cacheado — ver checagem acima

    if is_distributed:

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
        test_sampler  = DistributedSampler(test_dataset,  shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader, train_sampler, use_cache

    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader, None, use_cache


def get_data_imagecond(args, is_distributed=True):
    """
    Dataloader para condicionamento por imagem.

    Detecta automaticamente o cache em ./cache_encoder/ (gerado por
    `cache_arcface.py`). Se presente, retorna samples cacheados (5-tupla:
    latent, attrs, ref_img, id_emb, clip_tokens_raw) — ArcFace e CLIP
    não são executados durante o treino. Caso contrário, cai no modo
    live (3-tupla: latent, attrs, ref_img).

    Retorna (train_loader, val_loader, test_loader, train_sampler, use_cache).
    """

    latent_dir = "./cache_latent"

    image_dir = (
        "./CelebA_data/celeba/"
        "img_align_celeba/img_align_celeba"
    )

    txt_path = "./CelebA_data/celeba/list_attr_celeba.txt"
    csv_path = "./CelebA_data/celeba/list_attr_celeba.csv"

    if os.path.exists(txt_path):
        attr_path = txt_path
    elif os.path.exists(csv_path):
        attr_path = csv_path
    else:
        raise FileNotFoundError("Nenhum arquivo de atributos encontrado.")

    encoder_cache_dir = getattr(args, "encoder_cache_dir", "./cache_encoder")
    use_cache = os.path.isdir(encoder_cache_dir) and len(os.listdir(encoder_cache_dir)) > 0

    if use_cache:
        print(f"Cache de encoder detectado: {encoder_cache_dir} — usando embeddings pré-computados.")
        dataset = CelebALatentImageCondCachedDataset(
            latent_dir=latent_dir,
            attr_path=attr_path,
            image_dir=image_dir,
            encoder_cache_dir=encoder_cache_dir,
        )
    else:
        print(f"Cache de encoder NÃO encontrado em {encoder_cache_dir} — rodando ArcFace/CLIP live (lento).")
        dataset = CelebALatentIdentityDataset(
            latent_dir=latent_dir,
            attr_path=attr_path,
            image_dir=image_dir,
            arcface_dir=None,
        )

    train_size = int(0.70 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    print("\n===================================")
    print("DATASET SPLITS (ImageCond)")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")
    print("===================================\n")

    loader_kwargs = dict(
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if is_distributed:

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
        test_sampler  = DistributedSampler(test_dataset,  shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader, train_sampler, use_cache

    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader, None, use_cache