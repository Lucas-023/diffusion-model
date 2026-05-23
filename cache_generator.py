# ============================================================
# CACHE GENERATOR
# ============================================================

import os
import torch
import pandas as pd

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from vae.modules import VAE


# ============================================================
# LOAD ATTRIBUTES
# SUPORTA TXT E CSV
# ============================================================

def load_attributes(attr_path):

    attributes_dict = {}

    # ========================================================
    # TXT ORIGINAL
    # ========================================================

    if attr_path.endswith(".txt"):

        with open(attr_path, "r") as f:
            lines = f.readlines()

        attr_names = lines[1].split()

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

            attributes_dict[filename] = attrs

    # ========================================================
    # CSV KAGGLE
    # ========================================================

    else:

        df = pd.read_csv(attr_path)

        attr_names = list(df.columns[1:])

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

            attributes_dict[filename] = attrs

    return attributes_dict, attr_names


# ============================================================
# MAIN
# ============================================================

def make_cache():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\n🚀 Device: {device}")

    # ========================================================
    # PATHS
    # ========================================================

    image_dir = (
        "./CelebA_data/celeba/"
        "img_align_celeba/img_align_celeba"
    )

    txt_attr_path = (
        "./CelebA_data/celeba/"
        "list_attr_celeba.txt"
    )

    csv_attr_path = (
        "./CelebA_data/celeba/"
        "list_attr_celeba.csv"
    )

    cache_dir = "./cache_latent"

    os.makedirs(
        cache_dir,
        exist_ok=True
    )

    # ========================================================
    # DETECTA ARQUIVO DE ATRIBUTOS
    # ========================================================

    if os.path.exists(txt_attr_path):

        attr_path = txt_attr_path

    elif os.path.exists(csv_attr_path):

        attr_path = csv_attr_path

    else:

        raise FileNotFoundError(
            "Arquivo de atributos não encontrado."
        )

    # ========================================================
    # LOAD ATTRIBUTES
    # ========================================================

    print("\n📄 Carregando atributos...")

    attributes_dict, attr_names = load_attributes(
        attr_path
    )

    print(
        f"✅ {len(attr_names)} atributos encontrados"
    )

    # ========================================================
    # LOAD VAE
    # ========================================================

    print("\n🧠 Carregando VAE...")

    vae = VAE(
        in_channels=3,
        latent_dim=4
    ).to(device)

    vae.load_state_dict(
        torch.load(
            "vae/vae_epoch_62.pt",
            map_location=device
        )
    )

    vae.eval()

    for param in vae.parameters():

        param.requires_grad = False

    # ========================================================
    # TRANSFORM
    # ========================================================

    transform = transforms.Compose([

        transforms.CenterCrop(178),

        transforms.Resize((256, 256)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])

    # ========================================================
    # IMAGE FILES
    # ========================================================

    image_files = sorted([

        f for f in os.listdir(image_dir)

        if f.endswith(
            (".jpg", ".png")
        )
    ])

    print(
        f"\n🖼️ Total de imagens: "
        f"{len(image_files)}"
    )

    # ========================================================
    # CACHE LOOP
    # ========================================================

    print("\n⚡ Criando latent cache...\n")

    with torch.no_grad():

        for img_name in tqdm(image_files):

            try:

                # ============================================
                # LOAD IMAGE
                # ============================================

                img_path = os.path.join(
                    image_dir,
                    img_name
                )

                image = Image.open(
                    img_path
                ).convert("RGB")

                image = transform(image)

                image = image.unsqueeze(0).to(device)

                # ============================================
                # VAE ENCODE
                # ============================================

                # SUA VAE RETORNA:
                # mu, logvar

                mu, logvar = vae.encode(image)

                # ====================================================
                # LATENTE DETERMINÍSTICO
                # ====================================================

                # NÃO usamos sample()
                # NÃO usamos ruído aleatório
                # cache totalmente determinístico

                latents = mu

                # ====================================================
                # ESCALING FACTOR
                # ====================================================

                # mantém variância parecida com Stable Diffusion

                latents = latents * 0.18215

                latents = latents.squeeze(0).cpu()

                # ============================================
                # ATTRIBUTES
                # ============================================

                attrs = attributes_dict[img_name]

                # ============================================
                # SAVE
                # ============================================

                save_dict = {

                    "latent": latents,

                    "attrs": attrs
                }

                save_name = (
                    img_name
                    .replace(".jpg", ".pt")
                    .replace(".png", ".pt")
                )

                save_path = os.path.join(
                    cache_dir,
                    save_name
                )

                torch.save(
                    save_dict,
                    save_path
                )

            except Exception as e:

                print(f"\n❌ Erro em {img_name}")

                print(e)

    print("\n✅ Cache criado com sucesso!")

    print(f"📁 Salvo em: {cache_dir}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    make_cache()