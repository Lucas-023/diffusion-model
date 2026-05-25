import os
import random

random.seed(42)

latent_dir = "./cache_latent"

all_files = sorted([

    f.replace(".pt", "")

    for f in os.listdir(latent_dir)

    if f.endswith(".pt")

])

n = len(all_files)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_files = all_files[:train_end]
val_files = all_files[train_end:val_end]
test_files = all_files[val_end:]

os.makedirs("data", exist_ok=True)

with open("data/train.txt", "w") as f:

    for x in train_files:

        f.write(x + "\n")

with open("data/val.txt", "w") as f:

    for x in val_files:

        f.write(x + "\n")

with open("data/test.txt", "w") as f:

    for x in test_files:

        f.write(x + "\n")

print("\n✅ Splits criados")
print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")
print(f"Test:  {len(test_files)}")