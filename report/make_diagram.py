import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withStroke

fig, ax = plt.subplots(figsize=(7.0, 2.35))
ax.set_xlim(0, 100)
ax.set_ylim(0, 34)
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.35,rounding_size=2.5", linewidth=1.1)

def box(x, y, w, h, text, fc, ec="#333333", fs=7.4, tcolor="black"):
    p = FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=1.1,
                        boxstyle="round,pad=0.3,rounding_size=2.2", zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, color=tcolor, zorder=3, linespacing=1.3)
    return (x, y, w, h)

def arrow(b1, b2, side1="right", side2="left", color="#444444", style="-|>"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    p1 = (x1 + w1, y1 + h1/2) if side1 == "right" else (x1 + w1/2, y1)
    p2 = (x2, y2 + h2/2) if side2 == "left" else (x2 + w2/2, y2 + h2)
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=10,
                         color=color, lw=1.2, zorder=1,
                         connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)

# Column 1: inputs
ref = box(1, 22, 15, 9, "Foto de\nreferência\n(identidade)", "#dce9f9")
attrs = box(1, 3, 15, 9, "Atributos-alvo\ndesejados\n(vetor 40-d)", "#fdeadd")

# Column 2: encoders
enc = box(20, 22, 17, 9, "ArcFace + CLIP\n(congelados)\n→ id tokens", "#bcd8f5")
emb = box(20, 3, 17, 9, "Attribute\nEmbedder\n→ attr tokens", "#f9cba3")

arrow(ref, enc)
arrow(attrs, emb)

# Context concat
ctx = box(41, 12.5, 15, 9, "Contexto\n(concat tokens)\n[id | attr]", "#e4d3f2")
arrow(enc, ctx, side1="right", side2="left")
arrow(emb, ctx, side1="right", side2="left")

# UNet denoising loop
unet = box(60, 12.5, 20, 9, "U-Net latente\ncross-attention\nCFG composável\n($z_t \\to z_{t-1}$)", "#c9a8e6")
arrow(ctx, unet)

# noise input from top
noise = box(60, 25, 20, 6, "$z_T \\sim \\mathcal{N}(0, I)$", "#eeeeee", ec="#888888")
arrow(noise, unet, side1="right", side2="left", style="-|>")
a = FancyArrowPatch((70, 25), (70, 21.5), arrowstyle="-|>", mutation_scale=10,
                     color="#444444", lw=1.2, zorder=1)
ax.add_patch(a)

# VAE decode
dec = box(84, 12.5, 15, 9, "VAE\ndecoder", "#d7f0da")
arrow(unet, dec)

# output
out = box(84, 1, 15, 8, "Imagem editada\n(id. preservada,\natributo alterado)", "#fff2b3")
arrow(dec, out, side1="right", side2="left", style="-|>")
a = FancyArrowPatch((91.5, 12.5), (91.5, 9), arrowstyle="-|>", mutation_scale=10,
                     color="#444444", lw=1.2, zorder=1)
ax.add_patch(a)

plt.tight_layout(pad=0.15)
plt.savefig("report/figures/architecture.png", dpi=260, bbox_inches="tight")
print("ok")
