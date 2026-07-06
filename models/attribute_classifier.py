"""
Classificador de atributos CelebA fine-tunado sobre CLIP ViT-L/14.

Adicionado como um novo módulo (em vez de editar o AttributePredictor
existente em models/encoders.py) para não quebrar nada que já dependa
dele. Uso pretendido: trocar o AttributePredictor por
CLIPAttributeClassifier nos scripts de geração quando o checkpoint
fine-tunado estiver pronto.

Por que CLIP ViT-L/14:
    O objetivo é prever os 40 atributos do CelebA em fotos de pessoas
    que não estão no dataset. Um backbone ImageNet genérico ou um
    backbone de reconhecimento facial (ArcFace/VGGFace2) generaliza mal
    para essa tarefa — reconhecimento facial é treinado para ser
    INVARIANTE a expressão, maquiagem, cabelo etc., exatamente as
    variáveis que precisamos classificar. CLIP, treinado em ~400M
    imagens da internet, já viu uma enorme variedade de poses,
    iluminação e qualidade de câmera, o que o torna o mais robusto para
    fotos "em bruto" fora do domínio do CelebA.

Entrada esperada: rosto já alinhado (ver data/face_align.FaceAligner).
Treinar/inferir com rostos não-alinhados reintroduz o gap de domínio
que este design tenta evitar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


class CLIPAttributeClassifier(nn.Module):
    def __init__(
        self,
        num_attributes=40,
        clip_name="openai/clip-vit-large-patch14",
        unfreeze_last_n=4,
        pretrained_path=None,
    ):
        super().__init__()

        self.num_attributes = num_attributes

        self.clip = CLIPVisionModel.from_pretrained(clip_name)
        clip_dim = self.clip.config.hidden_size  # 1024 para ViT-L/14

        # backbone congelado por padrão; libera só os últimos N blocos
        # do transformer para fine-tuning — a maior parte do CLIP fica
        # intacta, só a parte final se adapta a atributos faciais.
        for p in self.clip.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            # versões recentes do transformers achatam CLIPVisionModel (o
            # transformer fica direto em .encoder), versões antigas o
            # envolvem em .vision_model.encoder — aceita os dois layouts.
            vision_model = getattr(self.clip, "vision_model", self.clip)
            for layer in vision_model.encoder.layers[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.head = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, clip_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(clip_dim // 2, num_attributes),
        )

        self.register_buffer(
            "clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

        # limiares por atributo, calibrados no treino (F1-ótimo na
        # validação) — atributos do CelebA são muito desbalanceados
        # ("Bald" ~2%, "Attractive" ~50%), um corte fixo em 0.5 penaliza
        # os raros. Default 0.5 até um checkpoint calibrado ser carregado.
        self.register_buffer("thresholds", torch.full((num_attributes,), 0.5))

        if pretrained_path is not None:
            self.load_checkpoint(pretrained_path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            self.load_state_dict(ckpt["model"], strict=False)
            if "thresholds" in ckpt:
                self.thresholds.copy_(torch.as_tensor(ckpt["thresholds"]))
        else:
            self.load_state_dict(ckpt, strict=False)

    def forward(self, img):
        # img : [B, 3, H, W]  valores em [-1, 1], rosto já alinhado
        # out : [B, num_attributes]  logits crus
        x = img.float() * 0.5 + 0.5
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.clip_mean) / self.clip_std

        pooled = self.clip(pixel_values=x).pooler_output  # [B, clip_dim]
        return self.head(pooled)

    @torch.no_grad()
    def predict(self, img, threshold=None):
        logits = self.forward(img)
        probs = torch.sigmoid(logits)
        thr = self.thresholds if threshold is None else threshold
        return (probs > thr).float()
