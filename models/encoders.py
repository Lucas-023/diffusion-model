import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# =========================================================
# ATTRIBUTE PREDICTOR
# =========================================================

class AttributePredictor(nn.Module):
    """
    Predicts the 40 CelebA binary attributes from an RGB image.

    Placeholder using a ResNet-50 backbone with ImageNet weights.
    Replace the backbone or load a checkpoint trained on CelebA
    attribute classification for accurate predictions.

    Usage during inference only — not part of the diffusion
    training loop (ground-truth attributes are used there).
    """

    def __init__(
        self,
        num_attributes=40,
        pretrained_path=None,
        freeze_backbone=False
    ):
        super().__init__()

        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
        )

        in_features = backbone.fc.in_features

        backbone.fc = nn.Linear(in_features, num_attributes)

        self.net = backbone

        if freeze_backbone:
            for name, param in self.net.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            self.load_state_dict(state)

    def forward(self, img):
        # img : [B, 3, H, W]  values in [-1, 1]
        # out : [B, num_attributes]  raw logits
        return self.net(img)

    @torch.no_grad()
    def predict(self, img, threshold=0.5):
        logits = self.forward(img)
        return (torch.sigmoid(logits) > threshold).float()


# =========================================================
# IDENTITY ENCODER
# =========================================================

class IdentityEncoder(nn.Module):
    """
    Encodes face identity from an RGB image into a fixed-size
    embedding vector.

    Placeholder using a ResNet-50 backbone with ImageNet weights.
    Backbone is frozen by default; only the projection head trains.

    Designed to be swapped for ArcFace / FaceNet / InsightFace
    without changing downstream code — just load the new weights
    and set identity_dim accordingly.
    """

    def __init__(
        self,
        identity_dim=512,
        pretrained_path=None,
        freeze_backbone=True
    ):
        super().__init__()

        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
        )

        in_features = backbone.fc.in_features

        backbone.fc = nn.Sequential(
            nn.Linear(in_features, identity_dim),
            nn.LayerNorm(identity_dim),
        )

        self.net = backbone
        self.identity_dim = identity_dim

        if freeze_backbone:
            for name, param in self.net.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            self.load_state_dict(state)

    def forward(self, img):
        # img : [B, 3, H, W]  values in [-1, 1]
        # out : [B, identity_dim]  L2-normalised embedding
        emb = self.net(img)
        return nn.functional.normalize(emb, dim=-1)


# =========================================================
# ARCFACE ENCODER
# =========================================================

_ARCFACE_MODEL_PATH = os.path.expanduser(
    "~/.insightface/models/buffalo_l/w600k_r50.onnx"
)


class ArcFaceEncoder(nn.Module):
    """
    Identity encoder using ArcFace (InsightFace buffalo_l / w600k_r50).
    Fully frozen ONNX model — no trainable parameters.

    Requires: pip install insightface onnxruntime-gpu  (or onnxruntime)
    Download model once:
        python -c "from insightface.model_zoo import get_model; \
                   get_model('buffalo_l/w600k_r50.onnx').prepare(ctx_id=0)"

    Input : [B, 3, H, W] float32 in [-1, 1], face-aligned images
    Output: [B, 512] L2-normalised identity embedding
    """

    def __init__(self, model_path: str = _ARCFACE_MODEL_PATH):
        super().__init__()
        self._model_path = model_path
        self._rec = None  # lazy init on first forward

    def _ensure_loaded(self):
        if self._rec is not None:
            return
        from insightface.model_zoo import get_model
        if not os.path.exists(self._model_path):
            from insightface.app import FaceAnalysis
            print("Baixando modelo ArcFace (buffalo_l)...")
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1)
            del app
        self._rec = get_model(
            self._model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._rec.prepare(ctx_id=-1)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, 3, H, W] in [-1, 1]  (face-aligned, any resolution)
        returns: [B, 512] L2-normalised embedding on same device as img
        """
        self._ensure_loaded()

        device = img.device

        # [B, 3, H, W] float [-1,1] → [B, H, W, 3] uint8 [0,255] BGR
        img_np = (
            img.float() * 127.5 + 127.5
        ).clamp(0, 255).byte().cpu().numpy()

        # ArcFace expects list of BGR HWC uint8; get_feat handles resize to 112×112
        imgs_bgr = [
            img_np[i].transpose(1, 2, 0)[:, :, ::-1].copy()
            for i in range(img_np.shape[0])
        ]

        feat = self._rec.get_feat(imgs_bgr)   # [B, 512] float32 ndarray

        emb = torch.from_numpy(feat.copy()).to(device)
        return F.normalize(emb.float(), dim=-1)
