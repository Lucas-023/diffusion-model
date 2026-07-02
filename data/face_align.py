"""
Alinhamento facial automático para uso tanto no treino do classificador
de atributos quanto na inferência com fotos "em bruto" (fora do CelebA).

Por que isso existe:
    O classificador de atributos (e o ArcFace) espera rostos recortados e
    alinhados de forma consistente. Se o treino usa um recorte e a
    inferência recebe uma foto qualquer sem alinhamento, o modelo sofre
    forte queda de desempenho fora do domínio do CelebA — esse é
    tipicamente o maior causador de "funciona no dataset, falha em fotos
    reais".

    Em vez de tentar reproduzir o alinhamento exato (não documentado)
    usado pelos autores originais do CelebA, este módulo define UM único
    template de alinhamento (baseado nos 5 pontos de referência do
    ArcFace: olho esq., olho dir., nariz, canto esq. da boca, canto dir.
    da boca) e aplica exatamente a mesma transformação tanto ao treinar
    o classificador quanto ao processar fotos novas na inferência. O que
    garante generalização não é bater com o template "oficial" do
    CelebA, e sim treino e inferência usarem sempre o mesmo alinhamento.

Requer: insightface (já é dependência via ArcFaceEncoder).
"""

import numpy as np
from PIL import Image
from insightface.utils import face_align as _ia_face_align


class FaceAligner:
    """
    Detecta o maior rosto na imagem e o alinha para um template fixo de
    5 pontos (olhos, nariz, cantos da boca), no estilo ArcFace.

    image_size deve ser múltiplo de 112 ou 128 (restrição do
    insightface.utils.face_align.estimate_norm). 224 é o valor default
    pois casa exatamente com a resolução de entrada do CLIP ViT-L/14.
    """

    def __init__(self, image_size=224, det_size=(640, 640)):
        self.image_size = image_size
        self._det_size = det_size
        self._app = None  # lazy init — evita custo de import/onnx em processos que não alinham

    def _ensure_loaded(self):
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=self._det_size)
        self._app = app

    def align_np(self, img_rgb: np.ndarray):
        """
        img_rgb: HWC uint8 RGB, qualquer resolução/enquadramento.
        Retorna: (aligned_rgb uint8 [image_size,image_size,3], found: bool)

        Se nenhum rosto for detectado, faz center-crop + resize como
        fallback (found=False) em vez de propagar uma exceção — a
        inferência não deve quebrar por causa de uma foto difícil.
        """
        self._ensure_loaded()

        img_bgr = img_rgb[:, :, ::-1].copy()
        faces = self._app.get(img_bgr)

        if not faces:
            return self._fallback_center_crop(img_rgb), False

        # maior rosto por área de bbox (evita alinhar em rostos de fundo)
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        aligned_bgr = _ia_face_align.norm_crop(
            img_bgr, face.kps, image_size=self.image_size, mode="arcface"
        )
        aligned_rgb = aligned_bgr[:, :, ::-1].copy()
        return aligned_rgb, True

    def _fallback_center_crop(self, img_rgb: np.ndarray):
        h, w = img_rgb.shape[:2]
        s = min(h, w)
        top, left = (h - s) // 2, (w - s) // 2
        crop = img_rgb[top:top + s, left:left + s]
        pil = Image.fromarray(crop).resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )
        return np.array(pil)

    def align_pil(self, img: Image.Image):
        img_rgb = np.array(img.convert("RGB"))
        aligned_rgb, found = self.align_np(img_rgb)
        return Image.fromarray(aligned_rgb), found
