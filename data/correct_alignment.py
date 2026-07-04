"""
Alinhamento para o convention NATIVO do CelebA (img_align_celeba) — em vez
do template ArcFace usado em data/face_align.py.

Por quê este módulo existe (e não basta usar FaceAligner):
    Todo o pipeline de difusão (VAE, ArcFaceEncoder, ImageConditionEncoder)
    foi treinado inteiramente sobre o crop nativo do CelebA
    (CenterCrop(178) + Resize(256) em cima de img_align_celeba) — NUNCA
    sobre o template de 5 pontos do ArcFace. Alinhar uma foto crua (ex.:
    uma selfie) com FaceAligner e alimentá-la nesse pipeline produz uma
    imagem fora da distribuição de treino (rosto em posição/escala
    diferente), degradando identidade e qualidade — ver discussão em
    data/face_align.py e CLAUDE.md.

    O algoritmo de alinhamento original do CelebA nunca foi publicado. Este
    módulo o reproduz empiricamente: o destino (template) usado abaixo é a
    média dos 5 pontos (olho esq., olho dir., nariz, canto esq. da boca,
    canto dir. da boca) de list_landmarks_align_celeba.csv sobre as 202599
    imagens de img_align_celeba — ou seja, é o ponto de convergência real
    do alinhamento que o CelebA já aplicou. Desvio-padrão medido por ponto:
    ~1-7px num canvas de 178x218, confirmando que o CelebA usa de fato um
    único template consistente (só não documentado).

Efeito esperado:
    - Numa imagem do próprio CelebA (já alinhada): os 5 pontos detectados
      já estão a poucos pixels do template — a transformação de
      similaridade estimada fica próxima da identidade, então a imagem de
      saída é imperceptivelmente diferente da entrada.
    - Numa foto qualquer fora do CelebA: a face é recortada/escalada/
      rotacionada para a MESMA posição e tamanho que o CelebA usa,
      tornando-a compatível com o pipeline de difusão treinado nesse
      formato — sem precisar mudar nada mais no pipeline (ArcFaceEncoder,
      ImageConditionEncoder, VAE continuam recebendo exatamente o mesmo
      tipo de entrada de sempre).

Requer: insightface (mesma dependência de detecção usada em face_align.py).
"""

import cv2
import numpy as np
from PIL import Image
from skimage import transform as sktrans


# Destino empírico (ver docstring acima). Ordem dos pontos igual à do
# insightface/RetinaFace (kps) e à das colunas do CSV do CelebA: olho
# esquerdo, olho direito, nariz, canto esquerdo da boca, canto direito da
# boca.
_CELEBA_TEMPLATE = np.array(
    [
        [69.3538665, 111.19798222],
        [107.64403082, 111.16160001],
        [88.0631395, 135.1020242],
        [71.24745927, 152.11301142],
        [105.58642935, 152.19466039],
    ],
    dtype=np.float32,
)

# (largura, altura) — mesmo tamanho de img_align_celeba. Mantendo esse
# tamanho, o resultado pode passar pelas MESMAS transformações downstream
# (CenterCrop(178) + Resize(256)) já usadas em todo o resto do pipeline.
_OUTPUT_SIZE = (178, 218)


def _estimate_celeba_norm(lmk: np.ndarray) -> np.ndarray:
    assert lmk.shape == (5, 2)
    tform = sktrans.SimilarityTransform()
    tform.estimate(lmk, _CELEBA_TEMPLATE)
    return tform.params[0:2, :]


class CelebAAligner:
    """
    Detecta o maior rosto na imagem e o alinha para o template nativo do
    CelebA (ver módulo). Mesma interface de FaceAligner (data/face_align.py).
    """

    def __init__(self, det_size=(640, 640)):
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
        Retorna: (aligned_rgb uint8 [218,178,3], found: bool)

        Se nenhum rosto for detectado, faz center-crop (respeitando a
        proporção 178:218) + resize como fallback (found=False) em vez de
        propagar uma exceção.
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

        M = _estimate_celeba_norm(face.kps)
        aligned_bgr = cv2.warpAffine(img_bgr, M, _OUTPUT_SIZE, borderValue=0.0)
        aligned_rgb = aligned_bgr[:, :, ::-1].copy()
        return aligned_rgb, True

    def _fallback_center_crop(self, img_rgb: np.ndarray):
        h, w = img_rgb.shape[:2]
        target_w, target_h = _OUTPUT_SIZE
        target_ratio = target_w / target_h
        cur_ratio = w / h

        if cur_ratio > target_ratio:
            new_w = max(1, int(h * target_ratio))
            left = (w - new_w) // 2
            crop = img_rgb[:, left:left + new_w]
        else:
            new_h = max(1, int(w / target_ratio))
            top = (h - new_h) // 2
            crop = img_rgb[top:top + new_h, :]

        pil = Image.fromarray(crop).resize(_OUTPUT_SIZE, Image.BICUBIC)
        return np.array(pil)

    def align_pil(self, img: Image.Image):
        img_rgb = np.array(img.convert("RGB"))
        aligned_rgb, found = self.align_np(img_rgb)
        return Image.fromarray(aligned_rgb), found
