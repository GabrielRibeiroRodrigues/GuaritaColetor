from __future__ import annotations

import logging

import cv2
import numpy as np

from .plates import read_plate_from_detections

log = logging.getLogger(__name__)


def rotate_crop(img: np.ndarray, angle: float) -> np.ndarray:
    """Mesma rotação do main.py atual (matriz com -angulo, borda replicada)."""
    if angle == 0:
        return img.copy()
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def to_ocr_input(crop: np.ndarray) -> np.ndarray:
    """Mesmo pré-processamento do leitor atual: cinza -> RGB."""
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def read_candidates(engine, crop: np.ndarray, angles, min_conf: float):
    """Lê o recorte em todos os ângulos; devolve [(placa, confiança)] com confiança > min_conf."""
    candidates = []
    for angle in angles:
        plate, conf = engine.read(to_ocr_input(rotate_crop(crop, angle)))
        if plate is not None and conf is not None and conf > min_conf:
            candidates.append((plate, float(conf)))
    return candidates


class PaddleEngine:
    """PaddleOCR com os mesmos parâmetros do leitor atual."""

    def __init__(self):
        # No Windows, se o paddleocr (via albumentations) importar o torch pela primeira vez depois
        # do paddle já estar carregado, o torch falha ao abrir `torch\\lib\\shm.dll` (WinError 127 —
        # conflito de DLL nativa). Importar o torch aqui, antes do paddleocr, evita o problema — vale
        # tanto no processo principal quanto no processo do pool de OCR, onde este é o primeiro import
        # pesado. Sem relação com o nosso código: é só para garantir a ordem de carregamento das DLLs.
        import torch  # noqa: F401
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(lang="pt", use_angle_cls=False, use_gpu=False, det_limit_side_len=640)

    def read(self, img_rgb: np.ndarray):
        try:
            results = self._ocr.ocr(img_rgb, cls=False)
            if not results or not results[0]:
                return None, None
            return read_plate_from_detections(results[0])
        except Exception:  # noqa: BLE001 - como o legado: um ângulo ruim não custa o veículo inteiro
            log.exception("Erro durante o OCR")
            return None, None


def make_paddle_engine() -> PaddleEngine:
    return PaddleEngine()
