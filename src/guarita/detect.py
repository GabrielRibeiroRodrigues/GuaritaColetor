from __future__ import annotations

import logging
import os

from .models import Detection

log = logging.getLogger(__name__)

DEFAULT_TRACKER_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack_plates.yaml")


class PlateDetector:
    """YOLO de placas com rastreio ByteTrack. Uma instância por câmera (o rastreador guarda estado)."""

    def __init__(self, weights: str, imgsz: int = 416, conf: float = 0.25, model=None, tracker_yaml: str = ""):
        if model is None:
            if not os.path.isfile(weights):
                raise FileNotFoundError(
                    f"Pesos do detector não encontrados: {weights}. O coletor não inicia sem eles "
                    "(não há fallback para yolov8n.pt, que detecta objetos COCO e não placas).")
            from .compat import prepare_ultralytics

            prepare_ultralytics()
            from ultralytics import YOLO

            model = YOLO(weights)
        self._model = model
        self._imgsz = imgsz
        self._conf = conf
        self._tracker = os.path.abspath(tracker_yaml) if tracker_yaml else DEFAULT_TRACKER_YAML

    def detect(self, frame) -> list[Detection]:
        result = self._model.track(
            frame, imgsz=self._imgsz, conf=self._conf, persist=True, tracker=self._tracker, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or getattr(boxes, "id", None) is None:
            return []
        detections = []
        for xyxy, conf, track_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.id.tolist()):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            detections.append(Detection(int(track_id), (x1, y1, x2, y2), float(conf)))
        return detections
