from __future__ import annotations

import threading

import numpy as np

from .models import Detection


class FrameStore:
    """Guarda só o quadro mais recente de cada câmera (nunca uma fila — não acumula memória)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[np.ndarray, list[Detection]]] = {}

    def update(self, camera_id: str, frame: np.ndarray, detections: list[Detection]) -> None:
        with self._lock:
            self._latest[camera_id] = (frame, detections)

    def snapshot(self, camera_id: str) -> tuple[np.ndarray, list[Detection]] | None:
        with self._lock:
            return self._latest.get(camera_id)
