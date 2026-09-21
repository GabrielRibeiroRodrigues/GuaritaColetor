from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 em pixels do quadro original
    conf: float


@dataclass
class Crop:
    image: np.ndarray
    ts: float
    score: float


@dataclass
class ClosedTrack:
    camera_id: str
    track_id: int
    started_at: float
    closed_at: float
    crops: list[Crop]  # melhor primeiro


@dataclass(frozen=True)
class Reading:
    camera_id: str
    external_id: str
    plate: str
    confidence: float  # 0..1
    timestamp: float  # epoch em segundos (do quadro do melhor recorte)
    direction: str
    image_jpeg: bytes
    processing_ms: int
