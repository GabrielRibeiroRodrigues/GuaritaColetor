from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import ClosedTrack, Crop, Detection


def crop_score(crop: np.ndarray, conf: float) -> float:
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    height, width = gray.shape[:2]
    return conf * height * width * (1.0 + sharpness / 100.0)


@dataclass
class _State:
    started_at: float
    last_seen: float
    crops: list[Crop] = field(default_factory=list)


class TrackManager:
    def __init__(self, camera_id: str, close_after_s: float, max_age_s: float, max_crops: int):
        self._camera_id = camera_id
        self._close_after = close_after_s
        self._max_age = max_age_s
        self._max_crops = max_crops
        self._tracks: dict[int, _State] = {}

    def update(self, frame: np.ndarray, detections: list[Detection], now: float) -> list[ClosedTrack]:
        height, width = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            state = self._tracks.get(det.track_id)
            if state is None:
                state = _State(started_at=now, last_seen=now)
                self._tracks[det.track_id] = state
            state.last_seen = now
            self._offer(state, Crop(crop.copy(), now, crop_score(crop, det.conf)))
        return self._collect(now, force=False)

    def flush(self, now: float) -> list[ClosedTrack]:
        return self._collect(now, force=True)

    def _offer(self, state: _State, crop: Crop) -> None:
        if len(state.crops) < self._max_crops:
            state.crops.append(crop)
            return
        worst = min(range(len(state.crops)), key=lambda i: state.crops[i].score)
        if crop.score > state.crops[worst].score:
            state.crops[worst] = crop

    def _collect(self, now: float, force: bool) -> list[ClosedTrack]:
        closed = []
        for track_id, state in list(self._tracks.items()):
            expired = (now - state.last_seen >= self._close_after) or (now - state.started_at >= self._max_age)
            if force or expired:
                del self._tracks[track_id]
                if state.crops:
                    crops = sorted(state.crops, key=lambda c: c.score, reverse=True)
                    closed.append(ClosedTrack(self._camera_id, track_id, state.started_at, now, crops))
        return closed
