from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray
    ts: float


class FrameThrottle:
    def __init__(self, target_fps: float):
        self._interval = 1.0 / target_fps
        self._last: float | None = None

    def should_process(self, ts: float) -> bool:
        if self._last is None or ts - self._last >= self._interval - 1e-9:
            self._last = ts
            return True
        return False


RTSP_TIMEOUT_MS = 5000


def open_rtsp(url: str):
    """Abre o RTSP com timeouts explícitos de abertura e de leitura (o `stimeout` do FFmpeg foi removido)."""
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_TIMEOUT_MS])


class RtspSource:
    exhausted = False

    def __init__(self, url: str, target_fps: float, open_fn=open_rtsp, clock=time.time,
                 monotonic=time.monotonic, max_backoff_s: float = 30.0, should_stop=lambda: False):
        self._url = url
        self._open_fn = open_fn
        self._clock = clock
        self._monotonic = monotonic
        self._max_backoff = max_backoff_s
        self._should_stop = should_stop
        self._throttle = FrameThrottle(target_fps)
        self._cap = None
        self._failures = 0
        self._next_open = 0.0

    def _schedule_retry(self) -> None:
        self._failures += 1
        delay = min(self._max_backoff, 2 ** (self._failures - 1))
        self._next_open = self._monotonic() + delay
        log.warning("Câmera indisponível; nova tentativa em %.0f s", delay)

    def _drop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._schedule_retry()

    def read(self) -> Frame | None:
        if self._cap is None:
            if self._monotonic() < self._next_open:
                return None
            cap = self._open_fn(self._url)
            if not cap.isOpened():
                cap.release()
                self._schedule_retry()
                return None
            self._cap = cap
            self._failures = 0
            log.info("Conectado ao stream da câmera")
        while not self._should_stop():
            if not self._cap.grab():
                self._drop()
                return None
            ts = self._clock()
            if self._throttle.should_process(ts):
                ok, image = self._cap.retrieve()
                if ok and image is not None:
                    return Frame(image, ts)
                self._drop()
                return None
        return None


class FileSource:
    def __init__(self, path: str, target_fps: float, open_fn=cv2.VideoCapture):
        self._cap = open_fn(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self._step = max(1, round(self.fps / target_fps))
        self._index = -1
        self.exhausted = False

    def read(self) -> Frame | None:
        while True:
            self._index += 1
            if not self._cap.grab():
                self.exhausted = True
                return None
            if self._index % self._step == 0:
                ok, image = self._cap.retrieve()
                if ok:
                    return Frame(image, self._index / self.fps)
