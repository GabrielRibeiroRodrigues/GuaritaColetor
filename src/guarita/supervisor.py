from __future__ import annotations

import threading
import time
from collections import defaultdict


class Health:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._last: float | None = None
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def touch(self) -> None:
        with self._lock:
            self._last = self._clock()

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def is_healthy(self, stale_after_s: float) -> bool:
        with self._lock:
            return self._last is not None and self._clock() - self._last < stale_after_s

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)


def metrics_line(camera_id: str, health: Health, ocr_pending: int, outbox_counts: dict[str, int]) -> str:
    snap = health.snapshot()
    return (
        f"[{camera_id}] frames={snap.get('frames', 0)} deteccoes={snap.get('detections', 0)} "
        f"rastreios={snap.get('tracks', 0)} readings={snap.get('readings', 0)} "
        f"ocr_pendentes={ocr_pending} fila_pendente={outbox_counts.get('pending', 0)} "
        f"entregues={outbox_counts.get('delivered', 0)} falhas={outbox_counts.get('failed', 0)}"
    )
