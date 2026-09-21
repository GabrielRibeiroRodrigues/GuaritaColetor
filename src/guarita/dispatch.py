from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from .models import ClosedTrack
from .ocr.worker import OcrJob, OcrResult, process_job

log = logging.getLogger(__name__)


@dataclass
class Outcome:
    track: ClosedTrack
    result: OcrResult | None
    error: str | None
    latency_ms: int


@dataclass
class _Pending:
    track: ClosedTrack
    job: OcrJob
    future: object
    submitted: float


class OcrDispatcher:
    def __init__(self, executor_factory, timeout_s: float, clock=time.monotonic):
        self._factory = executor_factory
        self._executor = executor_factory()
        self._timeout = timeout_s
        self._clock = clock
        self._pending: list[_Pending] = []
        self._lock = threading.Lock()

    def submit(self, track: ClosedTrack, job: OcrJob) -> None:
        with self._lock:
            try:
                future = self._executor.submit(process_job, job)
            except Exception:
                log.exception("Pool de OCR indisponível; recriando")
                self._restart_locked(self._clock())
                future = self._executor.submit(process_job, job)
            self._pending.append(_Pending(track, job, future, self._clock()))

    def poll(self) -> list[Outcome]:
        with self._lock:
            now = self._clock()
            outcomes, keep, timed_out = [], [], False
            for item in self._pending:
                latency = int((now - item.submitted) * 1000)
                if item.future.done():
                    try:
                        outcomes.append(Outcome(item.track, item.future.result(), None, latency))
                    except Exception as exc:
                        outcomes.append(Outcome(item.track, None, f"erro: {exc!r}", latency))
                elif now - item.submitted > self._timeout:
                    item.future.cancel()
                    outcomes.append(Outcome(item.track, None, "timeout", latency))
                    timed_out = True
                else:
                    keep.append(item)
            self._pending = keep
            if timed_out:
                log.error("Tarefa de OCR estourou o tempo; reiniciando o pool")
                self._restart_locked(now)
            return outcomes

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        with self._lock:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _restart_locked(self, now: float) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = self._factory()
        for item in self._pending:
            item.future = self._executor.submit(process_job, item.job)
            item.submitted = now
