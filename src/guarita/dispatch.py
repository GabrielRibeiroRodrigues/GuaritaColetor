from __future__ import annotations

import logging
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

from .models import ClosedTrack
from .ocr.worker import OcrJob, OcrResult, process_job, warmup

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


def hard_shutdown(executor, processes=None) -> None:
    """Encerra o executor E mata os processos de trabalho (best-effort; usa API interna do CPython)."""
    if processes is None:
        processes = list((getattr(executor, "_processes", None) or {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        log.exception("Falha ao encerrar o pool de OCR")
    for proc in processes:
        try:
            proc.terminate()
            proc.join(0.5)
            if proc.is_alive():
                proc.kill()
        except Exception:  # noqa: BLE001
            log.exception("Falha ao matar processo de OCR")


class OcrDispatcher:
    """Envia trilhas ao pool de OCR. O relógio do timeout de cada tarefa só corre depois que o pool
    terminou o aquecimento (spawn + carga do PaddleOCR), para que a partida a frio nunca conte."""

    def __init__(self, executor_factory, timeout_s: float, clock=time.monotonic, sleep=time.sleep,
                 workers: int = 1, warmup_timeout_s: float = 300.0):
        self._factory = executor_factory
        self._timeout = timeout_s
        self._clock = clock
        self._sleep = sleep
        self._workers = max(1, workers)
        self._warmup_timeout = warmup_timeout_s
        self._pending: list[_Pending] = []
        self._lock = threading.Lock()
        self._executor = None
        self._warm: list = []
        self._warm_started = 0.0
        self._ready_at: float | None = None
        self._start_pool_locked(self._clock())

    def _start_pool_locked(self, now: float) -> None:
        self._executor = self._factory()
        self._warm_started = now
        self._ready_at = None
        self._warm_times: list[float] = []
        self._warm = [self._executor.submit(warmup) for _ in range(self._workers)]
        for future in self._warm:
            future.add_done_callback(lambda _f, times=self._warm_times: times.append(self._clock()))

    def submit(self, track: ClosedTrack, job: OcrJob) -> None:
        with self._lock:
            try:
                future = self._executor.submit(process_job, job)
            except Exception:
                log.exception("Pool de OCR indisponível; recriando")
                self._restart_locked(self._clock())
                future = self._executor.submit(process_job, job)
            self._pending.append(_Pending(track, job, future, self._clock()))

    def wait_ready(self, timeout_s: float = 120.0, poll_s: float = 0.5) -> bool:
        """Espera (limitado) o aquecimento do pool; registra o progresso. Devolve True se pronto."""
        start = self._clock()
        last_log = start
        while True:
            with self._lock:
                if self._check_warm_locked(self._clock()) == "ready":
                    log.info("Pool de OCR aquecido em %.0f s", self._clock() - start)
                    return True
            now = self._clock()
            if now - start >= timeout_s:
                log.error("Pool de OCR não ficou pronto em %.0f s; seguindo sem aquecimento", timeout_s)
                return False
            if now - last_log >= 10:
                log.info("Aguardando o aquecimento do OCR (%.0f s)…", now - start)
                last_log = now
            self._sleep(poll_s)

    def _check_warm_locked(self, now: float) -> str:
        """'ready' | 'warming'. Reinicia o pool se o aquecimento falhou ou travou."""
        if self._ready_at is not None:
            return "ready"
        if all(f.done() for f in self._warm):
            failed = [f for f in self._warm if f.cancelled() or f.exception() is not None]
            if not failed:
                self._ready_at = max(self._warm_times, default=now)
                return "ready"
            log.error("Aquecimento do OCR falhou: %r; reiniciando o pool", failed[0].exception()
                      if not failed[0].cancelled() else "cancelado")
            self._restart_locked(now)
        elif now - self._warm_started > self._warmup_timeout:
            log.error("Aquecimento do OCR travou (> %.0f s); reiniciando o pool", self._warmup_timeout)
            self._restart_locked(now)
        return "warming"

    def poll(self) -> list[Outcome]:
        with self._lock:
            now = self._clock()
            ready = self._check_warm_locked(now) == "ready"
            outcomes, keep, restart = [], [], False
            for item in self._pending:
                latency = int((now - item.submitted) * 1000)
                if item.future.done():
                    try:
                        outcomes.append(Outcome(item.track, item.future.result(), None, latency))
                    except BrokenProcessPool as exc:
                        outcomes.append(Outcome(item.track, None, f"erro: {exc!r}", latency))
                        restart = True
                    except Exception as exc:  # noqa: BLE001
                        outcomes.append(Outcome(item.track, None, f"erro: {exc!r}", latency))
                elif ready and now - max(item.submitted, self._ready_at) > self._timeout:
                    item.future.cancel()
                    outcomes.append(Outcome(item.track, None, "timeout", latency))
                    restart = True
                else:
                    keep.append(item)
            self._pending = keep
            if restart:
                log.error("Pool de OCR travado ou quebrado; reiniciando")
                self._restart_locked(now)
            return outcomes

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        with self._lock:
            hard_shutdown(self._executor)

    def _restart_locked(self, now: float) -> None:
        hard_shutdown(self._executor)
        self._start_pool_locked(now)
        for item in self._pending:
            item.future = self._executor.submit(process_job, item.job)
            item.submitted = now
