from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

from .models import ClosedTrack
from .ocr.worker import OcrJob, OcrResult, process_job, warmup

log = logging.getLogger(__name__)

# Espera entre reinícios do pool quando o aquecimento falha (dobra até o teto) e quantas falhas
# seguidas bastam para declarar o despachante inutilizável.
WARMUP_BACKOFF_S = 5.0
WARMUP_BACKOFF_MAX_S = 300.0
MAX_WARMUP_FAILURES = 5


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
    if executor is None and processes is None:
        return
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
                 workers: int = 1, warmup_timeout_s: float = 300.0,
                 warmup_backoff_s: float = WARMUP_BACKOFF_S,
                 warmup_backoff_max_s: float = WARMUP_BACKOFF_MAX_S,
                 max_warmup_failures: int = MAX_WARMUP_FAILURES):
        self._factory = executor_factory
        self._timeout = timeout_s
        self._clock = clock
        self._sleep = sleep
        self._workers = max(1, workers)
        self._warmup_timeout = warmup_timeout_s
        self._backoff = warmup_backoff_s
        self._backoff_max = warmup_backoff_max_s
        self._max_warm_failures = max(1, max_warmup_failures)
        self._pending: list[_Pending] = []
        self._lock = threading.Lock()
        self._executor = None
        self._warm: list = []
        self._warm_times: list[float] = []
        self._warm_started = 0.0
        self._ready_at: float | None = None
        self._warm_failures = 0
        self._retry_at: float | None = None
        self._failed_reason: str | None = None
        self._start_pool_locked(self._clock())

    @property
    def failed(self) -> bool:
        """True quando o aquecimento falhou tantas vezes seguidas que o pool foi desistido."""
        return self._failed_reason is not None

    @property
    def failure_reason(self) -> str | None:
        return self._failed_reason

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
            now = self._clock()
            if self._executor is None:
                # Em espera entre reinícios (ou desistido): não há para onde enviar. A trilha vira uma
                # falha imediata, para que o Aggregator guarde o recorte em vez de ela sumir.
                future: object = Future()
                future.set_exception(RuntimeError(
                    self._failed_reason or "pool de OCR indisponível (aguardando novo aquecimento)"))
                self._pending.append(_Pending(track, job, future, now))
                return
            try:
                future = self._executor.submit(process_job, job)
            except Exception:
                log.exception("Pool de OCR indisponível; recriando")
                self._restart_locked(now)
                future = self._executor.submit(process_job, job)
            self._pending.append(_Pending(track, job, future, self._clock()))

    def wait_ready(self, timeout_s: float = 120.0, poll_s: float = 0.5, stop_event=None) -> bool:
        """Espera (limitado) o aquecimento do pool; registra o progresso. Devolve True se pronto.
        Com `stop_event`, desiste assim que a parada for pedida (SIGTERM durante a partida a frio)."""
        start = self._clock()
        last_log = start
        while True:
            with self._lock:
                state = self._check_warm_locked(self._clock())
            if state == "ready":
                log.info("Pool de OCR aquecido em %.0f s", self._clock() - start)
                return True
            if state == "failed":
                log.error("Pool de OCR desistido: %s", self._failed_reason)
                return False
            if stop_event is not None and stop_event.is_set():
                log.warning("Parada pedida durante o aquecimento do OCR; desistindo da espera")
                return False
            now = self._clock()
            if now - start >= timeout_s:
                log.error("Pool de OCR não ficou pronto em %.0f s", timeout_s)
                return False
            if now - last_log >= 10:
                log.info("Aguardando o aquecimento do OCR (%.0f s)…", now - start)
                last_log = now
            self._sleep(poll_s)

    def _check_warm_locked(self, now: float) -> str:
        """'ready' | 'warming' | 'failed'. Reinicia o pool (com espera crescente) se o aquecimento
        falhou ou travou; após `max_warmup_failures` falhas seguidas o despachante vira 'failed'."""
        if self._failed_reason is not None:
            return "failed"
        if self._ready_at is not None:
            return "ready"
        if self._retry_at is not None:                 # esperando antes de tentar de novo
            if now < self._retry_at:
                return "warming"
            self._retry_at = None
            self._restart_locked(now)
            return "warming"
        if self._warm and all(f.done() for f in self._warm):
            broken = [f for f in self._warm if f.cancelled() or f.exception() is not None]
            if not broken:
                self._ready_at = max(self._warm_times, default=now)
                self._warm_failures = 0
                return "ready"
            reason = "cancelado" if broken[0].cancelled() else repr(broken[0].exception())
            return self._warm_failed_locked(now, reason)
        if now - self._warm_started > self._warmup_timeout:
            return self._warm_failed_locked(now, f"travado (> {self._warmup_timeout:.0f} s sem terminar)")
        return "warming"

    def _warm_failed_locked(self, now: float, reason: str) -> str:
        """Derruba o pool e agenda o próximo reinício com espera crescente (ou desiste de vez)."""
        self._warm_failures += 1
        hard_shutdown(self._executor)
        self._executor = None
        self._warm = []
        if self._warm_failures >= self._max_warm_failures:
            self._failed_reason = (
                f"o aquecimento do OCR falhou {self._warm_failures} vezes seguidas; último erro: {reason}")
            self._retry_at = None
            log.critical("Pool de OCR desistido: %s", self._failed_reason)
            return "failed"
        delay = min(self._backoff_max, self._backoff * 2 ** (self._warm_failures - 1))
        self._retry_at = now + delay
        log.error("Aquecimento do OCR falhou (%d/%d): %s; nova tentativa em %.0f s",
                  self._warm_failures, self._max_warm_failures, reason, delay)
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
            if restart and not self.failed:
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
        self._executor = None
        try:
            self._start_pool_locked(now)
        except Exception as exc:  # noqa: BLE001 - não conseguir criar o pool não pode derrubar o poll()
            log.exception("Falha ao criar o pool de OCR")
            self._warm_failed_locked(now, repr(exc))
            return
        for item in self._pending:
            item.future = self._executor.submit(process_job, item.job)
            item.submitted = now
