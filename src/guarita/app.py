from __future__ import annotations

import logging
import threading
import time

from .config import Config
from .dispatch import OcrDispatcher
from .outbox import Outbox
from .pipeline import Aggregator, CameraRunner, FailureStore, drain_results
from .sender import ApiClient, run_sender
from .supervisor import Health, metrics_line

log = logging.getLogger(__name__)


class App:
    def __init__(self, cfg: Config, *, source_factory, detector_factory, executor_factory,
                 client: ApiClient, outbox: Outbox):
        self.cfg = cfg
        self._client = client
        self._outbox = outbox
        self.stop_event = threading.Event()
        self.dispatcher = OcrDispatcher(executor_factory, cfg.pipeline.ocr_timeout_s)
        self._failures = FailureStore(cfg.pipeline.failures_dir, cfg.pipeline.failures_ttl_h)
        self.health = {c.id: Health() for c in cfg.cameras}
        self._aggregators = {c.id: Aggregator(c, cfg.pipeline, self._failures) for c in cfg.cameras}
        self._runners = {
            c.id: CameraRunner(c, cfg.pipeline, source_factory(c), detector_factory(c), self.dispatcher,
                               self.health[c.id])
            for c in cfg.cameras
        }
        self._api_keys = {c.id: c.api_key for c in cfg.cameras}
        self._camera_threads: list[threading.Thread] = []
        self._results_thread: threading.Thread | None = None
        self._other_threads: list[threading.Thread] = []

    def start(self) -> None:
        for camera in self.cfg.cameras:
            thread = threading.Thread(target=self._camera_loop, args=(camera.id,), name=f"camera-{camera.id}", daemon=True)
            thread.start()
            self._camera_threads.append(thread)
        self._results_thread = threading.Thread(target=self._results_loop, name="resultados", daemon=True)
        self._results_thread.start()
        for name, target in (("envio", self._sender_loop), ("supervisor", self._supervisor_loop)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._other_threads.append(thread)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self._camera_threads:
            thread.join(timeout=10)
        for runner in self._runners.values():
            runner.finish()
        if self._results_thread:
            self._results_thread.join(timeout=10)
        deadline = time.monotonic() + self.cfg.pipeline.ocr_timeout_s + 5
        while self.dispatcher.pending_count() and time.monotonic() < deadline:
            self._drain()
            time.sleep(0.1)
        self._drain()
        for thread in self._other_threads:
            thread.join(timeout=10)
        self.dispatcher.shutdown()

    def _drain(self) -> None:
        for reading in drain_results(self.dispatcher, self._aggregators, self._outbox):
            self.health[reading.camera_id].count("readings")

    def _camera_loop(self, camera_id: str) -> None:
        runner = self._runners[camera_id]
        while not self.stop_event.is_set():
            try:
                state = runner.step()
            except Exception:
                log.exception("Erro no processamento da câmera %s", camera_id)
                self.stop_event.wait(1.0)
                continue
            if state == "done":
                runner.finish()
                log.info("Fonte da câmera %s terminou", camera_id)
                return
            if state == "idle":
                self.stop_event.wait(0.05)

    def _results_loop(self) -> None:
        last_housekeeping = time.monotonic()
        while not self.stop_event.is_set():
            try:
                self._drain()
                if time.monotonic() - last_housekeeping > 3600:
                    self._outbox.purge_delivered(self.cfg.outbox.delivered_ttl_h * 3600)
                    self._failures.purge()
                    last_housekeeping = time.monotonic()
            except Exception:
                log.exception("Erro ao processar resultados do OCR")
            self.stop_event.wait(0.2)

    def _sender_loop(self) -> None:
        run_sender(self._outbox, self._client, self._api_keys, self.stop_event)

    def _supervisor_loop(self) -> None:
        while not self.stop_event.is_set():
            for camera in self.cfg.cameras:
                health = self.health[camera.id]
                if health.is_healthy(self.cfg.stale_after_s):
                    if self._client.heartbeat(camera.api_key):
                        health.count("heartbeats")
                else:
                    log.warning("Câmera %s sem quadros há mais de %.0f s; heartbeat suspenso",
                                camera.id, self.cfg.stale_after_s)
                log.info(metrics_line(camera.id, health, self.dispatcher.pending_count(), self._outbox.counts()))
            self.stop_event.wait(self.cfg.heartbeat_interval_s)
