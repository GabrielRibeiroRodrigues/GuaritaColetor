from __future__ import annotations

import logging
import os
import time

import cv2

from .config import CameraConfig, PipelineConfig
from .dispatch import OcrDispatcher, Outcome
from .models import ClosedTrack, Reading
from .ocr.vote import vote_plate
from .ocr.worker import OcrJob
from .supervisor import Health
from .track import TrackManager

log = logging.getLogger(__name__)


def _jpeg(image) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else b""


def external_id(track: ClosedTrack) -> str:
    return f"{int(track.started_at * 1000)}-{track.track_id}"


class FailureStore:
    """Guarda o melhor recorte das leituras que falharam (24 h, para análise)."""

    def __init__(self, directory: str, ttl_h: float, clock=time.time):
        self._dir = directory
        self._ttl_s = ttl_h * 3600
        self._clock = clock

    def save(self, track: ClosedTrack, reason: str) -> str:
        os.makedirs(self._dir, exist_ok=True)
        path = os.path.join(self._dir, f"{external_id(track)}-{reason}.jpg")
        with open(path, "wb") as fh:
            fh.write(_jpeg(track.crops[0].image))
        return path

    def purge(self, now: float | None = None) -> int:
        now = self._clock() if now is None else now
        removed = 0
        if not os.path.isdir(self._dir):
            return 0
        for name in os.listdir(self._dir):
            path = os.path.join(self._dir, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > self._ttl_s:
                os.remove(path)
                removed += 1
        return removed


class Aggregator:
    """Transforma o resultado do OCR de uma placa rastreada em uma leitura (ou descarta)."""

    def __init__(self, camera: CameraConfig, cfg: PipelineConfig, failures: FailureStore | None):
        self._camera = camera
        self._cfg = cfg
        self._failures = failures
        self._last_emitted: dict[str, float] = {}

    def handle(self, outcome: Outcome) -> Reading | None:
        track = outcome.track
        if outcome.error is not None or outcome.result is None:
            reason = "timeout" if outcome.error == "timeout" else "erro"
            log.warning("OCR falhou para %s (%s)", external_id(track), outcome.error)
            if self._failures is not None:
                self._failures.save(track, reason)
            return None

        plate, confidence = vote_plate(outcome.result.candidates)
        if plate is None:
            log.info("Sem leitura válida para %s", external_id(track))
            return None

        best = track.crops[0]
        last = self._last_emitted.get(plate)
        if last is not None and abs(best.ts - last) < self._cfg.fuse_window_s:
            log.info("Placa %s repetida dentro da janela; fundida", plate)
            return None
        self._last_emitted[plate] = best.ts
        if len(self._last_emitted) > 1000:
            cutoff = best.ts - 10 * self._cfg.fuse_window_s
            self._last_emitted = {p: t for p, t in self._last_emitted.items() if t >= cutoff}

        return Reading(
            camera_id=self._camera.id, external_id=external_id(track), plate=plate, confidence=confidence,
            timestamp=best.ts, direction=self._camera.direction, image_jpeg=_jpeg(best.image),
            processing_ms=outcome.latency_ms)


class CameraRunner:
    """Um passo do fluxo por câmera: quadro -> detecção/rastreio -> trilhas fechadas -> OCR."""

    def __init__(self, camera: CameraConfig, cfg: PipelineConfig, source, detector, dispatcher: OcrDispatcher,
                 health: Health | None = None):
        self.camera = camera
        self._cfg = cfg
        self.source = source
        self._detector = detector
        self._dispatcher = dispatcher
        self._health = health
        self._tracks = TrackManager(camera.id, cfg.close_after_s, cfg.max_age_s, cfg.max_crops)
        self._last_ts = 0.0

    def step(self) -> str:
        frame = self.source.read()
        if frame is None:
            return "done" if getattr(self.source, "exhausted", False) else "idle"
        if self._health:
            self._health.touch()
            self._health.count("frames")
        detections = self._detector.detect(frame.image)
        self._last_ts = frame.ts
        if self._health and detections:
            self._health.count("detections", len(detections))
        for track in self._tracks.update(frame.image, detections, frame.ts):
            self._submit(track)
        return "frame"

    def finish(self) -> None:
        for track in self._tracks.flush(self._last_ts):
            self._submit(track)

    def _submit(self, track: ClosedTrack) -> None:
        if self._health:
            self._health.count("tracks")
        job = OcrJob([c.image for c in track.crops], self._cfg.rotation_angles, self._cfg.min_ocr_conf)
        self._dispatcher.submit(track, job)


class ListSink:
    """Destino em memória (bench/testes) com a mesma interface de Outbox.add."""

    def __init__(self):
        self.readings: list[Reading] = []

    def add(self, reading: Reading, now=None) -> bool:
        self.readings.append(reading)
        return True


def drain_results(dispatcher: OcrDispatcher, aggregators: dict[str, Aggregator], sink) -> list[Reading]:
    accepted = []
    for outcome in dispatcher.poll():
        reading = aggregators[outcome.track.camera_id].handle(outcome)
        if reading is not None and sink.add(reading):
            accepted.append(reading)
    return accepted
