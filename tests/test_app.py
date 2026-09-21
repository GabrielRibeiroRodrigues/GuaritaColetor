import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fakes import FakeEngine, ScriptedDetector, ScriptedSource
from guarita.app import App
from guarita.capture import Frame
from guarita.config import (ApiConfig, CameraConfig, Config, DetectorConfig, OutboxConfig, PipelineConfig)
from guarita.models import Detection
from guarita.ocr import worker
from guarita.outbox import Outbox
from guarita.sender import ApiClient


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_frames_become_a_delivered_reading_and_heartbeats_flow(tmp_path, api):
    cfg = Config(
        api=ApiConfig(base_url=api.base_url),
        detector=DetectorConfig(),
        pipeline=PipelineConfig(max_crops=2, close_after_s=1.5, failures_dir=str(tmp_path / "falhas")),
        outbox=OutboxConfig(path=str(tmp_path / "o.sqlite")),
        cameras=(CameraConfig("cam-01", "entrada", "", "KEY"),),
        heartbeat_interval_s=0.2, stale_after_s=30.0,
    )
    blank = np.zeros((100, 200, 3), np.uint8)
    frames = [Frame(blank, t) for t in (0.0, 0.5, 1.0, 2.5)]
    box = (10, 10, 60, 40)
    detections = [[Detection(1, box, 0.9)], [Detection(1, box, 0.9)], [], []]
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.8), ("ABC1234", 0.85), (None, None)])
    outbox = Outbox(cfg.outbox.path)

    def executor_factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: engine,))

    app = App(cfg, source_factory=lambda cam: ScriptedSource(frames),
              detector_factory=lambda cam: ScriptedDetector(detections),
              executor_factory=executor_factory, client=ApiClient(api.base_url), outbox=outbox)
    app.start()
    try:
        assert wait_for(lambda: outbox.counts()["delivered"] == 1)
        assert wait_for(lambda: any(r["path"].endswith("/heartbeat") for r in api.requests))
    finally:
        app.stop()
        outbox.close()

    readings = [r for r in api.requests if r["path"] == "/api/ingest/readings"]
    assert len(readings) == 1
    assert readings[0]["headers"]["x-api-key"] == "KEY"
    assert b"ABC1234" in readings[0]["body"]
    assert app.health["cam-01"].snapshot()["frames"] == 4
