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


def make_app(tmp_path, api, frames=(), detections=()):
    cfg = Config(
        api=ApiConfig(base_url=api.base_url), detector=DetectorConfig(),
        pipeline=PipelineConfig(failures_dir=str(tmp_path / "falhas")),
        outbox=OutboxConfig(path=str(tmp_path / "o.sqlite")),
        cameras=(CameraConfig("cam-01", "entrada", "", "KEY"),),
    )
    outbox = Outbox(cfg.outbox.path)

    def executor_factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: FakeEngine(),))

    app = App(cfg, source_factory=lambda cam: ScriptedSource(list(frames)),
              detector_factory=lambda cam: ScriptedDetector(list(detections)),
              executor_factory=executor_factory, client=ApiClient(api.base_url), outbox=outbox)
    return app, outbox


def test_start_waits_for_the_ocr_warmup_before_opening_cameras(tmp_path, api):
    app, outbox = make_app(tmp_path, api)
    order = []
    app.dispatcher.wait_ready = lambda timeout: order.append(("warm", timeout)) or True
    app._camera_loop = lambda camera_id: order.append("camera")
    try:
        app.start()
        app.stop()
    finally:
        outbox.close()
    assert order[0] == ("warm", 120.0)


def test_stop_always_shuts_the_dispatcher_down_even_when_steps_fail(tmp_path, api):
    app, outbox = make_app(tmp_path, api)
    calls = []
    real_shutdown = app.dispatcher.shutdown
    app.dispatcher.shutdown = lambda: calls.append("shutdown") or real_shutdown()
    app._runners["cam-01"].finish = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    app._drain = lambda: (_ for _ in ()).throw(RuntimeError("drain"))
    app.dispatcher.pending_count = lambda: 1
    try:
        app.stop()                       # não pode levantar
    finally:
        outbox.close()
    assert calls == ["shutdown"]


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
