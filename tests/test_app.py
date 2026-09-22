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
    app.dispatcher.wait_ready = lambda timeout, stop_event=None: order.append(("warm", timeout)) or True
    app._camera_loop = lambda camera_id: order.append("camera")
    try:
        app.start()
        app.stop()
    finally:
        outbox.close()
    assert order[0] == ("warm", 120.0)


def test_start_raises_when_the_ocr_pool_never_warms_up(tmp_path, api):
    import pytest

    app, outbox = make_app(tmp_path, api)
    app.dispatcher.wait_ready = lambda timeout, stop_event=None: False
    app.dispatcher._failed_reason = "aquecimento desistiu (teste)"  # atributo interno: só para o teste
    try:
        with pytest.raises(RuntimeError, match="aquecimento desistiu"):
            app.start()
    finally:
        outbox.close()


def test_start_returns_quietly_when_stop_is_requested_during_warmup(tmp_path, api):
    app, outbox = make_app(tmp_path, api)
    app.stop_event.set()
    app.dispatcher.wait_ready = lambda timeout, stop_event=None: False
    try:
        app.start()  # não deve levantar: a parada já foi pedida
    finally:
        outbox.close()


def test_detector_construction_never_touches_the_ocr_pool(tmp_path, api):
    """Detectores e fontes são construídos ANTES do pool de OCR: se os pesos faltarem, o erro sobe
    sem o pool ter sido criado (não há o que desligar)."""
    import pytest

    from guarita.config import ApiConfig, CameraConfig, Config, DetectorConfig, OutboxConfig, PipelineConfig
    from guarita.outbox import Outbox

    cfg = Config(
        api=ApiConfig(base_url=api.base_url), detector=DetectorConfig(),
        pipeline=PipelineConfig(failures_dir=str(tmp_path / "falhas")),
        outbox=OutboxConfig(path=str(tmp_path / "o.sqlite")),
        cameras=(CameraConfig("cam-01", "entrada", "", "KEY"),),
    )
    outbox = Outbox(cfg.outbox.path)
    executor_calls = []

    def executor_factory():
        executor_calls.append(True)
        raise AssertionError("o pool de OCR não deveria ser criado")

    def broken_detector(camera):
        raise FileNotFoundError("bestn.pt ausente (teste)")

    try:
        with pytest.raises(FileNotFoundError):
            App(cfg, source_factory=lambda cam: ScriptedSource([]), detector_factory=broken_detector,
                executor_factory=executor_factory, client=ApiClient(api.base_url), outbox=outbox)
    finally:
        outbox.close()
    assert executor_calls == []


def test_construction_shuts_the_pool_down_when_setup_fails_after_the_dispatcher(tmp_path, api, monkeypatch):
    """Se algo depois do pool (saúde, agregadores, runners) falhar na montagem, o pool não fica órfão."""
    import pytest

    import guarita.app as app_module
    from guarita.config import ApiConfig, CameraConfig, Config, DetectorConfig, OutboxConfig, PipelineConfig
    from guarita.dispatch import OcrDispatcher
    from guarita.outbox import Outbox

    shutdown_calls = []
    real_shutdown = OcrDispatcher.shutdown
    monkeypatch.setattr(OcrDispatcher, "shutdown",
                         lambda self: shutdown_calls.append(True) or real_shutdown(self))
    monkeypatch.setattr(app_module, "Health",
                         lambda: (_ for _ in ()).throw(RuntimeError("saúde não inicializa (teste)")))

    cfg = Config(
        api=ApiConfig(base_url=api.base_url), detector=DetectorConfig(),
        pipeline=PipelineConfig(failures_dir=str(tmp_path / "falhas")),
        outbox=OutboxConfig(path=str(tmp_path / "o.sqlite")),
        cameras=(CameraConfig("cam-01", "entrada", "", "KEY"),),
    )
    outbox = Outbox(cfg.outbox.path)

    def executor_factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: FakeEngine(),))

    try:
        with pytest.raises(RuntimeError, match="saúde não inicializa"):
            App(cfg, source_factory=lambda cam: ScriptedSource([]), detector_factory=lambda cam: ScriptedDetector([]),
                executor_factory=executor_factory, client=ApiClient(api.base_url), outbox=outbox)
    finally:
        outbox.close()
    assert shutdown_calls == [True]


def test_app_passes_on_frame_to_every_camera_runner(tmp_path, api):
    calls = []

    def hook(camera_id, img, detections):
        calls.append(camera_id)

    cfg = Config(
        api=ApiConfig(base_url=api.base_url), detector=DetectorConfig(),
        pipeline=PipelineConfig(failures_dir=str(tmp_path / "falhas")),
        outbox=OutboxConfig(path=str(tmp_path / "o.sqlite")),
        cameras=(CameraConfig("cam-01", "entrada", "", "KEY"),),
    )
    outbox = Outbox(cfg.outbox.path)

    def executor_factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: FakeEngine(),))

    app = App(cfg, source_factory=lambda cam: ScriptedSource([]), detector_factory=lambda cam: ScriptedDetector([]),
              executor_factory=executor_factory, client=ApiClient(api.base_url), outbox=outbox, on_frame=hook)
    try:
        assert app._runners["cam-01"]._on_frame is hook
    finally:
        outbox.close()


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
