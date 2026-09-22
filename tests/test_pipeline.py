import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fakes import FakeEngine, ScriptedDetector, ScriptedSource
from guarita.capture import Frame
from guarita.config import CameraConfig, PipelineConfig
from guarita.dispatch import OcrDispatcher, Outcome
from guarita.models import ClosedTrack, Crop, Detection
from guarita.ocr import worker
from guarita.ocr.worker import OcrResult
from guarita.pipeline import Aggregator, CameraRunner, FailureStore, ListSink, drain_results

CAMERA = CameraConfig(id="cam-01", direction="entrada", rtsp_url="", api_key="")


def track(ts=5.0, started=4.0, tid=7):
    crop = Crop(np.full((10, 30, 3), 128, np.uint8), ts=ts, score=1.0)
    return ClosedTrack("cam-01", tid, started, ts + 2, [crop])


def outcome(candidates, ts=5.0, tid=7):
    return Outcome(track(ts, tid=tid), OcrResult(candidates, 40), None, 350)


def test_aggregator_builds_a_reading_from_the_vote():
    agg = Aggregator(CAMERA, PipelineConfig(), None)
    reading = agg.handle(outcome([("ABC1234", 0.9), ("ABC1234", 0.7)]))
    assert reading.plate == "ABC1234"
    assert reading.confidence == pytest.approx(0.8)
    assert reading.camera_id == "cam-01" and reading.direction == "entrada"
    assert reading.external_id == "4000-7"
    assert reading.timestamp == 5.0
    assert reading.processing_ms == 350
    assert reading.image_jpeg[:2] == b"\xff\xd8"


def test_aggregator_returns_none_without_a_valid_plate():
    agg = Aggregator(CAMERA, PipelineConfig(), None)
    assert agg.handle(outcome([])) is None
    assert agg.handle(outcome([("ABCD123", 0.9)])) is None


def test_aggregator_saves_failures_for_timeouts_and_errors(tmp_path):
    store = FailureStore(str(tmp_path / "falhas"), ttl_h=24)
    agg = Aggregator(CAMERA, PipelineConfig(), store)
    assert agg.handle(Outcome(track(), None, "timeout", 10_000)) is None
    assert (tmp_path / "falhas" / "4000-7-timeout.jpg").exists()


def test_aggregator_fuses_the_same_plate_within_the_window():
    agg = Aggregator(CAMERA, PipelineConfig(fuse_window_s=20.0), None)
    assert agg.handle(outcome([("ABC1234", 0.9)], ts=100.0, tid=1)) is not None
    assert agg.handle(outcome([("ABC1234", 0.9)], ts=110.0, tid=2)) is None       # 10 s depois
    assert agg.handle(outcome([("ABC1235", 0.9)], ts=111.0, tid=3)) is not None   # outra placa
    assert agg.handle(outcome([("ABC1234", 0.9)], ts=125.0, tid=4)) is not None   # já passou a janela


def test_failure_store_purges_old_files(tmp_path):
    store = FailureStore(str(tmp_path), ttl_h=1)
    old = store.save(track(tid=1), "erro")
    new = store.save(track(tid=2), "erro")
    past = time.time() - 2 * 3600
    os.utime(old, (past, past))
    assert store.purge() == 1
    assert not os.path.exists(old) and os.path.exists(new)


def test_step_reports_idle_and_done():
    runner = CameraRunner(CAMERA, PipelineConfig(), ScriptedSource([]), ScriptedDetector([]), _dispatcher(FakeEngine()))
    assert runner.step() == "done"
    idle_source = ScriptedSource([])
    idle_source.read = lambda: None            # sem quadro, mas sem acabar
    runner = CameraRunner(CAMERA, PipelineConfig(), idle_source, ScriptedDetector([]), _dispatcher(FakeEngine()))
    assert runner.step() == "idle"


def _dispatcher(engine):
    def factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: engine,))
    return OcrDispatcher(factory, timeout_s=10.0)


def test_on_frame_hook_is_called_with_the_frame_and_detections():
    calls = []
    frame_img = np.zeros((10, 10, 3), np.uint8)
    dets = [Detection(1, (0, 0, 5, 5), 0.9)]
    runner = CameraRunner(
        CAMERA, PipelineConfig(), ScriptedSource([Frame(frame_img, 0.0)]), ScriptedDetector([dets]),
        _dispatcher(FakeEngine()), on_frame=lambda cid, img, d: calls.append((cid, img, d)))
    assert runner.step() == "frame"
    assert len(calls) == 1
    camera_id, img, detections = calls[0]
    assert camera_id == "cam-01"
    assert img is frame_img
    assert detections == dets


def test_on_frame_hook_is_not_required():
    # nenhum on_frame passado: comportamento igual ao de antes, sem lançar
    runner = CameraRunner(CAMERA, PipelineConfig(), ScriptedSource([]), ScriptedDetector([]), _dispatcher(FakeEngine()))
    assert runner.step() == "done"


def test_end_to_end_reading_lands_in_the_sink():
    cfg = PipelineConfig(max_crops=2, close_after_s=1.5)
    blank = np.zeros((100, 200, 3), np.uint8)
    frames = [Frame(blank, t) for t in (0.0, 0.5, 1.0, 2.5)]
    box = (10, 10, 60, 40)
    detections = [[Detection(1, box, 0.9)], [Detection(1, box, 0.9)], [], []]
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.8), ("ABC1234", 0.85), (None, None)])
    dispatcher = _dispatcher(engine)
    aggregators = {"cam-01": Aggregator(CAMERA, cfg, None)}
    sink = ListSink()
    runner = CameraRunner(CAMERA, cfg, ScriptedSource(frames), ScriptedDetector(detections), dispatcher)

    while runner.step() != "done":
        pass
    runner.finish()
    for _ in range(100):
        drain_results(dispatcher, aggregators, sink)
        if dispatcher.pending_count() == 0:
            break
        time.sleep(0.05)
    dispatcher.shutdown()

    assert len(engine.inputs) == 4                    # 2 recortes × (0° e 15°)
    assert len(sink.readings) == 1
    reading = sink.readings[0]
    assert reading.plate == "ABC1234"
    assert reading.confidence == pytest.approx(0.85)
    assert reading.external_id == "0-1"
    assert reading.timestamp == 0.0
    assert reading.direction == "entrada"


def test_failure_store_survives_oserror(tmp_path):
    blocker = tmp_path / "arquivo"
    blocker.write_text("x")
    store = FailureStore(str(blocker / "sub"), ttl_h=24)      # makedirs falha (pai é um arquivo)
    assert store.save(track(), "timeout") == ""


def test_drain_results_keeps_the_rest_of_the_batch_when_one_outcome_fails():
    class Fixed:
        def poll(self):
            return [outcome([("ABC1234", 0.9)], tid=1), outcome([("ABC1234", 0.9)], tid=2),
                    outcome([("XYZ9876", 0.9)], tid=3)]

    class Boom(Aggregator):
        calls = 0

        def handle(self, out):
            Boom.calls += 1
            if out.track.track_id == 1:
                raise RuntimeError("boom")
            return super().handle(out)

    sink = ListSink()
    accepted = drain_results(Fixed(), {"cam-01": Boom(CAMERA, PipelineConfig(), None)}, sink)
    assert Boom.calls == 3
    assert [r.plate for r in accepted] == ["ABC1234", "XYZ9876"]
