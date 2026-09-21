import numpy as np
import pytest

from fakes import FakeEngine, ScriptedDetector, ScriptedSource
from concurrent.futures import ThreadPoolExecutor

from guarita.capture import Frame
from guarita.config import CameraConfig, Config, ApiConfig, DetectorConfig, OutboxConfig, PipelineConfig
from guarita.evaluation import legacy_rows, read_rows, run_offline, score, write_rows
from guarita.models import Detection
from guarita.ocr import worker


def test_score_counts_correct_wrong_missed_and_extra():
    truth = [(10.0, "ABC1234"), (30.0, "DEF5678"), (60.0, "GHI9012")]
    results = [(11.0, "ABC1234"), (12.0, "ABC1234"),      # certo + duplicata (extra)
               (29.0, "DEF5670"),                         # leu outra placa
               (200.0, "ZZZ9999")]                        # sem veículo rotulado (extra)
    result = score(truth, results, window=5.0)
    assert (result.total, result.correct, result.wrong, result.missed, result.extra) == (3, 1, 1, 1, 2)
    assert result.accuracy == pytest.approx(1 / 3)


def test_score_window_boundary_and_order_independence():
    assert score([(10.0, "ABC1234")], [(15.0, "ABC1234")], window=5.0).correct == 1
    assert score([(10.0, "ABC1234")], [(15.1, "ABC1234")], window=5.0).missed == 1
    assert score([(20.0, "B"), (10.0, "A")], [(10.5, "A"), (20.5, "B")]).correct == 2


def test_score_with_no_truth():
    result = score([], [(1.0, "ABC1234")])
    assert result.total == 0 and result.extra == 1 and result.accuracy == 0.0


def test_legacy_rows_convert_frames_to_seconds():
    rows = [(250, "ABC1234", 0.9), (-1, "XYZ", 0.1), (0, "DEF5678", 0.8)]
    assert legacy_rows(rows, fps=25.0) == [(10.0, "ABC1234", 0.9), (0.0, "DEF5678", 0.8)]


def test_csv_round_trip(tmp_path):
    path = tmp_path / "r.csv"
    write_rows(str(path), [(1.5, "ABC1234", 0.9), (2.0, "DEF5678", 0.7)])
    assert read_rows(str(path)) == [(1.5, "ABC1234"), (2.0, "DEF5678")]


def test_run_offline_collects_readings_from_a_source():
    cfg = Config(
        api=ApiConfig(base_url="x"), detector=DetectorConfig(),
        pipeline=PipelineConfig(max_crops=2, close_after_s=1.5), outbox=OutboxConfig(),
        cameras=(CameraConfig("cam-01", "entrada", "", ""),))
    blank = np.zeros((100, 200, 3), np.uint8)
    frames = [Frame(blank, t) for t in (0.0, 0.5, 1.0, 2.5)]
    box = (10, 10, 60, 40)
    detections = [[Detection(1, box, 0.9)], [Detection(1, box, 0.9)], [], []]
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.8), ("ABC1234", 0.85), (None, None)])

    def executor_factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: engine,))

    rows, frame_count = run_offline(cfg, cfg.cameras[0], ScriptedSource(frames), ScriptedDetector(detections),
                                    executor_factory)
    assert frame_count == 4
    assert len(rows) == 1
    assert rows[0][0] == 0.0 and rows[0][1] == "ABC1234"
    assert rows[0][2] == pytest.approx(0.85)
