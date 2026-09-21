import numpy as np

from fakes import FakeEngine
from guarita.ocr import worker
from guarita.ocr.engine import read_candidates, rotate_crop, to_ocr_input


def color_crop(h=20, w=60):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_rotate_zero_returns_an_equal_copy():
    img = color_crop()
    out = rotate_crop(img, 0)
    assert np.array_equal(out, img)
    assert out is not img


def test_rotate_15_keeps_shape_and_changes_pixels():
    img = color_crop()
    out = rotate_crop(img, 15)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_to_ocr_input_from_bgr_and_gray():
    bgr = np.zeros((10, 30, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # vermelho puro
    out = to_ocr_input(bgr)
    assert out.shape == (10, 30, 3)
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    gray = np.full((10, 30), 100, dtype=np.uint8)
    assert to_ocr_input(gray).shape == (10, 30, 3)


def test_read_candidates_runs_every_angle_and_filters_by_confidence():
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.05)])
    out = read_candidates(engine, color_crop(), angles=(0, 15), min_conf=0.1)
    assert out == [("ABC1234", 0.9)]
    assert len(engine.inputs) == 2                       # 0° e 15° sempre
    assert all(i.ndim == 3 and i.shape[2] == 3 for i in engine.inputs)


def test_read_candidates_skips_unreadable_crops():
    engine = FakeEngine([(None, None), ("ABC1234", 0.8)])
    assert read_candidates(engine, color_crop(), (0, 15), 0.1) == [("ABC1234", 0.8)]


def test_process_job_reads_every_crop_at_every_angle():
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.8), (None, None), ("ABC1235", 0.7)])
    worker.init_worker(lambda: engine)
    try:
        result = worker.process_job(worker.OcrJob([color_crop(), color_crop()], (0.0, 15.0), 0.1))
    finally:
        worker.init_worker(lambda: None)
    assert len(engine.inputs) == 4
    assert result.candidates == [("ABC1234", 0.9), ("ABC1234", 0.8), ("ABC1235", 0.7)]
    assert result.elapsed_ms >= 0


def test_paddle_engine_swallows_unexpected_output_formats():
    from guarita.ocr.engine import PaddleEngine
    engine = PaddleEngine.__new__(PaddleEngine)             # sem importar o paddleocr
    engine._ocr = type("O", (), {"ocr": lambda self, img, cls=False: [[("formato", "inesperado")]]})()
    assert engine.read(np.zeros((10, 30, 3), np.uint8)) == (None, None)


def test_warmup_requires_an_initialised_engine():
    import pytest
    worker.init_worker(lambda: None)
    with pytest.raises(RuntimeError):
        worker.warmup()
    worker.init_worker(lambda: FakeEngine())
    try:
        assert worker.warmup() is True
    finally:
        worker.init_worker(lambda: None)
