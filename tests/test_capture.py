import itertools

import cv2
import numpy as np
import pytest

from guarita.capture import FileSource, FrameThrottle, RtspSource


class FakeCap:
    def __init__(self, grabs=(), opened=True, fps=25.0):
        self._grabs = list(grabs)
        self._opened = opened
        self._fps = fps
        self.retrieved = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def grab(self):
        return self._grabs.pop(0) if self._grabs else False

    def retrieve(self):
        self.retrieved += 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        self.released = True

    def get(self, prop):
        return self._fps if prop == cv2.CAP_PROP_FPS else 0.0


def test_throttle_limits_the_rate():
    throttle = FrameThrottle(8)  # intervalo de 0,125 s
    assert [throttle.should_process(t) for t in (0.0, 0.05, 0.13, 0.20, 0.26)] == [True, False, True, False, True]


def test_rtsp_source_skips_frames_with_grab_only():
    cap = FakeCap([True] * 10)
    ticks = itertools.count(0.0, 0.05)
    source = RtspSource("rtsp://x", 8, open_fn=lambda url: cap, clock=lambda: next(ticks))
    first = source.read()
    second = source.read()
    assert first.ts == 0.0
    assert second.ts == pytest.approx(0.15)
    assert cap.retrieved == 2  # os quadros descartados não foram decodificados por completo


def test_rtsp_source_reconnects_with_backoff():
    caps = [FakeCap([False]), FakeCap([True] * 3)]
    opened = []
    mono = [0.0]

    def open_fn(url):
        opened.append(url)
        return caps[len(opened) - 1]

    source = RtspSource("rtsp://x", 8, open_fn=open_fn, monotonic=lambda: mono[0])
    assert source.read() is None                 # 1º grab falha: conexão descartada
    assert caps[0].released
    mono[0] = 0.5
    assert source.read() is None and len(opened) == 1   # ainda em espera (1 s)
    mono[0] = 1.5
    frame = source.read()
    assert frame is not None and len(opened) == 2


def test_rtsp_open_failure_backs_off_exponentially():
    opened = []
    mono = [0.0]

    def open_fn(url):
        opened.append(url)
        return FakeCap(opened=False)

    source = RtspSource("rtsp://x", 8, open_fn=open_fn, monotonic=lambda: mono[0])
    source.read()                                # falha 1 -> próxima tentativa em 1 s
    mono[0] = 1.0
    source.read()                                # falha 2 -> próxima tentativa em 1 + 2 = 3 s
    mono[0] = 2.9
    source.read()
    assert len(opened) == 2
    mono[0] = 3.0
    source.read()
    assert len(opened) == 3


def test_file_source_steps_and_reports_video_time():
    cap = FakeCap([True] * 10, fps=25.0)
    source = FileSource("clip.mp4", target_fps=5, open_fn=lambda path: cap)  # 1 a cada 5 quadros
    assert source.fps == 25.0
    assert source.read().ts == 0.0
    assert source.read().ts == pytest.approx(0.2)
    assert source.read() is None
    assert source.exhausted is True
    assert cap.retrieved == 2


def test_file_source_missing_video():
    with pytest.raises(FileNotFoundError):
        FileSource("nada.mp4", 5, open_fn=lambda path: FakeCap(opened=False))


def test_open_rtsp_passes_explicit_timeouts(monkeypatch):
    import cv2
    from guarita import capture
    seen = {}
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: seen.setdefault("args", args))
    capture.open_rtsp("rtsp://x")
    url, backend, params = seen["args"]
    assert url == "rtsp://x" and backend == cv2.CAP_FFMPEG
    assert params == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
