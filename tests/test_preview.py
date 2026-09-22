import threading

import numpy as np
import pytest

from guarita.models import Detection
from guarita.preview import FrameStore


def frame(value=1):
    return np.full((10, 10, 3), value, np.uint8)


def test_snapshot_is_none_before_the_first_update():
    store = FrameStore()
    assert store.snapshot("cam-01") is None


def test_update_then_snapshot_returns_what_was_stored():
    store = FrameStore()
    detections = [Detection(1, (0, 0, 5, 5), 0.9)]
    store.update("cam-01", frame(7), detections)
    img, dets = store.snapshot("cam-01")
    assert img[0, 0, 0] == 7
    assert dets == detections


def test_each_camera_has_its_own_latest_frame():
    store = FrameStore()
    store.update("cam-01", frame(1), [])
    store.update("cam-02", frame(2), [])
    assert store.snapshot("cam-01")[0][0, 0, 0] == 1
    assert store.snapshot("cam-02")[0][0, 0, 0] == 2


def test_a_second_update_replaces_the_first_not_accumulates():
    store = FrameStore()
    store.update("cam-01", frame(1), [Detection(1, (0, 0, 1, 1), 0.1)])
    store.update("cam-01", frame(2), [])
    img, dets = store.snapshot("cam-01")
    assert img[0, 0, 0] == 2
    assert dets == []


def test_concurrent_update_and_snapshot_do_not_raise():
    store = FrameStore()
    store.update("cam-01", frame(), [])
    errors = []

    def writer():
        for i in range(200):
            try:
                store.update("cam-01", frame(i % 255), [])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def reader():
        for _ in range(200):
            try:
                store.snapshot("cam-01")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
