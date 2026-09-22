import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from guarita.models import Detection
from guarita.preview import FrameStore, PreviewServer


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


def start_server(frame_store, camera_ids=("cam-01",)):
    server = PreviewServer(frame_store, list(camera_ids), host="127.0.0.1", port=0)
    server.start()
    return server


def get(server, path):
    url = f"http://127.0.0.1:{server.port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.headers.get("Content-Type"), resp.read()
    except urllib.error.HTTPError as err:
        # 4xx/5xx: urlopen levanta em vez de devolver — err já é a resposta (tem status/headers/read()).
        return err.code, err.headers.get("Content-Type"), err.read()


def test_index_lists_the_cameras():
    store = FrameStore()
    server = start_server(store, ["cam-01", "cam-02"])
    try:
        status, content_type, body = get(server, "/")
        assert status == 200
        assert "text/html" in content_type
        assert b"cam-01" in body and b"cam-02" in body
        assert b"/preview/cam-01" in body and b"/preview/cam-02" in body
    finally:
        server.stop()


def test_camera_page_is_html_with_an_auto_refreshing_image():
    store = FrameStore()
    server = start_server(store)
    try:
        status, content_type, body = get(server, "/preview/cam-01")
        assert status == 200
        assert "text/html" in content_type
        assert b"cam-01" in body
        assert b"/preview/cam-01/frame.jpg" in body
        assert b"setInterval" in body
    finally:
        server.stop()


def test_frame_jpg_before_any_update_shows_a_waiting_placeholder():
    store = FrameStore()
    server = start_server(store)
    try:
        status, content_type, body = get(server, "/preview/cam-01/frame.jpg")
        assert status == 200
        assert content_type == "image/jpeg"
        assert body[:2] == b"\xff\xd8"  # cabeçalho JPEG
    finally:
        server.stop()


def test_frame_jpg_after_update_reflects_the_latest_frame():
    store = FrameStore()
    server = start_server(store)
    try:
        store.update("cam-01", frame(9), [Detection(1, (1, 1, 5, 5), 0.9)])
        status, content_type, body = get(server, "/preview/cam-01/frame.jpg")
        assert status == 200
        assert content_type == "image/jpeg"
        assert body[:2] == b"\xff\xd8"
    finally:
        server.stop()


def test_unknown_camera_and_unknown_path_are_404():
    store = FrameStore()
    server = start_server(store)
    try:
        assert get(server, "/preview/nao-existe")[0] == 404
        assert get(server, "/preview/nao-existe/frame.jpg")[0] == 404
        assert get(server, "/rota-qualquer")[0] == 404
    finally:
        server.stop()
