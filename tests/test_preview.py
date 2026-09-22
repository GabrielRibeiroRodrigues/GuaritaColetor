import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from guarita.models import Detection, Reading
from guarita.preview import FrameStore, PreviewServer, ReadingLog, Recorder


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


def reading(plate="ABC1234", ts=1_700_000_000.0, camera_id="cam-01"):
    return Reading(camera_id=camera_id, external_id="e1", plate=plate, confidence=0.9, timestamp=ts,
                    direction="entrada", image_jpeg=b"", processing_ms=100)


def start_server(frame_store, camera_ids=("cam-01",), reading_log=None, recorder=None):
    server = PreviewServer(frame_store, list(camera_ids), host="127.0.0.1", port=0,
                            reading_log=reading_log, recorder=recorder)
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


# ─── ReadingLog ─────────────────────────────────────────────────────────────


def test_reading_log_is_empty_before_anything_is_added():
    log = ReadingLog()
    assert log.recent("cam-01") == []


def test_reading_log_returns_newest_first():
    log = ReadingLog()
    log.add(reading("AAA0001", ts=1.0))
    log.add(reading("AAA0002", ts=2.0))
    assert [r.plate for r in log.recent("cam-01")] == ["AAA0002", "AAA0001"]


def test_reading_log_keeps_only_the_last_ten():
    log = ReadingLog()
    for i in range(15):
        log.add(reading(f"AAA{i:04d}", ts=float(i)))
    recent = log.recent("cam-01")
    assert len(recent) == 10
    assert recent[0].plate == "AAA0014"  # a mais nova
    assert recent[-1].plate == "AAA0005"  # a 10ª mais nova


def test_reading_log_is_per_camera():
    log = ReadingLog()
    log.add(reading("AAA0001", camera_id="cam-01"))
    log.add(reading("BBB0002", camera_id="cam-02"))
    assert [r.plate for r in log.recent("cam-01")] == ["AAA0001"]
    assert [r.plate for r in log.recent("cam-02")] == ["BBB0002"]


# ─── Recorder ───────────────────────────────────────────────────────────────


def test_recorder_is_not_recording_before_start(tmp_path):
    recorder = Recorder(str(tmp_path))
    assert recorder.is_recording("cam-01") is False


def test_recorder_writes_frames_to_a_video_file(tmp_path):
    recorder = Recorder(str(tmp_path), fps=5.0)
    assert recorder.start("cam-01") is True
    assert recorder.is_recording("cam-01") is True
    for _ in range(3):
        recorder.write("cam-01", frame())
    recorder.stop("cam-01")
    assert recorder.is_recording("cam-01") is False

    files = list((tmp_path / "cam-01").glob("*.mp4"))
    assert len(files) == 1
    assert files[0].stat().st_size > 0
    cap = cv2.VideoCapture(str(files[0]))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == 3


def test_recorder_write_without_start_does_nothing(tmp_path):
    recorder = Recorder(str(tmp_path))
    recorder.write("cam-01", frame())  # não deve lançar
    assert not (tmp_path / "cam-01").exists()


def test_recorder_starting_twice_keeps_recording_without_erroring(tmp_path):
    recorder = Recorder(str(tmp_path))
    assert recorder.start("cam-01") is True
    assert recorder.start("cam-01") is False  # já estava gravando; não reabre o arquivo
    recorder.write("cam-01", frame())
    recorder.stop("cam-01")
    assert len(list((tmp_path / "cam-01").glob("*.mp4"))) == 1


def test_recorder_stop_when_not_recording_is_a_safe_no_op(tmp_path):
    recorder = Recorder(str(tmp_path))
    recorder.stop("cam-01")  # não deve lançar


def test_recorder_auto_stops_after_the_max_duration(tmp_path):
    clock = [0.0]
    recorder = Recorder(str(tmp_path), fps=5.0, max_duration_s=1800.0, clock=lambda: clock[0])
    recorder.start("cam-01")
    clock[0] = 1801.0
    recorder.write("cam-01", frame())  # a escrita que cruza o limite fecha o arquivo
    assert recorder.is_recording("cam-01") is False


# ─── Rotas novas do PreviewServer: leituras e gravação ───────────────────────


def test_readings_json_is_empty_list_before_any_reading():
    server = start_server(FrameStore(), reading_log=ReadingLog())
    try:
        status, content_type, body = get(server, "/preview/cam-01/readings.json")
        assert status == 200
        assert content_type == "application/json"
        assert json.loads(body) == []
    finally:
        server.stop()


def test_readings_json_reflects_the_log():
    log = ReadingLog()
    log.add(reading("ABC1234", ts=1_700_000_000.0))
    server = start_server(FrameStore(), reading_log=log)
    try:
        _, _, body = get(server, "/preview/cam-01/readings.json")
        data = json.loads(body)
        assert len(data) == 1
        assert data[0]["plate"] == "ABC1234"
        assert "timestamp" in data[0]
    finally:
        server.stop()


def test_camera_page_includes_the_record_button_and_readings_list():
    server = start_server(FrameStore())
    try:
        _, _, body = get(server, "/preview/cam-01")
        assert b"readings.json" in body
        assert b"record/status" in body  # a rota de start/stop é montada em JS: 'record/' + action
        assert b"toggleRecording" in body and b"Gravar" in body
    finally:
        server.stop()


def post(server, path):
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers.get("Content-Type"), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type"), err.read()


def test_record_start_and_stop_via_http(tmp_path):
    recorder = Recorder(str(tmp_path), fps=5.0)
    server = start_server(FrameStore(), recorder=recorder)
    try:
        status, content_type, body = post(server, "/preview/cam-01/record/start")
        assert status == 200 and content_type == "application/json"
        assert json.loads(body) == {"recording": True}
        assert recorder.is_recording("cam-01") is True

        status, _, body = get(server, "/preview/cam-01/record/status")
        assert status == 200
        data = json.loads(body)
        assert data["recording"] is True
        assert data["seconds"] >= 0

        status, _, body = post(server, "/preview/cam-01/record/stop")
        assert status == 200
        assert json.loads(body) == {"recording": False}
        assert recorder.is_recording("cam-01") is False
    finally:
        server.stop()


def test_record_routes_404_for_unknown_camera(tmp_path):
    server = start_server(FrameStore(), recorder=Recorder(str(tmp_path)))
    try:
        assert post(server, "/preview/nao-existe/record/start")[0] == 404
        assert post(server, "/preview/nao-existe/record/stop")[0] == 404
        assert get(server, "/preview/nao-existe/record/status")[0] == 404
        assert get(server, "/preview/nao-existe/readings.json")[0] == 404
    finally:
        server.stop()
