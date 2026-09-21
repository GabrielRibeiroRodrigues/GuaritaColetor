import json
from dataclasses import replace
import threading
import time

import pytest

from fakes import make_reading
from guarita.outbox import Outbox
from guarita.sender import ApiClient, deliver_next, run_sender

KEYS = {"cam-01": "KEY"}


@pytest.fixture
def box(tmp_path):
    outbox = Outbox(str(tmp_path / "o.sqlite"))
    yield outbox
    outbox.close()


def deliver(box, client, keys=KEYS):
    return deliver_next(box, client, keys, now=time.time() + 1)


def test_idle_when_queue_is_empty(box, api):
    assert deliver(box, ApiClient(api.base_url)).outcome == "idle"


def test_delivers_multipart_with_image(box, api):
    box.add(make_reading(1))
    result = deliver(box, ApiClient(api.base_url))
    assert result.outcome == "delivered"
    req = api.requests[0]
    assert req["path"] == "/api/ingest/readings"
    assert req["headers"]["x-api-key"] == "KEY"
    assert "multipart/form-data" in req["headers"]["content-type"]
    body = req["body"]
    for fragment in (b'name="external_id"', b"ext-1", b'name="plate"', b"ABC1234", b'name="direction"',
                     b"entrada", b'name="confidence"', b"90.0", b'name="processing_ms"', b"120",
                     b'name="image"', b"jpg"):
        assert fragment in body
    assert box.counts()["delivered"] == 1


def test_timestamp_is_iso_8601_with_offset(box, api):
    box.add(make_reading(1))
    deliver(box, ApiClient(api.base_url))
    body = api.requests[0]["body"].decode("latin-1")
    stamp = body.split('name="timestamp"')[1].split("\r\n")[2]
    assert stamp[4] == "-" and stamp[10] == "T" and (stamp[-6] in "+-" or stamp.endswith("Z"))


def test_sends_json_when_there_is_no_image(box, api):
    box.add(make_reading(1, image=b""))
    assert deliver(box, ApiClient(api.base_url)).outcome == "delivered"
    req = api.requests[0]
    assert req["headers"]["content-type"].startswith("application/json")
    payload = json.loads(req["body"])
    assert payload["plate"] == "ABC1234"
    assert payload["confidence"] == pytest.approx(90.0)
    assert payload["processing_ms"] == 120
    assert payload["direction"] == "entrada"
    assert "image" not in payload


@pytest.mark.parametrize("status", [500, 503, 408, 429])
def test_server_errors_are_retried_with_backoff(box, api, status):
    box.add(make_reading(1))
    api.responses = [status]
    result = deliver(box, ApiClient(api.base_url))
    assert result.outcome == "retry"
    assert box.counts()["pending"] == 1
    assert box.next_due(now=time.time()) is None  # aguardando o atraso


def test_network_error_is_retried(box):
    box.add(make_reading(1))
    result = deliver(box, ApiClient("http://127.0.0.1:9/api", timeout_s=1))
    assert result.outcome == "retry"
    assert box.pending_count() == 1


def test_client_errors_are_final(box, api):
    box.add(make_reading(1))
    api.responses = [400]
    assert deliver(box, ApiClient(api.base_url)).outcome == "failed"
    assert box.counts()["failed"] == 1


def test_401_pauses_but_keeps_the_reading(box, api):
    box.add(make_reading(1))
    api.responses = [401]
    assert deliver(box, ApiClient(api.base_url)).outcome == "auth"
    assert box.pending_count() == 1
    assert box.next_due(now=time.time() + 1).attempts == 0


def test_413_drops_the_image_and_resends_as_json(box, api):
    box.add(make_reading(1))
    api.responses = [413]
    assert deliver(box, ApiClient(api.base_url)).outcome == "retry"
    assert box.next_due(now=time.time() + 1).image is None
    assert deliver(box, ApiClient(api.base_url)).outcome == "delivered"
    assert api.requests[1]["headers"]["content-type"].startswith("application/json")


def test_camera_without_key_is_failed(box, api):
    box.add(make_reading(1, camera_id="cam-99"))
    assert deliver(box, ApiClient(api.base_url)).outcome == "failed"
    assert api.requests == []


def test_heartbeat(api):
    client = ApiClient(api.base_url)
    assert client.heartbeat("KEY") is True
    assert api.requests[0]["path"] == "/api/ingest/heartbeat"
    assert api.requests[0]["headers"]["x-api-key"] == "KEY"
    api.responses = [401]
    assert client.heartbeat("KEY") is False
    assert ApiClient("http://127.0.0.1:9/api", timeout_s=1).heartbeat("KEY") is False


def test_epoch_zero_timestamp_is_delivered(box, api):
    box.add(replace(make_reading(1), timestamp=0.0))
    assert deliver(box, ApiClient(api.base_url)).outcome == "delivered"


def test_sender_thread_survives_unexpected_errors(box):
    box.add(make_reading(1))
    calls = []

    class Flaky:
        def post_reading(self, api_key, row):
            calls.append(row.id)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return 201, "ok"

    stop = threading.Event()
    thread = threading.Thread(
        target=run_sender, args=(box, Flaky(), KEYS, stop), kwargs={"idle_s": 0.05}, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and box.counts()["delivered"] < 1:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=5)
    assert box.counts()["delivered"] == 1
    assert len(calls) >= 2
