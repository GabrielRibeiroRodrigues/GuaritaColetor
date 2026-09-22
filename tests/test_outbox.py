import sqlite3
import time

import pytest

from fakes import make_reading
from guarita.outbox import Outbox


@pytest.fixture
def box(tmp_path):
    outbox = Outbox(str(tmp_path / "outbox.sqlite"))
    yield outbox
    outbox.close()


def test_add_and_next_due_in_insertion_order(box):
    assert box.add(make_reading(1)) is True
    assert box.add(make_reading(2)) is True
    row = box.next_due()
    assert (row.external_id, row.plate, row.camera_id, row.direction) == ("ext-1", "ABC1234", "cam-01", "entrada")
    assert row.confidence == pytest.approx(0.9)
    assert row.image == b"jpg"
    assert row.attempts == 0


def test_duplicate_external_id_per_camera_is_ignored(box):
    assert box.add(make_reading(1)) is True
    assert box.add(make_reading(1)) is False
    assert box.add(make_reading(1, camera_id="cam-02")) is True
    assert box.pending_count() == 2


def test_retry_backoff_grows_and_caps(box):
    box.add(make_reading(1))
    now = time.time()
    delays = []
    for _ in range(11):
        row = box.next_due(now=now + 10_000)
        delays.append(box.mark_retry(row.id, "erro", now=now))
    assert delays[:9] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert delays[9:] == [300, 300]


def test_retry_hides_row_until_due(box):
    box.add(make_reading(1))
    now = 1000.0
    row = box.next_due(now=now)
    box.mark_retry(row.id, "erro", now=now)  # próxima tentativa em 1 s
    assert box.next_due(now=now + 0.5) is None
    assert box.next_due(now=now + 1.5).attempts == 1


def test_delivered_and_failed_leave_the_queue(box):
    box.add(make_reading(1))
    box.add(make_reading(2))
    box.mark_delivered(box.next_due().id)
    box.mark_failed(box.next_due().id, "422")
    assert box.next_due() is None
    assert box.counts() == {"pending": 0, "delivered": 1, "failed": 1}


def test_drop_image(box):
    box.add(make_reading(1))
    box.drop_image(box.next_due().id)
    assert box.next_due().image is None


def test_survives_reopen(tmp_path):
    path = str(tmp_path / "o.sqlite")
    first = Outbox(path)
    first.add(make_reading(1))
    first.close()
    second = Outbox(path)
    assert second.next_due().external_id == "ext-1"
    second.close()


def test_trim_drops_oldest_images_beyond_max_pending(tmp_path):
    path = str(tmp_path / "o.sqlite")
    small = Outbox(path, max_pending=3)
    for n in range(1, 6):
        small.add(make_reading(n))
    rows = sqlite3.connect(path).execute("SELECT id, image IS NULL FROM outbox ORDER BY id").fetchall()
    assert [r[1] for r in rows] == [1, 1, 0, 0, 0]  # texto de todas mantido, imagem só das 3 mais novas
    assert small.pending_count() == 5
    small.close()


def test_purge_delivered(box):
    box.add(make_reading(1))
    box.add(make_reading(2))
    box.mark_delivered(box.next_due().id, now=1000.0)
    assert box.purge_delivered(older_than_s=3600, now=1000.0 + 3601) == 1
    assert box.counts()["delivered"] == 0
    assert box.pending_count() == 1


def test_mark_unexpected_backs_off_and_does_not_touch_attempts(box):
    box.add(make_reading(1))
    row_id = box.next_due().id
    now = 1000.0
    delays = []
    for n in range(1, 10):
        assert box.mark_unexpected(row_id, "erro", now=now) is False
        row = box.next_due(now=now + 1e6)
        assert row.unexpected_errors == n
        assert row.attempts == 0
        delays.append(row.unexpected_errors)
        now += 1e6
    assert box.mark_unexpected(row_id, "erro final", now=now) is True  # 10ª: falha de vez
    assert box.next_due(now=now + 1) is None
    assert box.counts() == {"pending": 0, "delivered": 0, "failed": 1}


def test_mark_unexpected_and_mark_retry_count_separately(box):
    box.add(make_reading(1))
    row_id = box.next_due().id
    now = 1000.0
    box.mark_retry(row_id, "rede", now=now)
    now += 1e6
    box.mark_unexpected(row_id, "bug", now=now)
    now += 1e6
    row = box.next_due(now=now + 1)
    assert (row.attempts, row.unexpected_errors) == (1, 1)
