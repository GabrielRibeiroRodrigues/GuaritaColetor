from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass

from .models import Reading

BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 300.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  external_id TEXT NOT NULL,
  camera_id TEXT NOT NULL,
  plate TEXT NOT NULL,
  direction TEXT NOT NULL,
  ts REAL NOT NULL,
  confidence REAL NOT NULL,
  processing_ms INTEGER NOT NULL,
  image BLOB,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at REAL NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at REAL NOT NULL,
  delivered_at REAL,
  UNIQUE (camera_id, external_id)
);
CREATE INDEX IF NOT EXISTS outbox_due_idx ON outbox(status, next_attempt_at, id);
"""


@dataclass(frozen=True)
class OutboxRow:
    id: int
    external_id: str
    camera_id: str
    plate: str
    direction: str
    ts: float
    confidence: float
    processing_ms: int
    image: bytes | None
    attempts: int


class Outbox:
    def __init__(self, path: str, max_pending: int = 50000):
        self.max_pending = max_pending
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)

    def add(self, reading: Reading, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO outbox(external_id, camera_id, plate, direction, ts, confidence,"
                " processing_ms, image, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (reading.external_id, reading.camera_id, reading.plate, reading.direction, reading.timestamp,
                 reading.confidence, reading.processing_ms, reading.image_jpeg, now))
            inserted = cur.rowcount == 1
            if inserted:
                self._trim_images()
            return inserted

    def _trim_images(self) -> None:
        with_images = self._db.execute(
            "SELECT count(*) FROM outbox WHERE status='pending' AND image IS NOT NULL").fetchone()[0]
        excess = with_images - self.max_pending
        if excess > 0:
            self._db.execute(
                "UPDATE outbox SET image = NULL WHERE id IN ("
                "SELECT id FROM outbox WHERE status='pending' AND image IS NOT NULL ORDER BY id LIMIT ?)",
                (excess,))

    def next_due(self, now: float | None = None) -> OutboxRow | None:
        now = time.time() if now is None else now
        with self._lock:
            row = self._db.execute(
                "SELECT id, external_id, camera_id, plate, direction, ts, confidence, processing_ms, image, attempts"
                " FROM outbox WHERE status='pending' AND next_attempt_at <= ? ORDER BY id LIMIT 1",
                (now,)).fetchone()
        return OutboxRow(*row) if row else None

    def mark_delivered(self, row_id: int, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._db.execute(
                "UPDATE outbox SET status='delivered', image=NULL, delivered_at=? WHERE id=?", (now, row_id))

    def mark_retry(self, row_id: int, error: str, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            attempts = self._db.execute("SELECT attempts FROM outbox WHERE id=?", (row_id,)).fetchone()[0]
            delay = min(MAX_BACKOFF_S, BASE_BACKOFF_S * 2 ** attempts)
            self._db.execute(
                "UPDATE outbox SET attempts=attempts+1, next_attempt_at=?, last_error=? WHERE id=?",
                (now + delay, error[:300], row_id))
        return delay

    def mark_failed(self, row_id: int, error: str) -> None:
        with self._lock:
            self._db.execute("UPDATE outbox SET status='failed', last_error=? WHERE id=?", (error[:300], row_id))

    def drop_image(self, row_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE outbox SET image=NULL WHERE id=?", (row_id,))

    def counts(self) -> dict[str, int]:
        with self._lock:
            found = dict(self._db.execute("SELECT status, count(*) FROM outbox GROUP BY status").fetchall())
        return {s: found.get(s, 0) for s in ("pending", "delivered", "failed")}

    def pending_count(self) -> int:
        return self.counts()["pending"]

    def purge_delivered(self, older_than_s: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM outbox WHERE status='delivered' AND delivered_at < ?", (now - older_than_s,))
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()
