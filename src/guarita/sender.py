from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .outbox import MAX_UNEXPECTED_ERRORS, Outbox, OutboxRow

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Delivery:
    outcome: str  # delivered | retry | failed | auth | idle
    detail: str = ""


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


AUTH_LOG_INTERVAL_S = 600.0
_auth_logged: dict[str, float] = {}


def _log_auth_failure(camera_id: str, now: float | None = None) -> None:
    """Loga o 401 no máximo uma vez por câmera a cada 10 min (não enche o log rotativo)."""
    now = time.monotonic() if now is None else now
    last = _auth_logged.get(camera_id)
    if last is None or now - last >= AUTH_LOG_INTERVAL_S:
        _auth_logged[camera_id] = now
        log.error("API recusou a chave da câmera %s (401); nova tentativa com espera crescente", camera_id)


class ApiClient:
    def __init__(self, base_url: str, timeout_s: float = 15.0, session: requests.Session | None = None):
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._session = session or requests.Session()

    def post_reading(self, api_key: str, row: OutboxRow) -> tuple[int, str]:
        headers = {"x-api-key": api_key}
        url = f"{self._base}/ingest/readings"
        if row.image:
            data = {
                "external_id": row.external_id, "plate": row.plate, "direction": row.direction,
                "timestamp": _iso(row.ts), "confidence": f"{row.confidence * 100:.1f}",
                "processing_ms": str(row.processing_ms),
            }
            files = {"image": ("plate.jpg", row.image, "image/jpeg")}
            resp = self._session.post(url, headers=headers, data=data, files=files, timeout=self._timeout)
        else:
            payload = {
                "external_id": row.external_id, "plate": row.plate, "direction": row.direction,
                "timestamp": _iso(row.ts), "confidence": round(row.confidence * 100, 1),
                "processing_ms": row.processing_ms,
            }
            resp = self._session.post(url, headers=headers, json=payload, timeout=self._timeout)
        return resp.status_code, resp.text[:200]

    def heartbeat(self, api_key: str) -> bool:
        try:
            resp = self._session.post(
                f"{self._base}/ingest/heartbeat", headers={"x-api-key": api_key}, timeout=self._timeout)
        except requests.RequestException as exc:
            log.warning("Heartbeat falhou: %s", exc)
            return False
        return resp.status_code == 204


def deliver_next(outbox: Outbox, client: ApiClient, api_keys: dict[str, str], now: float | None = None) -> Delivery:
    row = outbox.next_due(now)
    if row is None:
        return Delivery("idle")
    key = api_keys.get(row.camera_id)
    if not key:
        outbox.mark_failed(row.id, "sem chave de API para a câmera")
        return Delivery("failed", "sem chave de API")

    try:
        status, text = client.post_reading(key, row)
    except requests.RequestException as exc:
        outbox.mark_retry(row.id, f"rede: {exc}", now)
        return Delivery("retry", str(exc))
    except Exception as exc:  # noqa: BLE001 - linha envenenada não pode travar a cabeça da fila
        # Contado à parte de `attempts` (que é só de rede): uma linha que já esperou muitas vezes
        # pela rede não pode ser descartada no primeiro bug de serialização, e vice-versa.
        if outbox.mark_unexpected(row.id, f"inesperado: {exc!r}", now):
            log.error("Leitura %s descartada após %d erros inesperados: %r",
                      row.external_id, MAX_UNEXPECTED_ERRORS, exc)
            return Delivery("failed", f"inesperado: {exc!r}")
        log.warning("Erro inesperado ao enviar %s: %r", row.external_id, exc)
        return Delivery("retry", f"inesperado: {exc!r}")

    if status in (200, 201):
        outbox.mark_delivered(row.id, now)
        return Delivery("delivered")
    if status == 401:
        outbox.mark_retry(row.id, "401", now)  # espera crescente; as outras linhas/câmeras seguem
        _log_auth_failure(row.camera_id)
        return Delivery("auth", text)
    if status in (413, 415) and row.image:
        outbox.drop_image(row.id)
        return Delivery("retry", f"{status}: reenviando sem imagem")
    if status in (408, 429) or status >= 500:
        outbox.mark_retry(row.id, f"{status}: {text}", now)
        return Delivery("retry", str(status))
    outbox.mark_failed(row.id, f"{status}: {text}")
    log.error("Leitura %s recusada pela API (%s): %s", row.external_id, status, text)
    return Delivery("failed", f"{status}: {text}")


def run_sender(outbox: Outbox, client: ApiClient, api_keys: dict[str, str], stop: threading.Event,
               idle_s: float = 1.0) -> None:
    errors = 0
    while not stop.is_set():
        try:
            result = deliver_next(outbox, client, api_keys)
            errors = 0
            if result.outcome == "delivered":
                continue
            stop.wait(idle_s)
        except Exception:  # noqa: BLE001 - a thread de envio nunca pode morrer
            errors += 1
            log.exception("Erro inesperado no envio")
            stop.wait(min(60.0, idle_s * 2 ** min(errors, 6)))
