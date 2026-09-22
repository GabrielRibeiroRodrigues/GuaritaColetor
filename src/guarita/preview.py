from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .models import Detection

log = logging.getLogger(__name__)


class FrameStore:
    """Guarda só o quadro mais recente de cada câmera (nunca uma fila — não acumula memória)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[np.ndarray, list[Detection]]] = {}

    def update(self, camera_id: str, frame: np.ndarray, detections: list[Detection]) -> None:
        with self._lock:
            self._latest[camera_id] = (frame, detections)

    def snapshot(self, camera_id: str) -> tuple[np.ndarray, list[Detection]] | None:
        with self._lock:
            return self._latest.get(camera_id)


def _placeholder_jpeg() -> bytes:
    img = np.full((360, 640, 3), 40, np.uint8)
    cv2.putText(img, "aguardando sinal...", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (200, 200, 200), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> bytes:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def _page(camera_id: str) -> bytes:
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Guarita — {camera_id}</title>
<style>body{{background:#111;color:#eee;font-family:sans-serif;text-align:center}}
img{{max-width:95vw;margin-top:1em;border:1px solid #444}}</style>
</head><body>
<h1>{camera_id}</h1>
<img id="frame" src="/preview/{camera_id}/frame.jpg">
<script>
setInterval(function () {{
  document.getElementById('frame').src = '/preview/{camera_id}/frame.jpg?t=' + Date.now();
}}, 500);
</script>
</body></html>"""
    return html.encode("utf-8")


def _index(camera_ids: list[str]) -> bytes:
    items = "".join(f'<li><a href="/preview/{cid}">{cid}</a></li>' for cid in camera_ids)
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>Guarita — câmeras</title>
<style>body{{background:#111;color:#eee;font-family:sans-serif}}</style></head>
<body><h1>Câmeras</h1><ul>{items}</ul></body></html>"""
    return html.encode("utf-8")


class PreviewServer:
    """Página local (sem autenticação) para ver, ao vivo, a câmera enxergando a placa."""

    def __init__(self, frame_store: FrameStore, camera_ids: list[str], host: str = "0.0.0.0", port: int = 0):
        self._frame_store = frame_store
        self._camera_ids = camera_ids
        store = frame_store
        ids = camera_ids

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - nome exigido pelo http.server
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", _index(ids))
                elif path.startswith("/preview/") and path.endswith("/frame.jpg"):
                    camera_id = path[len("/preview/"):-len("/frame.jpg")]
                    if camera_id not in ids:
                        self._send(404, "text/plain; charset=utf-8", b"camera desconhecida")
                        return
                    found = store.snapshot(camera_id)
                    body = _draw_detections(*found) if found else _placeholder_jpeg()
                    self._send(200, "image/jpeg", body)
                elif path.startswith("/preview/"):
                    camera_id = path[len("/preview/"):]
                    if camera_id not in ids:
                        self._send(404, "text/plain; charset=utf-8", b"camera desconhecida")
                        return
                    self._send(200, "text/html; charset=utf-8", _page(camera_id))
                else:
                    self._send(404, "text/plain; charset=utf-8", b"nao encontrado")

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="preview-http", daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> None:
        self._thread.start()
        log.info("Preview web em http://%s:%d/", self._server.server_address[0], self.port)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
