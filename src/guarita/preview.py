from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .models import Detection, Reading

log = logging.getLogger(__name__)

READING_LOG_SIZE = 10
DEFAULT_RECORD_FPS = 8.0
DEFAULT_MAX_RECORD_S = 1800.0  # 30 min — trava de segurança se alguém esquecer o botão ligado


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


class ReadingLog:
    """Guarda as últimas leituras de cada câmera (em memória; não é a fila de envio — aquela é sobre
    entrega à API, esta é só "o que apareceu na tela"). Mais nova primeiro; só as últimas 10."""

    def __init__(self, size: int = READING_LOG_SIZE):
        self._size = size
        self._lock = threading.Lock()
        self._recent: dict[str, list[Reading]] = {}

    def add(self, reading: Reading) -> None:
        with self._lock:
            items = self._recent.setdefault(reading.camera_id, [])
            items.insert(0, reading)
            del items[self._size:]

    def recent(self, camera_id: str) -> list[Reading]:
        with self._lock:
            return list(self._recent.get(camera_id, []))


class Recorder:
    """Grava vídeo por câmera, só enquanto ligado pelo botão da página — nunca contínuo. Para
    sozinho depois de `max_duration_s`, para o caso de alguém esquecer o botão ligado."""

    def __init__(self, base_dir: str, fps: float = DEFAULT_RECORD_FPS,
                 max_duration_s: float = DEFAULT_MAX_RECORD_S, clock=time.monotonic):
        self._base_dir = base_dir
        self._fps = fps
        self._max_duration_s = max_duration_s
        self._clock = clock
        self._lock = threading.Lock()
        self._writers: dict[str, cv2.VideoWriter | None] = {}
        self._started_at: dict[str, float] = {}

    def is_recording(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._started_at

    def seconds_recording(self, camera_id: str) -> float:
        with self._lock:
            started = self._started_at.get(camera_id)
            return (self._clock() - started) if started is not None else 0.0

    def start(self, camera_id: str) -> bool:
        """Devolve True se começou agora; False se já estava gravando (a gravação em curso segue)."""
        with self._lock:
            if camera_id in self._started_at:
                return False
            self._started_at[camera_id] = self._clock()
            self._writers[camera_id] = None  # criado sob demanda no primeiro write() (só aí sabemos o tamanho do quadro)
            return True

    def write(self, camera_id: str, frame: np.ndarray) -> None:
        with self._lock:
            started = self._started_at.get(camera_id)
            if started is None:
                return
            if self._clock() - started > self._max_duration_s:
                log.info("Gravação de %s passou de %.0f s; parando sozinha", camera_id, self._max_duration_s)
                self._stop_locked(camera_id)
                return
            writer = self._writers.get(camera_id)
            if writer is None:
                writer = self._open_writer(camera_id, frame)
                self._writers[camera_id] = writer
            writer.write(frame)

    def stop(self, camera_id: str) -> None:
        with self._lock:
            self._stop_locked(camera_id)

    def _open_writer(self, camera_id: str, frame: np.ndarray) -> cv2.VideoWriter:
        dir_path = os.path.join(self._base_dir, camera_id)
        os.makedirs(dir_path, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(dir_path, f"{stamp}.mp4")
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        log.info("Gravando %s em %s", camera_id, path)
        return cv2.VideoWriter(path, fourcc, self._fps, (width, height))

    def _stop_locked(self, camera_id: str) -> None:
        writer = self._writers.pop(camera_id, None)
        self._started_at.pop(camera_id, None)
        if writer is not None:
            writer.release()


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


def _reading_dict(reading: Reading) -> dict:
    return {"plate": reading.plate, "timestamp": time.strftime("%H:%M:%S", time.localtime(reading.timestamp))}


def _page(camera_id: str) -> bytes:
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Guarita — {camera_id}</title>
<style>
body{{background:#111;color:#eee;font-family:sans-serif;text-align:center}}
img{{max-width:95vw;margin-top:1em;border:1px solid #444}}
button{{font-size:1em;padding:.5em 1.2em;margin-top:.8em;cursor:pointer}}
#readings{{list-style:none;padding:0;max-width:20em;margin:1em auto;text-align:left;font-family:monospace}}
#readings li{{padding:.2em 0;border-bottom:1px solid #333}}
</style>
</head><body>
<h1>{camera_id}</h1>
<img id="frame" src="/preview/{camera_id}/frame.jpg">
<div>
  <button id="recBtn" onclick="toggleRecording()">● Gravar</button>
  <div id="recStatus"></div>
</div>
<h2>Últimas leituras</h2>
<ul id="readings"></ul>
<script>
var cameraId = {camera_id!r};

setInterval(function () {{
  document.getElementById('frame').src = '/preview/' + cameraId + '/frame.jpg?t=' + Date.now();
}}, 500);

function refreshReadings() {{
  fetch('/preview/' + cameraId + '/readings.json').then(function (r) {{ return r.json(); }}).then(function (items) {{
    var list = document.getElementById('readings');
    list.innerHTML = '';
    items.forEach(function (r) {{
      var li = document.createElement('li');
      li.textContent = r.timestamp + '  ' + r.plate;
      list.appendChild(li);
    }});
  }});
}}

function refreshRecordStatus() {{
  fetch('/preview/' + cameraId + '/record/status').then(function (r) {{ return r.json(); }}).then(function (s) {{
    var btn = document.getElementById('recBtn');
    var status = document.getElementById('recStatus');
    if (s.recording) {{
      btn.textContent = '■ Parar gravação';
      status.textContent = 'Gravando há ' + Math.floor(s.seconds) + ' s';
    }} else {{
      btn.textContent = '● Gravar';
      status.textContent = '';
    }}
  }});
}}

function toggleRecording() {{
  fetch('/preview/' + cameraId + '/record/status').then(function (r) {{ return r.json(); }}).then(function (s) {{
    var action = s.recording ? 'stop' : 'start';
    fetch('/preview/' + cameraId + '/record/' + action, {{ method: 'POST' }}).then(refreshRecordStatus);
  }});
}}

refreshReadings();
refreshRecordStatus();
setInterval(refreshReadings, 2000);
setInterval(refreshRecordStatus, 2000);
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


def _split_preview_path(path: str) -> tuple[str | None, str]:
    """"/preview/<id>" -> (id, ""); "/preview/<id>/resto" -> (id, "resto"); outro caminho -> (None, "")."""
    if not path.startswith("/preview/"):
        return None, ""
    rest = path[len("/preview/"):]
    camera_id, _, sub = rest.partition("/")
    return camera_id, sub


class PreviewServer:
    """Página local (sem autenticação) para ver, ao vivo, a câmera enxergando a placa, com as
    últimas leituras e um botão para gravar vídeo sob demanda."""

    def __init__(self, frame_store: FrameStore, camera_ids: list[str], host: str = "0.0.0.0", port: int = 0,
                 reading_log: ReadingLog | None = None, recorder: Recorder | None = None):
        self._frame_store = frame_store
        self._camera_ids = camera_ids
        store = frame_store
        ids = camera_ids
        readings = reading_log if reading_log is not None else ReadingLog()
        recorder = recorder if recorder is not None else Recorder(os.path.join("data", "recordings"))

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, status: int, obj) -> None:
                self._send(status, "application/json", json.dumps(obj).encode("utf-8"))

            def _unknown_camera(self) -> None:
                self._send(404, "text/plain; charset=utf-8", b"camera desconhecida")

            def do_GET(self):  # noqa: N802 - nome exigido pelo http.server
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", _index(ids))
                    return
                camera_id, sub = _split_preview_path(path)
                if camera_id is None:
                    self._send(404, "text/plain; charset=utf-8", b"nao encontrado")
                    return
                if camera_id not in ids:
                    self._unknown_camera()
                    return
                if sub == "frame.jpg":
                    found = store.snapshot(camera_id)
                    body = _draw_detections(*found) if found else _placeholder_jpeg()
                    self._send(200, "image/jpeg", body)
                elif sub == "readings.json":
                    self._send_json(200, [_reading_dict(r) for r in readings.recent(camera_id)])
                elif sub == "record/status":
                    self._send_json(200, {
                        "recording": recorder.is_recording(camera_id),
                        "seconds": recorder.seconds_recording(camera_id),
                    })
                elif sub == "":
                    self._send(200, "text/html; charset=utf-8", _page(camera_id))
                else:
                    self._send(404, "text/plain; charset=utf-8", b"nao encontrado")

            def do_POST(self):  # noqa: N802 - nome exigido pelo http.server
                path = self.path.split("?", 1)[0]
                camera_id, sub = _split_preview_path(path)
                if camera_id is None or camera_id not in ids:
                    self._unknown_camera()
                    return
                if sub == "record/start":
                    recorder.start(camera_id)
                    self._send_json(200, {"recording": True})
                elif sub == "record/stop":
                    recorder.stop(camera_id)
                    self._send_json(200, {"recording": False})
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
