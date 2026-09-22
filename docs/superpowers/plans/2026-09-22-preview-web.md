# Preview web do coletor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Página web local, ligada por variável de ambiente, mostrando o vídeo mais recente de cada câmera com a caixa da detecção desenhada — para validar visualmente o coletor em teste presencial, sem depender de internet.

**Architecture:** `FrameStore` (guarda só o último quadro+detecções de cada câmera, thread-safe) e `PreviewServer` (`http.server.ThreadingHTTPServer`, biblioteca padrão) em um arquivo novo, `preview.py`. `CameraRunner` ganha um gancho `on_frame` opcional, chamado a cada quadro processado; sem ele (padrão), nada muda. `__main__.py` liga tudo isso só quando `GUARITA_PREVIEW=1`.

**Tech Stack:** Python 3.12, `http.server` (stdlib), OpenCV (já é dependência do projeto), pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-preview-web-design.md`

## Global Constraints

- Sem dependência nova: só `http.server`/`urllib` da biblioteca padrão.
- Desligado por padrão; liga com `GUARITA_PREVIEW=1`. Porta: `GUARITA_PREVIEW_PORT` (padrão `8888`).
- Sem autenticação (decisão do spec, seção 2).
- A página não mostra texto de placa, só o vídeo com a caixa (spec, seção 5).
- HTML/JS da página são autocontidos — sem CDN, sem framework (precisa funcionar sem internet).
- `CameraRunner` sem `on_frame` continua com exatamente o mesmo comportamento de hoje; todos os testes existentes de `tests/test_pipeline.py` continuam passando sem alteração.

---

### Task 1: `FrameStore`

**Files:**
- Create: `src/guarita/preview.py`, `tests/test_preview.py`

**Interfaces:**
- Consumes: `guarita.models.Detection` (`track_id: int, box: tuple[int,int,int,int], conf: float`).
- Produces: `FrameStore()` com `update(camera_id: str, frame: np.ndarray, detections: list[Detection]) -> None` e `snapshot(camera_id: str) -> tuple[np.ndarray, list[Detection]] | None`.

- [ ] **Step 1: Escrever o teste que falha — `tests/test_preview.py`**

```python
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
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_preview.py -q`
Expected: FAIL (`No module named 'guarita.preview'`).

- [ ] **Step 3: Implementar `FrameStore` em `src/guarita/preview.py`**

```python
from __future__ import annotations

import threading

import numpy as np

from .models import Detection


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
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_preview.py -q`
Expected: PASS (5 testes).

- [ ] **Step 5: Commit**

```bash
git add src/guarita/preview.py tests/test_preview.py
git commit -m "feat: add FrameStore holding each camera's latest frame for the preview page"
```

---

### Task 2: `PreviewServer`

**Files:**
- Modify: `src/guarita/preview.py`
- Modify: `tests/test_preview.py`

**Interfaces:**
- Consumes: `FrameStore` (Task 1).
- Produces: `PreviewServer(frame_store: FrameStore, camera_ids: list[str], host: str = "0.0.0.0", port: int = 0)` — um `http.server.ThreadingHTTPServer` já construído (não iniciado até `start()`), com:
  - `.start() -> None` — inicia a thread que serve as requisições.
  - `.stop() -> None` — encerra o servidor e a thread.
  - `.port -> int` — a porta real em uso (útil quando construído com `port=0`, porta aleatória — usado nos testes).
  - Rotas: `GET /`, `GET /preview/<id>`, `GET /preview/<id>/frame.jpg`; qualquer outra coisa dá `404`.

- [ ] **Step 1: Escrever o teste que falha — acrescentar a `tests/test_preview.py`**

```python
import urllib.request

from guarita.preview import PreviewServer


def start_server(frame_store, camera_ids=("cam-01",)):
    server = PreviewServer(frame_store, list(camera_ids), host="127.0.0.1", port=0)
    server.start()
    return server


def get(server, path):
    url = f"http://127.0.0.1:{server.port}{path}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.headers.get("Content-Type"), resp.read()


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
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_preview.py -q`
Expected: FAIL (`ImportError: cannot import name 'PreviewServer'`).

- [ ] **Step 3: Implementar `PreviewServer` — acrescentar a `src/guarita/preview.py`**

```python
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

log = logging.getLogger(__name__)


def _placeholder_jpeg() -> bytes:
    img = np.full((360, 640, 3), 40, np.uint8)
    cv2.putText(img, "aguardando sinal...", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
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
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_preview.py -q`
Expected: PASS (10 testes).

- [ ] **Step 5: Commit**

```bash
git add src/guarita/preview.py tests/test_preview.py
git commit -m "feat: add PreviewServer serving a local live-view page per camera"
```

---

### Task 3: Gancho `on_frame` no `CameraRunner`

**Files:**
- Modify: `src/guarita/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nenhuma interface nova de fora.
- Produces: `CameraRunner.__init__(..., health=None, on_frame=None)` — `on_frame`, se passado, é
  `Callable[[str, np.ndarray, list[Detection]], None]`, chamado em `step()` a cada quadro lido com
  sucesso (não chamado quando `step()` devolve `"idle"` ou `"done"`).

- [ ] **Step 1: Escrever o teste que falha — acrescentar a `tests/test_pipeline.py`**

```python
def test_on_frame_hook_is_called_with_the_frame_and_detections():
    calls = []
    frame_img = np.zeros((10, 10, 3), np.uint8)
    dets = [Detection(1, (0, 0, 5, 5), 0.9)]
    runner = CameraRunner(
        CAMERA, PipelineConfig(), ScriptedSource([Frame(frame_img, 0.0)]), ScriptedDetector([dets]),
        _dispatcher(FakeEngine()), on_frame=lambda cid, img, d: calls.append((cid, img, d)))
    assert runner.step() == "frame"
    assert len(calls) == 1
    camera_id, img, detections = calls[0]
    assert camera_id == "cam-01"
    assert img is frame_img
    assert detections == dets


def test_on_frame_hook_is_not_required():
    # nenhum on_frame passado: comportamento igual ao de antes, sem lançar
    runner = CameraRunner(CAMERA, PipelineConfig(), ScriptedSource([]), ScriptedDetector([]), _dispatcher(FakeEngine()))
    assert runner.step() == "done"
```

(`Frame` já é importado em `tests/test_pipeline.py` — ver `from guarita.capture import Frame` no topo do arquivo.)

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_pipeline.py -k on_frame -q`
Expected: FAIL (`TypeError: CameraRunner.__init__() got an unexpected keyword argument 'on_frame'`).

- [ ] **Step 3: Implementar o gancho em `src/guarita/pipeline.py`**

Trocar a assinatura e o corpo de `CameraRunner.__init__` (arquivo `src/guarita/pipeline.py`, dentro
da classe `CameraRunner`):

```python
    def __init__(self, camera: CameraConfig, cfg: PipelineConfig, source, detector, dispatcher: OcrDispatcher,
                 health: Health | None = None, on_frame=None):
        self.camera = camera
        self._cfg = cfg
        self.source = source
        self._detector = detector
        self._dispatcher = dispatcher
        self._health = health
        self._on_frame = on_frame
        self._tracks = TrackManager(camera.id, cfg.close_after_s, cfg.max_age_s, cfg.max_crops)
        self._last_ts = 0.0
```

E em `step()`, logo depois da linha `detections = self._detector.detect(frame.image)`:

```python
        detections = self._detector.detect(frame.image)
        if self._on_frame:
            self._on_frame(self.camera.id, frame.image, detections)
        self._last_ts = frame.ts
```

(substitui a linha `self._last_ts = frame.ts` que já existe logo depois do `detect()` — não duplicar).

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_pipeline.py -q`
Expected: PASS — inclusive todos os testes que já existiam em `test_pipeline.py`, sem nenhuma
alteração neles (confirma que o gancho é opcional e não muda o comportamento default).

- [ ] **Step 5: Commit**

```bash
git add src/guarita/pipeline.py tests/test_pipeline.py
git commit -m "feat: add optional on_frame hook to CameraRunner for the preview page"
```

---

### Task 4: Ligar tudo em `__main__.py` (variável de ambiente)

**Files:**
- Modify: `src/guarita/__main__.py`
- Modify: `src/guarita/app.py`
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: `FrameStore`, `PreviewServer` (Task 1-2), `CameraRunner(..., on_frame=...)` (Task 3).
- Produces: `App.__init__(..., on_frame=None)` — repassa `on_frame` para cada `CameraRunner` que
  constrói. Variáveis de ambiente lidas só em `__main__.py`: `GUARITA_PREVIEW` (`"1"` liga),
  `GUARITA_PREVIEW_PORT` (inteiro, padrão `8888`).

- [ ] **Step 1: Escrever o teste que falha — acrescentar a `tests/test_app.py`**

```python
def test_app_passes_on_frame_to_every_camera_runner(tmp_path, api):
    calls = []

    def hook(camera_id, img, detections):
        calls.append(camera_id)

    cfg = Config(
        api=ApiConfig(base_url=api.base_url), detector=DetectorConfig(),
        pipeline=PipelineConfig(failures_dir=str(tmp_path / "falhas")),
        outbox=OutboxConfig(path=str(tmp_path / "o.sqlite")),
        cameras=(CameraConfig("cam-01", "entrada", "", "KEY"),),
    )
    outbox = Outbox(cfg.outbox.path)

    def executor_factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: FakeEngine(),))

    app = App(cfg, source_factory=lambda cam: ScriptedSource([]), detector_factory=lambda cam: ScriptedDetector([]),
              executor_factory=executor_factory, client=ApiClient(api.base_url), outbox=outbox, on_frame=hook)
    try:
        assert app._runners["cam-01"]._on_frame is hook
    finally:
        outbox.close()
```

(`tests/test_app.py` já importa, no topo do arquivo, tudo que este teste usa: `ApiConfig`,
`CameraConfig`, `Config`, `DetectorConfig`, `OutboxConfig`, `PipelineConfig`, `Outbox`,
`ApiClient`, `ThreadPoolExecutor`, `worker`, `ScriptedSource`, `ScriptedDetector`, `FakeEngine`
- não é preciso reimportar nada. `Frame` já e importado em `tests/test_pipeline.py`, para o
teste da Task 3.)

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_app.py -k on_frame -q`
Expected: FAIL (`TypeError: App.__init__() got an unexpected keyword argument 'on_frame'`).

- [ ] **Step 3: Implementar em `src/guarita/app.py`**

Trocar a assinatura de `App.__init__` (adicionar `on_frame=None` ao final dos parâmetros) e passar
adiante na construção dos `CameraRunner`:

```python
    def __init__(self, cfg: Config, *, source_factory, detector_factory, executor_factory,
                 client: ApiClient, outbox: Outbox, stop_event: threading.Event | None = None, on_frame=None):
```

E, na montagem de `self._runners` (dentro do mesmo `__init__`):

```python
            self._runners = {
                c.id: CameraRunner(c, cfg.pipeline, sources[c.id], detectors[c.id], self.dispatcher,
                                   self.health[c.id], on_frame=on_frame)
                for c in cfg.cameras
            }
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_app.py -q`
Expected: PASS — inclusive todos os testes que já existiam em `test_app.py` sem alteração (o
parâmetro é opcional, com padrão `None`).

- [ ] **Step 5: Ligar por variável de ambiente em `src/guarita/__main__.py`**

No topo do arquivo, acrescentar aos imports:

```python
from .preview import FrameStore, PreviewServer
```

Dentro de `main()`, logo antes de `def executor_factory():` (que já existe), acrescentar:

```python
    preview_enabled = os.environ.get("GUARITA_PREVIEW") == "1"
    frame_store = FrameStore() if preview_enabled else None
    on_frame = frame_store.update if frame_store else None
    preview_server = None
```

Na chamada de `App(...)` já existente, acrescentar o parâmetro `on_frame=on_frame,` à lista de
argumentos.

Depois que `app.start()` é chamado com sucesso (mesmo bloco `try` que já envolve a criação do
`App`), subir o servidor:

```python
        if preview_enabled:
            preview_server = PreviewServer(frame_store, [c.id for c in cfg.cameras],
                                            port=int(os.environ.get("GUARITA_PREVIEW_PORT", "8888")))
            preview_server.start()
```

E no bloco `finally` existente (onde `app.stop()`/`outbox.close()` já acontecem), encerrar o
servidor antes do resto:

```python
    finally:
        if preview_server is not None:
            preview_server.stop()
```

(Ler o `main()` atual em `src/guarita/__main__.py` antes de editar, para encaixar essas três
inserções nos pontos certos do `try`/`finally` já existentes — a ordem exata das linhas ao redor
pode ter mudado desde que este plano foi escrito; o que importa é: `frame_store`/`on_frame`
definidos antes da construção do `App`, `preview_server.start()` só depois de `app.start()` ter
sucesso, `preview_server.stop()` no `finally`, antes de `outbox.close()`.)

- [ ] **Step 6: Rodar a suíte inteira**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (todos os testes, incluindo os 4 arquivos tocados nesta etapa).

- [ ] **Step 7: Commit**

```bash
git add src/guarita/app.py src/guarita/__main__.py tests/test_app.py
git commit -m "feat: wire the preview page behind GUARITA_PREVIEW=1"
```

- [ ] **Step 8: Verificação manual (ambiente real, fora do pytest)**

No ambiente completo (`.venv-run`, com PaddleOCR/YOLO instalados de verdade — ver `README.md`):

```powershell
$env:GUARITA_PREVIEW = "1"
.venv-run\Scripts\python -m guarita --config config.toml
```

Abrir `http://localhost:8888/` no navegador, clicar na câmera listada e confirmar visualmente que
o vídeo aparece e, quando uma placa é detectada, a caixa verde aparece sobre ela. Confirmar que sem
`GUARITA_PREVIEW` definida (`Remove-Item Env:\GUARITA_PREVIEW` ou uma nova janela de terminal) o
coletor sobe normalmente e a porta 8888 não abre.

## Verificação final

- [ ] `.venv\Scripts\python -m pytest -q` — todas as suítes passam (as já existentes + as 4 novas
      desta etapa).
- [ ] Passo 8 da Task 4 (verificação manual) concluído.
