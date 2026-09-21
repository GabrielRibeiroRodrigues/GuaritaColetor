# Coletor unificado · Etapa 2 — Pipeline (detecção, rastreio, OCR em pool, orquestração e ferramentas) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completar o coletor: captura RTSP com reconexão, detecção + rastreio (YOLO/ByteTrack), seleção dos melhores recortes, OCR em pool de processos com timeout, agregação por votação com fusão de duplicatas, orquestração por threads com envio e heartbeat, ponto de entrada (`python -m guarita`) e ferramentas de medição (`bench`) e de verificação do ambiente.

**Architecture:** Cada peça é uma classe pequena com dependências injetadas (fonte de quadros, detector, executor do OCR), então todo o fluxo é testado de ponta a ponta com falsos, sem câmera nem modelos. O modelo real só entra em `__main__.py`, `tools/` e `detect.py`/`ocr/engine.py`.

**Tech Stack:** Python 3.12, Ultralytics 8.0.114 (YOLO + ByteTrack), PaddleOCR, OpenCV, threads + `ProcessPoolExecutor`, SQLite, pytest. Depende da Etapa 1.

**Spec:** `docs/superpowers/specs/2026-09-21-coletor-unificado-design.md`

## Global Constraints

- Repositório `C:\Users\Gabriel\GuaritaColetor`; código em `src/guarita/`.
- O detector **não inicia** se o arquivo de pesos não existir (sem fallback para `yolov8n.pt`).
- Um detector (uma instância do YOLO com estado do ByteTrack) **por câmera**; um pool de OCR **compartilhado**.
- O recorte é sempre extraído do quadro em **resolução cheia**; o detector recebe o quadro inteiro e redimensiona internamente (`imgsz`).
- Mantidos da Etapa 1: OCR sempre em todos os ângulos configurados (0° e 15°), pré-processamento cinza→RGB, votação por posição, placa final só se válida, envio pela fila SQLite.
- Cada tarefa de OCR tem tempo limite (`ocr_timeout_s`, padrão 10 s); ao estourar, o pool é reiniciado e o recorte vai para a pasta de falhas (24 h).
- O heartbeat só é enviado enquanto os quadros avançam (`stale_after_s`, padrão 30 s).
- Uma leitura por veículo: a mesma placa em rastreios distintos dentro de `fuse_window_s` (20 s) é fundida.
- Timestamps: RTSP usa `time.time()` (epoch); arquivos de vídeo usam o tempo do vídeo (`quadro / fps`).
- Sem `print`; usar `logging`. Segredos só por variável de ambiente.

## Estrutura de arquivos desta etapa

```
src/guarita/
  compat.py  detect.py  track.py  dispatch.py  capture.py  pipeline.py
  supervisor.py  app.py  evaluation.py  __main__.py
tools/  bench.py  check_env.py
service/install-service.md   README.md
tests/  test_detect.py  test_track.py  test_dispatch.py  test_capture.py  test_pipeline.py
        test_supervisor.py  test_app.py  test_evaluation.py   (fakes.py é estendido)
```

---

### Task 1: Detector YOLO com rastreio

**Files:**
- Create: `src/guarita/compat.py`, `src/guarita/detect.py`, `tests/test_detect.py`

**Interfaces:**
- Consumes: `Detection`.
- Produces: `PlateDetector(weights, imgsz=416, conf=0.1, model=None)` — sem `model`, exige que `weights` exista (senão `FileNotFoundError`), prepara o Ultralytics e carrega o `YOLO`; `detect(frame) -> list[Detection]` (usa `model.track(..., persist=True, tracker='bytetrack.yaml')`; lista vazia quando não há trilhas confirmadas); `prepare_ultralytics()` (compatibilidade com pesos antigos).

- [ ] **Step 1: Escrever o teste que falha — `tests/test_detect.py`**

```python
from types import SimpleNamespace

import numpy as np
import pytest

from guarita.detect import PlateDetector
from guarita.models import Detection


class Vec(list):
    def tolist(self):
        return list(self)


class FakeYolo:
    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=self._boxes)]


FRAME = np.zeros((10, 10, 3), np.uint8)


def test_missing_weights_is_fatal(tmp_path):
    missing = tmp_path / "bestn.pt"
    with pytest.raises(FileNotFoundError, match="bestn.pt"):
        PlateDetector(str(missing))


def test_detect_returns_detections_with_track_ids():
    boxes = SimpleNamespace(
        id=Vec([3.0, 4.0]), conf=Vec([0.9, 0.4]), xyxy=Vec([[1.2, 2.7, 30.0, 40.9], [5, 5, 9, 9]]))
    model = FakeYolo(boxes)
    detections = PlateDetector("bestn.pt", imgsz=416, conf=0.1, model=model).detect(FRAME)
    assert detections == [Detection(3, (1, 2, 30, 40), 0.9), Detection(4, (5, 5, 9, 9), 0.4)]
    call = model.calls[0]
    assert call["imgsz"] == 416 and call["conf"] == 0.1
    assert call["persist"] is True and call["tracker"] == "bytetrack.yaml"


def test_no_confirmed_tracks_yields_empty_list():
    boxes = SimpleNamespace(id=None, conf=Vec([]), xyxy=Vec([]))
    assert PlateDetector("bestn.pt", model=FakeYolo(boxes)).detect(FRAME) == []
    assert PlateDetector("bestn.pt", model=FakeYolo(None)).detect(FRAME) == []
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_detect.py -q`
Expected: FAIL (`No module named 'guarita.detect'`).

- [ ] **Step 3: Implementar `src/guarita/compat.py`**

```python
from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def prepare_ultralytics() -> None:
    """Compatibilidade com pesos antigos (mesma preparação do main.py atual)."""
    try:
        import ultralytics.utils  # noqa: F401
    except Exception:
        try:
            import ultralytics.yolo.utils as legacy_utils

            sys.modules["ultralytics.utils"] = legacy_utils  # alias para o caminho antigo
        except Exception as exc:
            log.warning("Não foi possível preparar 'ultralytics.utils' (%s); o carregamento dos pesos pode falhar.", exc)

    import torch
    from ultralytics.nn.tasks import DetectionModel

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        try:
            torch.serialization.add_safe_globals([DetectionModel])
        except Exception as exc:
            log.warning("Falha ao ajustar torch safe globals: %s", exc)
```

- [ ] **Step 4: Implementar `src/guarita/detect.py`**

```python
from __future__ import annotations

import logging
import os

from .models import Detection

log = logging.getLogger(__name__)


class PlateDetector:
    """YOLO de placas com rastreio ByteTrack. Uma instância por câmera (o rastreador guarda estado)."""

    def __init__(self, weights: str, imgsz: int = 416, conf: float = 0.1, model=None):
        if model is None:
            if not os.path.isfile(weights):
                raise FileNotFoundError(
                    f"Pesos do detector não encontrados: {weights}. O coletor não inicia sem eles "
                    "(não há fallback para yolov8n.pt, que detecta objetos COCO e não placas).")
            from .compat import prepare_ultralytics

            prepare_ultralytics()
            from ultralytics import YOLO

            model = YOLO(weights)
        self._model = model
        self._imgsz = imgsz
        self._conf = conf

    def detect(self, frame) -> list[Detection]:
        result = self._model.track(
            frame, imgsz=self._imgsz, conf=self._conf, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        boxes = result.boxes
        if boxes is None or getattr(boxes, "id", None) is None:
            return []
        detections = []
        for xyxy, conf, track_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.id.tolist()):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            detections.append(Detection(int(track_id), (x1, y1, x2, y2), float(conf)))
        return detections
```

- [ ] **Step 5: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_detect.py -q`
Expected: PASS (3 testes).

- [ ] **Step 6: Commit**

```bash
git add src/guarita/compat.py src/guarita/detect.py tests/test_detect.py
git commit -m "feat: add YOLO plate detector with ByteTrack ids and fail-fast weights check"
```

---

### Task 2: Gerenciador de trilhas (melhores recortes)

**Files:**
- Create: `src/guarita/track.py`, `tests/test_track.py`

**Interfaces:**
- Consumes: `Detection`, `Crop`, `ClosedTrack`.
- Produces: `crop_score(crop, conf) -> float` (= `conf × altura × largura × (1 + nitidez/100)`, nitidez = variância do Laplaciano em cinza); `TrackManager(camera_id, close_after_s, max_age_s, max_crops)` com `update(frame, detections, now) -> list[ClosedTrack]` (a caixa é recortada do `frame` cheio, limitada aos bordos; recortes copiados; guarda os `max_crops` melhores) e `flush(now) -> list[ClosedTrack]`. Uma trilha fecha quando `now - último_visto >= close_after_s` ou `now - início >= max_age_s`. `ClosedTrack.crops` vem ordenado do melhor para o pior; trilhas sem recorte válido não são emitidas.

- [ ] **Step 1: Escrever o teste que falha — `tests/test_track.py`**

```python
import numpy as np

from guarita.models import Detection
from guarita.track import TrackManager, crop_score

BOX = (10, 10, 60, 40)


def frame_with(sharp=False):
    frame = np.full((100, 200, 3), 120, np.uint8)
    if sharp:
        checker = (np.indices((30, 50)).sum(axis=0) % 2 * 255).astype(np.uint8)
        frame[10:40, 10:60] = checker[..., None]
    return frame


def det(track_id=1, box=BOX, conf=0.9):
    return Detection(track_id, box, conf)


def manager(**kw):
    args = dict(camera_id="cam-01", close_after_s=1.5, max_age_s=20.0, max_crops=3)
    args.update(kw)
    return TrackManager(**args)


def test_closes_after_absence_and_reports_fields():
    tm = manager()
    frame = frame_with()
    assert tm.update(frame, [det()], 0.0) == []
    assert tm.update(frame, [det()], 0.5) == []
    assert tm.update(frame, [], 1.0) == []          # ausente há 0,5 s
    closed = tm.update(frame, [], 2.0)              # ausente há 1,5 s
    assert len(closed) == 1
    track = closed[0]
    assert (track.camera_id, track.track_id) == ("cam-01", 1)
    assert (track.started_at, track.closed_at) == (0.0, 2.0)
    assert len(track.crops) == 2
    assert track.crops[0].image.shape == (30, 50, 3)
    assert tm.update(frame, [], 3.0) == []          # já fechada e removida


def test_keeps_only_the_best_crops_best_first():
    tm = manager(max_crops=2)
    for now, sharp in [(0.0, False), (0.5, True), (1.0, False), (1.5, False)]:
        tm.update(frame_with(sharp), [det()], now)
    closed = tm.update(frame_with(), [], 10.0)[0]
    assert len(closed.crops) == 2
    assert closed.crops[0].ts == 0.5                # o quadro nítido vem primeiro
    assert closed.crops[0].score > closed.crops[1].score


def test_crop_score_prefers_sharp_and_confident_crops():
    sharp = frame_with(True)[10:40, 10:60]
    flat = frame_with(False)[10:40, 10:60]
    assert crop_score(sharp, 0.9) > crop_score(flat, 0.9)
    assert crop_score(flat, 0.9) > crop_score(flat, 0.4)


def test_max_age_forces_closing_and_a_new_track_starts():
    tm = manager(max_age_s=5.0)
    frame = frame_with()
    closed = []
    for now in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        closed += tm.update(frame, [det()], now)
    assert len(closed) == 1 and closed[0].started_at == 0.0 and closed[0].closed_at == 5.0
    assert tm.update(frame, [det()], 6.0) == []
    flushed = tm.flush(6.5)
    assert len(flushed) == 1 and flushed[0].started_at == 6.0


def test_flush_closes_everything_once():
    tm = manager()
    frame = frame_with()
    tm.update(frame, [det(1), det(2, (100, 10, 150, 40))], 0.0)
    flushed = tm.flush(0.1)
    assert sorted(t.track_id for t in flushed) == [1, 2]
    assert tm.flush(0.2) == []


def test_boxes_are_clipped_to_the_frame_and_outside_boxes_ignored():
    tm = manager()
    frame = frame_with()
    tm.update(frame, [det(1, (-10, -10, 300, 300)), det(2, (250, 250, 300, 300))], 0.0)
    flushed = tm.flush(0.1)
    assert [t.track_id for t in flushed] == [1]
    assert flushed[0].crops[0].image.shape == (100, 200, 3)


def test_crops_are_copies_of_the_frame():
    tm = manager()
    frame = frame_with()
    tm.update(frame, [det()], 0.0)
    frame[:] = 0
    crop = tm.flush(0.1)[0].crops[0].image
    assert crop.mean() > 0
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_track.py -q`
Expected: FAIL (módulo inexistente).

- [ ] **Step 3: Implementar `src/guarita/track.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import ClosedTrack, Crop, Detection


def crop_score(crop: np.ndarray, conf: float) -> float:
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    height, width = gray.shape[:2]
    return conf * height * width * (1.0 + sharpness / 100.0)


@dataclass
class _State:
    started_at: float
    last_seen: float
    crops: list[Crop] = field(default_factory=list)


class TrackManager:
    def __init__(self, camera_id: str, close_after_s: float, max_age_s: float, max_crops: int):
        self._camera_id = camera_id
        self._close_after = close_after_s
        self._max_age = max_age_s
        self._max_crops = max_crops
        self._tracks: dict[int, _State] = {}

    def update(self, frame: np.ndarray, detections: list[Detection], now: float) -> list[ClosedTrack]:
        height, width = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            state = self._tracks.get(det.track_id)
            if state is None:
                state = _State(started_at=now, last_seen=now)
                self._tracks[det.track_id] = state
            state.last_seen = now
            self._offer(state, Crop(crop.copy(), now, crop_score(crop, det.conf)))
        return self._collect(now, force=False)

    def flush(self, now: float) -> list[ClosedTrack]:
        return self._collect(now, force=True)

    def _offer(self, state: _State, crop: Crop) -> None:
        if len(state.crops) < self._max_crops:
            state.crops.append(crop)
            return
        worst = min(range(len(state.crops)), key=lambda i: state.crops[i].score)
        if crop.score > state.crops[worst].score:
            state.crops[worst] = crop

    def _collect(self, now: float, force: bool) -> list[ClosedTrack]:
        closed = []
        for track_id, state in list(self._tracks.items()):
            expired = (now - state.last_seen >= self._close_after) or (now - state.started_at >= self._max_age)
            if force or expired:
                del self._tracks[track_id]
                if state.crops:
                    crops = sorted(state.crops, key=lambda c: c.score, reverse=True)
                    closed.append(ClosedTrack(self._camera_id, track_id, state.started_at, now, crops))
        return closed
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_track.py -q`
Expected: PASS (7 testes).

- [ ] **Step 5: Commit**

```bash
git add src/guarita/track.py tests/test_track.py
git commit -m "feat: add track manager that keeps the best crops per plate"
```

---

### Task 3: Despachante do pool de OCR (timeout e reinício)

**Files:**
- Create: `src/guarita/dispatch.py`, `tests/test_dispatch.py`

**Interfaces:**
- Consumes: `ClosedTrack`, `OcrJob`, `OcrResult`, `process_job`.
- Produces: `Outcome(track, result: OcrResult | None, error: str | None, latency_ms: int)`; `OcrDispatcher(executor_factory, timeout_s, clock=time.monotonic)` com `submit(track, job)` (se o executor estiver quebrado, recria o pool e tenta de novo), `poll() -> list[Outcome]` (concluídos, com erro ou estourados; ao estourar o tempo o pool é recriado e os demais pendentes são reenviados ao novo pool), `pending_count()`, `shutdown()`. Thread-safe.

- [ ] **Step 1: Escrever o teste que falha — `tests/test_dispatch.py`**

```python
from concurrent.futures import Future

from guarita.dispatch import OcrDispatcher
from guarita.models import ClosedTrack
from guarita.ocr.worker import OcrJob, OcrResult


class FakeExecutor:
    def __init__(self, broken=False):
        self.submitted = []
        self.broken = broken
        self.shut_down = False

    def submit(self, fn, job):
        if self.broken:
            raise RuntimeError("pool quebrado")
        future = Future()
        self.submitted.append((job, future))
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shut_down = True


def make(broken_first=False):
    executors = []

    def factory():
        executors.append(FakeExecutor(broken=broken_first and not executors))
        return executors[-1]

    clock = [0.0]
    return OcrDispatcher(factory, timeout_s=10.0, clock=lambda: clock[0]), executors, clock


def track(tid=1):
    return ClosedTrack("cam-01", tid, 0.0, 1.0, [])


def job():
    return OcrJob([], (0.0, 15.0), 0.1)


def test_completed_job_becomes_an_outcome_with_latency():
    dispatcher, executors, clock = make()
    dispatcher.submit(track(), job())
    assert dispatcher.poll() == []
    executors[0].submitted[0][1].set_result(OcrResult([("ABC1234", 0.9)], 40))
    clock[0] = 2.5
    (outcome,) = dispatcher.poll()
    assert outcome.error is None and outcome.result.candidates == [("ABC1234", 0.9)]
    assert outcome.latency_ms == 2500
    assert dispatcher.pending_count() == 0


def test_failed_job_reports_the_error():
    dispatcher, executors, _ = make()
    dispatcher.submit(track(), job())
    executors[0].submitted[0][1].set_exception(ValueError("boom"))
    (outcome,) = dispatcher.poll()
    assert outcome.result is None and "boom" in outcome.error


def test_timeout_restarts_the_pool_and_resubmits_the_others():
    dispatcher, executors, clock = make()
    dispatcher.submit(track(1), job())
    clock[0] = 8.0
    dispatcher.submit(track(2), job())
    clock[0] = 11.0                                   # a 1ª passou de 10 s; a 2ª não
    (outcome,) = dispatcher.poll()
    assert outcome.track.track_id == 1 and outcome.error == "timeout"
    assert len(executors) == 2 and executors[0].shut_down
    assert dispatcher.pending_count() == 1
    assert len(executors[1].submitted) == 1           # a 2ª foi reenviada ao pool novo
    clock[0] = 15.0
    assert dispatcher.poll() == []                    # o relógio da 2ª reiniciou em 11 s


def test_broken_pool_is_replaced_on_submit():
    dispatcher, executors, _ = make(broken_first=True)
    dispatcher.submit(track(), job())
    assert len(executors) == 2
    assert len(executors[1].submitted) == 1
    assert dispatcher.pending_count() == 1
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_dispatch.py -q`
Expected: FAIL (módulo inexistente).

- [ ] **Step 3: Implementar `src/guarita/dispatch.py`**

```python
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from .models import ClosedTrack
from .ocr.worker import OcrJob, OcrResult, process_job

log = logging.getLogger(__name__)


@dataclass
class Outcome:
    track: ClosedTrack
    result: OcrResult | None
    error: str | None
    latency_ms: int


@dataclass
class _Pending:
    track: ClosedTrack
    job: OcrJob
    future: object
    submitted: float


class OcrDispatcher:
    def __init__(self, executor_factory, timeout_s: float, clock=time.monotonic):
        self._factory = executor_factory
        self._executor = executor_factory()
        self._timeout = timeout_s
        self._clock = clock
        self._pending: list[_Pending] = []
        self._lock = threading.Lock()

    def submit(self, track: ClosedTrack, job: OcrJob) -> None:
        with self._lock:
            try:
                future = self._executor.submit(process_job, job)
            except Exception:
                log.exception("Pool de OCR indisponível; recriando")
                self._restart_locked(self._clock())
                future = self._executor.submit(process_job, job)
            self._pending.append(_Pending(track, job, future, self._clock()))

    def poll(self) -> list[Outcome]:
        with self._lock:
            now = self._clock()
            outcomes, keep, timed_out = [], [], False
            for item in self._pending:
                latency = int((now - item.submitted) * 1000)
                if item.future.done():
                    try:
                        outcomes.append(Outcome(item.track, item.future.result(), None, latency))
                    except Exception as exc:
                        outcomes.append(Outcome(item.track, None, f"erro: {exc!r}", latency))
                elif now - item.submitted > self._timeout:
                    item.future.cancel()
                    outcomes.append(Outcome(item.track, None, "timeout", latency))
                    timed_out = True
                else:
                    keep.append(item)
            self._pending = keep
            if timed_out:
                log.error("Tarefa de OCR estourou o tempo; reiniciando o pool")
                self._restart_locked(now)
            return outcomes

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        with self._lock:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _restart_locked(self, now: float) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = self._factory()
        for item in self._pending:
            item.future = self._executor.submit(process_job, item.job)
            item.submitted = now
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_dispatch.py -q`
Expected: PASS (4 testes).

- [ ] **Step 5: Commit**

```bash
git add src/guarita/dispatch.py tests/test_dispatch.py
git commit -m "feat: add OCR dispatcher with per-task timeout and pool restart"
```

---

### Task 4: Captura (RTSP com reconexão e arquivo de vídeo)

**Files:**
- Create: `src/guarita/capture.py`, `tests/test_capture.py`

**Interfaces:**
- Produces: `Frame(image, ts)`; `FrameThrottle(target_fps).should_process(ts) -> bool`; `RtspSource(url, target_fps, open_fn=cv2.VideoCapture, clock=time.time, monotonic=time.monotonic, max_backoff_s=30.0, should_stop=lambda: False)` com `read() -> Frame | None` (usa `grab()` nos quadros descartados; `None` = sem quadro agora; reconecta com espera `min(30, 2^(falhas-1))` s) e `exhausted = False`; `FileSource(path, target_fps, open_fn=cv2.VideoCapture)` com `read()`, `fps`, `exhausted` (True ao acabar o vídeo; `ts = índice_do_quadro / fps`; levanta `FileNotFoundError` se não abrir).

- [ ] **Step 1: Escrever o teste que falha — `tests/test_capture.py`**

```python
import itertools

import cv2
import numpy as np
import pytest

from guarita.capture import FileSource, FrameThrottle, RtspSource


class FakeCap:
    def __init__(self, grabs=(), opened=True, fps=25.0):
        self._grabs = list(grabs)
        self._opened = opened
        self._fps = fps
        self.retrieved = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def grab(self):
        return self._grabs.pop(0) if self._grabs else False

    def retrieve(self):
        self.retrieved += 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        self.released = True

    def get(self, prop):
        return self._fps if prop == cv2.CAP_PROP_FPS else 0.0


def test_throttle_limits_the_rate():
    throttle = FrameThrottle(8)  # intervalo de 0,125 s
    assert [throttle.should_process(t) for t in (0.0, 0.05, 0.13, 0.20, 0.26)] == [True, False, True, False, True]


def test_rtsp_source_skips_frames_with_grab_only():
    cap = FakeCap([True] * 10)
    ticks = itertools.count(0.0, 0.05)
    source = RtspSource("rtsp://x", 8, open_fn=lambda url: cap, clock=lambda: next(ticks))
    first = source.read()
    second = source.read()
    assert first.ts == 0.0
    assert second.ts == pytest.approx(0.15)
    assert cap.retrieved == 2  # os quadros descartados não foram decodificados por completo


def test_rtsp_source_reconnects_with_backoff():
    caps = [FakeCap([False]), FakeCap([True] * 3)]
    opened = []
    mono = [0.0]

    def open_fn(url):
        opened.append(url)
        return caps[len(opened) - 1]

    source = RtspSource("rtsp://x", 8, open_fn=open_fn, monotonic=lambda: mono[0])
    assert source.read() is None                 # 1º grab falha: conexão descartada
    assert caps[0].released
    mono[0] = 0.5
    assert source.read() is None and len(opened) == 1   # ainda em espera (1 s)
    mono[0] = 1.5
    frame = source.read()
    assert frame is not None and len(opened) == 2


def test_rtsp_open_failure_backs_off_exponentially():
    opened = []
    mono = [0.0]

    def open_fn(url):
        opened.append(url)
        return FakeCap(opened=False)

    source = RtspSource("rtsp://x", 8, open_fn=open_fn, monotonic=lambda: mono[0])
    source.read()                                # falha 1 -> próxima tentativa em 1 s
    mono[0] = 1.0
    source.read()                                # falha 2 -> próxima tentativa em 1 + 2 = 3 s
    mono[0] = 2.9
    source.read()
    assert len(opened) == 2
    mono[0] = 3.0
    source.read()
    assert len(opened) == 3


def test_file_source_steps_and_reports_video_time():
    cap = FakeCap([True] * 10, fps=25.0)
    source = FileSource("clip.mp4", target_fps=5, open_fn=lambda path: cap)  # 1 a cada 5 quadros
    assert source.fps == 25.0
    assert source.read().ts == 0.0
    assert source.read().ts == pytest.approx(0.2)
    assert source.read() is None
    assert source.exhausted is True
    assert cap.retrieved == 2


def test_file_source_missing_video():
    with pytest.raises(FileNotFoundError):
        FileSource("nada.mp4", 5, open_fn=lambda path: FakeCap(opened=False))
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_capture.py -q`
Expected: FAIL (módulo inexistente).

- [ ] **Step 3: Implementar `src/guarita/capture.py`**

```python
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray
    ts: float


class FrameThrottle:
    def __init__(self, target_fps: float):
        self._interval = 1.0 / target_fps
        self._last: float | None = None

    def should_process(self, ts: float) -> bool:
        if self._last is None or ts - self._last >= self._interval - 1e-9:
            self._last = ts
            return True
        return False


class RtspSource:
    exhausted = False

    def __init__(self, url: str, target_fps: float, open_fn=cv2.VideoCapture, clock=time.time,
                 monotonic=time.monotonic, max_backoff_s: float = 30.0, should_stop=lambda: False):
        self._url = url
        self._open_fn = open_fn
        self._clock = clock
        self._monotonic = monotonic
        self._max_backoff = max_backoff_s
        self._should_stop = should_stop
        self._throttle = FrameThrottle(target_fps)
        self._cap = None
        self._failures = 0
        self._next_open = 0.0

    def _schedule_retry(self) -> None:
        self._failures += 1
        delay = min(self._max_backoff, 2 ** (self._failures - 1))
        self._next_open = self._monotonic() + delay
        log.warning("Câmera indisponível; nova tentativa em %.0f s", delay)

    def _drop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._schedule_retry()

    def read(self) -> Frame | None:
        if self._cap is None:
            if self._monotonic() < self._next_open:
                return None
            cap = self._open_fn(self._url)
            if not cap.isOpened():
                cap.release()
                self._schedule_retry()
                return None
            self._cap = cap
            self._failures = 0
            log.info("Conectado ao stream da câmera")
        while not self._should_stop():
            if not self._cap.grab():
                self._drop()
                return None
            ts = self._clock()
            if self._throttle.should_process(ts):
                ok, image = self._cap.retrieve()
                if ok and image is not None:
                    return Frame(image, ts)
                self._drop()
                return None
        return None


class FileSource:
    def __init__(self, path: str, target_fps: float, open_fn=cv2.VideoCapture):
        self._cap = open_fn(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self._step = max(1, round(self.fps / target_fps))
        self._index = -1
        self.exhausted = False

    def read(self) -> Frame | None:
        while True:
            self._index += 1
            if not self._cap.grab():
                self.exhausted = True
                return None
            if self._index % self._step == 0:
                ok, image = self._cap.retrieve()
                if ok:
                    return Frame(image, self._index / self.fps)
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_capture.py -q`
Expected: PASS (6 testes).

- [ ] **Step 5: Commit**

```bash
git add src/guarita/capture.py tests/test_capture.py
git commit -m "feat: add RTSP source with reconnect/backoff and video file source"
```

---

### Task 5: Agregação, falhas e execução por câmera

**Files:**
- Create: `src/guarita/supervisor.py`, `src/guarita/pipeline.py`, `tests/test_supervisor.py`, `tests/test_pipeline.py`
- Modify: `tests/fakes.py`

**Interfaces:**
- Consumes: `vote_plate`, `OcrDispatcher`, `Outcome`, `OcrJob`, `TrackManager`, `Frame`, `Outbox.add`.
- Produces:
  - `Health(clock=time.monotonic)`: `touch()`, `count(name, n=1)`, `is_healthy(stale_after_s) -> bool` (False antes do primeiro `touch`), `snapshot() -> dict`; `metrics_line(camera_id, health, ocr_pending, outbox_counts) -> str`.
  - `FailureStore(directory, ttl_h, clock=time.time)`: `save(track, reason) -> path` (JPEG do melhor recorte `<início_ms>-<id>-<motivo>.jpg`), `purge(now=None) -> int`.
  - `Aggregator(camera: CameraConfig, cfg: PipelineConfig, failures: FailureStore | None)`: `handle(outcome) -> Reading | None` — vota; erro/timeout → salva falha e devolve `None`; placa igual dentro de `fuse_window_s` (comparando o horário do melhor recorte) → `None`; `Reading.external_id = f"{int(track.started_at*1000)}-{track.track_id}"`, `timestamp` = horário do melhor recorte, `image_jpeg` = JPEG (qualidade 90) do melhor recorte, `processing_ms = outcome.latency_ms`.
  - `CameraRunner(camera, cfg, source, detector, dispatcher, health=None)`: `step() -> 'frame' | 'idle' | 'done'`, `finish()` (fecha trilhas abertas e as envia ao OCR).
  - `drain_results(dispatcher, aggregators: dict[camera_id, Aggregator], sink) -> list[Reading]` (`sink.add(reading)` deve devolver `True` se aceitou); `ListSink` (coleta em memória, para `bench`/testes).
  - Testes: `tests/fakes.py` ganha `ScriptedSource(frames)` (`read()` devolve os quadros em ordem; depois `None` e `exhausted=True`) e `ScriptedDetector(script)` (`detect()` devolve a próxima lista).

- [ ] **Step 1: Estender `tests/fakes.py`**

Acrescentar ao final do arquivo:
```python
class ScriptedSource:
    """Fonte de quadros roteirizada: devolve `frames` em ordem e depois marca exhausted."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.exhausted = False

    def read(self):
        if self._frames:
            return self._frames.pop(0)
        self.exhausted = True
        return None


class ScriptedDetector:
    """Detector roteirizado: cada chamada devolve a próxima lista de Detection."""

    def __init__(self, script):
        self._script = list(script)

    def detect(self, frame):
        return self._script.pop(0) if self._script else []
```

- [ ] **Step 2: Escrever os testes que falham**

`tests/test_supervisor.py`:
```python
from guarita.supervisor import Health, metrics_line


def test_health_is_stale_until_the_first_frame_and_after_silence():
    clock = [100.0]
    health = Health(clock=lambda: clock[0])
    assert health.is_healthy(30) is False
    health.touch()
    assert health.is_healthy(30) is True
    clock[0] = 129.0
    assert health.is_healthy(30) is True
    clock[0] = 131.0
    assert health.is_healthy(30) is False


def test_counters_and_metrics_line():
    health = Health()
    health.count("frames", 3)
    health.count("frames")
    health.count("readings")
    assert health.snapshot() == {"frames": 4, "readings": 1}
    line = metrics_line("cam-01", health, ocr_pending=2, outbox_counts={"pending": 5, "delivered": 9, "failed": 1})
    for fragment in ("cam-01", "frames=4", "readings=1", "ocr_pendentes=2", "fila_pendente=5", "entregues=9", "falhas=1"):
        assert fragment in line
```

`tests/test_pipeline.py`:
```python
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fakes import FakeEngine, ScriptedDetector, ScriptedSource
from guarita.capture import Frame
from guarita.config import CameraConfig, PipelineConfig
from guarita.dispatch import OcrDispatcher, Outcome
from guarita.models import ClosedTrack, Crop, Detection
from guarita.ocr import worker
from guarita.ocr.worker import OcrResult
from guarita.pipeline import Aggregator, CameraRunner, FailureStore, ListSink, drain_results

CAMERA = CameraConfig(id="cam-01", direction="entrada", rtsp_url="", api_key="")


def track(ts=5.0, started=4.0, tid=7):
    crop = Crop(np.full((10, 30, 3), 128, np.uint8), ts=ts, score=1.0)
    return ClosedTrack("cam-01", tid, started, ts + 2, [crop])


def outcome(candidates, ts=5.0, tid=7):
    return Outcome(track(ts, tid=tid), OcrResult(candidates, 40), None, 350)


def test_aggregator_builds_a_reading_from_the_vote():
    agg = Aggregator(CAMERA, PipelineConfig(), None)
    reading = agg.handle(outcome([("ABC1234", 0.9), ("ABC1234", 0.7)]))
    assert reading.plate == "ABC1234"
    assert reading.confidence == pytest.approx(0.8)
    assert reading.camera_id == "cam-01" and reading.direction == "entrada"
    assert reading.external_id == "4000-7"
    assert reading.timestamp == 5.0
    assert reading.processing_ms == 350
    assert reading.image_jpeg[:2] == b"\xff\xd8"


def test_aggregator_returns_none_without_a_valid_plate():
    agg = Aggregator(CAMERA, PipelineConfig(), None)
    assert agg.handle(outcome([])) is None
    assert agg.handle(outcome([("ABCD123", 0.9)])) is None


def test_aggregator_saves_failures_for_timeouts_and_errors(tmp_path):
    store = FailureStore(str(tmp_path / "falhas"), ttl_h=24)
    agg = Aggregator(CAMERA, PipelineConfig(), store)
    assert agg.handle(Outcome(track(), None, "timeout", 10_000)) is None
    assert (tmp_path / "falhas" / "4000-7-timeout.jpg").exists()


def test_aggregator_fuses_the_same_plate_within_the_window():
    agg = Aggregator(CAMERA, PipelineConfig(fuse_window_s=20.0), None)
    assert agg.handle(outcome([("ABC1234", 0.9)], ts=100.0, tid=1)) is not None
    assert agg.handle(outcome([("ABC1234", 0.9)], ts=110.0, tid=2)) is None       # 10 s depois
    assert agg.handle(outcome([("ABC1235", 0.9)], ts=111.0, tid=3)) is not None   # outra placa
    assert agg.handle(outcome([("ABC1234", 0.9)], ts=125.0, tid=4)) is not None   # já passou a janela


def test_failure_store_purges_old_files(tmp_path):
    store = FailureStore(str(tmp_path), ttl_h=1)
    old = store.save(track(tid=1), "erro")
    new = store.save(track(tid=2), "erro")
    past = time.time() - 2 * 3600
    os.utime(old, (past, past))
    assert store.purge() == 1
    assert not os.path.exists(old) and os.path.exists(new)


def test_step_reports_idle_and_done():
    runner = CameraRunner(CAMERA, PipelineConfig(), ScriptedSource([]), ScriptedDetector([]), _dispatcher(FakeEngine()))
    assert runner.step() == "done"
    idle_source = ScriptedSource([])
    idle_source.read = lambda: None            # sem quadro, mas sem acabar
    runner = CameraRunner(CAMERA, PipelineConfig(), idle_source, ScriptedDetector([]), _dispatcher(FakeEngine()))
    assert runner.step() == "idle"


def _dispatcher(engine):
    def factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: engine,))
    return OcrDispatcher(factory, timeout_s=10.0)


def test_end_to_end_reading_lands_in_the_sink():
    cfg = PipelineConfig(max_crops=2, close_after_s=1.5)
    blank = np.zeros((100, 200, 3), np.uint8)
    frames = [Frame(blank, t) for t in (0.0, 0.5, 1.0, 2.5)]
    box = (10, 10, 60, 40)
    detections = [[Detection(1, box, 0.9)], [Detection(1, box, 0.9)], [], []]
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.8), ("ABC1234", 0.85), (None, None)])
    dispatcher = _dispatcher(engine)
    aggregators = {"cam-01": Aggregator(CAMERA, cfg, None)}
    sink = ListSink()
    runner = CameraRunner(CAMERA, cfg, ScriptedSource(frames), ScriptedDetector(detections), dispatcher)

    while runner.step() != "done":
        pass
    runner.finish()
    for _ in range(100):
        drain_results(dispatcher, aggregators, sink)
        if dispatcher.pending_count() == 0:
            break
        time.sleep(0.05)
    dispatcher.shutdown()

    assert len(engine.inputs) == 4                    # 2 recortes × (0° e 15°)
    assert len(sink.readings) == 1
    reading = sink.readings[0]
    assert reading.plate == "ABC1234"
    assert reading.confidence == pytest.approx(0.85)
    assert reading.external_id == "0-1"
    assert reading.timestamp == 0.0
    assert reading.direction == "entrada"
```

- [ ] **Step 3: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_supervisor.py tests/test_pipeline.py -q`
Expected: FAIL (módulos inexistentes).

- [ ] **Step 4: Implementar `src/guarita/supervisor.py`**

```python
from __future__ import annotations

import threading
import time
from collections import defaultdict


class Health:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._last: float | None = None
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def touch(self) -> None:
        with self._lock:
            self._last = self._clock()

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def is_healthy(self, stale_after_s: float) -> bool:
        with self._lock:
            return self._last is not None and self._clock() - self._last < stale_after_s

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)


def metrics_line(camera_id: str, health: Health, ocr_pending: int, outbox_counts: dict[str, int]) -> str:
    snap = health.snapshot()
    return (
        f"[{camera_id}] frames={snap.get('frames', 0)} deteccoes={snap.get('detections', 0)} "
        f"rastreios={snap.get('tracks', 0)} readings={snap.get('readings', 0)} "
        f"ocr_pendentes={ocr_pending} fila_pendente={outbox_counts.get('pending', 0)} "
        f"entregues={outbox_counts.get('delivered', 0)} falhas={outbox_counts.get('failed', 0)}"
    )
```

- [ ] **Step 5: Implementar `src/guarita/pipeline.py`**

```python
from __future__ import annotations

import logging
import os
import time

import cv2

from .config import CameraConfig, PipelineConfig
from .dispatch import OcrDispatcher, Outcome
from .models import ClosedTrack, Reading
from .ocr.vote import vote_plate
from .ocr.worker import OcrJob
from .supervisor import Health
from .track import TrackManager

log = logging.getLogger(__name__)


def _jpeg(image) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else b""


def external_id(track: ClosedTrack) -> str:
    return f"{int(track.started_at * 1000)}-{track.track_id}"


class FailureStore:
    """Guarda o melhor recorte das leituras que falharam (24 h, para análise)."""

    def __init__(self, directory: str, ttl_h: float, clock=time.time):
        self._dir = directory
        self._ttl_s = ttl_h * 3600
        self._clock = clock

    def save(self, track: ClosedTrack, reason: str) -> str:
        os.makedirs(self._dir, exist_ok=True)
        path = os.path.join(self._dir, f"{external_id(track)}-{reason}.jpg")
        with open(path, "wb") as fh:
            fh.write(_jpeg(track.crops[0].image))
        return path

    def purge(self, now: float | None = None) -> int:
        now = self._clock() if now is None else now
        removed = 0
        if not os.path.isdir(self._dir):
            return 0
        for name in os.listdir(self._dir):
            path = os.path.join(self._dir, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > self._ttl_s:
                os.remove(path)
                removed += 1
        return removed


class Aggregator:
    """Transforma o resultado do OCR de uma placa rastreada em uma leitura (ou descarta)."""

    def __init__(self, camera: CameraConfig, cfg: PipelineConfig, failures: FailureStore | None):
        self._camera = camera
        self._cfg = cfg
        self._failures = failures
        self._last_emitted: dict[str, float] = {}

    def handle(self, outcome: Outcome) -> Reading | None:
        track = outcome.track
        if outcome.error is not None or outcome.result is None:
            reason = "timeout" if outcome.error == "timeout" else "erro"
            log.warning("OCR falhou para %s (%s)", external_id(track), outcome.error)
            if self._failures is not None:
                self._failures.save(track, reason)
            return None

        plate, confidence = vote_plate(outcome.result.candidates)
        if plate is None:
            log.info("Sem leitura válida para %s", external_id(track))
            return None

        best = track.crops[0]
        last = self._last_emitted.get(plate)
        if last is not None and abs(best.ts - last) < self._cfg.fuse_window_s:
            log.info("Placa %s repetida dentro da janela; fundida", plate)
            return None
        self._last_emitted[plate] = best.ts
        if len(self._last_emitted) > 1000:
            cutoff = best.ts - 10 * self._cfg.fuse_window_s
            self._last_emitted = {p: t for p, t in self._last_emitted.items() if t >= cutoff}

        return Reading(
            camera_id=self._camera.id, external_id=external_id(track), plate=plate, confidence=confidence,
            timestamp=best.ts, direction=self._camera.direction, image_jpeg=_jpeg(best.image),
            processing_ms=outcome.latency_ms)


class CameraRunner:
    """Um passo do fluxo por câmera: quadro -> detecção/rastreio -> trilhas fechadas -> OCR."""

    def __init__(self, camera: CameraConfig, cfg: PipelineConfig, source, detector, dispatcher: OcrDispatcher,
                 health: Health | None = None):
        self.camera = camera
        self._cfg = cfg
        self.source = source
        self._detector = detector
        self._dispatcher = dispatcher
        self._health = health
        self._tracks = TrackManager(camera.id, cfg.close_after_s, cfg.max_age_s, cfg.max_crops)
        self._last_ts = 0.0

    def step(self) -> str:
        frame = self.source.read()
        if frame is None:
            return "done" if getattr(self.source, "exhausted", False) else "idle"
        if self._health:
            self._health.touch()
            self._health.count("frames")
        detections = self._detector.detect(frame.image)
        self._last_ts = frame.ts
        if self._health and detections:
            self._health.count("detections", len(detections))
        for track in self._tracks.update(frame.image, detections, frame.ts):
            self._submit(track)
        return "frame"

    def finish(self) -> None:
        for track in self._tracks.flush(self._last_ts):
            self._submit(track)

    def _submit(self, track: ClosedTrack) -> None:
        if self._health:
            self._health.count("tracks")
        job = OcrJob([c.image for c in track.crops], self._cfg.rotation_angles, self._cfg.min_ocr_conf)
        self._dispatcher.submit(track, job)


class ListSink:
    """Destino em memória (bench/testes) com a mesma interface de Outbox.add."""

    def __init__(self):
        self.readings: list[Reading] = []

    def add(self, reading: Reading, now=None) -> bool:
        self.readings.append(reading)
        return True


def drain_results(dispatcher: OcrDispatcher, aggregators: dict[str, Aggregator], sink) -> list[Reading]:
    accepted = []
    for outcome in dispatcher.poll():
        reading = aggregators[outcome.track.camera_id].handle(outcome)
        if reading is not None and sink.add(reading):
            accepted.append(reading)
    return accepted
```

- [ ] **Step 6: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_supervisor.py tests/test_pipeline.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/guarita/supervisor.py src/guarita/pipeline.py tests/fakes.py tests/test_supervisor.py tests/test_pipeline.py
git commit -m "feat: add aggregator, failure store, per-camera runner and health metrics"
```

---

### Task 6: Aplicação (threads), ponto de entrada e envio integrados

**Files:**
- Create: `src/guarita/app.py`, `src/guarita/__main__.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: tudo acima, `Outbox`, `ApiClient`, `run_sender`, `Config`.
- Produces: `App(cfg, *, source_factory, detector_factory, executor_factory, client, outbox)` com `.health: dict[camera_id, Health]`, `start()`, `stop()`. `source_factory(camera)` → fonte; `detector_factory(camera)` → detector; `executor_factory()` → executor com `submit/shutdown`. Threads: uma por câmera (loop `step()`), resultados (`drain_results` a cada 0,2 s + limpeza horária), envio (`run_sender`) e supervisor (heartbeat por câmera saudável a cada `heartbeat_interval_s` + linha de métricas). `stop()` fecha as trilhas abertas, espera o OCR pendente (até `ocr_timeout_s + 5` s), encerra as threads e o pool. `__main__.main(argv)` monta tudo com o PaddleOCR/YOLO reais, `ProcessPoolExecutor(initializer=init_worker, initargs=(make_paddle_engine,))`, log rotativo em `logs/` e encerra em SIGINT/SIGTERM.

- [ ] **Step 1: Escrever o teste que falha — `tests/test_app.py`**

```python
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fakes import FakeEngine, ScriptedDetector, ScriptedSource
from guarita.app import App
from guarita.capture import Frame
from guarita.config import (ApiConfig, CameraConfig, Config, DetectorConfig, OutboxConfig, PipelineConfig)
from guarita.models import Detection
from guarita.ocr import worker
from guarita.outbox import Outbox
from guarita.sender import ApiClient


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_frames_become_a_delivered_reading_and_heartbeats_flow(tmp_path, api):
    cfg = Config(
        api=ApiConfig(base_url=api.base_url),
        detector=DetectorConfig(),
        pipeline=PipelineConfig(max_crops=2, close_after_s=1.5, failures_dir=str(tmp_path / "falhas")),
        outbox=OutboxConfig(path=str(tmp_path / "o.sqlite")),
        cameras=(CameraConfig("cam-01", "entrada", "", "KEY"),),
        heartbeat_interval_s=0.2, stale_after_s=30.0,
    )
    blank = np.zeros((100, 200, 3), np.uint8)
    frames = [Frame(blank, t) for t in (0.0, 0.5, 1.0, 2.5)]
    box = (10, 10, 60, 40)
    detections = [[Detection(1, box, 0.9)], [Detection(1, box, 0.9)], [], []]
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.8), ("ABC1234", 0.85), (None, None)])
    outbox = Outbox(cfg.outbox.path)

    def executor_factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: engine,))

    app = App(cfg, source_factory=lambda cam: ScriptedSource(frames),
              detector_factory=lambda cam: ScriptedDetector(detections),
              executor_factory=executor_factory, client=ApiClient(api.base_url), outbox=outbox)
    app.start()
    try:
        assert wait_for(lambda: outbox.counts()["delivered"] == 1)
        assert wait_for(lambda: any(r["path"].endswith("/heartbeat") for r in api.requests))
    finally:
        app.stop()
        outbox.close()

    readings = [r for r in api.requests if r["path"] == "/api/ingest/readings"]
    assert len(readings) == 1
    assert readings[0]["headers"]["x-api-key"] == "KEY"
    assert b"ABC1234" in readings[0]["body"]
    assert app.health["cam-01"].snapshot()["frames"] == 4
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_app.py -q`
Expected: FAIL (`No module named 'guarita.app'`).

- [ ] **Step 3: Implementar `src/guarita/app.py`**

```python
from __future__ import annotations

import logging
import threading
import time

from .config import Config
from .dispatch import OcrDispatcher
from .outbox import Outbox
from .pipeline import Aggregator, CameraRunner, FailureStore, drain_results
from .sender import ApiClient, run_sender
from .supervisor import Health, metrics_line

log = logging.getLogger(__name__)


class App:
    def __init__(self, cfg: Config, *, source_factory, detector_factory, executor_factory,
                 client: ApiClient, outbox: Outbox):
        self.cfg = cfg
        self._client = client
        self._outbox = outbox
        self.stop_event = threading.Event()
        self.dispatcher = OcrDispatcher(executor_factory, cfg.pipeline.ocr_timeout_s)
        self._failures = FailureStore(cfg.pipeline.failures_dir, cfg.pipeline.failures_ttl_h)
        self.health = {c.id: Health() for c in cfg.cameras}
        self._aggregators = {c.id: Aggregator(c, cfg.pipeline, self._failures) for c in cfg.cameras}
        self._runners = {
            c.id: CameraRunner(c, cfg.pipeline, source_factory(c), detector_factory(c), self.dispatcher,
                               self.health[c.id])
            for c in cfg.cameras
        }
        self._api_keys = {c.id: c.api_key for c in cfg.cameras}
        self._camera_threads: list[threading.Thread] = []
        self._results_thread: threading.Thread | None = None
        self._other_threads: list[threading.Thread] = []

    def start(self) -> None:
        for camera in self.cfg.cameras:
            thread = threading.Thread(target=self._camera_loop, args=(camera.id,), name=f"camera-{camera.id}", daemon=True)
            thread.start()
            self._camera_threads.append(thread)
        self._results_thread = threading.Thread(target=self._results_loop, name="resultados", daemon=True)
        self._results_thread.start()
        for name, target in (("envio", self._sender_loop), ("supervisor", self._supervisor_loop)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._other_threads.append(thread)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self._camera_threads:
            thread.join(timeout=10)
        for runner in self._runners.values():
            runner.finish()
        if self._results_thread:
            self._results_thread.join(timeout=10)
        deadline = time.monotonic() + self.cfg.pipeline.ocr_timeout_s + 5
        while self.dispatcher.pending_count() and time.monotonic() < deadline:
            self._drain()
            time.sleep(0.1)
        self._drain()
        for thread in self._other_threads:
            thread.join(timeout=10)
        self.dispatcher.shutdown()

    def _drain(self) -> None:
        for reading in drain_results(self.dispatcher, self._aggregators, self._outbox):
            self.health[reading.camera_id].count("readings")

    def _camera_loop(self, camera_id: str) -> None:
        runner = self._runners[camera_id]
        while not self.stop_event.is_set():
            try:
                state = runner.step()
            except Exception:
                log.exception("Erro no processamento da câmera %s", camera_id)
                self.stop_event.wait(1.0)
                continue
            if state == "done":
                runner.finish()
                log.info("Fonte da câmera %s terminou", camera_id)
                return
            if state == "idle":
                self.stop_event.wait(0.05)

    def _results_loop(self) -> None:
        last_housekeeping = time.monotonic()
        while not self.stop_event.is_set():
            try:
                self._drain()
                if time.monotonic() - last_housekeeping > 3600:
                    self._outbox.purge_delivered(self.cfg.outbox.delivered_ttl_h * 3600)
                    self._failures.purge()
                    last_housekeeping = time.monotonic()
            except Exception:
                log.exception("Erro ao processar resultados do OCR")
            self.stop_event.wait(0.2)

    def _sender_loop(self) -> None:
        run_sender(self._outbox, self._client, self._api_keys, self.stop_event)

    def _supervisor_loop(self) -> None:
        while not self.stop_event.is_set():
            for camera in self.cfg.cameras:
                health = self.health[camera.id]
                if health.is_healthy(self.cfg.stale_after_s):
                    if self._client.heartbeat(camera.api_key):
                        health.count("heartbeats")
                else:
                    log.warning("Câmera %s sem quadros há mais de %.0f s; heartbeat suspenso",
                                camera.id, self.cfg.stale_after_s)
                log.info(metrics_line(camera.id, health, self.dispatcher.pending_count(), self._outbox.counts()))
            self.stop_event.wait(self.cfg.heartbeat_interval_s)
```

- [ ] **Step 4: Implementar `src/guarita/__main__.py`**

```python
from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import RotatingFileHandler

from .app import App
from .capture import RtspSource
from .config import ConfigError, load_config
from .detect import PlateDetector
from .ocr.engine import make_paddle_engine
from .ocr.worker import init_worker
from .outbox import Outbox
from .sender import ApiClient


def setup_logging(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    handlers = [
        RotatingFileHandler(os.path.join(log_dir, "guarita.log"), maxBytes=10_000_000, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="guarita", description="Coletor unificado da guarita")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args(argv)
    setup_logging()
    log = logging.getLogger("guarita")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error("Configuração inválida: %s", exc)
        return 2

    os.makedirs(os.path.dirname(cfg.outbox.path) or ".", exist_ok=True)
    outbox = Outbox(cfg.outbox.path, cfg.outbox.max_pending)

    def executor_factory():
        return ProcessPoolExecutor(
            max_workers=cfg.pipeline.ocr_workers, initializer=init_worker, initargs=(make_paddle_engine,))

    app = App(
        cfg,
        source_factory=lambda cam: RtspSource(cam.rtsp_url, cfg.pipeline.target_fps),
        detector_factory=lambda cam: PlateDetector(cfg.detector.weights, cfg.detector.imgsz, cfg.detector.conf),
        executor_factory=executor_factory,
        client=ApiClient(cfg.api.base_url, cfg.api.timeout_s),
        outbox=outbox,
    )

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    log.info("Coletor iniciado (%d câmera(s))", len(cfg.cameras))
    app.start()
    try:
        stop.wait()
    finally:
        log.info("Encerrando…")
        app.stop()
        outbox.close()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
```

- [ ] **Step 5: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_app.py -q`
Expected: PASS (1 teste; leva alguns segundos).

- [ ] **Step 6: Rodar a suíte inteira e commitar**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (Etapas 1 e 2 até aqui).
```bash
git add src/guarita/app.py src/guarita/__main__.py tests/test_app.py
git commit -m "feat: add threaded app orchestration and service entry point"
```

---

### Task 7: Medição (`bench`), verificação do ambiente e documentação

**Files:**
- Create: `src/guarita/evaluation.py`, `tests/test_evaluation.py`, `tools/bench.py`, `tools/check_env.py`, `service/install-service.md`, `README.md`

**Interfaces:**
- Produces (em `guarita.evaluation`): `score(truth, results, window=5.0) -> Score(total, correct, wrong, missed, extra)` com `.accuracy` (`correct/total`; `truth`/`results` são listas de `(t_segundos, placa)`); `legacy_rows(rows, fps) -> [(t, placa, score)]` (`rows` = `(frame_nmr, placa, score)`; ignora `frame_nmr < 0`); `read_rows(path)`/`write_rows(path, rows)` (CSV `t_seconds,plate[,confidence]`); `run_offline(cfg, camera, source, detector, executor_factory) -> (rows, frames)` (roda o fluxo sobre uma fonte até o fim e devolve `[(t, placa, confiança)]`).
- CLI: `python tools/bench.py run --config config.toml --video clip.mp4 --out novos.csv`; `python tools/bench.py score --truth gabarito.csv --results novos.csv [--window 5]`; `python tools/bench.py legacy --dsn "..." --fps 25 --frame-min N --frame-max M --out antigos.csv`.

- [ ] **Step 1: Escrever o teste que falha — `tests/test_evaluation.py`**

```python
import numpy as np
import pytest

from fakes import FakeEngine, ScriptedDetector, ScriptedSource
from concurrent.futures import ThreadPoolExecutor

from guarita.capture import Frame
from guarita.config import CameraConfig, Config, ApiConfig, DetectorConfig, OutboxConfig, PipelineConfig
from guarita.evaluation import legacy_rows, read_rows, run_offline, score, write_rows
from guarita.models import Detection
from guarita.ocr import worker


def test_score_counts_correct_wrong_missed_and_extra():
    truth = [(10.0, "ABC1234"), (30.0, "DEF5678"), (60.0, "GHI9012")]
    results = [(11.0, "ABC1234"), (12.0, "ABC1234"),      # certo + duplicata (extra)
               (29.0, "DEF5670"),                         # leu outra placa
               (200.0, "ZZZ9999")]                        # sem veículo rotulado (extra)
    result = score(truth, results, window=5.0)
    assert (result.total, result.correct, result.wrong, result.missed, result.extra) == (3, 1, 1, 1, 2)
    assert result.accuracy == pytest.approx(1 / 3)


def test_score_window_boundary_and_order_independence():
    assert score([(10.0, "ABC1234")], [(15.0, "ABC1234")], window=5.0).correct == 1
    assert score([(10.0, "ABC1234")], [(15.1, "ABC1234")], window=5.0).missed == 1
    assert score([(20.0, "B"), (10.0, "A")], [(10.5, "A"), (20.5, "B")]).correct == 2


def test_score_with_no_truth():
    result = score([], [(1.0, "ABC1234")])
    assert result.total == 0 and result.extra == 1 and result.accuracy == 0.0


def test_legacy_rows_convert_frames_to_seconds():
    rows = [(250, "ABC1234", 0.9), (-1, "XYZ", 0.1), (0, "DEF5678", 0.8)]
    assert legacy_rows(rows, fps=25.0) == [(10.0, "ABC1234", 0.9), (0.0, "DEF5678", 0.8)]


def test_csv_round_trip(tmp_path):
    path = tmp_path / "r.csv"
    write_rows(str(path), [(1.5, "ABC1234", 0.9), (2.0, "DEF5678", 0.7)])
    assert read_rows(str(path)) == [(1.5, "ABC1234"), (2.0, "DEF5678")]


def test_run_offline_collects_readings_from_a_source():
    cfg = Config(
        api=ApiConfig(base_url="x"), detector=DetectorConfig(),
        pipeline=PipelineConfig(max_crops=2, close_after_s=1.5), outbox=OutboxConfig(),
        cameras=(CameraConfig("cam-01", "entrada", "", ""),))
    blank = np.zeros((100, 200, 3), np.uint8)
    frames = [Frame(blank, t) for t in (0.0, 0.5, 1.0, 2.5)]
    box = (10, 10, 60, 40)
    detections = [[Detection(1, box, 0.9)], [Detection(1, box, 0.9)], [], []]
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.8), ("ABC1234", 0.85), (None, None)])

    def executor_factory():
        return ThreadPoolExecutor(max_workers=1, initializer=worker.init_worker, initargs=(lambda: engine,))

    rows, frame_count = run_offline(cfg, cfg.cameras[0], ScriptedSource(frames), ScriptedDetector(detections),
                                    executor_factory)
    assert frame_count == 4
    assert len(rows) == 1
    assert rows[0][0] == 0.0 and rows[0][1] == "ABC1234"
    assert rows[0][2] == pytest.approx(0.85)
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_evaluation.py -q`
Expected: FAIL (`No module named 'guarita.evaluation'`).

- [ ] **Step 3: Implementar `src/guarita/evaluation.py`**

```python
from __future__ import annotations

import csv
import time
from dataclasses import dataclass

from .config import CameraConfig, Config
from .dispatch import OcrDispatcher
from .pipeline import Aggregator, CameraRunner, ListSink, drain_results


@dataclass(frozen=True)
class Score:
    total: int
    correct: int
    wrong: int
    missed: int
    extra: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def score(truth, results, window: float = 5.0) -> Score:
    """Compara leituras com o gabarito.

    `truth` e `results` são listas de (t_segundos, placa). Cada veículo do gabarito consome, no máximo,
    uma leitura dentro da janela: exata -> correct; só de outra placa -> wrong; nenhuma -> missed.
    Leituras que sobram (duplicatas ou veículos sem gabarito) contam como extra.
    """
    remaining = list(range(len(results)))
    correct = wrong = missed = 0
    for t, plate in sorted(truth):
        near = [i for i in remaining if abs(results[i][0] - t) <= window]
        exact = [i for i in near if results[i][1] == plate]
        if exact:
            remaining.remove(min(exact, key=lambda i: abs(results[i][0] - t)))
            correct += 1
        elif near:
            remaining.remove(min(near, key=lambda i: abs(results[i][0] - t)))
            wrong += 1
        else:
            missed += 1
    return Score(len(truth), correct, wrong, missed, len(remaining))


def legacy_rows(rows, fps: float):
    """(frame_nmr, placa, score) do banco antigo -> (segundos_de_vídeo, placa, score)."""
    return [(frame / fps, plate, conf) for frame, plate, conf in rows if frame >= 0]


def write_rows(path: str, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_seconds", "plate", "confidence"])
        for row in rows:
            writer.writerow(row if len(row) == 3 else (*row, ""))


def read_rows(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        return [(float(r["t_seconds"]), r["plate"]) for r in csv.DictReader(fh)]


def run_offline(cfg: Config, camera: CameraConfig, source, detector, executor_factory):
    """Roda o fluxo completo sobre uma fonte finita (arquivo de vídeo). Devolve (linhas, quadros)."""
    dispatcher = OcrDispatcher(executor_factory, cfg.pipeline.ocr_timeout_s)
    aggregators = {camera.id: Aggregator(camera, cfg.pipeline, None)}
    sink = ListSink()
    runner = CameraRunner(camera, cfg.pipeline, source, detector, dispatcher)
    frames = 0
    while True:
        state = runner.step()
        if state == "done":
            break
        if state == "frame":
            frames += 1
        drain_results(dispatcher, aggregators, sink)
    runner.finish()
    deadline = time.monotonic() + cfg.pipeline.ocr_timeout_s * 3
    while dispatcher.pending_count() and time.monotonic() < deadline:
        drain_results(dispatcher, aggregators, sink)
        time.sleep(0.05)
    drain_results(dispatcher, aggregators, sink)
    dispatcher.shutdown()
    rows = [(r.timestamp, r.plate, r.confidence) for r in sink.readings]
    return rows, frames
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_evaluation.py -q`
Expected: PASS (6 testes).

- [ ] **Step 5: Criar `tools/bench.py`**

```python
"""Medição do coletor em vídeos gravados (precisão e CPU).

  python tools/bench.py run    --config config.toml --video clip.mp4 --out novos.csv
  python tools/bench.py legacy --dsn "dbname=pci_transito user=postgres password=... host=localhost" \
                               --fps 25 --frame-min 0 --frame-max 90000 --out antigos.csv
  python tools/bench.py score  --truth gabarito.csv --results novos.csv [--window 5]

gabarito.csv: colunas t_seconds,plate (segundo do vídeo em que o veículo passa e a placa correta).
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guarita.capture import FileSource  # noqa: E402
from guarita.config import load_config  # noqa: E402
from guarita.evaluation import legacy_rows, read_rows, run_offline, score, write_rows  # noqa: E402


def cmd_run(args) -> int:
    import psutil

    from guarita.detect import PlateDetector
    from guarita.ocr.engine import make_paddle_engine
    from guarita.ocr.worker import init_worker

    cfg = load_config(args.config, require_secrets=False)
    camera = cfg.cameras[0]
    source = FileSource(args.video, cfg.pipeline.target_fps)
    detector = PlateDetector(cfg.detector.weights, cfg.detector.imgsz, cfg.detector.conf)

    def executor_factory():
        return ProcessPoolExecutor(max_workers=cfg.pipeline.ocr_workers, initializer=init_worker,
                                   initargs=(make_paddle_engine,))

    samples: list[float] = []
    stop = threading.Event()

    def sampler():
        psutil.cpu_percent(interval=None)
        while not stop.is_set():
            samples.append(psutil.cpu_percent(interval=1.0))

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    started = time.perf_counter()
    rows, frames = run_offline(cfg, camera, source, detector, executor_factory)
    elapsed = time.perf_counter() - started
    stop.set()
    thread.join(timeout=3)

    write_rows(args.out, rows)
    video_seconds = frames / cfg.pipeline.target_fps
    print(f"leituras: {len(rows)}  quadros processados: {frames}  tempo: {elapsed:.1f}s "
          f"(vídeo ~{video_seconds:.1f}s)")
    if samples:
        print(f"CPU da máquina: média {sum(samples) / len(samples):.0f}%  pico {max(samples):.0f}%")
    print(f"resultado gravado em {args.out}")
    return 0


def cmd_legacy(args) -> int:
    import psycopg2  # apenas para esta ferramenta

    with psycopg2.connect(args.dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT frame_nmr, license_number, license_number_score FROM transito_leitura_placa "
            "WHERE frame_nmr BETWEEN %s AND %s ORDER BY frame_nmr", (args.frame_min, args.frame_max))
        rows = cur.fetchall()
    write_rows(args.out, legacy_rows(rows, args.fps))
    print(f"{len(rows)} leituras do sistema antigo gravadas em {args.out}")
    return 0


def cmd_score(args) -> int:
    result = score(read_rows(args.truth), read_rows(args.results), args.window)
    print(f"veículos no gabarito: {result.total}")
    print(f"  placa exata:        {result.correct}  ({result.accuracy:.1%})")
    print(f"  placa errada:       {result.wrong}")
    print(f"  perdidos:           {result.missed}")
    print(f"leituras extras (duplicatas ou sem gabarito): {result.extra}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", default="config.toml")
    run.add_argument("--video", required=True)
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    legacy = sub.add_parser("legacy")
    legacy.add_argument("--dsn", required=True)
    legacy.add_argument("--fps", type=float, required=True)
    legacy.add_argument("--frame-min", type=int, default=0)
    legacy.add_argument("--frame-max", type=int, required=True)
    legacy.add_argument("--out", required=True)
    legacy.set_defaults(func=cmd_legacy)

    sc = sub.add_parser("score")
    sc.add_argument("--truth", required=True)
    sc.add_argument("--results", required=True)
    sc.add_argument("--window", type=float, default=5.0)
    sc.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
```

- [ ] **Step 6: Criar `tools/check_env.py`**

```python
"""Verifica se o ambiente completo (runtime) consegue carregar YOLO e PaddleOCR.

  python tools/check_env.py [caminho/para/bestn.pt]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ok = True
    for name in ("numpy", "cv2", "torch", "ultralytics", "paddle", "paddleocr", "requests", "psutil"):
        try:
            module = importlib.import_module(name)
            print(f"OK   {name} {getattr(module, '__version__', '')}")
        except Exception as exc:
            ok = False
            print(f"FALHA {name}: {exc}")

    if ok:
        import numpy as np

        from guarita.ocr.engine import PaddleEngine

        try:
            PaddleEngine().read(np.full((60, 200, 3), 255, np.uint8))
            print("OK   PaddleOCR inicializou e leu uma imagem em branco")
        except Exception as exc:
            ok = False
            print(f"FALHA PaddleOCR: {exc}")

    if len(sys.argv) > 1:
        from guarita.detect import PlateDetector

        try:
            detector = PlateDetector(sys.argv[1])
            detector.detect(np.zeros((480, 640, 3), np.uint8))
            print(f"OK   YOLO carregou {sys.argv[1]} e rodou o rastreio")
        except Exception as exc:
            ok = False
            print(f"FALHA YOLO: {exc}")
    else:
        print("(passe o caminho do bestn.pt para testar também o detector)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Criar `service/install-service.md`**

````markdown
# Instalar o coletor como serviço do Windows (NSSM)

1. Instale o ambiente completo (ver `README.md`) em `C:\guarita\.venv` e copie `bestn.pt` e `config.toml` para `C:\guarita\`.
2. Baixe o NSSM (https://nssm.cc) e, em um PowerShell **como administrador**:

```powershell
nssm install GuaritaColetor "C:\guarita\.venv\Scripts\python.exe" "-m guarita --config C:\guarita\config.toml"
nssm set GuaritaColetor AppDirectory C:\guarita
nssm set GuaritaColetor Start SERVICE_AUTO_START
nssm set GuaritaColetor AppExit Default Restart
nssm set GuaritaColetor AppRestartDelay 5000
nssm set GuaritaColetor AppStdout C:\guarita\logs\nssm.out.log
nssm set GuaritaColetor AppStderr C:\guarita\logs\nssm.err.log
# Segredos e opções do OpenCV (uma linha por variável):
nssm set GuaritaColetor AppEnvironmentExtra "GUARITA_RTSP_CAM01=rtsp://usuario:SENHA@10.12.8.63:554/cam/realmonitor?channel=1&subtype=0" "GUARITA_API_KEY_CAM01=CHAVE_DA_CAMERA" "OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|stimeout;5000000"
nssm start GuaritaColetor
```

- `OPENCV_FFMPEG_CAPTURE_OPTIONS` faz o RTSP usar TCP e desistir de leituras paradas em 5 s (senão um stream travado bloqueia o coletor sem erro).
- A chave da API é criada no dashboard: `docker compose exec api npm run create-camera -- cam-01 "Portaria Principal" "Entrada Leste" yolov8n`.
- No dashboard, ajuste `ALERT_CAMERA_OFFLINE_MIN` para 10–15 (padrão 120) para o alerta de "câmera sem sinal" chegar em tempo útil.
- Sincronize o relógio do Windows por NTP: a API recusa leituras com horário mais de 5 min no futuro.
- Logs: `C:\guarita\logs\guarita.log` (rotativo). Uma linha de métricas por minuto por câmera.
````

- [ ] **Step 8: Criar `README.md`**

````markdown
# GuaritaColetor

Coletor unificado da guarita (IFSULDEMINAS · Muzambinho): lê a câmera RTSP, detecta e rastreia placas (YOLO + ByteTrack),
lê cada veículo uma vez com PaddleOCR (recortes em 0° e 15°, votação por posição) e envia as leituras ao dashboard
`GestaoTransito` com fila local, reenvio e heartbeat. Substitui `GuaritaIdentificador` + `GuaritaLeitor` + o Flask antigo.

Especificação: `docs/superpowers/specs/2026-09-21-coletor-unificado-design.md`.

## Ambientes

Dois ambientes separados (não misture `opencv-python` com `opencv-python-headless`):

```powershell
# 1) Testes (leve, sem torch/paddle)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q

# 2) Runtime completo (guarita e bench)
python -m venv .venv-run
.venv-run\Scripts\python -m pip install -r requirements.txt
.venv-run\Scripts\python tools\check_env.py caminho\para\bestn.pt
```
Se o `pip` não resolver as versões juntas (torch/paddle/numpy), ajuste os pins de `requirements.txt` até o `check_env.py` passar.

## Configuração e execução

1. Copie `config.example.toml` para `config.toml` e ajuste (câmeras, direção fixa, limiares).
2. Defina `GUARITA_RTSP_CAM01` e `GUARITA_API_KEY_CAM01` como variáveis de ambiente (o `config.toml` nunca guarda segredos).
3. `python -m guarita --config config.toml` (ou como serviço: `service/install-service.md`).

O programa **não inicia** sem o arquivo de pesos do detector.

## Medir antes de trocar (critério de aceite)

```powershell
# vídeo gravado + gabarito (t_seconds,plate) de ~100 veículos
python tools\bench.py run    --config config.toml --video clip.mp4 --out novos.csv
python tools\bench.py legacy --dsn "dbname=pci_transito user=postgres password=... host=localhost" --fps 25 --frame-min 0 --frame-max 90000 --out antigos.csv
python tools\bench.py score  --truth gabarito.csv --results novos.csv
python tools\bench.py score  --truth gabarito.csv --results antigos.csv
```
Aceite da versão 1: precisão de placa exata **≥** a do sistema antigo, veículos perdidos **≤**, latência do fechamento até a
leitura pronta < 10 s (p95) e CPU com folga para uma segunda câmera. Só então desligue os programas antigos.

## Estrutura

`src/guarita/`: `capture` (RTSP/arquivo) → `detect` (YOLO+ByteTrack) → `track` (melhores recortes) → `dispatch` +
`ocr/` (pool de OCR, regras de placa, votação) → `pipeline` (agregação e fusão) → `outbox` (SQLite) → `sender` (API).
`app.py` liga tudo em threads; `supervisor.py` mantém saúde e métricas.
````

- [ ] **Step 9: Rodar a suíte inteira**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (todas as suítes das Etapas 1 e 2).

- [ ] **Step 10: Commit**

```bash
git add src/guarita/evaluation.py tests/test_evaluation.py tools service README.md
git commit -m "feat: add offline benchmark tools, env check, service and docs"
```

## Verificação final da etapa

- [ ] `.venv\Scripts\python -m pytest -q` — todas as suítes passam (as duas etapas).
- [ ] No ambiente completo (`.venv-run`): `python tools\check_env.py bestn.pt` imprime `OK` para todas as linhas. **Este é o primeiro contato real com Ultralytics 8.0.114 + ByteTrack + PaddleOCR juntos**: se a instalação ou o `model.track` falharem, ajustar `requirements.txt` (versões) e/ou `compat.py` antes de seguir.
- [ ] `python tools\bench.py run ...` em um trecho curto de vídeo real gera `novos.csv` e imprime CPU; `score` contra o gabarito e contra o CSV do sistema antigo compara precisão.
- [ ] Ainda a definir com dados reais (fora do código): valores de `target_fps`, `close_after_s`, `max_crops` e a decisão sobre exportar o detector para OpenVINO/ONNX (só se o `bench` mostrar resultado idêntico).
