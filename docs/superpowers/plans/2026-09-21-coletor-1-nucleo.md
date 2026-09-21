# Coletor unificado · Etapa 1 — Núcleo (config, regras de placa, votação, OCR, fila e envio) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir a parte do coletor que não depende de câmera nem de modelos pesados: configuração, regras de placa portadas do `util.py`, votação entre leituras, leitura OCR com rotações, fila SQLite (outbox) e cliente de envio ao dashboard.

**Architecture:** Pacote Python `guarita` (layout `src/`). Cada módulo tem uma responsabilidade e é testável sem torch/paddle (o PaddleOCR é injetado por uma interface `engine.read(img_rgb) -> (plate, conf)`). O envio usa uma fila SQLite durável e um cliente HTTP `requests` com classificação de respostas.

**Tech Stack:** Python 3.12, numpy, OpenCV, requests, SQLite (`sqlite3`), pytest. (PaddleOCR/Ultralytics só entram na Etapa 2.)

**Spec:** `docs/superpowers/specs/2026-09-21-coletor-unificado-design.md`

## Global Constraints

- Repositório: `C:\Users\Gabriel\GuaritaColetor`; código em `src/guarita/`, testes em `tests/`, ferramentas em `tools/`.
- O OCR roda **sempre** em todos os ângulos configurados (padrão **0° e 15°**), com a mesma matriz de rotação do `main.py` atual: `cv2.getRotationMatrix2D((w // 2, h // 2), -angulo, 1.0)`, `cv2.warpAffine(..., flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)`.
- Pré-processamento do OCR idêntico ao do leitor atual: recorte colorido → rotação → escala de cinza → RGB (`cv2.COLOR_GRAY2RGB`).
- As regras de correção de placa são **portadas literalmente** do `GuaritaLeitor09-07/util.py` (mapas `char_to_int`/`int_to_char`, padrões `LLLNNNN` e `LLLNLNN`, `PALAVRAS_IGNORAR`, conversões conservadoras). Nenhuma mudança de comportamento, mesmo onde o comportamento é peculiar (ver testes de caracterização).
- Confiança do OCR trabalhada em 0–1 internamente; convertida para 0–100 apenas no envio. Confiança mínima por candidato: `0.1` (mesmo valor do leitor atual).
- Placa final só é aceita se `license_complies_format` retornar padrão válido.
- Segredos (URL RTSP, chaves de API) vêm de variáveis de ambiente, nunca do arquivo de configuração nem do repositório.
- Envio: uma leitura só sai da fila após resposta `200`/`201`; backoff `min(300, 2^tentativas)` segundos; `401` pausa o envio; `4xx` (exceto 401/408/429/413/415) é falha definitiva; `413`/`415` descarta a imagem e reenvia.
- Direção fixa por câmera: `entrada` ou `saída`.
- Sem `print`: usar `logging`.

## Estrutura de arquivos desta etapa

```
pyproject.toml  requirements.txt  requirements-dev.txt  .gitignore  config.example.toml
src/guarita/
  __init__.py  models.py  config.py  outbox.py  sender.py
  ocr/__init__.py  ocr/plates.py  ocr/vote.py  ocr/engine.py  ocr/worker.py
tests/
  conftest.py  fakes.py  fake_api.py
  test_config.py  test_plates.py  test_vote.py  test_engine.py  test_outbox.py  test_sender.py
```

---

### Task 1: Scaffold do projeto, modelos de dados e configuração

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `config.example.toml`, `src/guarita/__init__.py`, `src/guarita/models.py`, `src/guarita/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Detection(track_id: int, box: (x1,y1,x2,y2), conf: float)` (frozen); `Crop(image: np.ndarray, ts: float, score: float)`; `ClosedTrack(camera_id, track_id, started_at, closed_at, crops: list[Crop])`; `Reading(camera_id, external_id, plate, confidence (0–1), timestamp (epoch s), direction, image_jpeg: bytes, processing_ms: int)` (frozen).
  - `load_config(path, env=None, require_secrets=True) -> Config`; `ConfigError`; dataclasses `Config(api, detector, pipeline, outbox, cameras, heartbeat_interval_s=60.0, stale_after_s=30.0)`, `ApiConfig(base_url, timeout_s=15.0)`, `DetectorConfig(weights='bestn.pt', imgsz=416, conf=0.1)`, `PipelineConfig(target_fps=8.0, close_after_s=1.5, max_age_s=20.0, max_crops=5, fuse_window_s=20.0, rotation_angles=(0.0, 15.0), min_ocr_conf=0.1, ocr_workers=2, ocr_timeout_s=10.0, failures_dir='data/failures', failures_ttl_h=24.0)`, `OutboxConfig(path='data/outbox.sqlite', max_pending=50000, delivered_ttl_h=24.0)`, `CameraConfig(id, direction, rtsp_url, api_key)`.

- [ ] **Step 1: Criar os arquivos de projeto**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "guarita-coletor"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```
`requirements-dev.txt` (ambiente leve para os testes; **não** instalar junto com `opencv-python` do runtime):
```
numpy>=2.0.0,<2.3.0
opencv-python-headless>=4.9
requests>=2.31
psutil>=5.9
pytest>=8.0
```
`requirements.txt` (runtime completo; mesmas versões que já funcionam nos dois programas atuais):
```
ultralytics==8.0.114
torch==2.5.1
torchvision==0.20.1
opencv-python==4.12.0.88
numpy>=2.0.0,<2.3.0
scipy==1.16.1
filterpy==1.4.5
lapx>=0.5.5
paddlepaddle==3.1.0
paddleocr==2.10.0
requests>=2.31
psutil>=5.9
```
`.gitignore`:
```
__pycache__/
*.py[cod]
.venv/
env-*/
data/
logs/
*.sqlite
*.sqlite-wal
*.sqlite-shm
.env
*.pt
.pytest_cache/
*.egg-info/
```
`config.example.toml`:
```toml
heartbeat_interval_s = 60
stale_after_s = 30

[api]
base_url = "https://campusinteligente.ifsuldeminas.edu.br/gestaotransito/api"
timeout_s = 15

[detector]
weights = "bestn.pt"
imgsz = 416
conf = 0.1

[pipeline]
target_fps = 8
close_after_s = 1.5
max_age_s = 20
max_crops = 5
fuse_window_s = 20
rotation_angles = [0, 15]
min_ocr_conf = 0.1
ocr_workers = 2
ocr_timeout_s = 10
failures_dir = "data/failures"
failures_ttl_h = 24

[outbox]
path = "data/outbox.sqlite"
max_pending = 50000
delivered_ttl_h = 24

# URL RTSP e chave da API vêm de variáveis de ambiente (nunca escreva o segredo aqui).
[[cameras]]
id = "cam-01"
direction = "entrada"          # "entrada" ou "saída"
rtsp_env = "GUARITA_RTSP_CAM01"
api_key_env = "GUARITA_API_KEY_CAM01"
```

- [ ] **Step 2: Criar `src/guarita/__init__.py` e `src/guarita/models.py`**

`src/guarita/__init__.py`:
```python
"""Coletor unificado de placas da guarita."""
```
`src/guarita/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 em pixels do quadro original
    conf: float


@dataclass
class Crop:
    image: np.ndarray
    ts: float
    score: float


@dataclass
class ClosedTrack:
    camera_id: str
    track_id: int
    started_at: float
    closed_at: float
    crops: list[Crop]  # melhor primeiro


@dataclass(frozen=True)
class Reading:
    camera_id: str
    external_id: str
    plate: str
    confidence: float  # 0..1
    timestamp: float  # epoch em segundos (do quadro do melhor recorte)
    direction: str
    image_jpeg: bytes
    processing_ms: int
```

- [ ] **Step 3: Escrever o teste que falha — `tests/test_config.py`**

```python
import pytest

from guarita.config import ConfigError, load_config

BASE = """
[api]
base_url = "https://exemplo.local/gestaotransito/api"

[[cameras]]
id = "cam-01"
direction = "entrada"
rtsp_env = "GUARITA_RTSP_CAM01"
api_key_env = "GUARITA_API_KEY_CAM01"
"""
ENV = {"GUARITA_RTSP_CAM01": "rtsp://u:p@10.0.0.1/stream", "GUARITA_API_KEY_CAM01": "chave"}


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_and_secrets_from_env(tmp_path):
    cfg = load_config(write(tmp_path, BASE), env=ENV)
    assert cfg.api.base_url.endswith("/api")
    assert cfg.api.timeout_s == 15.0
    assert cfg.detector.weights == "bestn.pt"
    assert cfg.pipeline.rotation_angles == (0.0, 15.0)
    assert cfg.pipeline.max_crops == 5
    assert cfg.pipeline.min_ocr_conf == 0.1
    cam = cfg.cameras[0]
    assert (cam.id, cam.direction) == ("cam-01", "entrada")
    assert cam.rtsp_url == "rtsp://u:p@10.0.0.1/stream"
    assert cam.api_key == "chave"


def test_overrides(tmp_path):
    text = BASE + "\n[pipeline]\ntarget_fps = 5\nrotation_angles = [0, 15, -15]\n"
    cfg = load_config(write(tmp_path, text), env=ENV)
    assert cfg.pipeline.target_fps == 5
    assert cfg.pipeline.rotation_angles == (0.0, 15.0, -15.0)


def test_accepts_saida_with_accent(tmp_path):
    cfg = load_config(write(tmp_path, BASE.replace("entrada", "saída")), env=ENV)
    assert cfg.cameras[0].direction == "saída"


def test_missing_env_var_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="GUARITA_RTSP_CAM01"):
        load_config(write(tmp_path, BASE), env={})


def test_secrets_optional_for_offline_tools(tmp_path):
    cfg = load_config(write(tmp_path, BASE), env={}, require_secrets=False)
    assert cfg.cameras[0].rtsp_url == ""
    assert cfg.cameras[0].api_key == ""


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="desconhecidas"):
        load_config(write(tmp_path, BASE + "\n[pipeline]\ntarget_fpss = 5\n"), env=ENV)
    with pytest.raises(ConfigError, match="desconhecidas"):
        load_config(write(tmp_path, BASE + "\nfoo = 1\n"), env=ENV)


def test_invalid_direction(tmp_path):
    with pytest.raises(ConfigError, match="direction"):
        load_config(write(tmp_path, BASE.replace("entrada", "norte")), env=ENV)


def test_duplicate_camera_ids(tmp_path):
    second = '\n[[cameras]]\nid = "cam-01"\ndirection = "saída"\nrtsp_env = "GUARITA_RTSP_CAM01"\napi_key_env = "GUARITA_API_KEY_CAM01"\n'
    with pytest.raises(ConfigError, match="duplicad"):
        load_config(write(tmp_path, BASE + second), env=ENV)


def test_requires_api_and_a_camera(tmp_path):
    with pytest.raises(ConfigError, match=r"\[api\]"):
        load_config(write(tmp_path, "[pipeline]\ntarget_fps = 5\n"), env=ENV)
    with pytest.raises(ConfigError, match="câmera"):
        load_config(write(tmp_path, '[api]\nbase_url = "x"\n'), env=ENV)


def test_invalid_values(tmp_path):
    with pytest.raises(ConfigError, match="target_fps"):
        load_config(write(tmp_path, BASE + "\n[pipeline]\ntarget_fps = 0\n"), env=ENV)
    with pytest.raises(ConfigError, match="rotation_angles"):
        load_config(write(tmp_path, BASE + "\n[pipeline]\nrotation_angles = []\n"), env=ENV)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="não encontrado"):
        load_config(tmp_path / "nao-existe.toml", env=ENV)
```

- [ ] **Step 4: Rodar e confirmar a falha**

Run: `python -m venv .venv && .venv\Scripts\python -m pip install -r requirements-dev.txt && .venv\Scripts\python -m pytest tests/test_config.py -q`
Expected: FAIL (`No module named 'guarita.config'`).

- [ ] **Step 5: Implementar `src/guarita/config.py`**

```python
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

DIRECTIONS = ("entrada", "saída")


class ConfigError(ValueError):
    """Configuração inválida ou incompleta."""


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    timeout_s: float = 15.0


@dataclass(frozen=True)
class DetectorConfig:
    weights: str = "bestn.pt"
    imgsz: int = 416
    conf: float = 0.1


@dataclass(frozen=True)
class PipelineConfig:
    target_fps: float = 8.0
    close_after_s: float = 1.5
    max_age_s: float = 20.0
    max_crops: int = 5
    fuse_window_s: float = 20.0
    rotation_angles: tuple[float, ...] = (0.0, 15.0)
    min_ocr_conf: float = 0.1
    ocr_workers: int = 2
    ocr_timeout_s: float = 10.0
    failures_dir: str = "data/failures"
    failures_ttl_h: float = 24.0


@dataclass(frozen=True)
class OutboxConfig:
    path: str = "data/outbox.sqlite"
    max_pending: int = 50000
    delivered_ttl_h: float = 24.0


@dataclass(frozen=True)
class CameraConfig:
    id: str
    direction: str
    rtsp_url: str
    api_key: str


@dataclass(frozen=True)
class Config:
    api: ApiConfig
    detector: DetectorConfig
    pipeline: PipelineConfig
    outbox: OutboxConfig
    cameras: tuple[CameraConfig, ...]
    heartbeat_interval_s: float = 60.0
    stale_after_s: float = 30.0


_TOP_LEVEL = {"api", "detector", "pipeline", "outbox", "cameras", "heartbeat_interval_s", "stale_after_s"}
_CAMERA_KEYS = {"id", "direction", "rtsp_env", "api_key_env"}


def _section(cls, data: dict, name: str):
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"[{name}] chaves desconhecidas: {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"[{name}] {exc}") from exc


def _validate_pipeline(p: PipelineConfig) -> None:
    if p.target_fps <= 0:
        raise ConfigError("[pipeline] target_fps deve ser maior que 0")
    if p.max_crops < 1:
        raise ConfigError("[pipeline] max_crops deve ser pelo menos 1")
    if not p.rotation_angles:
        raise ConfigError("[pipeline] rotation_angles não pode ser vazio")
    if p.ocr_workers < 1:
        raise ConfigError("[pipeline] ocr_workers deve ser pelo menos 1")


def _camera(raw: dict, env: Mapping[str, str], require_secrets: bool) -> CameraConfig:
    unknown = set(raw) - _CAMERA_KEYS
    if unknown:
        raise ConfigError(f"[[cameras]] chaves desconhecidas: {sorted(unknown)}")
    missing = _CAMERA_KEYS - set(raw)
    if missing:
        raise ConfigError(f"[[cameras]] faltam chaves: {sorted(missing)}")
    if raw["direction"] not in DIRECTIONS:
        raise ConfigError(f"câmera {raw['id']}: direction deve ser 'entrada' ou 'saída'")
    rtsp = env.get(raw["rtsp_env"], "")
    key = env.get(raw["api_key_env"], "")
    if require_secrets:
        for var, value in ((raw["rtsp_env"], rtsp), (raw["api_key_env"], key)):
            if not value:
                raise ConfigError(f"Variável de ambiente {var} não definida (câmera {raw['id']})")
    return CameraConfig(id=raw["id"], direction=raw["direction"], rtsp_url=rtsp, api_key=key)


def load_config(path, env: Mapping[str, str] | None = None, require_secrets: bool = True) -> Config:
    env = os.environ if env is None else env
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Arquivo de configuração não encontrado: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOML inválido: {exc}") from exc

    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ConfigError(f"chaves desconhecidas na raiz: {sorted(unknown)}")
    if "api" not in raw:
        raise ConfigError("a seção [api] é obrigatória")

    pipeline_raw = dict(raw.get("pipeline", {}))
    if "rotation_angles" in pipeline_raw:
        pipeline_raw["rotation_angles"] = tuple(float(a) for a in pipeline_raw["rotation_angles"])
    pipeline = _section(PipelineConfig, pipeline_raw, "pipeline")
    _validate_pipeline(pipeline)

    cameras = tuple(_camera(c, env, require_secrets) for c in raw.get("cameras", []))
    if not cameras:
        raise ConfigError("Defina ao menos uma câmera em [[cameras]]")
    ids = [c.id for c in cameras]
    if len(set(ids)) != len(ids):
        raise ConfigError("ids de câmera duplicados")

    extra = {k: raw[k] for k in ("heartbeat_interval_s", "stale_after_s") if k in raw}
    return Config(
        api=_section(ApiConfig, dict(raw["api"]), "api"),
        detector=_section(DetectorConfig, dict(raw.get("detector", {})), "detector"),
        pipeline=pipeline,
        outbox=_section(OutboxConfig, dict(raw.get("outbox", {})), "outbox"),
        cameras=cameras,
        **extra,
    )
```

- [ ] **Step 6: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_config.py -q`
Expected: PASS (11 testes).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt .gitignore config.example.toml src/guarita tests/test_config.py
git commit -m "feat: scaffold project, data models and config loader"
```

---

### Task 2: Regras de placa (portagem literal do `util.py`)

**Files:**
- Create: `src/guarita/ocr/__init__.py`, `src/guarita/ocr/plates.py`, `tests/test_plates.py`

**Interfaces:**
- Produces (em `guarita.ocr.plates`): `CHAR_TO_INT`, `INT_TO_CHAR`, `PALAVRAS_IGNORAR`; `is_letter_candidate(ch) -> bool`, `is_digit_candidate(ch) -> bool`, `license_complies_format(text) -> 'LLLNNNN' | 'LLLNLNN' | None`, `is_naturally_correct_for_position(ch, target) -> bool`, `format_char(ch, target) -> str`, `corrigir_placa(text, pattern_code) -> str`, `limpar_texto_placa(texto) -> str`, `read_plate_from_detections(detections) -> (plate | None, score | None)` onde `detections` é a lista `results[0]` do PaddleOCR (`[[box, (texto, score)], ...]`).

- [ ] **Step 1: Escrever o teste de caracterização que falha — `tests/test_plates.py`**

Estes testes fixam o comportamento **atual** do `util.py`, inclusive o fato de que, em posições de número, letras conversíveis (`O I J A G S B`) são mantidas como letra.

```python
import pytest

from guarita.ocr.plates import (
    PALAVRAS_IGNORAR,
    corrigir_placa,
    format_char,
    license_complies_format,
    limpar_texto_placa,
    read_plate_from_detections,
)


def box(x, y=0):
    return [[x, y], [x + 30, y], [x + 30, y + 10], [x, y + 10]]


@pytest.mark.parametrize("text,expected", [
    ("ABC1234", "LLLNNNN"),
    ("abc1234", "LLLNNNN"),
    ("ABC1D23", "LLLNLNN"),
    ("ABC1A23", "LLLNNNN"),   # 'A' também é candidato a dígito: o padrão clássico vence
    ("0BC1234", "LLLNNNN"),   # '0' é candidato a letra
    ("ABCD123", None),
    ("ABC123", None),
    ("", None),
    (None, None),
])
def test_license_complies_format(text, expected):
    assert license_complies_format(text) == expected


@pytest.mark.parametrize("text,pattern,expected", [
    ("ABC1234", "LLLNNNN", "ABC1234"),
    ("abc1234", "LLLNNNN", "ABC1234"),
    ("0BC1234", "LLLNNNN", "OBC1234"),   # dígito em posição de letra vira letra
    ("A1C1234", "LLLNNNN", "AIC1234"),
    ("ABCI234", "LLLNNNN", "ABCI234"),   # letra em posição de número é mantida
    ("ABC1D23", "LLLNLNN", "ABC1D23"),
    ("abc", None, "ABC"),                # sem padrão: só maiúsculas
    ("ABC1234", "OUTRO", "ABC1234"),
])
def test_corrigir_placa(text, pattern, expected):
    assert corrigir_placa(text, pattern) == expected


@pytest.mark.parametrize("ch,target,expected", [
    ("0", "L", "O"), ("1", "L", "I"), ("3", "L", "J"), ("4", "L", "A"),
    ("6", "L", "G"), ("5", "L", "S"), ("8", "L", "B"),
    ("2", "L", "2"),          # sem mapeamento: mantém
    ("a", "L", "A"),
    ("O", "N", "O"),          # letra conversível em posição de número: mantém
    ("5", "N", "5"),
    ("L", "N", "L"),
])
def test_format_char(ch, target, expected):
    assert format_char(ch, target) == expected


def test_limpar_texto_placa():
    assert limpar_texto_placa(" a-b.c ") == "ABC"
    assert limpar_texto_placa("AB C") == "AB C"
    assert limpar_texto_placa("ab!1@2#") == "AB12"


def test_palavras_ignorar_contem_termos_conhecidos():
    assert {"BRASIL", "MERCOSUL", "MUZAMBINHO", "MG", "GUAXUPE"} <= PALAVRAS_IGNORAR


def test_read_single_detection():
    assert read_plate_from_detections([[box(0), ("ABC-1234", 0.9)]]) == ("ABC1234", 0.9)


def test_read_combines_multiple_detections_in_x_order():
    detections = [[box(50), ("1234", 0.6)], [box(10), ("ABC", 0.8)]]
    plate, score = read_plate_from_detections(detections)
    assert plate == "ABC1234"
    assert score == pytest.approx(0.7)


def test_read_ignores_state_and_country_words():
    detections = [[box(10), ("BRASIL", 0.9)], [box(20), ("ABC1234", 0.8)]]
    assert read_plate_from_detections(detections) == ("ABC1234", 0.8)


def test_read_falls_back_to_individual_detection():
    detections = [[box(10), ("XX", 0.9)], [box(20), ("ABC1D23", 0.7)]]
    assert read_plate_from_detections(detections) == ("ABC1D23", 0.7)


@pytest.mark.parametrize("detections", [None, [], [[box(0), ("HELLO", 0.9)]]])
def test_read_returns_none_without_valid_plate(detections):
    assert read_plate_from_detections(detections) == (None, None)
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_plates.py -q`
Expected: FAIL (`No module named 'guarita.ocr'`).

- [ ] **Step 3: Implementar `src/guarita/ocr/__init__.py` e `src/guarita/ocr/plates.py`**

`src/guarita/ocr/__init__.py`: arquivo vazio.

`src/guarita/ocr/plates.py`:
```python
"""Regras de placa portadas literalmente de GuaritaLeitor09-07/util.py."""
from __future__ import annotations

import logging
import string

log = logging.getLogger(__name__)

CHAR_TO_INT = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8'}
INT_TO_CHAR = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B'}

PALAVRAS_IGNORAR = frozenset({
    'BRASIL', 'MERCOSUL', 'BRAZIL',
    'SP', 'SAOPAULO', 'RJ', 'RIODEJANEIRO', 'MG', 'MINASGERAIS',
    'ES', 'ESPIRITOSANTO', 'PR', 'PARANA', 'SC', 'SANTACATARINA', 'RS', 'RIOGRANDEDOSUL',
    'MS', 'MATOGROSSODOSUL', 'MT', 'MATOGROSSO', 'GO', 'GOIAS', 'DF', 'DISTRITOFEDERAL',
    'BA', 'BAHIA', 'SE', 'SERGIPE', 'AL', 'ALAGOAS', 'PE', 'PERNAMBUCO', 'PB', 'PARAIBA',
    'RN', 'RIOGRANDEDONORTE', 'CE', 'CEARA', 'PI', 'PIAUI', 'MA', 'MARANHAO', 'TO', 'TOCANTINS',
    'PA', 'PARA', 'AP', 'AMAPA', 'RR', 'RORAIMA', 'AM', 'AMAZONAS', 'AC', 'ACRE', 'RO', 'RONDONIA',
    'GUAXUPE', 'MUZAMBINHO', 'NOVAREZENDE', 'SAOPEDRODAUNIAO', 'JURUAIA',
    'PASSOS', 'POCOSDECALDAS', 'VARGINHA', 'ALFENAS', 'LAVRAS', 'UBERLANDIA', 'UBERABA',
    'BELOHORIZONTE', 'JUIZDEFORA', 'MONTESCLAROS', 'DIVINOPOLIS', 'GOVERNADORVALADARES',
    'CAMPINAS', 'RIBEIRAOPRETO', 'FRANCA', 'SAOJOSEDORIOPRETO', 'BAURU',
    'MUNICIPIO', 'ESTADO', 'UF', 'BR',
    'FRENTE', 'TRAS', 'LATERAL', 'AUTO', 'MOTO', 'CAMINHAO', 'CARRO',
})


def is_letter_candidate(char_val: str) -> bool:
    """Letra natural ou número conversível em letra."""
    return char_val.upper() in string.ascii_uppercase or char_val in INT_TO_CHAR


def is_digit_candidate(char_val: str) -> bool:
    """Dígito natural ou letra conversível em dígito."""
    return char_val.isdigit() or char_val.upper() in CHAR_TO_INT


def license_complies_format(text):
    if not text or len(text) != 7:
        return None

    # Padrão 1: LLLNNNN (clássico brasileiro)
    if (is_letter_candidate(text[0]) and is_letter_candidate(text[1]) and is_letter_candidate(text[2])
            and is_digit_candidate(text[3]) and is_digit_candidate(text[4])
            and is_digit_candidate(text[5]) and is_digit_candidate(text[6])):
        return "LLLNNNN"

    # Padrão 2: LLLNLNN (Mercosul)
    if (is_letter_candidate(text[0]) and is_letter_candidate(text[1]) and is_letter_candidate(text[2])
            and is_digit_candidate(text[3]) and is_letter_candidate(text[4])
            and is_digit_candidate(text[5]) and is_digit_candidate(text[6])):
        return "LLLNLNN"

    return None


def is_naturally_correct_for_position(char_val: str, target_type: str) -> bool:
    if target_type == 'L':
        return char_val.isalpha() and char_val.upper() in string.ascii_uppercase
    if target_type == 'N':
        return char_val.isdigit()
    return False


def format_char(char_val: str, target_type: str) -> str:
    """Corrige um caractere para letra ('L') ou dígito ('N'), de forma conservadora."""
    char_upper = char_val.upper()

    if is_naturally_correct_for_position(char_upper, target_type):
        return char_upper

    if target_type == 'L':
        if char_val in INT_TO_CHAR:
            converted = INT_TO_CHAR[char_val]
            log.debug("Convertendo número '%s' para letra '%s'", char_val, converted)
            return converted
        if char_upper.isalpha():
            return char_upper
        log.debug("Caractere '%s' mantido sem conversão (posição de letra)", char_val)
        return char_upper

    if target_type == 'N':
        if char_upper in CHAR_TO_INT:
            if char_upper.isalpha() and char_upper in string.ascii_uppercase:
                log.debug("Letra '%s' mantida (válida para posição de número em Mercosul)", char_upper)
                return char_upper
            converted = CHAR_TO_INT[char_upper]
            log.debug("Convertendo '%s' para número '%s' (ambíguo)", char_upper, converted)
            return converted
        if char_val.isdigit():
            return char_val
        log.debug("Caractere '%s' mantido sem conversão (posição de número)", char_val)
        return char_upper

    return char_upper


def corrigir_placa(text: str, pattern_code) -> str:
    if not pattern_code or len(text) != 7:
        return text.upper()
    if pattern_code not in ("LLLNNNN", "LLLNLNN"):
        return text.upper()

    corrected = list(text)
    for i in range(7):
        corrected[i] = format_char(corrected[i], pattern_code[i])
    final_plate = "".join(corrected)
    if final_plate != text:
        log.debug("Placa corrigida '%s' -> '%s' (%s)", text, final_plate, pattern_code)
    return final_plate


def limpar_texto_placa(texto: str) -> str:
    caracteres_remover = "-.!@#$%^&*()[]{};:,<>?/\\|`~'\""
    return ''.join(c for c in texto if c not in caracteres_remover).strip().upper()


def read_plate_from_detections(detections):
    """Lógica de ler_placas2() aplicada à saída do PaddleOCR (results[0]).

    Retorna (placa, score) ou (None, None).
    """
    if not detections:
        return None, None

    if len(detections) > 1:
        try:
            ordered = sorted(detections, key=lambda det: (det[0][0][0], det[0][0][1]))
        except TypeError:
            ordered = detections

        texts, scores = [], []
        for item in ordered:
            processed = limpar_texto_placa(item[1][0])
            if processed and processed not in PALAVRAS_IGNORAR:
                texts.append(processed)
                scores.append(item[1][1])

        if texts:
            combined = "".join(texts)
            pattern = license_complies_format(combined)
            if pattern:
                avg = sum(scores) / len(scores) if scores else 0.0
                return corrigir_placa(combined, pattern), avg

    for item in detections:
        processed = limpar_texto_placa(item[1][0])
        pattern = license_complies_format(processed)
        if pattern:
            return corrigir_placa(processed, pattern), item[1][1]

    return None, None
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_plates.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/guarita/ocr tests/test_plates.py
git commit -m "feat(ocr): port plate format rules from util.py with characterization tests"
```

---

### Task 3: Votação entre leituras

**Files:**
- Create: `src/guarita/ocr/vote.py`, `tests/test_vote.py`

**Interfaces:**
- Consumes: `license_complies_format`.
- Produces: `vote_plate(candidates: list[tuple[str, float]]) -> tuple[str | None, float | None]`. Considera só candidatos de 7 caracteres; vota posição a posição somando a confiança de cada caractere; desempate: maior confiança individual, depois o caractere menor; o resultado precisa validar em `license_complies_format`. Confiança final = `média das confianças × média da concordância por posição` (concordância = peso do vencedor ÷ peso total da posição).

- [ ] **Step 1: Escrever o teste que falha — `tests/test_vote.py`**

```python
import pytest

from guarita.ocr.vote import vote_plate


def test_single_candidate_is_returned_as_is():
    assert vote_plate([("ABC1234", 0.9)]) == ("ABC1234", pytest.approx(0.9))


def test_agreeing_candidates_average_confidence():
    plate, conf = vote_plate([("ABC1234", 0.9), ("ABC1234", 0.7)])
    assert plate == "ABC1234"
    assert conf == pytest.approx(0.8)


def test_majority_wins_and_confidence_is_penalized_by_disagreement():
    plate, conf = vote_plate([("ABC1234", 0.9), ("ABC1284", 0.6), ("ABC1234", 0.8)])
    assert plate == "ABC1234"
    assert conf == pytest.approx(0.7381, abs=1e-3)


def test_weights_beat_counts():
    # duas leituras fracas contra uma forte na posição 6
    plate, _ = vote_plate([("ABC1234", 0.3), ("ABC1234", 0.3), ("ABC1235", 0.9)])
    assert plate == "ABC1235"


def test_tie_prefers_higher_single_confidence_then_smaller_char():
    assert vote_plate([("ABC1234", 0.8), ("ABC1235", 0.9)])[0] == "ABC1235"
    assert vote_plate([("ABC1234", 0.5), ("ABC1235", 0.5)])[0] == "ABC1234"


def test_can_mix_positions_from_different_candidates():
    plate, _ = vote_plate([("ABC1D23", 0.9), ("ABC1234", 0.5)])
    assert plate == "ABC1D23"


def test_invalid_voted_plate_is_rejected():
    assert vote_plate([("ABCD123", 0.9)]) == (None, None)


def test_ignores_wrong_length_and_handles_empty():
    assert vote_plate([("ABC123", 0.9)]) == (None, None)
    assert vote_plate([]) == (None, None)
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_vote.py -q`
Expected: FAIL (módulo inexistente).

- [ ] **Step 3: Implementar `src/guarita/ocr/vote.py`**

```python
from __future__ import annotations

from collections import defaultdict

from .plates import license_complies_format


def vote_plate(candidates):
    """Combina leituras do mesmo veículo, posição a posição. Retorna (placa, confiança) ou (None, None)."""
    valid = [(p, c) for p, c in candidates if p and len(p) == 7]
    if not valid:
        return None, None

    chars: list[str] = []
    agreement: list[float] = []
    for i in range(7):
        weights: dict[str, float] = defaultdict(float)
        best_single: dict[str, float] = {}
        for plate, conf in valid:
            ch = plate[i]
            weights[ch] += conf
            best_single[ch] = max(best_single.get(ch, 0.0), conf)
        total = sum(weights.values())
        winner = max(weights, key=lambda ch: (weights[ch], best_single[ch], -ord(ch)))
        chars.append(winner)
        agreement.append(weights[winner] / total if total > 0 else 0.0)

    plate = "".join(chars)
    if license_complies_format(plate) is None:
        return None, None
    mean_conf = sum(c for _, c in valid) / len(valid)
    return plate, mean_conf * (sum(agreement) / 7)
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_vote.py -q`
Expected: PASS (8 testes).

- [ ] **Step 5: Commit**

```bash
git add src/guarita/ocr/vote.py tests/test_vote.py
git commit -m "feat(ocr): add weighted per-position plate voting"
```

---

### Task 4: Leitura OCR com rotações e worker

**Files:**
- Create: `src/guarita/ocr/engine.py`, `src/guarita/ocr/worker.py`, `tests/fakes.py`, `tests/fake_api.py`, `tests/conftest.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: `read_plate_from_detections`.
- Produces:
  - `rotate_crop(img, angle) -> np.ndarray`; `to_ocr_input(crop) -> np.ndarray (h,w,3) uint8`; `read_candidates(engine, crop, angles, min_conf) -> list[(plate, conf)]` (descarta `conf <= min_conf`); `PaddleEngine().read(img_rgb) -> (plate|None, score|None)`; `make_paddle_engine() -> PaddleEngine` (função de nível de módulo, serializável para o pool de processos).
  - `worker.OcrJob(crops: list[np.ndarray], angles: tuple[float,...], min_conf: float)`; `worker.OcrResult(candidates: list[(str,float)], elapsed_ms: int)`; `worker.init_worker(engine_factory)`; `worker.process_job(job) -> OcrResult`.
  - Testes: `tests/fakes.py` com `FakeEngine(script)` (devolve os resultados do `script` em ordem, `(None, None)` depois; guarda `inputs`) e `make_reading(n=1, plate='ABC1234', camera_id='cam-01', image=b'jpg') -> Reading`.

- [ ] **Step 1: Criar `tests/fakes.py` e `tests/conftest.py`**

`tests/fakes.py`:
```python
from __future__ import annotations

from guarita.models import Reading


class FakeEngine:
    """Substitui o PaddleOCR: devolve resultados em ordem e guarda as imagens recebidas."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.inputs = []

    def read(self, img_rgb):
        self.inputs.append(img_rgb)
        if self.script:
            return self.script.pop(0)
        return None, None


def make_reading(n=1, plate="ABC1234", camera_id="cam-01", image=b"jpg", confidence=0.9):
    return Reading(
        camera_id=camera_id, external_id=f"ext-{n}", plate=plate, confidence=confidence,
        timestamp=1_760_000_000.0 + n, direction="entrada", image_jpeg=image, processing_ms=120,
    )
```
`tests/fake_api.py` (servidor HTTP local usado pelos testes de envio na Tarefa 6):
```python
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeApi:
    """Servidor HTTP local que registra as requisições e responde com status roteirizados."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[int] = []  # status a devolver, em ordem; padrão 201 (204 no heartbeat)
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                api.requests.append({
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                })
                default = 204 if self.path.endswith("/heartbeat") else 201
                status = api.responses.pop(0) if api.responses else default
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/api"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
```
`tests/conftest.py`:
```python
import pytest

from fake_api import FakeApi


@pytest.fixture
def api():
    server = FakeApi()
    server.start()
    yield server
    server.stop()
```
Para o pytest achar `fakes.py`/`fake_api.py`, acrescente em `pyproject.toml`, em `[tool.pytest.ini_options]`: `pythonpath = ["src", "tests"]` (substitui a linha existente).

- [ ] **Step 2: Escrever o teste que falha — `tests/test_engine.py`**

```python
import numpy as np

from fakes import FakeEngine
from guarita.ocr import worker
from guarita.ocr.engine import read_candidates, rotate_crop, to_ocr_input


def color_crop(h=20, w=60):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_rotate_zero_returns_an_equal_copy():
    img = color_crop()
    out = rotate_crop(img, 0)
    assert np.array_equal(out, img)
    assert out is not img


def test_rotate_15_keeps_shape_and_changes_pixels():
    img = color_crop()
    out = rotate_crop(img, 15)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_to_ocr_input_from_bgr_and_gray():
    bgr = np.zeros((10, 30, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # vermelho puro
    out = to_ocr_input(bgr)
    assert out.shape == (10, 30, 3)
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    gray = np.full((10, 30), 100, dtype=np.uint8)
    assert to_ocr_input(gray).shape == (10, 30, 3)


def test_read_candidates_runs_every_angle_and_filters_by_confidence():
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.05)])
    out = read_candidates(engine, color_crop(), angles=(0, 15), min_conf=0.1)
    assert out == [("ABC1234", 0.9)]
    assert len(engine.inputs) == 2                       # 0° e 15° sempre
    assert all(i.ndim == 3 and i.shape[2] == 3 for i in engine.inputs)


def test_read_candidates_skips_unreadable_crops():
    engine = FakeEngine([(None, None), ("ABC1234", 0.8)])
    assert read_candidates(engine, color_crop(), (0, 15), 0.1) == [("ABC1234", 0.8)]


def test_process_job_reads_every_crop_at_every_angle():
    engine = FakeEngine([("ABC1234", 0.9), ("ABC1234", 0.8), (None, None), ("ABC1235", 0.7)])
    worker.init_worker(lambda: engine)
    try:
        result = worker.process_job(worker.OcrJob([color_crop(), color_crop()], (0.0, 15.0), 0.1))
    finally:
        worker.init_worker(lambda: None)
    assert len(engine.inputs) == 4
    assert result.candidates == [("ABC1234", 0.9), ("ABC1234", 0.8), ("ABC1235", 0.7)]
    assert result.elapsed_ms >= 0
```

- [ ] **Step 3: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_engine.py -q`
Expected: FAIL (`No module named 'guarita.ocr.engine'`).

- [ ] **Step 4: Implementar `src/guarita/ocr/engine.py`**

```python
from __future__ import annotations

import logging

import cv2
import numpy as np

from .plates import read_plate_from_detections

log = logging.getLogger(__name__)


def rotate_crop(img: np.ndarray, angle: float) -> np.ndarray:
    """Mesma rotação do main.py atual (matriz com -angulo, borda replicada)."""
    if angle == 0:
        return img.copy()
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def to_ocr_input(crop: np.ndarray) -> np.ndarray:
    """Mesmo pré-processamento do leitor atual: cinza -> RGB."""
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def read_candidates(engine, crop: np.ndarray, angles, min_conf: float):
    """Lê o recorte em todos os ângulos; devolve [(placa, confiança)] com confiança > min_conf."""
    candidates = []
    for angle in angles:
        plate, conf = engine.read(to_ocr_input(rotate_crop(crop, angle)))
        if plate is not None and conf is not None and conf > min_conf:
            candidates.append((plate, float(conf)))
    return candidates


class PaddleEngine:
    """PaddleOCR com os mesmos parâmetros do leitor atual."""

    def __init__(self):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(lang="pt", use_angle_cls=False, use_gpu=False, det_limit_side_len=640)

    def read(self, img_rgb: np.ndarray):
        try:
            results = self._ocr.ocr(img_rgb, cls=False)
        except Exception:
            log.exception("Erro durante o OCR")
            return None, None
        if not results or not results[0]:
            return None, None
        return read_plate_from_detections(results[0])


def make_paddle_engine() -> PaddleEngine:
    return PaddleEngine()
```

- [ ] **Step 5: Implementar `src/guarita/ocr/worker.py`**

```python
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .engine import read_candidates

_engine = None


@dataclass
class OcrJob:
    crops: list[np.ndarray]
    angles: tuple[float, ...]
    min_conf: float


@dataclass
class OcrResult:
    candidates: list[tuple[str, float]]
    elapsed_ms: int


def init_worker(engine_factory) -> None:
    """Inicializador do processo/thread do pool: cria o motor de OCR uma vez."""
    global _engine
    _engine = engine_factory()


def process_job(job: OcrJob) -> OcrResult:
    start = time.perf_counter()
    candidates: list[tuple[str, float]] = []
    for crop in job.crops:
        candidates.extend(read_candidates(_engine, crop, job.angles, job.min_conf))
    return OcrResult(candidates, int((time.perf_counter() - start) * 1000))
```

- [ ] **Step 6: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_engine.py -q`
Expected: PASS (6 testes).

- [ ] **Step 7: Commit**

```bash
git add src/guarita/ocr/engine.py src/guarita/ocr/worker.py tests/fakes.py tests/conftest.py tests/fake_api.py tests/test_engine.py pyproject.toml
git commit -m "feat(ocr): add rotated OCR reading, paddle engine wrapper and pool worker"
```

---

### Task 5: Fila SQLite (outbox)

**Files:**
- Create: `src/guarita/outbox.py`, `tests/test_outbox.py`

**Interfaces:**
- Consumes: `Reading`.
- Produces: `Outbox(path, max_pending=50000)` com: `add(reading, now=None) -> bool` (False se `(camera_id, external_id)` já existe); `next_due(now=None) -> OutboxRow | None` (mais antigo `pending` com `next_attempt_at <= now`); `mark_delivered(id, now=None)` (limpa a imagem); `mark_retry(id, error, now=None) -> float` (atraso `min(300, 2^tentativas_anteriores)`); `mark_failed(id, error)`; `drop_image(id)`; `counts() -> {'pending','delivered','failed'}`; `pending_count()`; `purge_delivered(older_than_s, now=None) -> int`; `close()`. `OutboxRow(id, external_id, camera_id, plate, direction, ts, confidence, processing_ms, image, attempts)`. Quando o número de pendentes **com imagem** passa de `max_pending`, as imagens das mais antigas são descartadas (o texto fica).

- [ ] **Step 1: Escrever o teste que falha — `tests/test_outbox.py`**

```python
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
```

- [ ] **Step 2: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_outbox.py -q`
Expected: FAIL (módulo inexistente).

- [ ] **Step 3: Implementar `src/guarita/outbox.py`**

```python
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
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_outbox.py -q`
Expected: PASS (9 testes).

- [ ] **Step 5: Commit**

```bash
git add src/guarita/outbox.py tests/test_outbox.py
git commit -m "feat: add durable SQLite outbox with backoff and image trimming"
```

---

### Task 6: Cliente de envio ao dashboard

**Files:**
- Create: `src/guarita/sender.py`, `tests/test_sender.py` (o servidor de teste `tests/fake_api.py` já foi criado na Tarefa 4)

**Interfaces:**
- Consumes: `Outbox`, `OutboxRow`.
- Produces: `ApiClient(base_url, timeout_s=15.0, session=None)` com `post_reading(api_key, row) -> (status_code, text)` e `heartbeat(api_key) -> bool`; `Delivery(outcome, detail='')` com `outcome ∈ {'delivered','retry','failed','auth','idle'}`; `deliver_next(outbox, client, api_keys: dict[camera_id, key], now=None) -> Delivery`; `run_sender(outbox, client, api_keys, stop: threading.Event, idle_s=1.0, auth_pause_s=30.0)`.
- Regras: com imagem → `multipart/form-data` (campos `external_id, plate, direction, timestamp, confidence (0–100), processing_ms`, arquivo `image`); sem imagem → JSON (`application/json`). Timestamp ISO 8601 com fuso local. `200/201` → entregue; `401` → `auth` (linha continua pendente); `413/415` com imagem → descarta a imagem e devolve `retry` sem atraso; `408/429/5xx` e erro de rede → `retry` com backoff; demais `4xx` → `failed`; câmera sem chave → `failed`.

- [ ] **Step 1: Criar `tests/fake_api.py`**

```python
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeApi:
    """Servidor HTTP local que registra as requisições e responde com status roteirizados."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[int] = []  # status a devolver, em ordem; padrão 201 (204 no heartbeat)
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                api.requests.append({
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                })
                default = 204 if self.path.endswith("/heartbeat") else 201
                status = api.responses.pop(0) if api.responses else default
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/api"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
```

- [ ] **Step 2: Escrever o teste que falha — `tests/test_sender.py`**

```python
import json
import time

import pytest

from fakes import make_reading
from guarita.outbox import Outbox
from guarita.sender import ApiClient, deliver_next

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
```

- [ ] **Step 3: Rodar e confirmar a falha**

Run: `.venv\Scripts\python -m pytest tests/test_sender.py -q`
Expected: FAIL (`No module named 'guarita.sender'`).

- [ ] **Step 4: Implementar `src/guarita/sender.py`**

```python
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime

import requests

from .outbox import Outbox, OutboxRow

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Delivery:
    outcome: str  # delivered | retry | failed | auth | idle
    detail: str = ""


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")


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

    if status in (200, 201):
        outbox.mark_delivered(row.id, now)
        return Delivery("delivered")
    if status == 401:
        log.error("API recusou a chave da câmera %s (401); envio pausado", row.camera_id)
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
               idle_s: float = 1.0, auth_pause_s: float = 30.0) -> None:
    while not stop.is_set():
        result = deliver_next(outbox, client, api_keys)
        if result.outcome == "delivered":
            continue
        stop.wait(auth_pause_s if result.outcome == "auth" else idle_s)
```

- [ ] **Step 5: Rodar e confirmar que passa**

Run: `.venv\Scripts\python -m pytest tests/test_sender.py -q`
Expected: PASS.

- [ ] **Step 6: Rodar a suíte inteira e commitar**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (todas as suítes da Etapa 1).

```bash
git add src/guarita/sender.py tests/fake_api.py tests/test_sender.py
git commit -m "feat: add dashboard API client and outbox delivery with retry rules"
```

## Verificação final da etapa

- [ ] `.venv\Scripts\python -m pytest -q` — todas as suítes passam (config, plates, vote, engine, outbox, sender).
- [ ] Conferir que `tests/test_plates.py` cobre o comportamento peculiar do `util.py` (letras mantidas em posições de número) — esse é o contrato de "nenhuma mudança de precisão".
