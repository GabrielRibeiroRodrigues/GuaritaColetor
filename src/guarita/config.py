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
