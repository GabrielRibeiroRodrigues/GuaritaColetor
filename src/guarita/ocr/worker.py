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


def warmup() -> bool:
    """Tarefa trivial que força o pool a subir os processos e a terminar o `init_worker`
    (spawn + imports + carga dos modelos). Falha se o motor não foi inicializado."""
    if _engine is None:
        raise RuntimeError("motor de OCR não inicializado no trabalhador")
    return True


def process_job(job: OcrJob) -> OcrResult:
    start = time.perf_counter()
    candidates: list[tuple[str, float]] = []
    for crop in job.crops:
        candidates.extend(read_candidates(_engine, crop, job.angles, job.min_conf))
    return OcrResult(candidates, int((time.perf_counter() - start) * 1000))
