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
