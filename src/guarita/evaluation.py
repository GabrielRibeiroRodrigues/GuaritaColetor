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
    dispatcher = OcrDispatcher(executor_factory, cfg.pipeline.ocr_timeout_s, workers=cfg.pipeline.ocr_workers)
    dispatcher.wait_ready()
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
