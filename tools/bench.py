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
    detector = PlateDetector(cfg.detector.weights, cfg.detector.imgsz, cfg.detector.conf,
                             tracker_yaml=cfg.detector.tracker_yaml)

    mp_context = multiprocessing.get_context("spawn")  # ver comentário em __main__.py

    def executor_factory():
        return ProcessPoolExecutor(max_workers=cfg.pipeline.ocr_workers, initializer=init_worker,
                                   initargs=(make_paddle_engine,), mp_context=mp_context)

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
