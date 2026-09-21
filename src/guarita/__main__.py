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

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def executor_factory():
        return ProcessPoolExecutor(
            max_workers=cfg.pipeline.ocr_workers, initializer=init_worker, initargs=(make_paddle_engine,))

    app = None
    try:
        app = App(
            cfg,
            source_factory=lambda cam: RtspSource(cam.rtsp_url, cfg.pipeline.target_fps, should_stop=stop.is_set),
            detector_factory=lambda cam: PlateDetector(
                cfg.detector.weights, cfg.detector.imgsz, cfg.detector.conf, tracker_yaml=cfg.detector.tracker_yaml),
            executor_factory=executor_factory,
            client=ApiClient(cfg.api.base_url, cfg.api.timeout_s),
            outbox=outbox,
        )
        log.info("Coletor iniciado (%d câmera(s))", len(cfg.cameras))
        app.start()
        stop.wait()
    finally:
        log.info("Encerrando…")
        stop.set()
        try:
            if app is not None:
                app.stop()
        finally:
            outbox.close()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
