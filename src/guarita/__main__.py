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
from .preview import FrameStore, PreviewServer, ReadingLog, Recorder
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

    # mp_context explícito: por padrão o Linux usa "fork", que copia as threads (câmeras, envio,
    # supervisor) já abertas no processo pai para dentro de cada worker — não é seguro aqui (o
    # processo já importou torch/paddle e abriu threads antes do pool existir). No Windows o spawn
    # já é obrigatório; fixar isso deixa o comportamento igual se um dia isto rodar em Linux.
    mp_context = multiprocessing.get_context("spawn")

    # Página local de vídeo ao vivo, para validar o coletor em teste presencial sem depender de
    # internet — desligada por padrão (custo zero) e ligada só com GUARITA_PREVIEW=1. A gravação
    # (botão na página) é sempre sob demanda, mesmo com o preview ligado — nunca contínua.
    preview_enabled = os.environ.get("GUARITA_PREVIEW") == "1"
    frame_store = FrameStore() if preview_enabled else None
    reading_log = ReadingLog() if preview_enabled else None
    recordings_dir = os.path.join(os.path.dirname(cfg.outbox.path) or ".", "recordings")
    recorder = Recorder(recordings_dir, fps=cfg.pipeline.target_fps) if preview_enabled else None
    on_reading = reading_log.add if reading_log else None
    preview_server = None

    on_frame = None
    if preview_enabled:
        def on_frame(camera_id, frame, detections):
            frame_store.update(camera_id, frame, detections)
            recorder.write(camera_id, frame)  # write() é um no-op se ninguém apertou "Gravar"

    def executor_factory():
        return ProcessPoolExecutor(
            max_workers=cfg.pipeline.ocr_workers, initializer=init_worker, initargs=(make_paddle_engine,),
            mp_context=mp_context)

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
            stop_event=stop,
            on_frame=on_frame,
            on_reading=on_reading,
        )
        log.info("Coletor iniciado (%d câmera(s))", len(cfg.cameras))
        app.start()
        if preview_enabled:
            preview_server = PreviewServer(frame_store, [c.id for c in cfg.cameras],
                                            port=int(os.environ.get("GUARITA_PREVIEW_PORT", "8888")),
                                            reading_log=reading_log, recorder=recorder)
            preview_server.start()
        stop.wait()
    except (RuntimeError, FileNotFoundError) as exc:
        # Sem OCR ou sem pesos o coletor não tem função. Sai com 3 para que o gerenciador de serviço
        # (NSSM) reinicie com atraso, em vez de girar em falso.
        log.error("Partida abortada: %s", exc)
        return 3
    finally:
        log.info("Encerrando…")
        stop.set()
        if preview_server is not None:
            preview_server.stop()
        try:
            if app is not None:
                app.stop()
        finally:
            outbox.close()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
