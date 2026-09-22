"""Verificação manual com o ambiente REAL (YOLO + PaddleOCR + pool de processos do Windows).

Não faz parte do pytest (usa modelos e pode baixar arquivos na primeira vez). Rode no ambiente
completo (`.venv-run`), depois de `python tools/check_env.py caminho/bestn.pt` passar.

  python tools/smoke_real.py detect <modelo.pt>   # ids de rastreio reais do YOLO/ByteTrack
  python tools/smoke_real.py ocr                  # PaddleOCR real em placas sintéticas, 0° e 15°
  python tools/smoke_real.py pool                 # pool de processos real (aquecimento, timeout, travamento)
  python tools/smoke_real.py e2e <modelo.pt>       # fluxo completo com OCR real (detector real ou simulado)
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def synthetic_plate(text: str, size=(400, 130), noise=False, blur=False) -> np.ndarray:
    """Placa sintética: fundo cinza-claro, moldura preta grossa, texto grande e escuro."""
    height, width = size[1], size[0]
    img = np.full((height, width, 3), 235, np.uint8)
    cv2.rectangle(img, (4, 4), (width - 5, height - 5), (20, 20, 20), 8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = width / 420.0
    (tw, th), _ = cv2.getTextSize(text, font, scale * 2.2, 6)
    org = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(img, text, org, font, scale * 2.2, (15, 15, 15), 6, cv2.LINE_AA)
    if blur:
        img = cv2.GaussianBlur(img, (5, 5), 1.2)
    if noise:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    return img


def cmd_detect(args) -> int:
    from guarita.detect import PlateDetector

    asset = Path(args.model).resolve()
    detector = PlateDetector(str(asset), imgsz=640, conf=0.25)

    import ultralytics

    sample = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    base = cv2.imread(str(sample))
    if base is None:
        print(f"FALHA: não abriu a imagem de exemplo em {sample}")
        return 1

    ids_seen = set()
    for i in range(30):
        # pequenos deslocamentos simulam quadros sucessivos de vídeo (mesmos objetos, posição muda)
        shifted = np.roll(base, shift=(i % 5, 0), axis=(0, 1))
        detections = detector.detect(shifted)
        if detections:
            ids_seen.update(d.track_id for d in detections)
        print(f"quadro {i:02d}: {len(detections)} detecção(ões) — ids {[d.track_id for d in detections]}")

    if not ids_seen:
        print("FALHA: nenhuma detecção em 30 quadros (modelo genérico + conf 0.25; esperado ao menos alguma)")
        return 1
    print(f"OK   ids de rastreio observados: {sorted(ids_seen)} em 30 quadros "
          "(bus.jpg tem ~6 objetos reais; o que importa é não crescer sem parar)")
    return 0


def cmd_ocr(args) -> int:
    from guarita.ocr.engine import PaddleEngine, read_candidates

    engine = PaddleEngine()
    cases = [
        ("clássica, nítida", synthetic_plate("ABC1234")),
        ("mercosul, nítida", synthetic_plate("ABC1D23")),
        ("clássica, com ruído", synthetic_plate("ABC1234", noise=True)),
        ("clássica, desfocada", synthetic_plate("ABC1234", blur=True)),
    ]
    any_ok = False
    for label, img in cases:
        started = time.perf_counter()
        candidates = read_candidates(engine, img, angles=(0.0, 15.0), min_conf=0.1)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"{label:22s} -> {candidates}  ({elapsed_ms:.0f} ms, 2 ângulos)")
        any_ok = any_ok or bool(candidates)
    if not any_ok:
        print("FALHA: nenhuma leitura em nenhum caso (verifique o PaddleOCR ou o gerador de placas sintéticas)")
        return 1
    print("OK   ocr: o PaddleOCR real leu ao menos um caso")
    return 0


def cmd_pool(args) -> int:
    from guarita.dispatch import OcrDispatcher
    from guarita.models import ClosedTrack as CT, Crop
    from guarita.ocr.engine import make_paddle_engine
    from guarita.ocr.worker import OcrJob, init_worker

    def executor_factory():
        return ProcessPoolExecutor(max_workers=1, initializer=init_worker, initargs=(make_paddle_engine,))

    print("Aquecendo o pool real (spawn + carga do PaddleOCR)…")
    started = time.perf_counter()
    dispatcher = OcrDispatcher(executor_factory, timeout_s=10.0, workers=1)
    ready = dispatcher.wait_ready(timeout_s=180.0)
    warm_s = time.perf_counter() - started
    print(f"{'OK' if ready else 'FALHA'} aquecimento em {warm_s:.1f} s")
    if not ready:
        return 1

    plate = synthetic_plate("ABC1234")
    track = CT("cam-teste", 1, 0.0, 1.0, [Crop(plate, 0.0, 1.0)])
    job = OcrJob([plate], (0.0, 15.0), 0.1)
    started = time.perf_counter()
    dispatcher.submit(track, job)
    outcomes = []
    while not outcomes and time.perf_counter() - started < 15.0:
        outcomes = dispatcher.poll()
        time.sleep(0.1)
    if not outcomes:
        print("FALHA: nenhum resultado em 15 s (o job não deveria estourar logo após o aquecimento)")
        return 1
    outcome = outcomes[0]
    print(f"OK   1º job depois do aquecimento: erro={outcome.error} candidatos={outcome.result and outcome.result.candidates} "
          f"latência={outcome.latency_ms}ms (não deve ser 'timeout')")
    if outcome.error == "timeout":
        print("FALHA: o job estourou por causa da partida a frio (bug C1 reapareceu)")
        return 1

    print("Simulando travamento (tempo limite de 1 s, sem enviar resultado)…")
    dispatcher2 = OcrDispatcher(executor_factory, timeout_s=1.0, workers=1)
    if not dispatcher2.wait_ready(180.0):
        print("FALHA: o segundo pool não aqueceu")
        return 1
    hang_job = OcrJob([plate] * 50, tuple(range(0, 90, 5)), 0.1)  # muitos ângulos: deve estourar o 1 s
    dispatcher2.submit(track, hang_job)
    recovered = []
    started = time.perf_counter()
    while not recovered and time.perf_counter() - started < 20.0:
        recovered = dispatcher2.poll()
        time.sleep(0.2)
    ok = bool(recovered) and recovered[0].error == "timeout"
    print(f"{'OK' if ok else 'FALHA'} recuperação de travamento: {recovered}")
    dispatcher.shutdown()
    dispatcher2.shutdown()
    return 0 if ok else 1


def cmd_e2e(args) -> int:
    from guarita.config import CameraConfig, PipelineConfig
    from guarita.dispatch import OcrDispatcher
    from guarita.models import Detection
    from guarita.ocr.engine import make_paddle_engine
    from guarita.ocr.worker import init_worker
    from guarita.outbox import Outbox
    from guarita.pipeline import Aggregator, CameraRunner, drain_results

    class StubSource:
        def __init__(self, frames):
            self._frames = list(frames)
            self.exhausted = False

        def read(self):
            if self._frames:
                return self._frames.pop(0)
            self.exhausted = True
            return None

    class StubDetector:
        """Devolve sempre a mesma caixa/id: exercita o pool de OCR real, o TrackManager, o
        Aggregator e a fila SQLite real, sem depender de o modelo genérico enxergar a placa."""

        def __init__(self, box, frames_with_detection):
            self._box = box
            self._left = frames_with_detection

        def detect(self, frame):
            if self._left <= 0:
                return []
            self._left -= 1
            return [Detection(1, self._box, 0.9)]

    from guarita.capture import Frame

    plate_img = synthetic_plate("ABC1234")
    ph, pw = plate_img.shape[:2]
    big = np.full((1080, 1920, 3), 200, np.uint8)
    x0, y0 = 700, 400
    big[y0:y0 + ph, x0:x0 + pw] = plate_img
    box = (x0, y0, x0 + pw, y0 + ph)

    frames = [Frame(big, t) for t in (0.0, 0.3, 0.6, 2.5)]  # as 3 primeiras têm o carro; a 4ª fecha a trilha
    camera = CameraConfig("cam-01", "entrada", "", "")
    cfg = PipelineConfig(close_after_s=1.5, max_crops=3)

    def executor_factory():
        return ProcessPoolExecutor(max_workers=1, initializer=init_worker, initargs=(make_paddle_engine,))

    outbox_path = Path("data") / "smoke_e2e.sqlite"
    outbox_path.parent.mkdir(exist_ok=True)
    outbox_path.unlink(missing_ok=True)
    outbox = Outbox(str(outbox_path))

    dispatcher = OcrDispatcher(executor_factory, timeout_s=10.0, workers=1)
    print("Aquecendo…")
    if not dispatcher.wait_ready(180.0):
        print("FALHA: pool não aqueceu")
        return 1

    # Só a fila real (como App.py de verdade): dispatcher.poll() é destrutivo — chamar drain_results
    # duas vezes por rodada (uma para cada destino) faria a segunda achar a fila já vazia.
    aggregators = {"cam-01": Aggregator(camera, cfg, None)}
    runner = CameraRunner(camera, cfg, StubSource(frames), StubDetector(box, frames_with_detection=3),
                          dispatcher)
    while runner.step() != "done":
        drain_results(dispatcher, aggregators, outbox)
    runner.finish()

    started = time.perf_counter()
    while dispatcher.pending_count() and time.perf_counter() - started < 20.0:
        drain_results(dispatcher, aggregators, outbox)
        time.sleep(0.2)
    drain_results(dispatcher, aggregators, outbox)
    dispatcher.shutdown()

    row = outbox.next_due(now=time.time() + 1)
    outbox.close()
    if row is None:
        print("FALHA: nenhuma leitura chegou à fila")
        return 1
    print(f"OK   leitura na fila real: placa={row.plate} confiança={row.confidence:.2f} "
          f"direção={row.direction} id={row.external_id} imagem={'sim' if row.image else 'não'}")
    ok = row.plate == "ABC1234"
    print("OK   fim a fim com OCR real" if ok else "FALHA: placa não bate com o esperado")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    detect = sub.add_parser("detect")
    detect.add_argument("model")
    detect.set_defaults(func=cmd_detect)

    sub.add_parser("ocr").set_defaults(func=cmd_ocr)
    sub.add_parser("pool").set_defaults(func=cmd_pool)

    e2e = sub.add_parser("e2e")
    e2e.add_argument("model", nargs="?", default=None)
    e2e.set_defaults(func=cmd_e2e)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
