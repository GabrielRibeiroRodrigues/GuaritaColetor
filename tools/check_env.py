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
        import numpy as np

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
