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


class ScriptedSource:
    """Fonte de quadros roteirizada: devolve `frames` em ordem e depois marca exhausted."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.exhausted = False

    def read(self):
        if self._frames:
            return self._frames.pop(0)
        self.exhausted = True
        return None


class ScriptedDetector:
    """Detector roteirizado: cada chamada devolve a próxima lista de Detection."""

    def __init__(self, script):
        self._script = list(script)

    def detect(self, frame):
        return self._script.pop(0) if self._script else []
