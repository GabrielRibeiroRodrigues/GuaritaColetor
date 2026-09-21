import os
from types import SimpleNamespace

import numpy as np
import pytest

from guarita.detect import DEFAULT_TRACKER_YAML, PlateDetector
from guarita.models import Detection


class Vec(list):
    def tolist(self):
        return list(self)


class FakeYolo:
    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=self._boxes)]


FRAME = np.zeros((10, 10, 3), np.uint8)


def test_missing_weights_is_fatal(tmp_path):
    missing = tmp_path / "bestn.pt"
    with pytest.raises(FileNotFoundError, match="bestn.pt"):
        PlateDetector(str(missing))


def test_detect_returns_detections_with_track_ids():
    boxes = SimpleNamespace(
        id=Vec([3.0, 4.0]), conf=Vec([0.9, 0.4]), xyxy=Vec([[1.2, 2.7, 30.0, 40.9], [5, 5, 9, 9]]))
    model = FakeYolo(boxes)
    detections = PlateDetector("bestn.pt", imgsz=416, conf=0.25, model=model).detect(FRAME)
    assert detections == [Detection(3, (1, 2, 30, 40), 0.9), Detection(4, (5, 5, 9, 9), 0.4)]
    call = model.calls[0]
    assert call["imgsz"] == 416 and call["conf"] == 0.25
    assert call["persist"] is True and os.path.isabs(call["tracker"])
    assert call["tracker"] == DEFAULT_TRACKER_YAML and os.path.isfile(DEFAULT_TRACKER_YAML)


def test_no_confirmed_tracks_yields_empty_list():
    boxes = SimpleNamespace(id=None, conf=Vec([]), xyxy=Vec([]))
    assert PlateDetector("bestn.pt", model=FakeYolo(boxes)).detect(FRAME) == []
    assert PlateDetector("bestn.pt", model=FakeYolo(None)).detect(FRAME) == []


def test_bundled_tracker_thresholds_match_detector_conf():
    text = open(DEFAULT_TRACKER_YAML, encoding="utf-8").read()
    values = dict(line.split(":", 1) for line in text.splitlines() if line and not line.startswith("#"))
    assert values["tracker_type"].strip() == "bytetrack"
    assert float(values["new_track_thresh"]) == float(values["track_high_thresh"]) == 0.25
    assert float(values["track_low_thresh"]) == 0.1


def test_tracker_yaml_override_is_used():
    model = FakeYolo(SimpleNamespace(id=None, conf=Vec([]), xyxy=Vec([])))
    PlateDetector("bestn.pt", model=model, tracker_yaml="meu.yaml").detect(FRAME)
    assert model.calls[0]["tracker"] == os.path.abspath("meu.yaml")
