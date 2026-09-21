import numpy as np

from guarita.models import Detection
from guarita.track import TrackManager, crop_score

BOX = (10, 10, 60, 40)


def frame_with(sharp=False):
    frame = np.full((100, 200, 3), 120, np.uint8)
    if sharp:
        checker = (np.indices((30, 50)).sum(axis=0) % 2 * 255).astype(np.uint8)
        frame[10:40, 10:60] = checker[..., None]
    return frame


def det(track_id=1, box=BOX, conf=0.9):
    return Detection(track_id, box, conf)


def manager(**kw):
    args = dict(camera_id="cam-01", close_after_s=1.5, max_age_s=20.0, max_crops=3)
    args.update(kw)
    return TrackManager(**args)


def test_closes_after_absence_and_reports_fields():
    tm = manager()
    frame = frame_with()
    assert tm.update(frame, [det()], 0.0) == []
    assert tm.update(frame, [det()], 0.5) == []
    assert tm.update(frame, [], 1.0) == []          # ausente há 0,5 s
    closed = tm.update(frame, [], 2.0)              # ausente há 1,5 s
    assert len(closed) == 1
    track = closed[0]
    assert (track.camera_id, track.track_id) == ("cam-01", 1)
    assert (track.started_at, track.closed_at) == (0.0, 2.0)
    assert len(track.crops) == 2
    assert track.crops[0].image.shape == (30, 50, 3)
    assert tm.update(frame, [], 3.0) == []          # já fechada e removida


def test_keeps_only_the_best_crops_best_first():
    tm = manager(max_crops=2)
    for now, sharp in [(0.0, False), (0.5, True), (1.0, False), (1.5, False)]:
        tm.update(frame_with(sharp), [det()], now)
    closed = tm.update(frame_with(), [], 10.0)[0]
    assert len(closed.crops) == 2
    assert closed.crops[0].ts == 0.5                # o quadro nítido vem primeiro
    assert closed.crops[0].score > closed.crops[1].score


def test_crop_score_prefers_sharp_and_confident_crops():
    sharp = frame_with(True)[10:40, 10:60]
    flat = frame_with(False)[10:40, 10:60]
    assert crop_score(sharp, 0.9) > crop_score(flat, 0.9)
    assert crop_score(flat, 0.9) > crop_score(flat, 0.4)


def test_max_age_forces_closing_and_a_new_track_starts():
    tm = manager(max_age_s=5.0)
    frame = frame_with()
    closed = []
    for now in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        closed += tm.update(frame, [det()], now)
    assert len(closed) == 1 and closed[0].started_at == 0.0 and closed[0].closed_at == 5.0
    assert tm.update(frame, [det()], 6.0) == []
    flushed = tm.flush(6.5)
    assert len(flushed) == 1 and flushed[0].started_at == 6.0


def test_flush_closes_everything_once():
    tm = manager()
    frame = frame_with()
    tm.update(frame, [det(1), det(2, (100, 10, 150, 40))], 0.0)
    flushed = tm.flush(0.1)
    assert sorted(t.track_id for t in flushed) == [1, 2]
    assert tm.flush(0.2) == []


def test_boxes_are_clipped_to_the_frame_and_outside_boxes_ignored():
    tm = manager()
    frame = frame_with()
    tm.update(frame, [det(1, (-10, -10, 300, 300)), det(2, (250, 250, 300, 300))], 0.0)
    flushed = tm.flush(0.1)
    assert [t.track_id for t in flushed] == [1]
    assert flushed[0].crops[0].image.shape == (100, 200, 3)


def test_crops_are_copies_of_the_frame():
    tm = manager()
    frame = frame_with()
    tm.update(frame, [det()], 0.0)
    frame[:] = 0
    crop = tm.flush(0.1)[0].crops[0].image
    assert crop.mean() > 0
