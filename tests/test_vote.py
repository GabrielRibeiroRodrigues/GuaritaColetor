import pytest

from guarita.ocr.vote import vote_plate


def test_single_candidate_is_returned_as_is():
    assert vote_plate([("ABC1234", 0.9)]) == ("ABC1234", pytest.approx(0.9))


def test_agreeing_candidates_average_confidence():
    plate, conf = vote_plate([("ABC1234", 0.9), ("ABC1234", 0.7)])
    assert plate == "ABC1234"
    assert conf == pytest.approx(0.8)


def test_majority_wins_and_confidence_is_penalized_by_disagreement():
    plate, conf = vote_plate([("ABC1234", 0.9), ("ABC1284", 0.6), ("ABC1234", 0.8)])
    assert plate == "ABC1234"
    assert conf == pytest.approx(0.7381, abs=1e-3)


def test_weights_beat_counts():
    # duas leituras fracas contra uma forte na posição 6
    plate, _ = vote_plate([("ABC1234", 0.3), ("ABC1234", 0.3), ("ABC1235", 0.9)])
    assert plate == "ABC1235"


def test_tie_prefers_higher_single_confidence_then_smaller_char():
    assert vote_plate([("ABC1234", 0.8), ("ABC1235", 0.9)])[0] == "ABC1235"
    assert vote_plate([("ABC1234", 0.5), ("ABC1235", 0.5)])[0] == "ABC1234"


def test_can_mix_positions_from_different_candidates():
    plate, _ = vote_plate([("ABC1D23", 0.9), ("ABC1234", 0.5)])
    assert plate == "ABC1D23"


def test_invalid_voted_plate_is_rejected():
    assert vote_plate([("ABCD123", 0.9)]) == (None, None)


def test_ignores_wrong_length_and_handles_empty():
    assert vote_plate([("ABC123", 0.9)]) == (None, None)
    assert vote_plate([]) == (None, None)
