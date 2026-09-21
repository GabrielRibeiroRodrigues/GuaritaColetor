import pytest

from guarita.ocr.plates import (
    PALAVRAS_IGNORAR,
    corrigir_placa,
    format_char,
    license_complies_format,
    limpar_texto_placa,
    read_plate_from_detections,
)


def box(x, y=0):
    return [[x, y], [x + 30, y], [x + 30, y + 10], [x, y + 10]]


@pytest.mark.parametrize("text,expected", [
    ("ABC1234", "LLLNNNN"),
    ("abc1234", "LLLNNNN"),
    ("ABC1D23", "LLLNLNN"),
    ("ABC1A23", "LLLNNNN"),   # 'A' também é candidato a dígito: o padrão clássico vence
    ("0BC1234", "LLLNNNN"),   # '0' é candidato a letra
    ("ABCD123", None),
    ("ABC123", None),
    ("", None),
    (None, None),
])
def test_license_complies_format(text, expected):
    assert license_complies_format(text) == expected


@pytest.mark.parametrize("text,pattern,expected", [
    ("ABC1234", "LLLNNNN", "ABC1234"),
    ("abc1234", "LLLNNNN", "ABC1234"),
    ("0BC1234", "LLLNNNN", "OBC1234"),   # dígito em posição de letra vira letra
    ("A1C1234", "LLLNNNN", "AIC1234"),
    ("ABCI234", "LLLNNNN", "ABCI234"),   # letra em posição de número é mantida
    ("ABC1D23", "LLLNLNN", "ABC1D23"),
    ("abc", None, "ABC"),                # sem padrão: só maiúsculas
    ("ABC1234", "OUTRO", "ABC1234"),
])
def test_corrigir_placa(text, pattern, expected):
    assert corrigir_placa(text, pattern) == expected


@pytest.mark.parametrize("ch,target,expected", [
    ("0", "L", "O"), ("1", "L", "I"), ("3", "L", "J"), ("4", "L", "A"),
    ("6", "L", "G"), ("5", "L", "S"), ("8", "L", "B"),
    ("2", "L", "2"),          # sem mapeamento: mantém
    ("a", "L", "A"),
    ("O", "N", "O"),          # letra conversível em posição de número: mantém
    ("5", "N", "5"),
    ("L", "N", "L"),
])
def test_format_char(ch, target, expected):
    assert format_char(ch, target) == expected


def test_limpar_texto_placa():
    assert limpar_texto_placa(" a-b.c ") == "ABC"
    assert limpar_texto_placa("AB C") == "AB C"
    assert limpar_texto_placa("ab!1@2#") == "AB12"


def test_palavras_ignorar_contem_termos_conhecidos():
    assert {"BRASIL", "MERCOSUL", "MUZAMBINHO", "MG", "GUAXUPE"} <= PALAVRAS_IGNORAR


def test_read_single_detection():
    assert read_plate_from_detections([[box(0), ("ABC-1234", 0.9)]]) == ("ABC1234", 0.9)


def test_read_combines_multiple_detections_in_x_order():
    detections = [[box(50), ("1234", 0.6)], [box(10), ("ABC", 0.8)]]
    plate, score = read_plate_from_detections(detections)
    assert plate == "ABC1234"
    assert score == pytest.approx(0.7)


def test_read_ignores_state_and_country_words():
    detections = [[box(10), ("BRASIL", 0.9)], [box(20), ("ABC1234", 0.8)]]
    assert read_plate_from_detections(detections) == ("ABC1234", 0.8)


def test_read_falls_back_to_individual_detection():
    detections = [[box(10), ("XX", 0.9)], [box(20), ("ABC1D23", 0.7)]]
    assert read_plate_from_detections(detections) == ("ABC1D23", 0.7)


@pytest.mark.parametrize("detections", [None, [], [[box(0), ("HELLO", 0.9)]]])
def test_read_returns_none_without_valid_plate(detections):
    assert read_plate_from_detections(detections) == (None, None)
