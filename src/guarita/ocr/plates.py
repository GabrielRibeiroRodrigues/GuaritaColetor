"""Regras de placa portadas literalmente de GuaritaLeitor09-07/util.py."""
from __future__ import annotations

import logging
import string

log = logging.getLogger(__name__)

CHAR_TO_INT = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8'}
INT_TO_CHAR = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B'}

PALAVRAS_IGNORAR = frozenset({
    'BRASIL', 'MERCOSUL', 'BRAZIL',
    'SP', 'SAOPAULO', 'RJ', 'RIODEJANEIRO', 'MG', 'MINASGERAIS',
    'ES', 'ESPIRITOSANTO', 'PR', 'PARANA', 'SC', 'SANTACATARINA', 'RS', 'RIOGRANDEDOSUL',
    'MS', 'MATOGROSSODOSUL', 'MT', 'MATOGROSSO', 'GO', 'GOIAS', 'DF', 'DISTRITOFEDERAL',
    'BA', 'BAHIA', 'SE', 'SERGIPE', 'AL', 'ALAGOAS', 'PE', 'PERNAMBUCO', 'PB', 'PARAIBA',
    'RN', 'RIOGRANDEDONORTE', 'CE', 'CEARA', 'PI', 'PIAUI', 'MA', 'MARANHAO', 'TO', 'TOCANTINS',
    'PA', 'PARA', 'AP', 'AMAPA', 'RR', 'RORAIMA', 'AM', 'AMAZONAS', 'AC', 'ACRE', 'RO', 'RONDONIA',
    'GUAXUPE', 'MUZAMBINHO', 'NOVAREZENDE', 'SAOPEDRODAUNIAO', 'JURUAIA',
    'PASSOS', 'POCOSDECALDAS', 'VARGINHA', 'ALFENAS', 'LAVRAS', 'UBERLANDIA', 'UBERABA',
    'BELOHORIZONTE', 'JUIZDEFORA', 'MONTESCLAROS', 'DIVINOPOLIS', 'GOVERNADORVALADARES',
    'CAMPINAS', 'RIBEIRAOPRETO', 'FRANCA', 'SAOJOSEDORIOPRETO', 'BAURU',
    'MUNICIPIO', 'ESTADO', 'UF', 'BR',
    'FRENTE', 'TRAS', 'LATERAL', 'AUTO', 'MOTO', 'CAMINHAO', 'CARRO',
})


def is_letter_candidate(char_val: str) -> bool:
    """Letra natural ou número conversível em letra."""
    return char_val.upper() in string.ascii_uppercase or char_val in INT_TO_CHAR


def is_digit_candidate(char_val: str) -> bool:
    """Dígito natural ou letra conversível em dígito."""
    return char_val.isdigit() or char_val.upper() in CHAR_TO_INT


def license_complies_format(text):
    if not text or len(text) != 7:
        return None

    # Padrão 1: LLLNNNN (clássico brasileiro)
    if (is_letter_candidate(text[0]) and is_letter_candidate(text[1]) and is_letter_candidate(text[2])
            and is_digit_candidate(text[3]) and is_digit_candidate(text[4])
            and is_digit_candidate(text[5]) and is_digit_candidate(text[6])):
        return "LLLNNNN"

    # Padrão 2: LLLNLNN (Mercosul)
    if (is_letter_candidate(text[0]) and is_letter_candidate(text[1]) and is_letter_candidate(text[2])
            and is_digit_candidate(text[3]) and is_letter_candidate(text[4])
            and is_digit_candidate(text[5]) and is_digit_candidate(text[6])):
        return "LLLNLNN"

    return None


def is_naturally_correct_for_position(char_val: str, target_type: str) -> bool:
    if target_type == 'L':
        return char_val.isalpha() and char_val.upper() in string.ascii_uppercase
    if target_type == 'N':
        return char_val.isdigit()
    return False


def format_char(char_val: str, target_type: str) -> str:
    """Corrige um caractere para letra ('L') ou dígito ('N'), de forma conservadora."""
    char_upper = char_val.upper()

    if is_naturally_correct_for_position(char_upper, target_type):
        return char_upper

    if target_type == 'L':
        if char_val in INT_TO_CHAR:
            converted = INT_TO_CHAR[char_val]
            log.debug("Convertendo número '%s' para letra '%s'", char_val, converted)
            return converted
        if char_upper.isalpha():
            return char_upper
        log.debug("Caractere '%s' mantido sem conversão (posição de letra)", char_val)
        return char_upper

    if target_type == 'N':
        if char_upper in CHAR_TO_INT:
            if char_upper.isalpha() and char_upper in string.ascii_uppercase:
                log.debug("Letra '%s' mantida (válida para posição de número em Mercosul)", char_upper)
                return char_upper
            converted = CHAR_TO_INT[char_upper]
            log.debug("Convertendo '%s' para número '%s' (ambíguo)", char_upper, converted)
            return converted
        if char_val.isdigit():
            return char_val
        log.debug("Caractere '%s' mantido sem conversão (posição de número)", char_val)
        return char_upper

    return char_upper


def corrigir_placa(text: str, pattern_code) -> str:
    if not pattern_code or len(text) != 7:
        return text.upper()
    if pattern_code not in ("LLLNNNN", "LLLNLNN"):
        return text.upper()

    corrected = list(text)
    for i in range(7):
        corrected[i] = format_char(corrected[i], pattern_code[i])
    final_plate = "".join(corrected)
    if final_plate != text:
        log.debug("Placa corrigida '%s' -> '%s' (%s)", text, final_plate, pattern_code)
    return final_plate


def limpar_texto_placa(texto: str) -> str:
    caracteres_remover = "-.!@#$%^&*()[]{};:,<>?/\\|`~'\""
    return ''.join(c for c in texto if c not in caracteres_remover).strip().upper()


def read_plate_from_detections(detections):
    """Lógica de ler_placas2() aplicada à saída do PaddleOCR (results[0]).

    Retorna (placa, score) ou (None, None).
    """
    if not detections:
        return None, None

    if len(detections) > 1:
        try:
            ordered = sorted(detections, key=lambda det: (det[0][0][0], det[0][0][1]))
        except TypeError:
            ordered = detections

        texts, scores = [], []
        for item in ordered:
            processed = limpar_texto_placa(item[1][0])
            if processed and processed not in PALAVRAS_IGNORAR:
                texts.append(processed)
                scores.append(item[1][1])

        if texts:
            combined = "".join(texts)
            pattern = license_complies_format(combined)
            if pattern:
                avg = sum(scores) / len(scores) if scores else 0.0
                return corrigir_placa(combined, pattern), avg

    for item in detections:
        processed = limpar_texto_placa(item[1][0])
        pattern = license_complies_format(processed)
        if pattern:
            return corrigir_placa(processed, pattern), item[1][1]

    return None, None
