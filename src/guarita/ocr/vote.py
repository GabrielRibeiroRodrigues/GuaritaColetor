from __future__ import annotations

from collections import defaultdict

from .plates import license_complies_format


def vote_plate(candidates):
    """Combina leituras do mesmo veículo, posição a posição. Retorna (placa, confiança) ou (None, None)."""
    valid = [(p, c) for p, c in candidates if p and len(p) == 7]
    if not valid:
        return None, None

    chars: list[str] = []
    agreement: list[float] = []
    for i in range(7):
        weights: dict[str, float] = defaultdict(float)
        best_single: dict[str, float] = {}
        for plate, conf in valid:
            ch = plate[i]
            weights[ch] += conf
            best_single[ch] = max(best_single.get(ch, 0.0), conf)
        total = sum(weights.values())
        winner = max(weights, key=lambda ch: (weights[ch], best_single[ch], -ord(ch)))
        chars.append(winner)
        agreement.append(weights[winner] / total if total > 0 else 0.0)

    plate = "".join(chars)
    if license_complies_format(plate) is None:
        return None, None
    mean_conf = sum(c for _, c in valid) / len(valid)
    return plate, mean_conf * (sum(agreement) / 7)
