from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def prepare_ultralytics() -> None:
    """Compatibilidade com pesos antigos (mesma preparação do main.py atual)."""
    try:
        import ultralytics.utils  # noqa: F401
    except Exception:
        try:
            import ultralytics.yolo.utils as legacy_utils

            sys.modules["ultralytics.utils"] = legacy_utils  # alias para o caminho antigo
        except Exception as exc:
            log.warning("Não foi possível preparar 'ultralytics.utils' (%s); o carregamento dos pesos pode falhar.", exc)

    import torch
    from ultralytics.nn.tasks import DetectionModel

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        try:
            torch.serialization.add_safe_globals([DetectionModel])
        except Exception as exc:
            log.warning("Falha ao ajustar torch safe globals: %s", exc)
