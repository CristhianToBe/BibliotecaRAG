from __future__ import annotations

import logging
import sys

def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO

    # force=True evita que se queden handlers viejos que duplican salida
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,   # todo a stdout (más ordenado en Windows)
        force=True,
    )

    # 🔇 ruido de red SIEMPRE
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)