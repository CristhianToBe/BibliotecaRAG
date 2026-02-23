from __future__ import annotations

import argparse

from worklib.logging_utils import setup_logging
from .pipeline import pro_query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="Pregunta del usuario")
    ap.add_argument("--manifest", default=None, help="Ruta a manifest.json")
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-confirm", action="store_true", help="Desactiva loop de confirmación")
    ap.add_argument("--confirm-rounds", type=int, default=4)
    ap.add_argument("--no-glimpse", action="store_true", help="No hacer mini-retrieval en confirmación")
    args = ap.parse_args()

    setup_logging(debug=args.debug)

    print(
        pro_query(
            args.question,
            manifest_path=args.manifest,
            max_workers=args.max_workers,
            debug=args.debug,
            confirm=(not args.no_confirm),
            confirm_rounds=args.confirm_rounds,
            confirm_glimpse=(not args.no_glimpse),
        )
    )