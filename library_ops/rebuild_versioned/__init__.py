from __future__ import annotations

import argparse
from pathlib import Path

from .orchestrator import run as orchestrator_run


def run(
    *,
    manifest_override: Path | None,
    apply: bool,
    mode: str,
    debug: bool,
    batch_size: int,
    prefix_batch: int,
    max_rounds: int,
    max_docs: int,
    cleanup_empty: bool,
    create_vector_stores: bool,
) -> int:
    return orchestrator_run(
        manifest_override=manifest_override,
        apply=apply,
        mode=mode,
        debug=debug,
        batch_size=batch_size,
        prefix_batch=prefix_batch,
        max_rounds=max_rounds,
        max_docs=max_docs,
        cleanup_empty=cleanup_empty,
        create_vector_stores=create_vector_stores,
    )


def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("rebuild-versioned", help="Rebuild library into Biblioteca/biblioteca/vN with AUTOR/AÑO prefix + GPT taxonomy")
    p.add_argument("--manifest", default=None, help="Ruta a library.json (override)")
    p.add_argument("--apply", action="store_true", help="Aplica cambios. Si no, solo plan.")
    p.add_argument("--mode", choices=["copy", "move"], default="copy")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--prefix-batch", type=int, default=20)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--cleanup-empty", action="store_true")
    p.add_argument("--create-vector-stores", action="store_true")

    def _cmd(args: argparse.Namespace) -> int:
        return run(
            manifest_override=Path(args.manifest).expanduser() if args.manifest else None,
            apply=args.apply,
            mode=args.mode,
            debug=args.debug,
            batch_size=args.batch_size,
            prefix_batch=args.prefix_batch,
            max_rounds=args.max_rounds,
            max_docs=args.max_docs,
            cleanup_empty=args.cleanup_empty,
            create_vector_stores=args.create_vector_stores,
        )

    p.set_defaults(func=_cmd)
