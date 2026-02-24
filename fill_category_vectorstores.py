"""Wrapper para ejecutar fill-category-vectorstores con manifest explícito."""

from __future__ import annotations

import argparse
from pathlib import Path

from library_ops.fill_vectorstores import run


def main() -> int:
    ap = argparse.ArgumentParser(description="Llena vector stores por categoría usando un manifest")
    ap.add_argument("--manifest", required=True, help="Ruta al manifest JSON")
    ap.add_argument("--out", default="")
    ap.add_argument("--only-empty", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--vs-name-prefix", default="")
    ap.add_argument("--file-batch-size", type=int, default=200)
    ap.add_argument("--no-upload-missing", dest="upload_missing", action="store_false")
    ap.set_defaults(upload_missing=True)
    ap.add_argument("--max-cats", type=int, default=0)
    args = ap.parse_args()

    return run(
        manifest=Path(args.manifest).expanduser(),
        out=Path(args.out).expanduser() if args.out else None,
        only_empty=args.only_empty,
        dry_run=args.dry_run,
        debug=args.debug,
        vs_name_prefix=args.vs_name_prefix,
        file_batch_size=args.file_batch_size,
        upload_missing=args.upload_missing,
        max_cats=args.max_cats,
    )


if __name__ == "__main__":
    raise SystemExit(main())
