"""CLI principal de `library_ops` con subcomandos de mantenimiento."""

from __future__ import annotations

import argparse

from . import rebuild_versioned, fill_vectorstores, fill_keywords, delete_old_vectorstore_files, orchestrate

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="library_ops", description="BibliotecaRAG ops (refactor)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    rebuild_versioned.build_parser(sp)
    fill_vectorstores.build_parser(sp)
    fill_keywords.build_parser(sp)
    delete_old_vectorstore_files.build_parser(sp)
    orchestrate.build_parser(sp)

    args = ap.parse_args(argv)
    func = getattr(args, "func", None)
    if not func:
        ap.print_help()
        return 2
    return int(func(args))

if __name__ == "__main__":
    raise SystemExit(main())
