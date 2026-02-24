"""CLI principal de `library_ops` con subcomandos de mantenimiento."""

from __future__ import annotations

import argparse

from . import bootstrap_manifest, rebuild_versioned, fill_vectorstores, fill_keywords, delete_old_vectorstore_files, orchestrate, relink_openai_files, verify_vectorstore

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="library_ops", description="BibliotecaRAG ops (refactor)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    bootstrap_manifest.build_parser(sp)
    rebuild_versioned.build_parser(sp)
    fill_vectorstores.build_parser(sp)
    fill_vectorstores.build_parser_alias(sp)
    fill_keywords.build_parser(sp)
    delete_old_vectorstore_files.build_parser(sp)
    orchestrate.build_parser(sp)
    relink_openai_files.build_parser(sp)
    verify_vectorstore.build_parser(sp)

    args = ap.parse_args(argv)
    func = getattr(args, "func", None)
    if not func:
        ap.print_help()
        return 2
    return int(func(args))

if __name__ == "__main__":
    raise SystemExit(main())
