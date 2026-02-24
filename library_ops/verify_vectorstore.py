"""Comando para verificar cantidad de archivos adjuntos en un vector store."""

from __future__ import annotations

import argparse

from .openai_utils import get_client_from_env


def run(*, vector_store_id: str, debug: bool = False) -> int:
    client = get_client_from_env()
    files = list(client.vector_stores.files.list(vector_store_id))
    print("=== VERIFY VECTOR STORE ===")
    print(f"- vector_store_id: {vector_store_id}")
    print(f"- attached_files: {len(files)}")
    if debug:
        for item in files[:20]:
            print(f"  - {getattr(item, 'id', '')}")
    return 0


def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("verify-vectorstore", help="Muestra cantidad de archivos adjuntos de un vector store")
    p.add_argument("--vector-store-id", required=True)
    p.add_argument("--debug", action="store_true")

    def _cmd(args: argparse.Namespace) -> int:
        return run(vector_store_id=args.vector_store_id, debug=args.debug)

    p.set_defaults(func=_cmd)
