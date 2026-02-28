from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .manifest_json import safe_json_load
from .openai_utils import get_client_from_env

def run(
    *,
    manifest: Path,
    dry_run: bool = False,
    debug: bool = False,
    sleep_ms: int = 0,
    max_deletes: int = 0,
    skip_vs: str = "",
) -> int:
    client = get_client_from_env()

    if not manifest.exists():
        print(f"❌ No existe: {manifest}")
        return 2

    data = safe_json_load(manifest)
    docs: Dict[str, Any] = data.get("docs", {}) or {}

    skip_set: Set[str] = set([x.strip() for x in (skip_vs or "").split(",") if x.strip()])

    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for d in docs.values():
        vsid = (d.get("vector_store_id") or "").strip()
        fid = (d.get("openai_file_id") or "").strip()
        if not vsid or not fid:
            continue
        if vsid in skip_set:
            continue
        key = (vsid, fid)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    vs_count = len(set(v for v, _ in pairs))
    print("=== INVENTARIO DELETE ===")
    print(f"- manifest: {manifest}")
    print(f"- docs en manifest: {len(docs)}")
    print(f"- pares (vsid,file_id) únicos a eliminar: {len(pairs)}")
    print(f"- vector stores involucrados: {vs_count}")
    print(f"- dry-run: {dry_run}")

    if dry_run:
        if debug:
            print("\n--- PARES (debug) ---")
            for vsid, fid in pairs[:500]:
                print(f"- {vsid} <- {fid}")
            print("--- FIN ---")
        print("\n📝 Dry-run: no se ejecutó ningún delete.")
        return 0

    print("\n🧹 Eliminando archivos de vector stores antiguos (delete)...")
    deleted = 0
    errors = 0

    for vsid, fid in pairs:
        if max_deletes and deleted >= max_deletes:
            print(f"⏹️ Alcanzado max-deletes={max_deletes}.")
            break
        try:
            if debug:
                print(f"[DEL] vs={vsid} file={fid}")
            client.vector_stores.files.delete(vector_store_id=vsid, file_id=fid)
            deleted += 1
        except Exception as e:
            errors += 1
            print(f"⚠️ Error vs={vsid} file={fid}: {e}")
        if sleep_ms and sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    print("\n✅ RESUMEN ===")
    print(f"- deletes ejecutados: {deleted}")
    print(f"- errores: {errors}")
    print("Nota: la eliminación es eventualmente consistente; puede tardar un poco en reflejarse en búsquedas.")
    return 0



def _configure_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", required=True, help="Ruta al library.json viejo")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--sleep-ms", type=int, default=0)
    p.add_argument("--max-deletes", type=int, default=0)
    p.add_argument("--skip-vs", default="")


def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("delete-old-vs-files", help="Desvincula archivos de vector stores antiguos usando un manifest viejo")
    _configure_parser(p)

    def _cmd(args: argparse.Namespace) -> int:
        return run(
            manifest=Path(args.manifest).expanduser(),
            dry_run=args.dry_run,
            debug=args.debug,
            sleep_ms=args.sleep_ms,
            max_deletes=args.max_deletes,
            skip_vs=args.skip_vs,
        )

    p.set_defaults(func=_cmd)


def build_parser_alias(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("delete-old-vectors", help="Alias explícito para eliminar archivos de vector stores antiguos")
    _configure_parser(p)

    def _cmd(args: argparse.Namespace) -> int:
        return run(
            manifest=Path(args.manifest).expanduser(),
            dry_run=args.dry_run,
            debug=args.debug,
            sleep_ms=args.sleep_ms,
            max_deletes=args.max_deletes,
            skip_vs=args.skip_vs,
        )

    p.set_defaults(func=_cmd)
