from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from .manifest_json import safe_json_load, safe_json_dump, backup_file, index_docs_by_category
from .openai_utils import get_client, create_vector_store, attach_files, upload_file

def run(
    *,
    manifest: Path,
    out: Path | None = None,
    only_empty: bool = False,
    dry_run: bool = False,
    debug: bool = False,
    vs_name_prefix: str = "",
    file_batch_size: int = 200,
    upload_missing: bool = False,
    max_cats: int = 0,
) -> int:
    load_dotenv()
    client = get_client()

    if not manifest.exists():
        print(f"❌ No existe: {manifest}")
        return 2

    out_path = out or manifest

    data = safe_json_load(manifest)
    cats: Dict[str, Any] = data.get("categories", {}) or {}
    docs: Dict[str, Any] = data.get("docs", {}) or {}
    docs_by_cat = index_docs_by_category(docs)

    total_cats = len(cats)
    empty_vs = sum(1 for c in cats.values() if not (c.get("vector_store_id") or "").strip())
    print("=== INVENTARIO ===")
    print(f"- manifest: {manifest}")
    print(f"- categories: {total_cats} | empty_vector_store_id: {empty_vs}")
    print(f"- docs: {len(docs)}")

    processed = created = attached_total = uploaded_total = 0

    for cat_name, cat in cats.items():
        if max_cats and processed >= max_cats:
            break

        current_vsid = (cat.get("vector_store_id") or "").strip()
        if only_empty and current_vsid:
            continue

        cat_docs = docs_by_cat.get(cat_name, [])
        file_ids: List[str] = []
        for d in cat_docs:
            fid = (d.get("openai_file_id") or "").strip()
            if not fid and upload_missing:
                abs_path = (d.get("abs_path") or "").strip()
                if abs_path:
                    if dry_run:
                        fid = ""
                    else:
                        try:
                            fid = upload_file(client, abs_path, debug=debug)
                            d["openai_file_id"] = fid
                            uploaded_total += 1
                        except Exception as e:
                            if debug:
                                print(f"  ! no pude subir {abs_path}: {e}")
                            fid = ""
            if fid:
                file_ids.append(fid)

        vsid = current_vsid
        if not vsid:
            vs_name = cat_name
            if vs_name_prefix:
                vs_name = f"{vs_name_prefix}:{cat_name}"

            if debug or dry_run:
                print(f"\n[CAT] {cat_name}")
                print(f"  - docs en categoría: {len(cat_docs)} | file_ids disponibles: {len(file_ids)}")
                print(f"  - creando vector store: {vs_name}" if not dry_run else f"  - (dry-run) crearía vector store: {vs_name}")

            if dry_run:
                vsid = ""
            else:
                vsid = create_vector_store(client, vs_name)
                created += 1
                cat["vector_store_id"] = vsid
        else:
            if debug:
                print(f"\n[CAT] {cat_name}")
                print(f"  - ya tiene vector_store_id: {vsid}")
                print(f"  - docs en categoría: {len(cat_docs)} | file_ids disponibles: {len(file_ids)}")

        if vsid and file_ids:
            if dry_run:
                if debug:
                    print(f"  - (dry-run) adjuntaría {len(file_ids)} files a {vsid}")
            else:
                attach_files(client, vsid, file_ids, batch_size=file_batch_size, debug=debug)
                attached_total += len(file_ids)

        if vsid:
            for d in cat_docs:
                d["vector_store_id"] = vsid

        processed += 1

    if dry_run:
        print("\n📝 Dry-run: no se guardó nada.")
        print(f"- categorías procesadas: {processed}")
        return 0

    if out_path == manifest:
        bk = backup_file(manifest)
        print(f"\n🧷 Backup: {bk}")

    safe_json_dump(out_path, data)

    print("\n✅ LISTO ===")
    print(f"- guardado: {out_path}")
    print(f"- categorías procesadas: {processed}")
    print(f"- vector stores creados: {created}")
    print(f"- archivos adjuntados: {attached_total}")
    print(f"- archivos subidos (upload-missing): {uploaded_total}")
    return 0

def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("fill-vectorstores", help="Crea/llena vector_store_id por categoría y adjunta documentos")
    p.add_argument("--manifest", required=True, help="Ruta a vN/_state/library.json")
    p.add_argument("--out", default="", help="Salida (default: sobreescribe el manifest)")
    p.add_argument("--only-empty", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--vs-name-prefix", default="")
    p.add_argument("--file-batch-size", type=int, default=200)
    p.add_argument("--upload-missing", action="store_true")
    p.add_argument("--max-cats", type=int, default=0)
    def _cmd(args: argparse.Namespace) -> int:
        out = Path(args.out).expanduser() if args.out else None
        return run(
            manifest=Path(args.manifest).expanduser(),
            out=out,
            only_empty=args.only_empty,
            dry_run=args.dry_run,
            debug=args.debug,
            vs_name_prefix=args.vs_name_prefix,
            file_batch_size=args.file_batch_size,
            upload_missing=args.upload_missing,
            max_cats=args.max_cats,
        )
    p.set_defaults(func=_cmd)
