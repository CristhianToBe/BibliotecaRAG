"""Comando CLI para crear/llenar vector stores por categoría."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .manifest_json import safe_json_load, index_docs_by_category, persist_manifest
from .openai_utils import get_client_from_env, create_vector_store, upload_file


def _list_attached_file_ids(client: Any, vector_store_id: str) -> set[str]:
    attached: set[str] = set()
    pager = client.vector_stores.files.list(vector_store_id)
    for item in pager:
        fid = getattr(item, "id", None)
        if fid:
            attached.add(str(fid))
    return attached


def _attach_file(client: Any, vector_store_id: str, file_id: str) -> None:
    client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_id)

def run(
    *,
    manifest: Path,
    out: Path | None = None,
    only_empty: bool = False,
    dry_run: bool = False,
    debug: bool = False,
    vs_name_prefix: str = "",
    file_batch_size: int = 200,
    upload_missing: bool = True,
    max_cats: int = 0,
) -> int:
    if not manifest.exists():
        print(f"❌ No existe: {manifest}")
        return 2

    client = get_client_from_env()

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
    category_rows: list[dict[str, Any]] = []

    for cat_name, cat in cats.items():
        if max_cats and processed >= max_cats:
            break

        current_vsid = (cat.get("vector_store_id") or "").strip()
        if only_empty and current_vsid:
            continue

        cat_docs = docs_by_cat.get(cat_name, [])
        file_ids: List[str] = []
        uploaded_cat = 0
        for d in cat_docs:
            fid = (d.get("openai_file_id") or "").strip()
            if not fid and upload_missing:
                abs_path = (d.get("abs_path") or "").strip()
                if not abs_path:
                    print(f"⚠️ [{cat_name}] doc sin abs_path, se omite upload: {d.get('filename') or d.get('doc_id') or 'N/A'}")
                    continue
                if dry_run:
                    fid = ""
                else:
                    try:
                        fid = upload_file(client, abs_path, debug=debug)
                        d["openai_file_id"] = fid
                        uploaded_total += 1
                        uploaded_cat += 1
                    except Exception as e:
                        print(f"⚠️ [{cat_name}] no pude subir {abs_path}: {e}")
                        fid = ""
            if fid:
                file_ids.append(fid)

        file_ids = list(dict.fromkeys(file_ids))

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

        attached_cat = 0
        final_vs_count = 0
        if vsid and file_ids:
            if dry_run:
                if debug:
                    print(f"  - (dry-run) adjuntaría {len(file_ids)} files a {vsid}")
            else:
                already_attached = _list_attached_file_ids(client, vsid)
                for fid in file_ids:
                    if fid in already_attached:
                        if debug:
                            print(f"  - ya adjunto: {fid}")
                        continue
                    _attach_file(client, vsid, fid)
                    already_attached.add(fid)
                    attached_cat += 1
                attached_total += attached_cat
                final_vs_count = len(already_attached)
        elif vsid and not dry_run:
            final_vs_count = len(_list_attached_file_ids(client, vsid))

        if vsid:
            for d in cat_docs:
                d["vector_store_id"] = vsid

        category_rows.append(
            {
                "name": cat_name,
                "vector_store_id": vsid,
                "docs": len(cat_docs),
                "uploaded": uploaded_cat,
                "attached": attached_cat,
                "final_vs_count": final_vs_count,
            }
        )

        processed += 1

    if dry_run:
        print("\n📝 Dry-run: no se guardó nada.")
        print(f"- categorías procesadas: {processed}")
        return 0

    saved_path, backup_path = persist_manifest(manifest, out, data)
    if backup_path:
        print(f"\n🧷 Backup: {backup_path}")

    print("\n✅ LISTO ===")
    print(f"- guardado: {saved_path}")
    print(f"- categorías procesadas: {processed}")
    print(f"- vector stores creados: {created}")
    print(f"- archivos adjuntados: {attached_total}")
    print(f"- archivos subidos (upload-missing): {uploaded_total}")
    print("\n=== RESUMEN POR CATEGORÍA ===")
    for row in category_rows:
        print(
            f"- {row['name']} | vs={row['vector_store_id'] or 'N/A'} | "
            f"docs={row['docs']} | uploaded={row['uploaded']} | "
            f"attached={row['attached']} | final_count={row['final_vs_count']}"
        )
    return 0

def _configure_fill_vectorstores_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", required=True, help="Ruta a vN/_state/library.json")
    p.add_argument("--out", default="", help="Salida (default: sobreescribe el manifest)")
    p.add_argument("--only-empty", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--vs-name-prefix", default="")
    p.add_argument("--file-batch-size", type=int, default=200)
    p.add_argument("--upload-missing", dest="upload_missing", action="store_true", default=True)
    p.add_argument("--no-upload-missing", dest="upload_missing", action="store_false")
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


def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("fill-vectorstores", help="Crea/llena vector_store_id por categoría y adjunta documentos")
    _configure_fill_vectorstores_parser(p)


def build_parser_alias(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("fill-category-vectorstores", help="Alias de fill-vectorstores")
    _configure_fill_vectorstores_parser(p)
