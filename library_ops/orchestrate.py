from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import rebuild_versioned, fill_vectorstores, fill_keywords, delete_old_vectorstore_files

def run(
    *,
    manifest: Optional[Path],
    apply: bool,
    mode: str,
    debug: bool,
    dry_run: bool,
    # fill steps
    only_empty_keywords: bool,
    only_empty_vectorstores: bool,
    vs_name_prefix: str,
    upload_missing: bool,
    file_batch_size: int,
    # delete step
    old_manifest: Optional[Path],
    skip_vs: str,
) -> int:
    # 1) rebuild
    print("\n=== STEP 1: rebuild-versioned ===")
    rc = rebuild_versioned.run(
        manifest_override=manifest,
        apply=(apply and (not dry_run)),
        mode=mode,
        debug=debug,
        batch_size=10,
        prefix_batch=20,
        max_rounds=3,
        max_docs=0,
        cleanup_empty=False,
        create_vector_stores=False,
    )
    if rc != 0:
        return rc

    # figure out newest version folder to find its _state/library.json
    # We rely on worklib.config.default_config().library_dir = .../Biblioteca/biblioteca
    from worklib.config import default_config
    cfg = default_config()
    lib_dir = Path(cfg.library_dir)
    versions = sorted([p for p in lib_dir.glob('v*') if p.is_dir()], key=lambda p: p.name)
    if not versions:
        print("❌ No encontré carpetas vN en library_dir")
        return 2
    newest = versions[-1]
    new_manifest = newest / "_state" / "library.json"
    if not new_manifest.exists():
        print(f"❌ No encontré manifest generado: {new_manifest}")
        return 2

    # 2) fill vectorstores + keywords in parallel
    print("\n=== STEP 2: fill-vectorstores + fill-keywords (parallel) ===")
    tasks = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        tasks.append(ex.submit(
            fill_vectorstores.run,
            manifest=new_manifest,
            out=None,
            only_empty=only_empty_vectorstores,
            dry_run=dry_run,
            debug=debug,
            vs_name_prefix=vs_name_prefix,
            file_batch_size=file_batch_size,
            upload_missing=upload_missing,
            max_cats=0,
        ))
        tasks.append(ex.submit(
            fill_keywords.run,
            manifest=new_manifest,
            out=None,
            model="gpt-5-nano",
            per_category_docs=4,
            max_chars_per_doc=10000,
            only_empty=only_empty_keywords,
            debug=debug,
        ))
        for fut in as_completed(tasks):
            rc2 = fut.result()
            if rc2 != 0:
                return rc2

    # 3) delete old files from old manifest
    if old_manifest:
        print("\n=== STEP 3: delete-old-vs-files ===")
        rc3 = delete_old_vectorstore_files.run(
            manifest=old_manifest,
            dry_run=dry_run,
            debug=debug,
            sleep_ms=0,
            max_deletes=0,
            skip_vs=skip_vs,
        )
        return rc3

    print("\n✅ Orchestrate listo.")
    return 0

def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("orchestrate", help="Orquesta: rebuild -> (fill vectorstores + fill keywords) -> delete old")
    p.add_argument("--manifest", default=None, help="Manifest base (si se omite, usa worklib default)")
    p.add_argument("--apply", action="store_true", help="Aplica (si no, solo plan).")

    p.add_argument("--mode", choices=["copy","move"], default="copy")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="No hace cambios: apply forzado a False + fills/delete en dry-run")

    p.add_argument("--only-empty-keywords", action="store_true")
    p.add_argument("--only-empty-vectorstores", action="store_true")
    p.add_argument("--vs-name-prefix", default="")
    p.add_argument("--upload-missing", action="store_true")
    p.add_argument("--file-batch-size", type=int, default=200)

    p.add_argument("--old-manifest", default=None, help="Manifest viejo (para delete-old-vs-files)")
    p.add_argument("--skip-vs", default="", help="VS a omitir (vs_...,vs_...)")
    def _cmd(args: argparse.Namespace) -> int:
        return run(
            manifest=Path(args.manifest).expanduser() if args.manifest else None,
            apply=args.apply,
            mode=args.mode,
            debug=args.debug,
            dry_run=args.dry_run,
            only_empty_keywords=args.only_empty_keywords,
            only_empty_vectorstores=args.only_empty_vectorstores,
            vs_name_prefix=args.vs_name_prefix,
            upload_missing=args.upload_missing,
            file_batch_size=args.file_batch_size,
            old_manifest=Path(args.old_manifest).expanduser() if args.old_manifest else None,
            skip_vs=args.skip_vs,
        )
    p.set_defaults(func=_cmd)
