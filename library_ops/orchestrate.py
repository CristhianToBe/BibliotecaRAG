from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from . import rebuild_versioned, fill_vectorstores, fill_keywords, delete_old_vectorstore_files


def _log_step(status: str, step_name: str, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    print(f"{status}: {step_name}{suffix}")


def _discover_manifests() -> tuple[Optional[Path], Optional[Path]]:
    """Retorna (new_manifest, old_manifest) a partir de library_dir/vN."""
    from worklib.config import default_config

    cfg = default_config()
    lib_dir = Path(cfg.library_dir)
    versions = sorted([p for p in lib_dir.glob("v*") if p.is_dir()], key=lambda p: p.name)
    if not versions:
        return None, None

    new_manifest = versions[-1] / "_state" / "library.json"
    previous_manifest: Optional[Path] = None
    if len(versions) > 1:
        previous_manifest = versions[-2] / "_state" / "library.json"
    return new_manifest, previous_manifest

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
    steps = [
        ("rebuild_library_versioned", True, ""),
        ("fill_category_vectorstores", not dry_run, "dry-run mode" if dry_run else ""),
        ("fill_category_keywords", not dry_run, "dry-run mode" if dry_run else ""),
        ("delete_old_vectorstore_files", not dry_run, "dry-run mode" if dry_run else ""),
    ]

    if dry_run:
        print("\n=== ORCHESTRATION PLAN (dry-run) ===")
        for step_name, should_run, reason in steps:
            plan_label = "WOULD RUN" if should_run else "WOULD SKIP"
            detail = reason or "enabled"
            print(f"PLAN: {step_name}: {plan_label} ({detail})")

    vectorstores_ran = False
    delete_ran = False

    # 1) rebuild
    _log_step("STEP START", "rebuild_library_versioned")
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
        _log_step("STEP DONE", "rebuild_library_versioned", f"rc={rc}")
        return rc
    _log_step("STEP DONE", "rebuild_library_versioned")

    new_manifest, discovered_old_manifest = _discover_manifests()
    if not new_manifest:
        print("❌ No encontré carpetas vN en library_dir")
        return 2
    if not new_manifest.exists():
        print(f"❌ No encontré manifest generado: {new_manifest}")
        return 2

    old_manifest_to_use = old_manifest or discovered_old_manifest
    if not old_manifest:
        if old_manifest_to_use:
            _log_step("STEP SKIPPED", "old_manifest_override", f"using auto-discovered {old_manifest_to_use}")
        else:
            _log_step("STEP SKIPPED", "old_manifest_override", "no override provided")

    if dry_run:
        _log_step("STEP SKIPPED", "fill_category_vectorstores", "dry-run mode")
        _log_step("STEP SKIPPED", "fill_category_keywords", "dry-run mode")
    else:
        # 2) fill vectorstores + keywords in parallel
        _log_step("STEP START", "fill_category_vectorstores")
        _log_step("STEP START", "fill_category_keywords")
        with ThreadPoolExecutor(max_workers=2) as ex:
            future_vs = ex.submit(
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
            )
            future_kw = ex.submit(
                fill_keywords.run,
                manifest=new_manifest,
                out=None,
                model="gpt-5-nano",
                per_category_docs=4,
                max_chars_per_doc=10000,
                only_empty=only_empty_keywords,
                debug=debug,
            )

            rc_vs = future_vs.result()
            rc_kw = future_kw.result()

        _log_step("STEP DONE", "fill_category_vectorstores", f"rc={rc_vs}")
        _log_step("STEP DONE", "fill_category_keywords", f"rc={rc_kw}")
        vectorstores_ran = True

        if rc_vs != 0:
            return rc_vs
        if rc_kw != 0:
            return rc_kw

    if dry_run:
        _log_step("STEP SKIPPED", "delete_old_vectorstore_files", "dry-run mode")
    elif not old_manifest_to_use:
        _log_step("STEP DONE", "delete_old_vectorstore_files", "rc=1")
        print("❌ STEP FAILED: delete_old_vectorstore_files requires an old manifest (override or auto-discovered previous version).")
        return 1
    else:
        _log_step("STEP START", "delete_old_vectorstore_files")
        rc3 = delete_old_vectorstore_files.run(
            manifest=old_manifest_to_use,
            dry_run=dry_run,
            debug=debug,
            sleep_ms=0,
            max_deletes=0,
            skip_vs=skip_vs,
        )
        _log_step("STEP DONE", "delete_old_vectorstore_files", f"rc={rc3}")
        delete_ran = True
        if rc3 != 0:
            return rc3

    if not dry_run and not vectorstores_ran:
        print("❌ STEP FAILED: fill_category_vectorstores did not run in non-dry-run mode.")
        return 1
    if not dry_run and not delete_ran:
        print("❌ STEP FAILED: delete_old_vectorstore_files did not run in non-dry-run mode.")
        return 1

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
