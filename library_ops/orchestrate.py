from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from worklib.versioning import (
    archive_version_folder,
    get_biblioteca_root,
    latest_manifest_path,
    latest_version,
    list_versions,
)

from . import rebuild_versioned, fill_vectorstores, fill_keywords, delete_old_vectorstore_files


def _log_step(status: str, step_name: str, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    print(f"{status}: {step_name}{suffix}")


def _discover_manifests() -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Retorna (new_manifest, old_manifest, old_version_dir)."""
    versions = list_versions(get_biblioteca_root())
    if not versions:
        return None, None, None

    new_manifest = versions[-1][2] / "_state" / "library.json"
    previous_manifest: Optional[Path] = None
    previous_dir: Optional[Path] = None
    if len(versions) > 1:
        previous_manifest = versions[-2][2] / "_state" / "library.json"
        previous_dir = versions[-2][2]
    return new_manifest, previous_manifest, previous_dir


def _run_smoke_test(*, manifest_path: Path, smoke_query: str, debug: bool) -> int:
    _log_step("STEP START", "smoke_test_query", f'manifest={manifest_path} query="{smoke_query}"')
    try:
        from worklib.query.pipeline import pro_query_with_meta

        out = pro_query_with_meta(smoke_query, manifest_path=str(manifest_path), debug=debug, confirm=True)
    except Exception as exc:
        _log_step("STEP DONE", "smoke_test_query", f"rc=1 error={exc}")
        print(f"❌ Smoke test falló por excepción: {exc}")
        return 1

    answer = str(out.get("answer") or "").strip()
    if not answer:
        _log_step("STEP DONE", "smoke_test_query", "rc=1 empty_answer")
        print("❌ Smoke test falló: respuesta vacía.")
        return 1

    _log_step("STEP DONE", "smoke_test_query", "rc=0")
    print("✅ Smoke test OK.")
    return 0


def run(
    *,
    manifest: Optional[Path],
    apply: bool,
    mode: str,
    debug: bool,
    dry_run: bool,
    only_empty_keywords: bool,
    only_empty_vectorstores: bool,
    vs_name_prefix: str,
    upload_missing: bool,
    file_batch_size: int,
    old_manifest: Optional[Path],
    skip_vs: str,
    delete_old_vectors: bool,
    archive_previous: bool,
    keep_old_folder: bool,
    smoke_test: bool,
    smoke_test_query: str,
) -> int:
    detected_latest = latest_version(get_biblioteca_root())
    _log_step("DEBUG", "latest_version_before_run", str(detected_latest or "none"))

    if dry_run:
        print("\n=== ORCHESTRATION PLAN (dry-run) ===")
        print("PLAN: rebuild_library_versioned: WOULD RUN")
        print("PLAN: fill_category_vectorstores: WOULD SKIP (dry-run mode)")
        print("PLAN: fill_category_keywords: WOULD SKIP (dry-run mode)")
        print("PLAN: delete_old_vectorstore_files: WOULD SKIP (manual-only; requires --delete-old-vectors)")
        print("PLAN: smoke_test_query: WOULD SKIP (dry-run mode)")
        print("PLAN: archive_previous_version: WOULD SKIP (dry-run mode)")

    vectorstores_ran = False

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

    new_manifest, discovered_old_manifest, discovered_old_dir = _discover_manifests()
    if not new_manifest or not new_manifest.exists():
        print("❌ No encontré el manifest de la versión más reciente en Biblioteca/vN/_state/library.json")
        return 2

    _log_step("DEBUG", "latest_manifest", str(new_manifest))
    old_manifest_to_use = old_manifest or discovered_old_manifest

    if dry_run:
        _log_step("STEP SKIPPED", "fill_category_vectorstores", "dry-run mode")
        _log_step("STEP SKIPPED", "fill_category_keywords", "dry-run mode")
    else:
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
    elif not delete_old_vectors:
        _log_step("STEP SKIPPED", "delete_old_vectorstore_files", "manual-only (use --delete-old-vectors)")
    elif not old_manifest_to_use:
        _log_step("STEP DONE", "delete_old_vectorstore_files", "rc=1")
        print("❌ delete-old-vectors solicitado pero no hay manifest viejo disponible.")
        return 1
    else:
        _log_step("STEP START", "delete_old_vectorstore_files", f"manifest={old_manifest_to_use}")
        rc3 = delete_old_vectorstore_files.run(
            manifest=old_manifest_to_use,
            dry_run=dry_run,
            debug=debug,
            sleep_ms=0,
            max_deletes=0,
            skip_vs=skip_vs,
        )
        _log_step("STEP DONE", "delete_old_vectorstore_files", f"rc={rc3}")
        if rc3 != 0:
            return rc3

    if not dry_run and not vectorstores_ran:
        print("❌ STEP FAILED: fill_category_vectorstores did not run in non-dry-run mode.")
        return 1

    if dry_run:
        _log_step("STEP SKIPPED", "smoke_test_query", "dry-run mode")
    elif smoke_test:
        rc_smoke = _run_smoke_test(manifest_path=new_manifest, smoke_query=smoke_test_query, debug=debug)
        if rc_smoke != 0:
            return rc_smoke
    else:
        _log_step("STEP SKIPPED", "smoke_test_query", "disabled by --no-smoke-test")

    if dry_run:
        _log_step("STEP SKIPPED", "archive_previous_version", "dry-run mode")
    elif not archive_previous:
        _log_step("STEP SKIPPED", "archive_previous_version", "disabled by --no-archive")
    elif not discovered_old_dir or not discovered_old_dir.exists():
        _log_step("STEP SKIPPED", "archive_previous_version", "no previous version")
    else:
        _log_step("STEP START", "archive_previous_version", f"source={discovered_old_dir}")
        archive_path = archive_version_folder(discovered_old_dir, keep_old_folder=keep_old_folder)
        detail = f"zip={archive_path}"
        if keep_old_folder:
            detail += " | old_folder_kept=true"
        else:
            detail += " | old_folder_removed=true"
        _log_step("STEP DONE", "archive_previous_version", detail)

    latest_after = latest_manifest_path(get_biblioteca_root())
    _log_step("DEBUG", "latest_manifest_after_run", str(latest_after) if latest_after else "none")
    print("\n✅ Orchestrate listo.")
    return 0


def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("orchestrate", help="Orquesta: rebuild -> fill vectorstores/keywords -> smoke test -> archive")
    p.add_argument("--manifest", default=None, help="Manifest base (si se omite, usa último Biblioteca/vN/_state/library.json)")
    p.add_argument("--apply", action="store_true", help="Aplica (si no, solo plan).")

    p.add_argument("--mode", choices=["copy", "move"], default="copy")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="No hace cambios: apply forzado a False + fills/smoke/archive en dry-run")

    p.add_argument("--only-empty-keywords", action="store_true")
    p.add_argument("--only-empty-vectorstores", action="store_true")
    p.add_argument("--vs-name-prefix", default="")
    p.add_argument("--upload-missing", action="store_true")
    p.add_argument("--file-batch-size", type=int, default=200)

    p.add_argument("--old-manifest", default=None, help="Manifest viejo (solo para --delete-old-vectors)")
    p.add_argument("--skip-vs", default="", help="VS a omitir (vs_...,vs_...)")
    p.add_argument("--delete-old-vectors", action="store_true", help="(MANUAL) Ejecuta limpieza de archivos viejos en vector stores")

    p.add_argument("--no-archive", action="store_true", help="No zippea/borra la versión anterior")
    p.add_argument("--keep-old-folder", action="store_true", help="Zippea la versión anterior, pero no elimina la carpeta")

    p.add_argument("--smoke-test", dest="smoke_test", action="store_true", default=True, help="Ejecuta smoke test E2E al final (default)")
    p.add_argument("--no-smoke-test", dest="smoke_test", action="store_false", help="Desactiva smoke test E2E")
    p.add_argument("--smoke-test-query", default="¿qué es el SIAR?", help="Pregunta del smoke test")

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
            delete_old_vectors=args.delete_old_vectors,
            archive_previous=(not args.no_archive),
            keep_old_folder=args.keep_old_folder,
            smoke_test=args.smoke_test,
            smoke_test_query=args.smoke_test_query,
        )

    p.set_defaults(func=_cmd)
