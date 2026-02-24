"""Comando CLI para relinkear openai_file_id usando archivos ya existentes en OpenAI."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

from .manifest_json import persist_manifest, safe_json_load
from .openai_utils import get_client_from_env


def _iter_openai_files(client: Any, purpose: str) -> Iterable[Any]:
    page = client.files.list(purpose=purpose, limit=100)
    for item in getattr(page, "data", []) or []:
        yield item
    while getattr(page, "has_next_page", lambda: False)():
        page = page.get_next_page()
        for item in getattr(page, "data", []) or []:
            yield item


def _doc_size(doc: Dict[str, Any]) -> int:
    raw = doc.get("size_bytes") or doc.get("bytes") or doc.get("size") or 0
    try:
        return int(raw)
    except Exception:
        return 0


def _choose_best_match(candidates: list[dict[str, Any]], target_size: int, prefer_size: bool) -> tuple[dict[str, Any] | None, bool]:
    if not candidates:
        return None, False

    if prefer_size and target_size > 0:
        size_matches = [c for c in candidates if int(c.get("bytes") or 0) == target_size]
        if len(size_matches) == 1:
            return size_matches[0], False
        if len(size_matches) > 1:
            size_matches.sort(key=lambda c: int(c.get("created_at") or 0), reverse=True)
            return size_matches[0], True

    if len(candidates) == 1:
        return candidates[0], False

    candidates.sort(key=lambda c: int(c.get("created_at") or 0), reverse=True)
    return candidates[0], True


def run(
    *,
    manifest: Path,
    out: Path | None = None,
    purpose: str = "assistants",
    match_by: str = "filename",
    also_match_by_bytes: bool = False,
    dry_run: bool = False,
    debug: bool = False,
) -> int:
    if match_by != "filename":
        print(f"❌ match-by no soportado: {match_by}")
        return 2
    if not manifest.exists():
        print(f"❌ No existe: {manifest}")
        return 2

    client = get_client_from_env()
    data = safe_json_load(manifest)
    docs: Dict[str, Any] = data.get("docs", {}) or {}

    files_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_files = 0
    for f in _iter_openai_files(client, purpose=purpose):
        row = {
            "id": str(getattr(f, "id", "") or ""),
            "filename": str(getattr(f, "filename", "") or ""),
            "bytes": int(getattr(f, "bytes", 0) or 0),
            "created_at": int(getattr(f, "created_at", 0) or 0),
        }
        if not row["id"] or not row["filename"]:
            continue
        files_index[row["filename"]].append(row)
        total_files += 1

    missing_before = 0
    matched = 0
    ambiguous = 0
    for doc in docs.values():
        fid = (doc.get("openai_file_id") or "").strip()
        if fid:
            continue

        missing_before += 1
        filename = (doc.get("filename") or "").strip()
        if not filename:
            continue

        candidates = list(files_index.get(filename, []))
        chosen, was_ambiguous = _choose_best_match(
            candidates,
            target_size=_doc_size(doc),
            prefer_size=also_match_by_bytes,
        )
        if not chosen:
            if debug:
                print(f"- sin match: {doc.get('doc_id') or 'N/A'} filename={filename}")
            continue

        if was_ambiguous:
            ambiguous += 1
        doc["openai_file_id"] = chosen["id"]
        matched += 1
        if debug:
            print(
                f"- relink {doc.get('doc_id') or 'N/A'} -> {chosen['id']} "
                f"(filename={filename}, bytes={chosen['bytes']})"
            )

    still_missing = sum(1 for d in docs.values() if not (d.get("openai_file_id") or "").strip())

    print("=== RELINK OPENAI FILES ===")
    print(f"- manifest: {manifest}")
    print(f"- purpose: {purpose}")
    print(f"- match_by: {match_by} | also_match_by_bytes: {also_match_by_bytes}")
    print(f"- openai files indexados: {total_files}")
    print(f"- docs_missing_before: {missing_before}")
    print(f"- matched: {matched}")
    print(f"- still_missing: {still_missing}")
    print(f"- duplicates_ambiguous: {ambiguous}")

    if dry_run:
        print("📝 Dry-run: no se guardó nada.")
        return 0

    saved_path, backup_path = persist_manifest(manifest, out, data)
    if backup_path:
        print(f"🧷 Backup: {backup_path}")
    print(f"✅ guardado: {saved_path}")
    return 0


def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("relink-openai-files", help="Relinkea openai_file_id desde archivos existentes en OpenAI")
    p.add_argument("--manifest", required=True, help="Ruta a manifest JSON")
    p.add_argument("--out", default="", help="Salida (default: sobreescribe el manifest)")
    p.add_argument("--purpose", default="assistants")
    p.add_argument("--match-by", default="filename", choices=["filename"])
    p.add_argument("--also-match-by-bytes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")

    def _cmd(args: argparse.Namespace) -> int:
        out = Path(args.out).expanduser() if args.out else None
        return run(
            manifest=Path(args.manifest).expanduser(),
            out=out,
            purpose=args.purpose,
            match_by=args.match_by,
            also_match_by_bytes=args.also_match_by_bytes,
            dry_run=args.dry_run,
            debug=args.debug,
        )

    p.set_defaults(func=_cmd)
