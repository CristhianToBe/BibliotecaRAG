from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from worklib.config import default_config
from worklib.store import Doc, Manifest, load_manifest

from .arbitrate import arbitrate
from .confirm import confirm_loop
from .paths import resolve_local_path
from .pick import pick_categories
from .refine import refine_all
from .retrieve import retrieve_via_tool
from .write import write_answer


def _default_manifest_path() -> str:
    _cfg = default_config()
    return (
        os.getenv("RAG_MANIFEST_PATH")
        or os.getenv("WORKLIB_MANIFEST_PATH")
        or str(_cfg.manifest_path)
    )


def build_vs_ids(manifest: Manifest, selected_categories: List[str]) -> List[str]:
    vs: List[str] = []
    for cname in selected_categories:
        cat = manifest.categories.get(cname)
        if not cat:
            continue
        vs_id = getattr(cat, "vector_store_id", "") or ""
        if vs_id:
            vs.append(vs_id)
    seen = set()
    out: List[str] = []
    for x in vs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def merge_and_dedupe_evidence(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for h in hits:
        key = (h.get("file_id") or "", h.get("filename") or "", (h.get("text") or "")[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def attach_local_paths(
    manifest: Manifest,
    hits: List[Dict[str, Any]],
    *,
    library_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        lp = resolve_local_path(
            manifest,
            file_id=h.get("file_id"),
            filename=h.get("filename"),
            library_root=library_root,
        )
        hh = dict(h)
        hh["local_path"] = lp
        out.append(hh)
    return out


def _find_doc_reference(manifest: Manifest, hit: Dict[str, Any]) -> Optional[Doc]:
    fid = str(hit.get("file_id") or "").strip()
    local_path = str(hit.get("local_path") or "").strip()
    filename = str(hit.get("filename") or "").strip().lower()

    docs = manifest.docs.values() if isinstance(manifest.docs, dict) else []

    if fid:
        for d in docs:
            if str(getattr(d, "openai_file_id", "") or "").strip() == fid:
                return d

    if local_path:
        for d in docs:
            if str(getattr(d, "abs_path", "") or "").strip() == local_path:
                return d

    if filename:
        for d in docs:
            if str(getattr(d, "filename", "") or "").strip().lower() == filename:
                return d

    return None


def _collect_references(manifest: Manifest, hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    seen: set[str] = set()

    for h in hits:
        d = _find_doc_reference(manifest, h)
        if not d:
            continue
        doc_id = str(getattr(d, "doc_id", "") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        refs.append(
            {
                "doc_id": doc_id,
                "filename": str(getattr(d, "filename", "") or ""),
                "abs_path": str(getattr(d, "abs_path", "") or ""),
            }
        )

    return refs


def pro_query_with_meta(
    question: str,
    *,
    manifest_path: Optional[str] = None,
    max_workers: int = 3,
    debug: bool = False,
    confirm: bool = True,
    confirm_rounds: int = 4,
    confirm_glimpse: bool = True,
) -> Dict[str, Any]:
    mp = manifest_path or _default_manifest_path()
    manifest = load_manifest(Path(mp))

    library_root = None
    try:
        library_root = Path(mp).parent
    except Exception:
        library_root = None

    picked = pick_categories(question, manifest.categories, debug=debug)
    selected = list(picked.get("selected", []) or [])[:2]

    q_final = question
    cats_final = selected
    if confirm:
        q_final, cats_final, picked = confirm_loop(
            question,
            picked=picked,
            manifest=manifest,
            max_rounds=confirm_rounds,
            use_glimpse=confirm_glimpse,
            debug=debug,
        )

    must_terms = list(picked.get("must_include_terms", []) or [])
    avoid_terms = list(picked.get("avoid_terms", []) or [])

    refiners = refine_all(q_final, must_terms, avoid_terms, max_workers=max_workers, debug=debug)

    vs_ids = build_vs_ids(manifest, cats_final)
    if not vs_ids:
        for _, cat in manifest.categories.items():
            vs_id = getattr(cat, "vector_store_id", "") or ""
            if vs_id:
                vs_ids.append(vs_id)
            if len(vs_ids) >= 2:
                break

    all_hits: List[Dict[str, Any]] = []
    all_hits.extend(retrieve_via_tool(vs_ids, q_final, max_num_results=12, debug=debug, max_workers=max_workers))
    for r in refiners:
        rq = (r.get("query") or "").strip()
        if rq and rq != q_final:
            all_hits.extend(retrieve_via_tool(vs_ids, rq, max_num_results=8, debug=False, max_workers=max_workers))

    all_hits = merge_and_dedupe_evidence(all_hits)
    all_hits = attach_local_paths(manifest, all_hits, library_root=library_root)

    arb = arbitrate(q_final, refiners, all_hits[:25], debug=debug)
    best_query = (arb.get("best_query") or q_final).strip()
    if best_query and best_query != q_final:
        all_hits.extend(retrieve_via_tool(vs_ids, best_query, max_num_results=10, debug=False, max_workers=max_workers))
        all_hits = merge_and_dedupe_evidence(all_hits)
        all_hits = attach_local_paths(manifest, all_hits, library_root=library_root)

    answer = write_answer(q_final, all_hits[:30], debug=debug)
    references = _collect_references(manifest, all_hits)

    return {
        "answer": answer,
        "selected_categories": list(cats_final or []),
        "references": references,
    }


def pro_query(
    question: str,
    *,
    manifest_path: Optional[str] = None,
    max_workers: int = 3,
    debug: bool = False,
    confirm: bool = True,
    confirm_rounds: int = 4,
    confirm_glimpse: bool = True,
) -> str:
    data = pro_query_with_meta(
        question,
        manifest_path=manifest_path,
        max_workers=max_workers,
        debug=debug,
        confirm=confirm,
        confirm_rounds=confirm_rounds,
        confirm_glimpse=confirm_glimpse,
    )
    return str(data.get("answer", ""))
