from __future__ import annotations

import os
import re
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


def _count_docs_by_category(manifest: Manifest) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in manifest.docs.values() if isinstance(manifest.docs, dict) else []:
        cname = str(getattr(d, "category", "") or "").strip()
        if not cname:
            continue
        counts[cname] = counts.get(cname, 0) + 1
    return counts


def _simplify_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{2,}", query or "")
    # Prefer acronyms like SIAR.
    for t in tokens:
        if t.isupper() and len(t) >= 3:
            return t
    if tokens:
        return max(tokens, key=len)
    return (query or "").strip()


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

    doc_counts = _count_docs_by_category(manifest)
    category_infos: List[Dict[str, Any]] = []
    for cname in cats_final:
        cat = manifest.categories.get(cname)
        vs_id = str(getattr(cat, "vector_store_id", "") or "").strip() if cat else ""
        docs_in_cat = int(doc_counts.get(cname, 0))
        category_infos.append(
            {
                "category": cname,
                "vector_store_id": vs_id,
                "docs": docs_in_cat,
                "vs_exists": bool(vs_id and docs_in_cat > 0),
            }
        )

    if debug:
        from .llm import eprint

        eprint(f"\n[DEBUG] selected categories final list: {cats_final}")
        for info in category_infos:
            eprint(
                "[DEBUG] category=",
                info["category"],
                "vector_store_id=",
                info["vector_store_id"] or "<missing>",
                "docs=",
                info["docs"],
                "vs_exists=",
                info["vs_exists"],
            )

    category_vs_ids = [str(i["vector_store_id"]).strip() for i in category_infos if str(i["vector_store_id"]).strip()]
    if not category_vs_ids and debug:
        from .llm import eprint

        eprint("[WARN] No selected categories with vector_store_id. Consider running fill_category_vectorstores/fill_vectorstores.")

    fallback_vs_ids: List[str] = []
    if not category_vs_ids:
        for _, cat in manifest.categories.items():
            vs_id = str(getattr(cat, "vector_store_id", "") or "").strip()
            if vs_id:
                fallback_vs_ids.append(vs_id)
            if len(fallback_vs_ids) >= 2:
                break
    vs_ids = category_vs_ids or fallback_vs_ids

    def _retrieve_all_for_query(query_text: str, *, max_results: int, debug_query: bool) -> List[Dict[str, Any]]:
        from .llm import eprint

        combined: List[Dict[str, Any]] = []
        if debug_query:
            eprint(f"[DEBUG] retrieval query used: {query_text}")
        if category_infos:
            for info in category_infos:
                cname = str(info["category"])
                vs_id = str(info["vector_store_id"] or "").strip()
                if not vs_id:
                    if debug_query:
                        eprint(f"[WARN] category={cname} has blank vector_store_id; skipping retrieval for this category.")
                    continue
                hits = retrieve_via_tool([vs_id], query_text, max_num_results=max_results, debug=debug_query, max_workers=max_workers)
                if debug_query:
                    eprint(f"[DEBUG] hits returned category={cname}: {len(hits)}")
                combined.extend(hits)
        elif vs_ids:
            combined.extend(retrieve_via_tool(vs_ids, query_text, max_num_results=max_results, debug=debug_query, max_workers=max_workers))
            if debug_query:
                eprint(f"[DEBUG] hits returned fallback categories(total): {len(combined)}")
        return combined

    all_hits: List[Dict[str, Any]] = []
    all_hits.extend(_retrieve_all_for_query(q_final, max_results=12, debug_query=debug))
    for r in refiners:
        rq = (r.get("query") or "").strip()
        if rq and rq != q_final:
            all_hits.extend(_retrieve_all_for_query(rq, max_results=8, debug_query=False))

    if not all_hits:
        fallback_query = _simplify_query(q_final)
        if fallback_query and fallback_query != q_final:
            if debug:
                from .llm import eprint

                eprint(f"[WARN] no hits for original/refined queries; trying fallback query: {fallback_query}")
            all_hits.extend(_retrieve_all_for_query(fallback_query, max_results=10, debug_query=debug))

    all_hits = merge_and_dedupe_evidence(all_hits)
    all_hits = attach_local_paths(manifest, all_hits, library_root=library_root)
    if debug:
        from .llm import eprint

        eprint(f"[DEBUG] total hits after dedupe/paths: {len(all_hits)}")

    arbiter_hits = all_hits[:25]
    if debug:
        from .llm import eprint

        first_keys = sorted(list(arbiter_hits[0].keys())) if arbiter_hits else []
        eprint(f"[DEBUG] arbiter hits payload length: {len(arbiter_hits)} first_item_keys: {first_keys}")

    arb = arbitrate(q_final, refiners, arbiter_hits, debug=debug)
    best_query = (arb.get("best_query") or q_final).strip()
    if best_query and best_query != q_final:
        all_hits.extend(_retrieve_all_for_query(best_query, max_results=10, debug_query=False))
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
