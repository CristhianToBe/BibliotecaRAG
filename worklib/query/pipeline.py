from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from worklib.config import default_config
from worklib.store import Doc, Manifest, load_manifest

from .arbitrate import arbitrate
from .confirm import confirm_once_non_interactive
from .llm import eprint
from .paths import resolve_local_path
from .pick import pick_categories
from .refine import refine_all
from .retrieve import retrieve_via_tool
from .telemetry import RequestTelemetry, get_telemetry
from .write import write_answer


_PICK_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_RETRIEVAL_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return max(0.1, float(raw))
    except Exception:
        return default


def _normalize_question(question: str) -> str:
    s = re.sub(r"\s+", " ", (question or "").strip().lower())
    return s


def _manifest_version(manifest_path: str) -> str:
    p = Path(manifest_path)
    try:
        st = p.stat()
        return f"{p}:{st.st_mtime_ns}:{st.st_size}"
    except Exception:
        return str(p)


def _cache_get(cache: Dict[str, Tuple[float, Any]], key: str) -> Any | None:
    now = time.time()
    with _CACHE_LOCK:
        item = cache.get(key)
        if not item:
            return None
        exp, value = item
        if exp < now:
            cache.pop(key, None)
            return None
        return value


def _cache_set(cache: Dict[str, Tuple[float, Any]], key: str, value: Any, ttl_s: float) -> None:
    with _CACHE_LOCK:
        cache[key] = (time.time() + ttl_s, value)


def _timed_stage(name: str, fn: Callable[[], Any]) -> Any:
    telemetry = get_telemetry()
    started = time.perf_counter()
    try:
        return fn()
    finally:
        if telemetry:
            telemetry.mark_stage(name, time.perf_counter() - started)


def _run_with_timeout(timeout_s: float, fn: Callable[[], Any], fallback: Any) -> Any:
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except FuturesTimeoutError:
        return fallback
    except Exception:
        return fallback
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _default_manifest_path() -> str:
    _cfg = default_config()
    return os.getenv("RAG_MANIFEST_PATH") or os.getenv("WORKLIB_MANIFEST_PATH") or str(_cfg.manifest_path)


def _count_docs_by_category(manifest: Manifest) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in manifest.docs.values() if isinstance(manifest.docs, dict) else []:
        cname = str(getattr(d, "category", "") or "").strip()
        if cname:
            counts[cname] = counts.get(cname, 0) + 1
    return counts


def _simplify_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{2,}", query or "")
    for t in tokens:
        if t.isupper() and len(t) >= 3:
            return t
    return max(tokens, key=len) if tokens else (query or "").strip()


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


def attach_local_paths(manifest: Manifest, hits: List[Dict[str, Any]], *, library_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    return [dict(h, local_path=resolve_local_path(manifest, file_id=h.get("file_id"), filename=h.get("filename"), library_root=library_root)) for h in hits]


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
        refs.append({"doc_id": doc_id, "filename": str(getattr(d, "filename", "") or ""), "abs_path": str(getattr(d, "abs_path", "") or "")})
    return refs


def _cap_evidence_chars(hits: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    used = 0
    out: List[Dict[str, Any]] = []
    for h in hits:
        text = str(h.get("text") or "")
        if not text:
            continue
        room = max_chars - used
        if room <= 0:
            break
        hh = dict(h)
        if len(text) > room:
            hh["text"] = text[:room]
            out.append(hh)
            break
        out.append(hh)
        used += len(text)
    return out


def _retrieve_by_category(
    *,
    q_final: str,
    refiners: List[Dict[str, Any]],
    category_infos: List[Dict[str, Any]],
    max_workers: int,
    retrieve_timeout_s: float,
    top_k: int,
    debug: bool,
) -> List[Dict[str, Any]]:
    telemetry = get_telemetry()
    if telemetry:
        telemetry.add_event("retrieving", categories=[x["category"] for x in category_infos])

    queries = [q_final] + [str(r.get("query") or "").strip() for r in refiners]
    queries = [q for i, q in enumerate(queries) if q and q not in queries[:i]]

    def _run_category(info: Dict[str, Any]) -> List[Dict[str, Any]]:
        cname = str(info["category"])
        vs_id = str(info["vector_store_id"] or "").strip()
        if not vs_id:
            if telemetry:
                telemetry.mark_retrieval_category(cname, 0.0, "missing_vs")
            return []
        started = time.perf_counter()
        out: List[Dict[str, Any]] = []
        status = "ok"
        try:
            for q in queries:
                out.extend(retrieve_via_tool([vs_id], q, max_num_results=top_k, debug=False, max_workers=max_workers))
        except Exception:
            status = "failed"
        elapsed = time.perf_counter() - started
        if elapsed > retrieve_timeout_s:
            status = "timeout"
            return []
        if telemetry:
            telemetry.mark_retrieval_category(cname, elapsed, status)
        return out

    workers = max(1, min(2, len(category_infos)))
    all_hits: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_category, info): info for info in category_infos}
        for fut in as_completed(futs, timeout=max(1.0, retrieve_timeout_s * max(1, len(category_infos)))):
            try:
                all_hits.extend(fut.result(timeout=retrieve_timeout_s))
            except Exception:
                info = futs[fut]
                if telemetry:
                    telemetry.mark_retrieval_category(str(info["category"]), retrieve_timeout_s, "timeout")
    return all_hits


def _pick_with_cache(question: str, categories: Dict[str, Any], pick_ttl_s: float, debug: bool) -> Tuple[Dict[str, Any], bool]:
    key = _normalize_question(question)
    cached = _cache_get(_PICK_CACHE, key)
    if cached is not None:
        return dict(cached), True
    picked = pick_categories(question, categories, debug=debug)
    _cache_set(_PICK_CACHE, key, dict(picked), pick_ttl_s)
    return picked, False


def run_pick_confirm(
    question: str,
    *,
    manifest_path: Optional[str] = None,
    debug: bool = False,
    confirm_glimpse: bool = True,
) -> Dict[str, Any]:
    mp = manifest_path or _default_manifest_path()
    manifest = load_manifest(Path(mp))
    pick_ttl_s = _env_float("WEBAPP_PICK_CACHE_TTL_S", 600)

    picked, pick_cache_hit = _timed_stage("pick", lambda: _pick_with_cache(question, manifest.categories, pick_ttl_s, debug))
    telemetry = get_telemetry()
    if telemetry:
        telemetry.add_event("picked", cache_hit=pick_cache_hit)

    # Non-interactive default optimization: if there is no user reply loop, auto-accept.
    confirm_data = _timed_stage(
        "confirm",
        lambda: {
            "action": "PASS",
            "categories_final": list((picked.get("selected") or [])[:_env_int("WEBAPP_MAX_CATEGORIES", 2)]),
            "suggested_categories": list((picked.get("selected") or [])[:_env_int("WEBAPP_MAX_CATEGORIES", 2)]),
            "selector_instruction": "",
        },
    )

    if confirm_glimpse:
        # Keep compatibility hook: allow explicit confirm policy via env.
        if os.getenv("WEBAPP_FORCE_CONFIRM", "0") == "1":
            confirm_data = _timed_stage(
                "confirm",
                lambda: confirm_once_non_interactive(question, picked=picked, manifest=manifest, use_glimpse=True, debug=debug),
            )

    if telemetry:
        telemetry.add_event("confirmed", action=confirm_data.get("action", "PASS"))

    return {"manifest_path": mp, "question": question, "picked": picked, "pick_cache_hit": pick_cache_hit, "confirm": confirm_data}


def run_after_confirm(
    question: str,
    *,
    selected_categories: List[str],
    manifest_path: Optional[str] = None,
    max_workers: int = 3,
    debug: bool = False,
    picked: Optional[Dict[str, Any]] = None,
    selector_instruction: str = "",
    user_reply: str = "",
) -> Dict[str, Any]:
    mp = manifest_path or _default_manifest_path()
    manifest = load_manifest(Path(mp))
    telemetry = get_telemetry()

    retrieve_timeout_s = _env_float("WEBAPP_RETRIEVE_TIMEOUT_S", 20)
    answer_timeout_s = _env_float("WEBAPP_ANSWER_TIMEOUT_S", 45)
    max_categories = _env_int("WEBAPP_MAX_CATEGORIES", 2)
    top_k = _env_int("WEBAPP_TOP_K", 8)
    max_prompt_chars = _env_int("WEBAPP_MAX_PROMPT_CHARS", 12000)
    retrieval_ttl_s = _env_float("WEBAPP_RETRIEVAL_CACHE_TTL_S", 1800)

    library_root = Path(mp).parent
    picked_curr = picked or pick_categories(question, manifest.categories, debug=debug)

    valid_set = set(manifest.categories.keys())
    cats_final = [c for c in (selected_categories or []) if c in valid_set][:max_categories]
    if not cats_final:
        cats_final = [c for c in (picked_curr.get("selected", []) or []) if c in valid_set][:max_categories]

    q_final = question
    if selector_instruction or user_reply:
        q_final = (question + "\n\nInstrucción de ajuste: " + (selector_instruction or "") + "\nRespuesta del usuario: " + (user_reply or "")).strip()

    must_terms = list(picked_curr.get("must_include_terms", []) or [])
    avoid_terms = list(picked_curr.get("avoid_terms", []) or [])

    refiners = _timed_stage(
        "refine",
        lambda: _run_with_timeout(
            retrieve_timeout_s,
            lambda: refine_all(q_final, must_terms, avoid_terms, max_workers=max_workers, debug=debug),
            [{"name": "A?", "query": q_final, "constraints": {"prefer_norma_first": True}}],
        ),
    )

    doc_counts = _count_docs_by_category(manifest)
    category_infos: List[Dict[str, Any]] = []
    for cname in cats_final:
        cat = manifest.categories.get(cname)
        vs_id = str(getattr(cat, "vector_store_id", "") or "").strip() if cat else ""
        category_infos.append({"category": cname, "vector_store_id": vs_id, "docs": int(doc_counts.get(cname, 0)), "vs_exists": bool(vs_id)})

    cache_input = {
        "q": _normalize_question(q_final),
        "cats": [x["category"] for x in category_infos],
        "manifest": _manifest_version(mp),
    }
    retrieval_key = hashlib.sha1(json.dumps(cache_input, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    retrieval_cache_hit = False
    all_hits = _cache_get(_RETRIEVAL_CACHE, retrieval_key)
    if all_hits is None:
        all_hits = _timed_stage(
            "retrieval",
            lambda: _retrieve_by_category(
                q_final=q_final,
                refiners=refiners,
                category_infos=category_infos,
                max_workers=max_workers,
                retrieve_timeout_s=retrieve_timeout_s,
                top_k=top_k,
                debug=debug,
            ),
        )
        if not all_hits:
            fallback_query = _simplify_query(q_final)
            if fallback_query and fallback_query != q_final:
                all_hits = _timed_stage(
                    "retrieval",
                    lambda: _retrieve_by_category(
                        q_final=fallback_query,
                        refiners=[],
                        category_infos=category_infos,
                        max_workers=max_workers,
                        retrieve_timeout_s=retrieve_timeout_s,
                        top_k=top_k,
                        debug=debug,
                    ),
                )
        _cache_set(_RETRIEVAL_CACHE, retrieval_key, list(all_hits), retrieval_ttl_s)
    else:
        retrieval_cache_hit = True
        if telemetry:
            telemetry.mark_stage("retrieval", 0.0)

    if telemetry:
        telemetry.add_event("retrieving", cache_hit=retrieval_cache_hit)

    all_hits = attach_local_paths(manifest, merge_and_dedupe_evidence(all_hits), library_root=library_root)
    all_hits = _cap_evidence_chars(all_hits[:30], max_prompt_chars)

    if telemetry:
        telemetry.add_event("answering")

    answer = _timed_stage(
        "answer_generation",
        lambda: _run_with_timeout(answer_timeout_s, lambda: write_answer(q_final, all_hits, debug=debug), "No fue posible completar la respuesta dentro del tiempo límite. Aquí tienes la evidencia recuperada para continuar."),
    )

    references = _collect_references(manifest, all_hits)
    if telemetry:
        telemetry.add_event("done")

    return {
        "answer": answer,
        "selected_categories": list(cats_final or []),
        "references": references,
        "retrieval_cache_hit": retrieval_cache_hit,
    }


def pro_query_non_interactive(
    question: str,
    *,
    manifest_path: Optional[str] = None,
    max_workers: int = 3,
    debug: bool = False,
    confirm: bool = True,
    confirm_rounds: int = 4,
    confirm_glimpse: bool = True,
) -> Dict[str, Any]:
    _ = confirm_rounds
    pre = run_pick_confirm(question, manifest_path=manifest_path, debug=debug, confirm_glimpse=confirm_glimpse if confirm else False)
    picked = dict(pre.get("picked") or {})
    confirm_data = dict(pre.get("confirm") or {})
    selected_categories = list(confirm_data.get("categories_final") or [])
    if not selected_categories:
        selected_categories = list(confirm_data.get("suggested_categories") or [])

    out = run_after_confirm(
        question,
        selected_categories=selected_categories,
        manifest_path=manifest_path,
        max_workers=max_workers,
        debug=debug,
        picked=picked,
    )
    out["pick_cache_hit"] = bool(pre.get("pick_cache_hit"))
    return out


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
    return pro_query_non_interactive(
        question,
        manifest_path=manifest_path,
        max_workers=max_workers,
        debug=debug,
        confirm=confirm,
        confirm_rounds=confirm_rounds,
        confirm_glimpse=confirm_glimpse,
    )


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
    return str(
        pro_query_with_meta(
            question,
            manifest_path=manifest_path,
            max_workers=max_workers,
            debug=debug,
            confirm=confirm,
            confirm_rounds=confirm_rounds,
            confirm_glimpse=confirm_glimpse,
        ).get("answer", "")
    )
