from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from worklib.config import default_config
from worklib.store import Doc, Manifest, load_manifest

from .arbitrate import arbitrate
from . import confirm as confirm_stage
from .llm import eprint
from .paths import resolve_local_path
from . import pick as pick_stage
from . import refine as refine_stage
from . import retrieve as retrieve_stage
from .telemetry import RequestTelemetry, get_telemetry, reset_current_stage, reset_debug_enabled, reset_telemetry, set_current_stage, set_debug_enabled, set_telemetry, stage_ctx
from . import write as write_stage



# Backward-compat aliases for existing tests/patch points
pick_categories = pick_stage.run
refine_all = refine_stage.run
retrieve_via_tool = retrieve_stage.run
write_answer = write_stage.run
confirm_loop = confirm_stage.confirm_loop

_PICK_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_RETRIEVAL_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class PipelineTimeouts:
    total_s: float = 120.0
    # debug stabilization; pick may be slow when prompt is large
    pick_s: float = 30.0
    # temporary to reduce false timeouts during debugging; adjust later
    confirm_s: float = 20.0
    refine_s: float = 20.0
    # debug stabilization; retrieval with tool-calls can be slow
    retrieve_per_cat_s: float = 30.0
    write_s: float = 45.0


@dataclass
class PipelineOptions:
    use_picker: bool = True
    use_confirmer: bool = True
    use_refiner: bool = True
    refine_a1: bool = True
    refine_a2: bool = True
    refine_a3: bool = True
    use_arbiter: bool = True
    use_retrieve: bool = True
    use_write: bool = True
    max_categories: int = 2
    top_k: int = 8
    max_context_chars: int = 12000
    timeouts: PipelineTimeouts = field(default_factory=PipelineTimeouts)
    unbounded: bool = True


TIMEOUT_GRACE_MS = 250
LATE_RESULT_GRACE_S = 2.0

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


def _top_category_token(c: str) -> str:
    return str(c or "").split("__", 1)[0].strip().upper()


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


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _fallback_pick_categories(question: str, valid_keys: List[str], max_categories: int) -> List[str]:
    text = str(question or "").upper()
    ranked: List[str] = []
    if "SARE" in text or "TRIBUT" in text or "DIAN" in text:
        ranked.extend([k for k in valid_keys if _top_category_token(k) == "DIAN"])
    if not ranked:
        ranked.extend(valid_keys)
    return _dedupe_preserve_order(ranked)[: max(1, max_categories)]


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


def _shrink_evidence_payload(hits: List[Dict[str, Any]], *, max_hits: int = 5, max_per_hit_chars: int = 1200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in (hits or [])[:max_hits]:
        hh = dict(h)
        text = str(hh.get("text") or "")
        if len(text) > max_per_hit_chars:
            hh["text"] = text[:max_per_hit_chars]
        out.append(hh)
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
    unbounded: bool,
) -> List[Dict[str, Any]]:
    telemetry = get_telemetry()
    if telemetry:
        telemetry.add_event("retrieving", categories=[x["category"] for x in category_infos])

    queries = [q_final] + [str(r.get("query") or "").strip() for r in refiners]
    queries = [q for i, q in enumerate(queries) if q and q not in queries[:i]]

    def _run_category(info: Dict[str, Any]) -> List[Dict[str, Any]]:
        cname = str(info["category"])
        vs_id = str(info["vector_store_id"] or "").strip()
        trace_id = getattr(telemetry, "trace_id", "no-trace") if telemetry else "no-trace"
        if not vs_id:
            if debug:
                print("[DEBUG] RETRIEVAL_CAT_END", {"trace_id": trace_id, "category": cname, "elapsed_ms": 0.0, "hits_count": 0, "reason": "missing_vs"})
            if telemetry:
                telemetry.mark_retrieval_category(cname, 0.0, "missing_vs")
            return []
        started = time.perf_counter()
        out: List[Dict[str, Any]] = []
        status = "ok"
        try:
            for q in queries:
                if debug:
                    print("[DEBUG] RETRIEVAL_CAT_BEGIN", {
                        "trace_id": trace_id,
                        "category": cname,
                        "vector_store_id": vs_id,
                        "budget_s": retrieve_timeout_s,
                        "query": q,
                    })
                out.extend(retrieve_via_tool([vs_id], q, max_num_results=top_k, debug=False, max_workers=max_workers))
        except Exception:
            status = "failed"
        elapsed = time.perf_counter() - started
        if (not unbounded) and elapsed > retrieve_timeout_s:
            # Bounded mode only: mark over-budget but keep hits.
            status = "timeout_over_budget_kept_hits"
            if debug:
                print("[DEBUG] RETRIEVAL_CAT_END", {
                    "trace_id": trace_id,
                    "category": cname,
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "hits_count": len(out),
                    "reason": status,
                })
            if telemetry:
                telemetry.mark_retrieval_category(cname, elapsed, status)
            return out
        if debug:
            print("[DEBUG] RETRIEVAL_CAT_END", {"trace_id": trace_id, "category": cname, "elapsed_ms": round(elapsed * 1000, 2), "hits_count": len(out), "reason": status})
        if telemetry:
            telemetry.mark_retrieval_category(cname, elapsed, status)
        return out

    workers = max(1, min(2, len(category_infos)))
    all_hits: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_category, info): info for info in category_infos}
        for fut in as_completed(futs):
            try:
                all_hits.extend(fut.result())
            except Exception:
                info = futs[fut]
                if telemetry:
                    telemetry.mark_retrieval_category(str(info["category"]), 0.0, "failed")
    return all_hits


def _retrieve_quick_fallback(
    *,
    q_final: str,
    category_infos: List[Dict[str, Any]],
    top_k: int,
    max_workers: int,
    debug: bool,
) -> List[Dict[str, Any]]:
    """Best-effort fallback: run a single simplified query per category."""
    q_simple = _simplify_query(q_final)
    out: List[Dict[str, Any]] = []
    for info in category_infos:
        vs_id = str(info.get("vector_store_id") or "").strip()
        if not vs_id:
            continue
        try:
            hits = retrieve_via_tool([vs_id], q_simple, max_num_results=max(2, min(6, top_k)), debug=False, max_workers=max_workers)
            out.extend(hits)
        except Exception:
            if debug:
                print("[DEBUG] RETRIEVAL_FALLBACK_USED", {
                    "reason": "selector_timeout",
                    "returned_candidates": len(out),
                })
    return out


def _pick_with_cache(question: str, categories: Dict[str, Any], pick_ttl_s: float, debug: bool) -> Tuple[Dict[str, Any], bool]:
    key = _normalize_question(question)
    cached = _cache_get(_PICK_CACHE, key)
    if cached is not None:
        return dict(cached), True
    picked = pick_categories(question, categories, debug=debug)
    _cache_set(_PICK_CACHE, key, dict(picked), pick_ttl_s)
    return picked, False


def _coerce_pipeline_options(raw: Optional[Dict[str, Any]]) -> PipelineOptions:
    opts = PipelineOptions()
    if not raw:
        opts.unbounded = True
        return opts

    t_raw = raw.get("timeouts") if isinstance(raw, dict) else None
    unbounded_raw = raw.get("unbounded") if isinstance(raw, dict) else None

    if unbounded_raw is True:
        opts.unbounded = True
    elif t_raw is None:
        opts.unbounded = True
    elif isinstance(t_raw, dict) and (t_raw.get("enabled") is False):
        opts.unbounded = True
    else:
        opts.unbounded = False

    if isinstance(t_raw, dict) and opts.unbounded is False:
        opts.timeouts = PipelineTimeouts(
            total_s=max(5.0, float(t_raw.get("total_s", opts.timeouts.total_s))),
            pick_s=max(1.0, float(t_raw.get("pick_s", opts.timeouts.pick_s))),
            confirm_s=max(1.0, float(t_raw.get("confirm_s", opts.timeouts.confirm_s))),
            refine_s=max(1.0, float(t_raw.get("refine_s", opts.timeouts.refine_s))),
            retrieve_per_cat_s=max(1.0, float(t_raw.get("retrieve_per_cat_s", opts.timeouts.retrieve_per_cat_s))),
            write_s=max(1.0, float(t_raw.get("write_s", opts.timeouts.write_s))),
        )

    opts.use_picker = bool(raw.get("use_picker", opts.use_picker))
    opts.use_confirmer = bool(raw.get("use_confirmer", opts.use_confirmer))
    opts.use_refiner = bool(raw.get("use_refiner", opts.use_refiner))
    opts.refine_a1 = bool(raw.get("refine_a1", opts.refine_a1))
    opts.refine_a2 = bool(raw.get("refine_a2", opts.refine_a2))
    opts.refine_a3 = bool(raw.get("refine_a3", opts.refine_a3))
    opts.use_arbiter = bool(raw.get("use_arbiter", opts.use_arbiter))
    opts.use_retrieve = bool(raw.get("use_retrieve", opts.use_retrieve))
    opts.use_write = bool(raw.get("use_write", opts.use_write))
    opts.max_categories = max(1, min(3, int(raw.get("max_categories", opts.max_categories))))
    opts.top_k = max(1, min(30, int(raw.get("top_k", opts.top_k))))
    opts.max_context_chars = max(1000, min(30000, int(raw.get("max_context_chars", opts.max_context_chars))))
    return opts


def run_pipeline_resilient(
    question: str,
    *,
    manifest_path: Optional[str] = None,
    max_workers: int = 3,
    debug: bool = False,
    pipeline: Optional[Dict[str, Any]] = None,
    manual_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper around the centralized query orchestrator."""
    return run_query(
        question,
        manifest_path=manifest_path,
        max_workers=max_workers,
        debug=debug,
        pipeline=pipeline,
        manual_categories=manual_categories,
    )


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

    if debug:
        eprint(f"[DEBUG] STAGE_BEGIN trace_id={(telemetry.trace_id if telemetry else 'no-trace')} stage_name=confirm")
    confirm_started = time.perf_counter()
    confirm_data = _timed_stage(
        "confirm",
        lambda: confirm_stage.confirm_once_non_interactive(question, picked=picked, manifest=manifest, use_glimpse=confirm_glimpse, debug=debug),
    )
    if debug:
        eprint(
            "[DEBUG] STAGE_END",
            {
                "trace_id": telemetry.trace_id if telemetry else "no-trace",
                "stage_name": "confirm",
                "elapsed_ms": round((time.perf_counter() - confirm_started) * 1000, 2),
                "decision_action": str(confirm_data.get("action") or ""),
                "decision_confidence": confirm_data.get("confidence"),
                "decision_message_to_user_present": bool(str(confirm_data.get("message_to_user") or "").strip()),
            },
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
            lambda: refine_stage.run(q_final, must_terms, avoid_terms, max_workers=max_workers, debug=False),
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
                unbounded=True,
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


def run_query(
    question: str,
    *,
    manifest_path: Optional[str] = None,
    max_workers: int = 3,
    debug: bool = False,
    continue_from: Optional[Dict[str, Any]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
    pipeline: Optional[Dict[str, Any]] = None,
    manual_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Central orchestrator: pick -> confirm loop -> refine -> arbiter -> retrieve -> write."""
    opts = _coerce_pipeline_options(pipeline)
    mp = manifest_path or _default_manifest_path()
    manifest = load_manifest(Path(mp))
    valid_set = set(manifest.categories.keys())
    telemetry = get_telemetry()
    trace_id = getattr(telemetry, "trace_id", "no-trace")

    def _dbg(label: str, payload: Dict[str, Any]) -> None:
        if debug:
            base = {"trace_id": trace_id}
            base.update(payload)
            print(f"[DEBUG] {label}", base)

    def _stage(stage_name: str, fn: Callable[[], Any], *, meta: Optional[Dict[str, Any]] = None) -> Any:
        _dbg("STAGE_BEGIN", {"stage_name": stage_name})
        started = time.perf_counter()
        with stage_ctx(stage_name):
            out = fn()
        end_payload = {
            "stage_name": stage_name,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "next_stage": "pending",
        }
        if meta:
            end_payload.update(meta)
        _dbg("STAGE_END", end_payload)
        if telemetry:
            telemetry.mark_stage(stage_name, time.perf_counter() - started)
        return out

    manual_categories = [str(c).strip() for c in (manual_categories or []) if str(c).strip()]
    if not opts.use_picker and not manual_categories and not (continue_from and continue_from.get("stage") == "confirm_refine"):
        return {"error": "missing_minimum", "details": "Debes activar Picker o indicar categorías manuales.", "trace_id": trace_id}

    if continue_from and (continue_from.get("stage") == "confirm_refine"):
        picked_curr = dict((conversation_state or {}).get("picked") or {})
        q_curr = str((conversation_state or {}).get("last_question") or question or "")
        suggested = [c for c in ((continue_from.get("selected_categories") or (conversation_state or {}).get("suggested_categories") or [])) if c in valid_set]
        user_reply = str(continue_from.get("user_reply") or "")
    else:
        q_curr = question
        if opts.use_picker:
            picked_curr = _stage("pick", lambda: pick_stage.run(q_curr, manifest.categories, debug=debug))
            suggested = [c for c in (picked_curr.get("selected") or []) if c in valid_set][: opts.max_categories]
        else:
            picked_curr = {"selected": list(manual_categories)}
            suggested = [c for c in manual_categories if c in valid_set][: opts.max_categories]
        user_reply = ""

    if not suggested:
        return {"error": "missing_categories", "trace_id": trace_id}

    if opts.use_confirmer:
        confirm_out = _stage(
            "confirm",
            lambda: confirm_stage.run(
                q_curr,
                picked=picked_curr,
                manifest=manifest,
                user_reply=user_reply,
                suggested_categories=suggested,
                use_glimpse=True,
                debug=debug,
            ),
        )
    else:
        confirm_out = {
            "decision": "CONFIRMED",
            "reason": "confirm_disabled",
            "rewritten_prompt": "",
            "categories_final": suggested,
            "suggested_categories": suggested,
            "message_to_user": "",
            "raw": {"action": "PASS"},
        }
        _dbg("STAGE_BEGIN", {"stage_name": "confirm"})
        _dbg("STAGE_END", {"stage_name": "confirm", "elapsed_ms": 0.0, "next_stage": "refine", "reason": "disabled"})

    _dbg("CONFIRM_DECISION", {
        "decision": confirm_out.get("decision"),
        "reason": confirm_out.get("reason", ""),
        "rewritten_prompt": confirm_out.get("rewritten_prompt") or "",
    })

    if confirm_out.get("decision") in {"REPICK", "PARTIAL"}:
        rewritten = str(confirm_out.get("rewritten_prompt") or q_curr).strip() or q_curr
        _dbg("LOOP_REPICK_TRIGGERED", {})
        repicked = _stage("pick", lambda: pick_stage.run(rewritten, manifest.categories, debug=debug))
        state = {
            "last_question": rewritten,
            "manifest_path": mp,
            "picked": repicked,
            "suggested_categories": [c for c in (repicked.get("selected") or []) if c in valid_set][: opts.max_categories],
            "selected_categories": [c for c in (repicked.get("selected") or []) if c in valid_set][: opts.max_categories],
        }
        return {
            "status": "needs_confirmation",
            "trace_id": trace_id,
            "question": rewritten,
            "picked_curr": repicked,
            "confirm": confirm_out,
            "suggested_categories": state["suggested_categories"],
            "state": state,
            "message": confirm_out.get("message_to_user") or "Confirma o ajusta las categorías sugeridas antes de continuar.",
        }

    selected_categories = [
        c for c in (confirm_out.get("categories_final") or confirm_out.get("suggested_categories") or suggested) if c in valid_set
    ][: opts.max_categories]

    must_terms = list(picked_curr.get("must_include_terms", []) or [])
    avoid_terms = list(picked_curr.get("avoid_terms", []) or [])
    enabled_variants: List[str] = []
    if opts.use_refiner:
        if opts.refine_a1:
            enabled_variants.append("A1")
        if opts.refine_a2:
            enabled_variants.append("A2")
        if opts.refine_a3:
            enabled_variants.append("A3")

    refiners = _stage(
        "refine",
        lambda: refine_stage.run(
            q_curr,
            must_terms,
            avoid_terms,
            max_workers=max_workers,
            enabled_variants=enabled_variants,
            debug=debug,
        ),
        meta={"variants_enabled": enabled_variants},
    ) if opts.use_refiner else []

    chosen_candidate: Dict[str, Any]
    arbiter_meta: Dict[str, Any]
    if not refiners:
        chosen_candidate = {"name": "FALLBACK", "query": q_curr, "constraints": {"must_include_terms": must_terms, "avoid_terms": avoid_terms}}
        arbiter_meta = {"winner": "FALLBACK", "reason": "no_refine_candidates", "considered": []}
        _dbg("ARBITER_DECISION", arbiter_meta)
    elif opts.use_arbiter and len(refiners) > 1:
        arbiter_out = _stage(
            "arbiter",
            lambda: arbitrate(
                q_curr,
                refiners,
                [],
                categories=selected_categories,
                selector_instruction=str((confirm_out.get("raw") or {}).get("selector_instruction") or ""),
                debug=debug,
            ),
        )
        chosen_candidate = {
            "name": arbiter_out.get("chosen_variant_name") or "A1",
            "query": arbiter_out.get("chosen_query") or q_curr,
            "constraints": arbiter_out.get("chosen_constraints") or {},
        }
        arbiter_meta = {
            "winner": chosen_candidate.get("name"),
            "reason": arbiter_out.get("rationale", ""),
            "considered": arbiter_out.get("considered", []),
        }
        _dbg("ARBITER_DECISION", arbiter_meta)
    else:
        if opts.use_arbiter:
            _dbg("STAGE_BEGIN", {"stage_name": "arbiter"})
            _dbg("STAGE_END", {"stage_name": "arbiter", "elapsed_ms": 0.0, "next_stage": "retrieve", "reason": "single_candidate"})
        chosen_candidate = refiners[0]
        arbiter_meta = {
            "winner": chosen_candidate.get("name", "A1"),
            "reason": "arbiter_disabled_or_single_candidate",
            "considered": [r.get("name") for r in refiners],
        }
        _dbg("ARBITER_DECISION", arbiter_meta)

    q_final = str(chosen_candidate.get("query") or q_curr)

    category_infos = []
    for cname in selected_categories:
        cat = manifest.categories.get(cname)
        category_infos.append({"category": cname, "vector_store_id": str(getattr(cat, "vector_store_id", "") or "")})

    hits: List[Dict[str, Any]] = []
    if opts.use_retrieve:
        hits = _stage(
            "retrieve",
            lambda: _retrieve_by_category(
                q_final=q_final,
                refiners=[chosen_candidate],
                category_infos=category_infos,
                max_workers=max_workers,
                retrieve_timeout_s=_env_float("WEBAPP_RETRIEVE_TIMEOUT_S", 30),
                top_k=opts.top_k,
                debug=debug,
                unbounded=True,
            ),
            meta={"query_used": q_final},
        )
        hits = attach_local_paths(manifest, merge_and_dedupe_evidence(hits or []), library_root=Path(mp).parent)
        hits = _shrink_evidence_payload((hits or []), max_hits=5, max_per_hit_chars=1200)
        hits = _cap_evidence_chars(hits, opts.max_context_chars)
    else:
        _dbg("STAGE_BEGIN", {"stage_name": "retrieve"})
        _dbg("STAGE_END", {"stage_name": "retrieve", "elapsed_ms": 0.0, "next_stage": "write", "reason": "disabled"})

    refs = _collect_references(manifest, hits)
    if opts.use_write:
        answer = _stage("write", lambda: write_stage.run(q_final, hits, debug=debug))
        status = "ok"
    else:
        _dbg("STAGE_BEGIN", {"stage_name": "write"})
        _dbg("STAGE_END", {"stage_name": "write", "elapsed_ms": 0.0, "next_stage": "done", "reason": "disabled"})
        status = "ok"
        answer = "Etapa write deshabilitada por configuración de pipeline."

    return {
        "status": status,
        "trace_id": trace_id,
        "answer": answer,
        "references": refs,
        "selected_categories": selected_categories,
        "confirm": confirm_out,
        "chosen_candidate": chosen_candidate,
        "arbiter": arbiter_meta,
        "warnings": [] if opts.use_retrieve else ["retrieve_disabled"],
        "degrade_steps": [] if opts.use_write else ["write_disabled"],
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
    _ = (confirm_rounds, confirm_glimpse)
    return run_query(
        question,
        manifest_path=manifest_path,
        max_workers=max_workers,
        debug=debug,
        pipeline={"use_confirmer": bool(confirm)},
    )


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
