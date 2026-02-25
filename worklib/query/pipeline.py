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
from .confirm import confirm_once_non_interactive
from .llm import eprint
from .paths import resolve_local_path
from .pick import pick_categories
from .refine import refine_all
from .retrieve import retrieve_via_tool
from .telemetry import RequestTelemetry, get_telemetry, reset_current_stage, reset_debug_enabled, reset_telemetry, set_current_stage, set_debug_enabled, set_telemetry
from .write import write_answer


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
    use_confirmer: bool = False
    use_refiner: bool = False
    max_categories: int = 2
    top_k: int = 8
    max_context_chars: int = 12000
    timeouts: PipelineTimeouts = field(default_factory=PipelineTimeouts)


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
        if elapsed > retrieve_timeout_s:
            status = "timeout"
            if debug:
                print("[DEBUG] RETRIEVAL_CAT_END", {"trace_id": trace_id, "category": cname, "elapsed_ms": round(elapsed * 1000, 2), "hits_count": len(out), "reason": "timeout"})
            return []
        if debug:
            print("[DEBUG] RETRIEVAL_CAT_END", {"trace_id": trace_id, "category": cname, "elapsed_ms": round(elapsed * 1000, 2), "hits_count": len(out), "reason": status})
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
        return opts
    t_raw = raw.get("timeouts") if isinstance(raw, dict) else None
    if isinstance(t_raw, dict):
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
    opts = _coerce_pipeline_options(pipeline)
    warnings: List[str] = []
    degrade_steps: List[str] = []
    manual_categories = [str(c).strip() for c in (manual_categories or []) if str(c).strip()]

    mp = manifest_path or _default_manifest_path()
    manifest = load_manifest(Path(mp))
    valid_set = set(manifest.categories.keys())

    telemetry = get_telemetry()
    trace_id = getattr(telemetry, "trace_id", "no-trace")
    debug_pipeline: Dict[str, Any] = {"trace_id": trace_id, "stages": {}, "query_evolution": {}}

    def _debug_log(label: str, payload: Dict[str, Any]) -> None:
        if not debug:
            return
        base = {"trace_id": trace_id}
        base.update(payload)
        print(f"[DEBUG] {label}", base)

    def _stage_begin(stage_name: str, *, skipped: bool = False, reason: str = "") -> float:
        t0 = time.perf_counter()
        _debug_log("STAGE_BEGIN", {
            "stage_name": stage_name,
            "start_ts": t0,
            "skipped": skipped,
            "reason": reason,
        })
        return t0

    def _stage_end(stage_name: str, *, started_at: float, skipped: bool = False, degraded: bool = False, reason: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
        t1 = time.perf_counter()
        payload: Dict[str, Any] = {
            "stage_name": stage_name,
            "start_ts": started_at,
            "end_ts": t1,
            "elapsed_ms": round((t1 - started_at) * 1000, 2),
            "skipped": skipped,
            "degraded": degraded,
            "reason": reason,
        }
        if extra:
            payload.update(extra)
        debug_pipeline["stages"][stage_name] = payload
        _debug_log("STAGE_END", payload)

    def _run_stage(stage_name: str, fn: Callable[[], Any], fallback: Any) -> Tuple[Any, str]:
        started_at = _stage_begin(stage_name)
        calls_before = telemetry.model_calls if telemetry else 0
        task_started = time.perf_counter()
        result_used = "stage_output"
        try:
            tele_token = set_telemetry(telemetry)
            stage_token = set_current_stage(stage_name)
            debug_token = set_debug_enabled(debug)
            try:
                value = fn()
            finally:
                reset_debug_enabled(debug_token)
                reset_current_stage(stage_token)
                reset_telemetry(tele_token)
        except Exception as exc:
            value = fallback
            result_used = "fallback"
            warnings.append(f"Fallo en etapa {stage_name}: {type(exc).__name__}")
            _debug_log("STAGE_ERROR", {"stage_name": stage_name, "error": str(exc)})
        task_elapsed_s = max(0.0, time.perf_counter() - task_started)
        if telemetry:
            telemetry.mark_stage(stage_name, task_elapsed_s)
        calls_after = telemetry.model_calls if telemetry else calls_before
        _stage_end(
            stage_name,
            started_at=started_at,
            reason="ok" if result_used == "stage_output" else "fallback_exception",
            extra={
                "llm_calls_made": max(0, calls_after - calls_before),
                "stage_result_used": result_used,
                "task_elapsed_ms": round(task_elapsed_s * 1000, 2),
            },
        )
        return value, result_used

    if not opts.use_picker and not manual_categories:
        out = {"error": "missing_minimum", "details": "Debes activar Picker o indicar categorías manuales."}
        if debug:
            out["debug_picker"] = {"valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [], "picked_curr": {}, "selected_categories": []}
            out["debug_pipeline"] = debug_pipeline
        return out

    selected_categories: List[str] = []
    picked_curr: Dict[str, Any] = {}
    pick_ttl_s = _env_float("WEBAPP_PICK_CACHE_TTL_S", 600)

    if opts.use_picker:
        _debug_log("PICK_INPUT_SUMMARY", {
            "user_len": len(question or ""),
            "model": os.getenv("MODEL_PICK", "gpt-5-nano"),
            "valid_set_len": len(valid_set),
        })
        pick_result, _ = _run_stage("pick", lambda: _pick_with_cache(question, manifest.categories, pick_ttl_s, False), ({}, False))
        picked_curr = pick_result[0] if isinstance(pick_result, tuple) else {}
        raw_selected = picked_curr.get("selected") or []
        if debug:
            _debug_log("PICKER_MATCH", {
                "valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [],
                "picked_curr": picked_curr,
                "membership_map": {c: (c in valid_set) for c in raw_selected},
            })
        selected_categories = [c for c in raw_selected if c in valid_set][: opts.max_categories]
        if not selected_categories and raw_selected:
            valid_keys = list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else []
            expanded: List[str] = []
            for token in [_top_category_token(c) for c in raw_selected]:
                expanded.extend([k for k in valid_keys if _top_category_token(k) == token])
            selected_categories = _dedupe_preserve_order(expanded)[: opts.max_categories]
        if not selected_categories:
            valid_keys = list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else []
            selected_categories = _fallback_pick_categories(question, valid_keys, opts.max_categories)
            if selected_categories:
                _debug_log("PICK_FALLBACK_USED", {"strategy": "heuristic_defaults", "selected_categories": selected_categories})
    else:
        s0 = _stage_begin("pick", skipped=True, reason="disabled")
        if telemetry:
            telemetry.mark_stage("pick", 0.0)
        _stage_end("pick", started_at=s0, skipped=True, reason="disabled")

    if not selected_categories and manual_categories:
        selected_categories = [c for c in manual_categories if c in valid_set][: opts.max_categories]

    if not selected_categories:
        out = {"error": "missing_minimum", "details": "Debes activar Picker o indicar categorías manuales."}
        if debug:
            out["debug_picker"] = {"valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [], "picked_curr": picked_curr, "selected_categories": selected_categories}
            out["debug_pipeline"] = debug_pipeline
        return out

    confirm_data: Dict[str, Any] = {}
    if opts.use_confirmer and selected_categories:
        confirm_data, _ = _run_stage(
            "confirm",
            lambda: confirm_once_non_interactive(question, picked=picked_curr, manifest=manifest, use_glimpse=False, debug=False),
            {"categories_final": selected_categories, "suggested_categories": selected_categories},
        )
        selected_categories = [c for c in (confirm_data.get("categories_final") or confirm_data.get("suggested_categories") or selected_categories) if c in valid_set][: opts.max_categories]
    else:
        s0 = _stage_begin("confirm", skipped=True, reason="disabled_or_no_categories")
        if telemetry:
            telemetry.mark_stage("confirm", 0.0)
        _stage_end("confirm", started_at=s0, skipped=True, reason="disabled_or_no_categories")

    must_terms = list(picked_curr.get("must_include_terms", []) or [])
    avoid_terms = list(picked_curr.get("avoid_terms", []) or [])
    q_final = question

    refiners: List[Dict[str, Any]] = []
    if opts.use_refiner:
        refiners, _ = _run_stage("refine", lambda: refine_all(q_final, must_terms, avoid_terms, max_workers=max_workers, debug=False), [])
    else:
        s0 = _stage_begin("refine", skipped=True, reason="disabled")
        if telemetry:
            telemetry.mark_stage("refine", 0.0)
        _stage_end("refine", started_at=s0, skipped=True, reason="disabled")

    refined_queries = [str(r.get("query") or "").strip() for r in refiners if str(r.get("query") or "").strip()]
    if refined_queries:
        q_final = refined_queries[0]

    debug_pipeline["query_evolution"] = {
        "original_question": question,
        "refined_queries": refined_queries,
        "final_query_for_retrieval": q_final,
        "selector_instruction": str(confirm_data.get("selector_instruction") or ""),
        "must_include_terms": must_terms,
        "avoid_terms": avoid_terms,
    }
    _debug_log("QUERY_EVOLUTION", debug_pipeline["query_evolution"])

    doc_counts = _count_docs_by_category(manifest)
    category_infos: List[Dict[str, Any]] = []
    for cname in selected_categories[: opts.max_categories]:
        cat = manifest.categories.get(cname)
        vs_id = str(getattr(cat, "vector_store_id", "") or "").strip() if cat else ""
        category_infos.append({"category": cname, "vector_store_id": vs_id, "docs": int(doc_counts.get(cname, 0)), "vs_exists": bool(vs_id)})

    retrieval_key = hashlib.sha1(
        json.dumps({"q": _normalize_question(q_final), "cats": [x["category"] for x in category_infos], "manifest": _manifest_version(mp)}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    all_hits = _cache_get(_RETRIEVAL_CACHE, retrieval_key)
    retrieval_cache_hit = all_hits is not None
    cached_hits_count = len(all_hits) if isinstance(all_hits, list) else 0
    _debug_log("CACHE_LOOKUP", {
        "cache": "retrieval",
        "cache_key_sha1": retrieval_key,
        "cache_hit": retrieval_cache_hit,
        "cached_object_type": type(all_hits).__name__ if all_hits is not None else "NoneType",
        "cached_hits_count": cached_hits_count,
        "cached_payload_size": len(json.dumps(all_hits, ensure_ascii=False)) if all_hits is not None else 0,
    })

    if retrieval_cache_hit and cached_hits_count == 0:
        _debug_log("CACHE_HIT_EMPTY", {"cache_key_sha1": retrieval_key, "action": "treat_as_miss_and_execute_retrieval"})
        retrieval_cache_hit = False
        all_hits = None

    if all_hits is None:
        _debug_log("RETRIEVAL_EXECUTED", {
            "categories": [x.get("category") for x in category_infos],
            "category_infos": category_infos,
            "queries": [q_final] + [str(r.get("query") or "").strip() for r in refiners if str(r.get("query") or "").strip()],
            "top_k": opts.top_k,
        })
        all_hits, retrieval_result_used = _run_stage(
            "retrieval",
            lambda: _retrieve_by_category(
                q_final=q_final,
                refiners=refiners,
                category_infos=category_infos,
                max_workers=max_workers,
                retrieve_timeout_s=opts.timeouts.retrieve_per_cat_s,
                top_k=opts.top_k,
                debug=debug,
            ),
            [],
        )
        if retrieval_result_used != "stage_output" and not all_hits:
            all_hits = _retrieve_quick_fallback(q_final=q_final, category_infos=category_infos, top_k=opts.top_k, max_workers=max_workers, debug=debug)
            _debug_log("RETRIEVAL_FALLBACK_USED", {"reason": "stage_exception", "returned_candidates": len(all_hits or [])})
        _cache_set(_RETRIEVAL_CACHE, retrieval_key, list(all_hits or []), _env_float("WEBAPP_RETRIEVAL_CACHE_TTL_S", 1800))
    else:
        s0 = _stage_begin("retrieval", skipped=True, reason="cache_hit")
        if telemetry:
            telemetry.mark_stage("retrieval", 0.0)
        _debug_log("RETRIEVAL_SKIPPED", {"reason": "cache_hit", "cached_hits_count": len(all_hits or [])})
        _stage_end("retrieval", started_at=s0, skipped=True, reason="cache_hit", extra={"hits_len": len(all_hits or [])})

    all_hits = attach_local_paths(manifest, merge_and_dedupe_evidence(all_hits or []), library_root=Path(mp).parent)
    all_hits = _shrink_evidence_payload((all_hits or []), max_hits=5, max_per_hit_chars=1200)
    all_hits = _cap_evidence_chars(all_hits, opts.max_context_chars)
    references = _collect_references(manifest, all_hits)

    retrieval_degraded = False
    if telemetry:
        status_map = telemetry.summary().get("retrieval_status_by_category", {})
        retrieval_degraded = any(v in ("timeout", "failed") for v in status_map.values())
        if retrieval_degraded and all_hits:
            warnings.append("Retrieval parcial: al menos una categoría agotó tiempo o falló.")

    if debug:
        timings = telemetry.summary().get("timings_ms", {}) if telemetry else {}
        print("[DEBUG] run_pipeline_resilient snapshot", {
            "trace_id": trace_id,
            "selected_categories": selected_categories,
            "category_infos": [{"category": x.get("category"), "docs": x.get("docs"), "vs_exists": x.get("vs_exists")} for x in category_infos],
            "retrieval_cache_hit": retrieval_cache_hit,
            "all_hits_len": len(all_hits or []),
            "timings_ms": {
                "pick": timings.get("pick"),
                "confirm": timings.get("confirm"),
                "refine": timings.get("refine"),
                "retrieval": timings.get("retrieval"),
                "answer_generation": timings.get("answer_generation"),
            },
        })

    if not all_hits:
        warning = "No se recuperó evidencia del repositorio en esta ejecución (0 fragmentos)."
        warnings.append(warning)

    answer, writer_result_used = _run_stage("answer_generation", lambda: write_answer(q_final, all_hits, debug=debug), "")
    _debug_log("WRITER_STATUS", {"writer_result_used": writer_result_used})

    if debug:
        print("[DEBUG] run_pipeline_resilient answer", {
            "trace_id": trace_id,
            "answer_len": len(answer or ""),
            "answer_preview": str(answer or "")[:220],
            "warnings": warnings,
            "degrade_steps": degrade_steps,
        })

    if not answer:
        if all_hits:
            warning = f"No alcancé a redactar la respuesta completa dentro del tiempo, pero sí recuperé {len(all_hits)} fragmentos."
            warnings.append(warning)
            refs_top = references[:3]
            excerpts = [str(h.get("text") or "").strip()[:300] for h in (all_hits or [])[:3] if str(h.get("text") or "").strip()]
            refs_lines = [f"- {r.get('filename') or 'sin_nombre'} ({r.get('abs_path') or 'sin_ruta'})" for r in refs_top]
            ex_lines = [f"- {x}" for x in excerpts]
            answer = warning + "\n\nReferencias principales:\n" + ("\n".join(refs_lines) if refs_lines else "- (sin referencias)") + "\n\nExtractos:\n" + ("\n".join(ex_lines) if ex_lines else "- (sin extractos)")
        else:
            warning = "No se recuperó evidencia del repositorio en esta ejecución (0 fragmentos)."
            warnings.append(warning)
            labels = ", ".join(selected_categories) if selected_categories else "sin categorías"
            answer = f"{warning}\n\nResumen rápido: {question}\nCategorías consideradas: {labels}."

    if debug:
        timings = telemetry.summary().get("timings_ms", {}) if telemetry else {}
        print("[DEBUG] run_pipeline_resilient done", {
            "trace_id": trace_id,
            "timings_ms": {
                "pick": timings.get("pick"),
                "confirm": timings.get("confirm"),
                "refine": timings.get("refine"),
                "retrieval": timings.get("retrieval"),
                "answer_generation": timings.get("answer_generation"),
            },
            "selected_categories": selected_categories,
            "category_infos": [{"category": x.get("category"), "docs": x.get("docs"), "vs_exists": x.get("vs_exists")} for x in category_infos],
            "retrieval_cache_hit": retrieval_cache_hit,
            "all_hits_len": len(all_hits or []),
            "warnings": warnings,
            "degrade_steps": degrade_steps,
        })

    out = {
        "status": "ok",
        "answer": answer,
        "references": references,
        "selected_categories": selected_categories,
        "warnings": warnings,
        "degrade_steps": degrade_steps,
        "retrieval_cache_hit": retrieval_cache_hit,
    }
    if debug:
        out["debug_picker"] = {"valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [], "picked_curr": picked_curr, "selected_categories": selected_categories}
        out["debug_pipeline"] = debug_pipeline
        out["debug_pipeline"].update({
            "picked_curr": picked_curr,
            "selected_categories": selected_categories,
            "all_hits_len": len(all_hits or []),
            "retrieval_degraded": retrieval_degraded,
            "confirm_refine_executed": {
                "confirm": bool(debug_pipeline["stages"].get("confirm") and not debug_pipeline["stages"]["confirm"].get("skipped")),
                "refine": bool(debug_pipeline["stages"].get("refine") and not debug_pipeline["stages"]["refine"].get("skipped")),
            },
        })
    return out


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
            lambda: refine_all(q_final, must_terms, avoid_terms, max_workers=max_workers, debug=False),
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
