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
from .telemetry import RequestTelemetry, get_telemetry
from .write import write_answer


_PICK_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_RETRIEVAL_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class PipelineTimeouts:
    total_s: float = 120.0
    pick_s: float = 10.0
    confirm_s: float = 10.0
    refine_s: float = 10.0
    retrieve_per_cat_s: float = 15.0
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
    started = time.perf_counter()
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

    def _stage_begin(stage_name: str, *, stage_budget_s: Optional[float], skipped: bool = False, reason: str = "") -> float:
        t0 = time.perf_counter()
        _debug_log("STAGE_BEGIN", {
            "stage_name": stage_name,
            "start_ts": t0,
            "stage_budget_s": stage_budget_s,
            "skipped": skipped,
            "reason": reason,
        })
        return t0

    def _stage_end(
        stage_name: str,
        *,
        started_at: float,
        stage_budget_s: Optional[float],
        skipped: bool = False,
        degraded: bool = False,
        timed_out: bool = False,
        reason: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        t1 = time.perf_counter()
        payload: Dict[str, Any] = {
            "stage_name": stage_name,
            "start_ts": started_at,
            "end_ts": t1,
            "elapsed_ms": round((t1 - started_at) * 1000, 2),
            "stage_budget_s": stage_budget_s,
            "skipped": skipped,
            "degraded": degraded,
            "timed_out": timed_out,
            "reason": reason,
        }
        if extra:
            payload.update(extra)
        debug_pipeline["stages"][stage_name] = payload
        _debug_log("STAGE_END", payload)

    def _run_stage_with_timeout(stage_name: str, budget_s: float, fn: Callable[[], Any], fallback: Any) -> Tuple[Any, bool, float, int]:
        s0 = _stage_begin(stage_name, stage_budget_s=budget_s)
        calls_before = telemetry.model_calls if telemetry else 0
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn)
        timeout_state = {"timed_out": False}

        if debug:
            def _on_done(_fut):
                if timeout_state["timed_out"]:
                    _debug_log("STAGE_LATE_RESULT", {
                        "stage_name": stage_name,
                        "note": "received output after stage timed out",
                        "output_disposition": "discarded_fallback_used",
                    })
            fut.add_done_callback(_on_done)

        timed_out = False
        try:
            value = fut.result(timeout=budget_s)
        except FuturesTimeoutError:
            timeout_state["timed_out"] = True
            timed_out = True
            value = fallback
        except Exception as exc:
            value = fallback
            _debug_log("STAGE_ERROR", {"stage_name": stage_name, "error": str(exc)})
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        elapsed_s = max(0.0, time.perf_counter() - s0)
        if telemetry:
            telemetry.mark_stage(stage_name, elapsed_s)
        calls_after = telemetry.model_calls if telemetry else calls_before
        _stage_end(
            stage_name,
            started_at=s0,
            stage_budget_s=budget_s,
            timed_out=timed_out,
            reason="timeout" if timed_out else "ok",
            extra={
                "timeout_policy": "future.result(timeout=budget_s)",
                "llm_calls_made": max(0, calls_after - calls_before),
            },
        )
        return value, timed_out, elapsed_s, max(0, calls_after - calls_before)

    if not opts.use_picker and not manual_categories:
        out = {
            "error": "missing_minimum",
            "details": "Debes activar Picker o indicar categorías manuales.",
        }
        if debug:
            out["debug_picker"] = {
                "valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [],
                "picked_curr": {},
                "selected_categories": [],
            }
            out["debug_pipeline"] = debug_pipeline
        return out

    def _elapsed_s() -> float:
        return max(0.0, time.perf_counter() - started)

    def _remaining_s() -> float:
        return max(0.0, opts.timeouts.total_s - _elapsed_s())

    def _degrade_once(reason: str) -> bool:
        if opts.use_confirmer:
            opts.use_confirmer = False
            msg = f"Se desactivó confirmer por tiempo ({reason})."
        elif opts.use_refiner:
            opts.use_refiner = False
            msg = f"Se desactivó refiner por tiempo ({reason})."
        elif opts.max_categories > 1:
            opts.max_categories = 1
            msg = f"Se redujo max_categories a 1 por tiempo ({reason})."
        elif opts.top_k > 4 or opts.max_context_chars > 6000:
            opts.top_k = min(opts.top_k, 4)
            opts.max_context_chars = min(opts.max_context_chars, 6000)
            msg = f"Se redujeron top_k/contexto por tiempo ({reason})."
        else:
            return False
        warnings.append(msg)
        degrade_steps.append(msg)
        return True

    if _remaining_s() < 5 and not _degrade_once("presupuesto crítico inicial"):
        warnings.append("Presupuesto de tiempo extremadamente bajo.")

    selected_categories: List[str] = []
    picked_curr: Dict[str, Any] = {}
    pick_ttl_s = _env_float("WEBAPP_PICK_CACHE_TTL_S", 600)

    if opts.use_picker:
        pick_timeout = min(opts.timeouts.pick_s, max(1.0, _remaining_s() - 2))
        pick_result, pick_timed_out, _, _ = _run_stage_with_timeout(
            "pick",
            pick_timeout,
            lambda: _pick_with_cache(question, manifest.categories, pick_ttl_s, debug),
            ({}, False),
        )
        picked_curr = pick_result[0] if isinstance(pick_result, tuple) else {}
        raw_selected = picked_curr.get("selected") or []

        if debug:
            membership_map = {c: (c in valid_set) for c in raw_selected}
            _debug_log("PICKER_MATCH", {
                "valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [],
                "picked_curr": picked_curr,
                "membership_map": membership_map,
            })

        selected_categories = [c for c in raw_selected if c in valid_set][: opts.max_categories]
        if not selected_categories and raw_selected:
            valid_keys = list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else []
            expanded: List[str] = []
            for token in [_top_category_token(c) for c in raw_selected]:
                expanded.extend([k for k in valid_keys if _top_category_token(k) == token])
            selected_categories = _dedupe_preserve_order(expanded)[: opts.max_categories]

        if pick_timed_out:
            warnings.append("Picker alcanzó timeout; se aplicó fallback de selección.")
        if not selected_categories:
            warnings.append("Picker no devolvió categorías válidas; se intentarán categorías manuales.")
    else:
        s = _stage_begin("pick", stage_budget_s=opts.timeouts.pick_s, skipped=True, reason="disabled")
        if telemetry:
            telemetry.mark_stage("pick", 0.0)
        _stage_end("pick", started_at=s, stage_budget_s=opts.timeouts.pick_s, skipped=True, reason="disabled")

    if not selected_categories and manual_categories:
        selected_categories = [c for c in manual_categories if c in valid_set][: opts.max_categories]

    if not selected_categories:
        out = {
            "error": "missing_minimum",
            "details": "Debes activar Picker o indicar categorías manuales.",
        }
        if debug:
            out["debug_picker"] = {
                "valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [],
                "picked_curr": picked_curr,
                "selected_categories": selected_categories,
            }
            out["debug_pipeline"] = debug_pipeline
        return out

    confirm_data: Dict[str, Any] = {}
    if opts.use_confirmer and selected_categories:
        confirm_timeout = min(opts.timeouts.confirm_s, max(1.0, _remaining_s() - 2))
        confirm_data, confirm_timed_out, confirm_elapsed_s, confirm_calls = _run_stage_with_timeout(
            "confirm",
            confirm_timeout,
            lambda: confirm_once_non_interactive(question, picked=picked_curr, manifest=manifest, use_glimpse=True, debug=debug),
            {"categories_final": selected_categories, "suggested_categories": selected_categories, "_timeout": True},
        )
        _debug_log("CONFIRM_TIMEOUT_DIAGNOSTICS", {
            "confirm_budget_s": confirm_timeout,
            "confirm_elapsed_ms": round(confirm_elapsed_s * 1000, 2),
            "confirm_llm_calls": confirm_calls,
            "confirm_timeout_flag": bool(confirm_timed_out or confirm_data.get("_timeout")),
            "confirm_used_output": "fallback" if (confirm_timed_out or confirm_data.get("_timeout")) else "stage_output",
        })
        if confirm_data.get("_timeout"):
            _degrade_once("timeout en confirmer")
        selected_categories = [c for c in (confirm_data.get("categories_final") or confirm_data.get("suggested_categories") or selected_categories) if c in valid_set][: opts.max_categories]
    else:
        s = _stage_begin("confirm", stage_budget_s=opts.timeouts.confirm_s, skipped=True, reason="disabled_or_no_categories")
        if telemetry:
            telemetry.mark_stage("confirm", 0.0)
        _stage_end("confirm", started_at=s, stage_budget_s=opts.timeouts.confirm_s, skipped=True, reason="disabled_or_no_categories")

    while _remaining_s() < 8 and _degrade_once("presupuesto crítico"):
        pass

    must_terms = list(picked_curr.get("must_include_terms", []) or [])
    avoid_terms = list(picked_curr.get("avoid_terms", []) or [])
    q_final = question

    refiners: List[Dict[str, Any]] = []
    if opts.use_refiner:
        refine_timeout = min(opts.timeouts.refine_s, max(1.0, _remaining_s() - 2))
        refiners, refine_timed_out, refine_elapsed_s, refine_calls = _run_stage_with_timeout(
            "refine",
            refine_timeout,
            lambda: refine_all(q_final, must_terms, avoid_terms, max_workers=max_workers, debug=debug),
            [],
        )
        _debug_log("REFINE_TIMEOUT_DIAGNOSTICS", {
            "refine_budget_s": refine_timeout,
            "refine_elapsed_ms": round(refine_elapsed_s * 1000, 2),
            "refine_llm_calls": refine_calls,
            "refiner_variants_returned": len(refiners or []),
            "refiner_variant_names": [str(r.get("name") or "") for r in (refiners or [])],
            "refine_timeout_flag": bool(refine_timed_out),
        })
        if not refiners:
            _degrade_once("timeout en refiner")
    else:
        s = _stage_begin("refine", stage_budget_s=opts.timeouts.refine_s, skipped=True, reason="disabled")
        if telemetry:
            telemetry.mark_stage("refine", 0.0)
        _stage_end("refine", started_at=s, stage_budget_s=opts.timeouts.refine_s, skipped=True, reason="disabled")

    debug_pipeline["query_evolution"] = {
        "original_question": question,
        "refined_queries": [str(r.get("query") or "") for r in refiners if str(r.get("query") or "").strip()],
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
        _debug_log("CACHE_HIT_EMPTY", {
            "cache_key_sha1": retrieval_key,
            "action": "treat_as_miss_and_execute_retrieval",
        })
        retrieval_cache_hit = False
        all_hits = None

    retrieval_stage_start = _stage_begin("retrieval", stage_budget_s=opts.timeouts.retrieve_per_cat_s)
    if all_hits is None and _remaining_s() > 6:
        retrieve_timeout = min(opts.timeouts.retrieve_per_cat_s, max(1.0, _remaining_s() / max(1, len(category_infos))))
        _debug_log("RETRIEVAL_EXECUTED", {
            "categories": [x.get("category") for x in category_infos],
            "category_infos": category_infos,
            "queries": [q_final] + [str(r.get("query") or "").strip() for r in refiners if str(r.get("query") or "").strip()],
            "top_k": opts.top_k,
            "retrieve_timeout_s": retrieve_timeout,
        })
        rt = time.perf_counter()
        all_hits = _retrieve_by_category(
            q_final=q_final,
            refiners=refiners,
            category_infos=category_infos,
            max_workers=max_workers,
            retrieve_timeout_s=retrieve_timeout,
            top_k=opts.top_k,
            debug=debug,
        )
        if telemetry:
            telemetry.mark_stage("retrieval", max(0.0, time.perf_counter() - rt))
        _cache_set(_RETRIEVAL_CACHE, retrieval_key, list(all_hits or []), _env_float("WEBAPP_RETRIEVAL_CACHE_TTL_S", 1800))
        _stage_end("retrieval", started_at=retrieval_stage_start, stage_budget_s=opts.timeouts.retrieve_per_cat_s, reason="executed", extra={"hits_len": len(all_hits or [])})
    elif all_hits is None:
        all_hits = []
        _degrade_once("sin tiempo para retrieval")
        if telemetry:
            telemetry.mark_stage("retrieval", 0.0)
        _debug_log("RETRIEVAL_SKIPPED", {"reason": "insufficient_remaining_budget", "remaining_s": _remaining_s()})
        _stage_end("retrieval", started_at=retrieval_stage_start, stage_budget_s=opts.timeouts.retrieve_per_cat_s, skipped=True, degraded=True, reason="insufficient_remaining_budget", extra={"hits_len": 0})
    else:
        if telemetry:
            telemetry.mark_stage("retrieval", 0.0)
        _debug_log("RETRIEVAL_SKIPPED", {"reason": "cache_hit", "cached_hits_count": len(all_hits or [])})
        _stage_end("retrieval", started_at=retrieval_stage_start, stage_budget_s=opts.timeouts.retrieve_per_cat_s, skipped=True, reason="cache_hit", extra={"hits_len": len(all_hits or [])})

    all_hits = attach_local_paths(manifest, merge_and_dedupe_evidence(all_hits or []), library_root=Path(mp).parent)
    all_hits = _cap_evidence_chars((all_hits or [])[:30], opts.max_context_chars)
    references = _collect_references(manifest, all_hits)

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

    if _remaining_s() < 5 and not all_hits:
        warning = "No alcancé a consultar documentos a tiempo"
        warnings.append(warning)
        labels = ", ".join(selected_categories) if selected_categories else "sin categorías"
        answer = f"{warning}.\n\nRespuesta de mejor esfuerzo para: '{question}'.\nCategorías consideradas: {labels}."
        if debug:
            print("[DEBUG] run_pipeline_resilient fallback", {
                "trace_id": trace_id,
                "reason": "sin tiempo o evidencia (mejor esfuerzo)",
                "answer_len": len(answer),
                "answer_preview": answer[:220],
                "warnings": warnings,
                "degrade_steps": degrade_steps,
            })
        out = {
            "status": "ok",
            "answer": answer,
            "references": [],
            "selected_categories": selected_categories,
            "warnings": warnings,
            "degrade_steps": degrade_steps,
            "retrieval_cache_hit": retrieval_cache_hit,
        }
        if debug:
            out["debug_picker"] = {
                "valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [],
                "picked_curr": picked_curr,
                "selected_categories": selected_categories,
            }
            out["debug_pipeline"] = debug_pipeline
            out["debug_pipeline"].update({
                "picked_curr": picked_curr,
                "selected_categories": selected_categories,
                "all_hits_len": len(all_hits or []),
                "confirm_refine_executed": {
                    "confirm": bool(debug_pipeline["stages"].get("confirm") and not debug_pipeline["stages"]["confirm"].get("skipped")),
                    "refine": bool(debug_pipeline["stages"].get("refine") and not debug_pipeline["stages"]["refine"].get("skipped")),
                },
            })
        return out

    write_timeout = min(opts.timeouts.write_s, max(1.0, _remaining_s() - 1))
    answer, writer_timed_out, _, _ = _run_stage_with_timeout(
        "answer_generation",
        write_timeout,
        lambda: write_answer(q_final, all_hits, debug=debug),
        "",
    )
    _debug_log("WRITER_STATUS", {"timed_out": writer_timed_out, "write_budget_s": write_timeout})

    if debug:
        print("[DEBUG] run_pipeline_resilient answer", {
            "trace_id": trace_id,
            "answer_len": len(answer or ""),
            "answer_preview": str(answer or "")[:220],
            "warnings": warnings,
            "degrade_steps": degrade_steps,
        })

    if not answer:
        warning = "No alcancé a consultar documentos a tiempo"
        warnings.append(warning)
        labels = ", ".join(selected_categories) if selected_categories else "sin categorías"
        answer = f"{warning}.\n\nResumen rápido: {question}\nCategorías consideradas: {labels}."
        if debug:
            print("[DEBUG] run_pipeline_resilient fallback", {
                "trace_id": trace_id,
                "reason": "answer vacía (mejor esfuerzo)",
                "answer_len": len(answer),
                "answer_preview": answer[:220],
                "warnings": warnings,
                "degrade_steps": degrade_steps,
            })

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
        out["debug_picker"] = {
            "valid_set": list(manifest.categories.keys()) if isinstance(manifest.categories, dict) else [],
            "picked_curr": picked_curr,
            "selected_categories": selected_categories,
        }
        out["debug_pipeline"] = debug_pipeline
        out["debug_pipeline"].update({
            "picked_curr": picked_curr,
            "selected_categories": selected_categories,
            "all_hits_len": len(all_hits or []),
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
