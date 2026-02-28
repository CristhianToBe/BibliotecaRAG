from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from worklib.prompt_loader import load_prompt
from .llm import call_text, eprint, MODEL_REFINE
from .telemetry import (
    get_current_stage,
    get_telemetry,
    is_debug_enabled,
    reset_current_stage,
    reset_debug_enabled,
    reset_telemetry,
    set_current_stage,
    set_debug_enabled,
    set_telemetry,
)

SYSTEM_REFINER_A1 = load_prompt("query_refiner_a1_system")
SYSTEM_REFINER_A2 = load_prompt("query_refiner_a2_system")
SYSTEM_REFINER_A3 = load_prompt("query_refiner_a3_system")

_VARIANT_PROMPTS = {
    "A1": SYSTEM_REFINER_A1,
    "A2": SYSTEM_REFINER_A2,
    "A3": SYSTEM_REFINER_A3,
}


def refine_one(system_prompt: str, base_context: Dict[str, Any], *, debug: bool = False) -> Dict[str, Any]:
    resp = call_text(MODEL_REFINE, system_prompt, json.dumps(base_context, ensure_ascii=False), debug=debug)
    txt = resp.output_text or ""
    if debug:
        eprint("\n[DEBUG] refiner raw output_text:")
        eprint(txt)

    try:
        obj = json.loads(txt)
    except Exception:
        obj = {
            "query": base_context["question"],
            "constraints": {
                "prefer_norma_first": True,
                "header_phrases": [],
                "must_include_terms": [],
                "avoid_terms": [],
            },
        }

    obj.setdefault("constraints", {})
    obj["constraints"]["prefer_norma_first"] = True
    obj["constraints"].setdefault("header_phrases", [])
    obj["constraints"].setdefault("must_include_terms", base_context.get("must_include_terms", []))
    obj["constraints"].setdefault("avoid_terms", base_context.get("avoid_terms", []))
    obj.setdefault("query", base_context["question"])
    return obj


def refine_all(
    question: str,
    must_include_terms: List[str],
    avoid_terms: List[str],
    *,
    max_workers: int = 3,
    enabled_variants: Optional[List[str]] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    base_context = {"question": question, "must_include_terms": must_include_terms, "avoid_terms": avoid_terms}
    telemetry = get_telemetry()
    stage_name = get_current_stage() or "refine"
    debug_ctx = bool(debug or is_debug_enabled())

    variants = [v for v in (enabled_variants or ["A1", "A2", "A3"]) if v in _VARIANT_PROMPTS]
    if debug_ctx:
        eprint("[DEBUG] refine variants enabled:", variants)

    tele_token = set_telemetry(telemetry)
    stage_token = set_current_stage(stage_name)
    debug_token = set_debug_enabled(debug_ctx)
    try:
        if not variants:
            if debug_ctx:
                eprint("[DEBUG] refine variants enabled: [] (all disabled)")
            return []

        out: List[Dict[str, Any]] = []

        def _run_variant(variant_name: str) -> Dict[str, Any]:
            variant_stage_name = f"{stage_name}/{variant_name}"
            if debug_ctx:
                eprint("[DEBUG] REFINE_VARIANT_BEGIN", {
                    "trace_id": getattr(telemetry, "trace_id", "no-trace") if telemetry else "no-trace",
                    "variant": variant_name,
                    "stage_name": variant_stage_name,
                })
            variant_tele = set_telemetry(telemetry)
            variant_stage = set_current_stage(variant_stage_name)
            variant_dbg = set_debug_enabled(debug_ctx)
            try:
                result = refine_one(_VARIANT_PROMPTS[variant_name], base_context, debug=debug_ctx)
                result["name"] = variant_name
                return result
            finally:
                reset_debug_enabled(variant_dbg)
                reset_current_stage(variant_stage)
                reset_telemetry(variant_tele)

        workers = max(1, min(3, int(max_workers), len(variants)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_variant, name): name for name in variants}
            for fut in as_completed(futures):
                variant_name = futures[fut]
                try:
                    out.append(fut.result())
                except Exception as exc:
                    if debug_ctx:
                        eprint("[DEBUG] refine variant failed:", {"name": variant_name, "error": str(exc)})

                if debug_ctx:
                    eprint("[DEBUG] REFINE_VARIANT_END", {
                        "trace_id": getattr(telemetry, "trace_id", "no-trace") if telemetry else "no-trace",
                        "variant": variant_name,
                    })

        out.sort(key=lambda x: variants.index(str(x.get("name") or "A1")) if str(x.get("name") or "A1") in variants else 999)
        if debug_ctx:
            eprint("[DEBUG] refine results:", [
                {
                    "name": r.get("name"),
                    "query_len": len(str(r.get("query") or "")),
                    "must_include_terms": len((r.get("constraints") or {}).get("must_include_terms") or []),
                    "avoid_terms": len((r.get("constraints") or {}).get("avoid_terms") or []),
                }
                for r in out
            ])
        return out
    finally:
        reset_debug_enabled(debug_token)
        reset_current_stage(stage_token)
        reset_telemetry(tele_token)


def run(
    question: str,
    must_include_terms: List[str],
    avoid_terms: List[str],
    *,
    max_workers: int = 3,
    enabled_variants: Optional[List[str]] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    return refine_all(
        question,
        must_include_terms,
        avoid_terms,
        max_workers=max_workers,
        enabled_variants=enabled_variants,
        debug=debug,
    )
