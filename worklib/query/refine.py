from __future__ import annotations

import json
from typing import Any, Dict, List

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
            "name": "A1",
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
    obj.setdefault("name", "A1")
    obj.setdefault("query", base_context["question"])
    return obj


def refine_all(
    question: str,
    must_include_terms: List[str],
    avoid_terms: List[str],
    *,
    max_workers: int = 3,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    # Debug stabilization: single-call refiner to avoid stage timeouts from multi-variant fan-out.
    base_context = {"question": question, "must_include_terms": must_include_terms, "avoid_terms": avoid_terms}
    telemetry = get_telemetry()
    stage_name = get_current_stage() or "refine"
    debug_ctx = bool(debug or is_debug_enabled())

    tele_token = set_telemetry(telemetry)
    stage_token = set_current_stage(stage_name)
    debug_token = set_debug_enabled(debug_ctx)
    try:
        out = refine_one(SYSTEM_REFINER_A1, base_context, debug=debug_ctx)
    finally:
        reset_debug_enabled(debug_token)
        reset_current_stage(stage_token)
        reset_telemetry(tele_token)

    return [out]


def run(question: str, must_include_terms: List[str], avoid_terms: List[str], *, max_workers: int = 3, debug: bool = False) -> List[Dict[str, Any]]:
    return refine_all(question, must_include_terms, avoid_terms, max_workers=max_workers, debug=debug)
