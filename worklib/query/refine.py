from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from worklib.prompt_loader import load_prompt
from .llm import call_text, eprint, MODEL_FAST

SYSTEM_REFINER_A1 = load_prompt("query_refiner_a1_system")
SYSTEM_REFINER_A2 = load_prompt("query_refiner_a2_system")
SYSTEM_REFINER_A3 = load_prompt("query_refiner_a3_system")


def refine_one(system_prompt: str, base_context: Dict[str, Any], *, debug: bool = False) -> Dict[str, Any]:
    resp = call_text(MODEL_FAST, system_prompt, json.dumps(base_context, ensure_ascii=False), debug=debug)
    txt = resp.output_text or ""
    if debug:
        eprint("\n[DEBUG] refiner raw output_text:")
        eprint(txt)

    try:
        obj = json.loads(txt)
    except Exception:
        obj = {
            "name": "A?",
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
    obj.setdefault("name", "A?")
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
    base_context = {"question": question, "must_include_terms": must_include_terms, "avoid_terms": avoid_terms}
    systems = [SYSTEM_REFINER_A1, SYSTEM_REFINER_A2, SYSTEM_REFINER_A3]
    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(systems))) as ex:
        futs = [ex.submit(refine_one, sys_p, base_context, debug=debug) for sys_p in systems]
        for fut in as_completed(futs):
            out.append(fut.result())
    order = {"A1": 0, "A2": 1, "A3": 2}
    out.sort(key=lambda x: order.get(x.get("name", ""), 99))
    return out