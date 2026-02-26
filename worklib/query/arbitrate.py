from __future__ import annotations

import json
from typing import Any, Dict, List

from worklib.prompt_loader import load_prompt
from .llm import call_text, eprint, MODEL_ARBITRATE

SYSTEM_ARBITER = load_prompt("query_arbiter_system")


def arbitrate(
    question: str,
    refiners: List[Dict[str, Any]],
    hits: List[Dict[str, Any]],
    *,
    categories: List[str] | None = None,
    selector_instruction: str = "",
    debug: bool = False,
) -> Dict[str, Any]:
    payload = {
        "question": question,
        "categories": categories or [],
        "selector_instruction": selector_instruction,
        "refiners": [
            {
                "name": r.get("name"),
                "query": r.get("query"),
                "constraints": r.get("constraints") or {},
            }
            for r in refiners
        ],
        "hits": [
            {
                "file_id": h.get("file_id"),
                "filename": h.get("filename"),
                "score": h.get("score"),
                "text": h.get("text"),
            }
            for h in (hits or [])[:5]
        ],
    }
    resp = call_text(MODEL_ARBITRATE, SYSTEM_ARBITER, json.dumps(payload, ensure_ascii=False), debug=debug)
    txt = resp.output_text or ""
    if debug:
        eprint("\n[DEBUG] arbiter raw output_text:")
        eprint(txt)
    try:
        parsed = json.loads(txt)
    except Exception:
        parsed = {}

    by_name = {str(r.get("name") or "").upper(): r for r in refiners}
    preferred = str(parsed.get("chosen_variant_name") or parsed.get("winner") or "").upper().strip()
    selected = by_name.get(preferred)
    if not selected:
        selected = refiners[0] if refiners else {"name": "A1", "query": question, "constraints": {}}

    return {
        "chosen_variant_name": str(selected.get("name") or "A1"),
        "chosen_query": str(selected.get("query") or question),
        "chosen_constraints": dict(selected.get("constraints") or {}),
        "rationale": str(parsed.get("rationale") or parsed.get("reason") or "fallback_first_variant").strip(),
        "considered": [str(r.get("name") or "") for r in refiners],
        "raw": parsed,
    }
