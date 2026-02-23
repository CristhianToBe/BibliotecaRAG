from __future__ import annotations

import json
from typing import Any, Dict, List

from worklib.prompt_loader import load_prompt
from .llm import call_text, eprint, MODEL_SMART

SYSTEM_ARBITER = load_prompt("query_arbiter_system")


def arbitrate(question: str, refiners: List[Dict[str, Any]], hits: List[Dict[str, Any]], *, debug: bool = False) -> Dict[str, Any]:
    payload = {
        "question": question,
        "refiners": refiners,
        "hits": [
            {
                "file_id": h.get("file_id"),
                "filename": h.get("filename"),
                "score": h.get("score"),
                "text": h.get("text"),
            }
            for h in hits
        ],
    }
    resp = call_text(MODEL_SMART, SYSTEM_ARBITER, json.dumps(payload, ensure_ascii=False), debug=debug)
    txt = resp.output_text or ""
    if debug:
        eprint("\n[DEBUG] arbiter raw output_text:")
        eprint(txt)
    try:
        return json.loads(txt)
    except Exception:
        return {"best_query": question, "reason": "fallback"}