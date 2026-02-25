from __future__ import annotations

import json
from typing import Any, Dict

from worklib.store import Category
from worklib.prompt_loader import load_prompt

from .llm import call_text, eprint, MODEL_PICK

SYSTEM_PICK = load_prompt("query_pick_system")


def pick_categories(question: str, cats: Dict[str, Category], *, debug: bool = False) -> Dict[str, Any]:
    cat_view = {k: {"keywords": v.keywords[:25]} for k, v in cats.items()}
    payload = {"question": question, "categories": cat_view}
    resp = call_text(MODEL_PICK, SYSTEM_PICK, json.dumps(payload, ensure_ascii=False), debug=debug)
    txt = resp.output_text or ""
    if debug:
        eprint("\n[DEBUG] pick_categories raw output_text:")
        eprint(txt)

    try:
        obj = json.loads(txt)
    except Exception:
        obj = {"selected": list(cats.keys())[:2], "reason": "fallback", "must_include_terms": [], "avoid_terms": []}

    selected = [c for c in obj.get("selected", []) if c in cats][:2]
    if not selected:
        selected = list(cats.keys())[:2]

    return {
        "selected": selected,
        "reason": obj.get("reason", ""),
        "must_include_terms": (obj.get("must_include_terms", []) or [])[:10],
        "avoid_terms": (obj.get("avoid_terms", []) or [])[:10],
    }