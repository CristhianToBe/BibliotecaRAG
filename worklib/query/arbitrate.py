from __future__ import annotations

import json
from typing import Any, Dict, List

from worklib.prompt_loader import load_prompt

from .arbiter_utils import normalize_indexes, validate_evidence_indexes
from .llm import MODEL_ARBITRATE, call_text, eprint

SYSTEM_ARBITER = load_prompt("query_arbiter_system")


def _score_fallback_order(evidences: List[Dict[str, Any]]) -> List[int]:
    return sorted(
        range(len(evidences)),
        key=lambda idx: float((evidences[idx] or {}).get("score") or 0.0),
        reverse=True,
    )


def arbitrate(
    question: str,
    refiners: List[Dict[str, Any]],
    hits: List[Dict[str, Any]],
    *,
    categories: List[str] | None = None,
    selector_instruction: str = "",
    debug: bool = False,
) -> Dict[str, Any]:
    _ = refiners
    evidences = list(hits or [])

    payload = {
        "question": question,
        "categories": categories or [],
        "selector_instruction": selector_instruction,
        "evidences": [
            {
                "file_id": h.get("file_id"),
                "filename": h.get("filename"),
                "local_path": h.get("local_path"),
                "score": h.get("score"),
                "text": h.get("text"),
            }
            for h in evidences
        ],
    }

    if debug:
        eprint("[DEBUG] ARBITER_INPUT", {
            "question": question,
            "categories": categories or [],
            "evidence_count": len(evidences),
        })

    parsed: Dict[str, Any] = {}
    if evidences:
        try:
            resp = call_text(MODEL_ARBITRATE, SYSTEM_ARBITER, json.dumps(payload, ensure_ascii=False), debug=debug)
            txt = resp.output_text or ""
            if debug:
                eprint("\n[DEBUG] arbiter raw output_text:")
                eprint(txt)
            parsed = json.loads(txt)
        except Exception:
            parsed = {}

    selected_indexes_raw = normalize_indexes(parsed.get("selected_indexes"))
    also_indexes_raw = normalize_indexes(parsed.get("also_indexes"))
    fallback_order = _score_fallback_order(evidences)

    selected_indexes, also_indexes, used_fallback = validate_evidence_indexes(
        selected_indexes=selected_indexes_raw,
        also_indexes=also_indexes_raw,
        evidence_len=len(evidences),
        fallback_order=fallback_order,
    )

    if used_fallback:
        eprint("[WARN] ARBITER_EVIDENCE_INVALID_INDEXES", {
            "selected_indexes_raw": selected_indexes_raw,
            "also_indexes_raw": also_indexes_raw,
            "evidence_len": len(evidences),
            "fallback_order": fallback_order,
            "selected_indexes": selected_indexes,
            "also_indexes": also_indexes,
        })

    why = str(parsed.get("why") or parsed.get("rationale") or ("evidence_index_fallback" if used_fallback else "selected_by_arbiter"))
    decision = {
        "selected_indexes": selected_indexes,
        "also_indexes": also_indexes,
        "why": why,
    }
    if debug:
        eprint("[DEBUG] ARBITER_DECISION", decision)
    return decision
