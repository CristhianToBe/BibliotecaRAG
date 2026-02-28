from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from worklib.prompt_loader import load_prompt
from .llm import call_text, eprint, MODEL_ARBITRATE

SYSTEM_ARBITER = load_prompt("query_arbiter_system")


def _term_coverage(query: str, must_terms: List[str], texts: List[str]) -> float:
    terms = [t.strip().lower() for t in (must_terms or []) if str(t).strip()]
    if not terms:
        return 1.0
    corpus = "\n".join(texts).lower()
    matched = sum(1 for t in terms if t in corpus or t in (query or "").lower())
    return round(matched / max(1, len(terms)), 3)


def _build_signal_summary(candidate_packages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for item in candidate_packages:
        name = str(item.get("candidate") or item.get("name") or "A1")
        query = str(item.get("query_used") or item.get("query") or "")
        hits = list(item.get("hits") or [])
        constraints = dict(item.get("constraints") or {})
        must_terms = list(constraints.get("must_include_terms") or [])
        texts = [str(h.get("text") or "")[:1200] for h in hits]
        unique_docs = len({str(h.get("doc_id") or h.get("file_id") or h.get("filename") or "") for h in hits if (h.get("doc_id") or h.get("file_id") or h.get("filename"))})
        summary[name] = {
            "hits_count": int(item.get("hits_count", len(hits))),
            "unique_docs": unique_docs,
            "must_terms_coverage": _term_coverage(query, must_terms, texts),
        }
    return summary


def _deterministic_pick(signal_summary: Dict[str, Dict[str, Any]], considered: List[str]) -> str:
    ranked = sorted(
        considered,
        key=lambda name: (
            float(signal_summary.get(name, {}).get("must_terms_coverage", 0.0)),
            int(signal_summary.get(name, {}).get("hits_count", 0)),
            int(signal_summary.get(name, {}).get("unique_docs", 0)),
            -int(re.sub(r"\D", "", name or "0") or 0),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else "A1"


def arbitrate(
    question: str,
    refiners: List[Dict[str, Any]],
    hits: List[Dict[str, Any]],
    *,
    categories: List[str] | None = None,
    selector_instruction: str = "",
    debug: bool = False,
) -> Dict[str, Any]:
    _ = hits  # backward-compatible argument; evidence now comes via refiners packages.

    candidate_packages = [
        {
            "candidate": str(r.get("candidate") or r.get("name") or "A1"),
            "query_used": str(r.get("query_used") or r.get("query") or question),
            "constraints": dict(r.get("constraints") or {}),
            "hits": list(r.get("hits") or []),
            "hits_count": int(r.get("hits_count", len(list(r.get("hits") or [])))),
        }
        for r in (refiners or [])
    ]
    considered = [str(x.get("candidate") or "A1") for x in candidate_packages]
    signal_summary = _build_signal_summary(candidate_packages)

    if debug:
        eprint("[DEBUG] ARBITER_INPUT", {
            "question": question,
            "categories": categories or [],
            "considered": considered,
            "signal_summary": signal_summary,
        })

    if not candidate_packages:
        return {
            "winner": "A1",
            "considered": [],
            "why": "no_candidates_fallback",
            "signal_summary": {},
            "selected_indexes": [],
            "also_indexes": [],
            "chosen_variant_name": "A1",
            "chosen_query": question,
            "chosen_constraints": {},
            "rationale": "no_candidates_fallback",
        }

    all_zero_hits = all(int(signal_summary.get(name, {}).get("hits_count", 0)) <= 0 for name in considered)
    deterministic_winner = _deterministic_pick(signal_summary, considered)

    payload = {
        "question": question,
        "categories": categories or [],
        "selector_instruction": selector_instruction,
        "candidates": candidate_packages,
        "signal_summary": signal_summary,
    }

    parsed: Dict[str, Any] = {}
    if not all_zero_hits:
        try:
            resp = call_text(MODEL_ARBITRATE, SYSTEM_ARBITER, json.dumps(payload, ensure_ascii=False), debug=debug)
            txt = resp.output_text or ""
            if debug:
                eprint("\n[DEBUG] arbiter raw output_text:")
                eprint(txt)
            parsed = json.loads(txt)
        except Exception:
            parsed = {}

    winner = str(parsed.get("winner") or parsed.get("chosen_variant_name") or deterministic_winner).strip().upper()
    if winner not in considered:
        winner = deterministic_winner

    winner_pkg = next((x for x in candidate_packages if str(x.get("candidate") or "").upper() == winner), candidate_packages[0])
    winner_hits = list(winner_pkg.get("hits") or [])
    selected_indexes = list(parsed.get("selected_indexes") or list(range(min(3, len(winner_hits)))))
    also_indexes = list(parsed.get("also_indexes") or [])

    if all_zero_hits:
        why = "all_zero_hits_fallback"
    else:
        why = str(parsed.get("why") or parsed.get("rationale") or "selected_by_evidence_signals")

    decision = {
        "winner": winner,
        "considered": considered,
        "why": why,
        "signal_summary": signal_summary,
        "selected_indexes": selected_indexes,
        "also_indexes": also_indexes,
        "chosen_variant_name": winner,
        "chosen_query": str(winner_pkg.get("query_used") or question),
        "chosen_constraints": dict(winner_pkg.get("constraints") or {}),
        "rationale": why,
    }
    if debug:
        eprint("[DEBUG] ARBITER_DECISION", decision)
    return decision
