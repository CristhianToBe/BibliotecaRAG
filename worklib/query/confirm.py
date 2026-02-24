from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from worklib.store import Manifest

from .llm import MODEL_CONFIRM, SYSTEM_CONFIRM, call_text, clip, eprint
from .pick import pick_categories
from .retrieve import retrieve_via_tool


def _safe_json_load(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            return json.loads(s[i : j + 1])
        raise


def confirm_decision(
    question: str,
    *,
    suggested: List[str],
    valid_categories: List[str],
    user_reply: str,
    glimpses: Optional[List[Dict[str, Any]]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    payload = {
        "question": question,
        "suggested_categories": suggested,
        "valid_categories": valid_categories,
        "glimpses": glimpses or [],
        "user_reply": user_reply,
    }
    resp = call_text(MODEL_CONFIRM, SYSTEM_CONFIRM, json.dumps(payload, ensure_ascii=False), debug=debug)
    txt = resp.output_text or ""
    if debug:
        eprint("\n[DEBUG] confirm_decision raw output_text:")
        eprint(txt)

    try:
        obj = _safe_json_load(txt)
    except Exception:
        obj = {}

    action = str(obj.get("action", "REFINE")).strip().upper()
    if action not in ("PASS", "REFINE", "REPHRASE"):
        action = "REFINE"

    valid_set = set(valid_categories)
    cats_final = [c for c in (obj.get("categories_final") or []) if c in valid_set]
    if action == "PASS" and not cats_final:
        cats_final = [c for c in suggested if c in valid_set]

    return {
        "message_to_user": str(obj.get("message_to_user", "") or "").strip(),
        "action": action,
        "categories_final": cats_final,
        "selector_instruction": str(obj.get("selector_instruction", "") or "").strip(),
        "rephrased_question": str(obj.get("rephrased_question", "") or "").strip(),
        "confidence": float(obj.get("confidence", 0.0) or 0.0),
    }


def _build_glimpses(
    manifest: Manifest,
    suggested: List[str],
    question: str,
) -> List[Dict[str, Any]]:
    glimpses: List[Dict[str, Any]] = []
    for cname in suggested[:2]:
        cat = manifest.categories.get(cname)
        vs_id = getattr(cat, "vector_store_id", "") if cat else ""
        if not vs_id:
            continue
        try:
            hits = retrieve_via_tool([vs_id], question, max_num_results=2, debug=False)
        except Exception:
            hits = []
        glimpses.append(
            {
                "category": cname,
                "hits": [
                    {
                        "title": (h.get("filename") or h.get("file_id") or ""),
                        "snippet": clip(h.get("text") or "", 280),
                        "score": h.get("score"),
                    }
                    for h in hits[:2]
                ],
            }
        )
    return glimpses


def confirm_loop(
    question: str,
    *,
    picked: Dict[str, Any],
    manifest: Manifest,
    max_rounds: int = 4,
    use_glimpse: bool = True,
    debug: bool = False,
    ask_user: Callable[[str], str] = input,
    emit: Callable[[str], None] = print,
) -> Tuple[str, List[str], Dict[str, Any]]:
    valid = list(manifest.categories.keys())
    suggested = list(picked.get("selected", []) or [])[:2]

    q_curr = question
    picked_curr = picked

    for _ in range(max_rounds):
        glimpses: List[Dict[str, Any]] = []
        if use_glimpse and suggested:
            glimpses = _build_glimpses(manifest, suggested, q_curr)

        prompt_dec = confirm_decision(
            q_curr,
            suggested=suggested,
            valid_categories=valid,
            glimpses=glimpses,
            user_reply="",
            debug=debug,
        )
        msg0 = prompt_dec.get("message_to_user")
        if msg0:
            emit(msg0)
        else:
            emit("Creo que tu consulta va por estas categorías: " + (", ".join(suggested) if suggested else "(ninguna)"))

        user_reply = ask_user("Confirma/ajusta (respuesta libre) > ").strip()

        decision = confirm_decision(
            q_curr,
            suggested=suggested,
            valid_categories=valid,
            glimpses=glimpses,
            user_reply=user_reply,
            debug=debug,
        )

        msg = decision.get("message_to_user")
        if msg:
            emit(msg)

        action = decision["action"]
        if action == "PASS":
            return (q_curr, decision["categories_final"], picked_curr)

        if action == "REFINE":
            instr = decision.get("selector_instruction", "")
            q_for_selector = q_curr if not instr else (q_curr + "\n\nInstrucción adicional para seleccionar categorías: " + instr)
            picked_curr = pick_categories(q_for_selector, manifest.categories, debug=debug)
            suggested = list(picked_curr.get("selected", []) or [])[:2]
            continue

        if action == "REPHRASE":
            q_new = decision.get("rephrased_question") or q_curr
            q_curr = q_new
            picked_curr = pick_categories(q_curr, manifest.categories, debug=debug)
            suggested = list(picked_curr.get("selected", []) or [])[:2]
            continue

    return (q_curr, suggested, picked_curr)


def confirm_loop_non_interactive(
    question: str,
    *,
    picked: Dict[str, Any],
    manifest: Manifest,
    use_glimpse: bool = True,
    debug: bool = False,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Non-interactive confirm flow for server/API usage.

    This never asks for console input. We automatically accept the model proposal
    from the first confirm step by default.
    """
    valid = list(manifest.categories.keys())
    suggested = list(picked.get("selected", []) or [])[:2]
    glimpses: List[Dict[str, Any]] = _build_glimpses(manifest, suggested, question) if (use_glimpse and suggested) else []

    decision = confirm_decision(
        question,
        suggested=suggested,
        valid_categories=valid,
        glimpses=glimpses,
        user_reply="",
        debug=debug,
    )

    cats_final = list(decision.get("categories_final") or [])
    if not cats_final:
        cats_final = [c for c in suggested if c in set(valid)]

    return (question, cats_final, picked)
