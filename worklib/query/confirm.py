from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from worklib.store import Manifest

from .llm import MODEL_CONFIRM, SYSTEM_CONFIRM, call_text, clip, eprint
from .pick import pick_categories
from .retrieve import retrieve_via_tool
from .confirm_rules import enforce_confirm_schema, fallback_clarification, strip_category_mentions


def run(
    question: str,
    *,
    picked: Dict[str, Any],
    manifest: Manifest,
    user_reply: str = "",
    suggested_categories: Optional[List[str]] = None,
    use_glimpse: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """Single confirmation decision used by pipeline orchestration.

    This is the only place that decides whether categories are confirmed or a
    repick loop must continue.
    """
    valid = list(manifest.categories.keys())
    valid_set = set(valid)
    suggested = [
        c for c in (suggested_categories or picked.get("selected") or []) if c in valid_set
    ][:2]
    glimpses: List[Dict[str, Any]] = _build_glimpses(manifest, suggested, question) if (use_glimpse and suggested) else []

    decision = confirm_decision(
        question,
        suggested=suggested,
        valid_categories=valid,
        glimpses=glimpses,
        user_reply=(user_reply or "").strip(),
        debug=debug,
    )
    action = str(decision.get("action") or "REFINE").strip().upper()

    pipeline_decision = "REPICK"
    rewritten_prompt = ""
    if action == "PASS":
        pipeline_decision = "CONFIRMED"
    elif action == "REPHRASE":
        pipeline_decision = "PARTIAL"
        rewritten_prompt = str(decision.get("rephrased_question") or "").strip()
    else:
        pipeline_decision = "REPICK"
        selector_instruction = str(decision.get("selector_instruction") or "").strip()
        if selector_instruction:
            rewritten_prompt = f"{question}\n\nInstrucción adicional para seleccionar categorías: {selector_instruction}".strip()

    return {
        "decision": pipeline_decision,
        "reason": str(decision.get("message_to_user") or decision.get("selector_instruction") or "").strip(),
        "rewritten_prompt": rewritten_prompt,
        "suggested_categories": suggested,
        "message_to_user": str(decision.get("message_to_user") or "").strip(),
        "raw": decision,
    }


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
    obj: Dict[str, Any] = {}
    try:
        resp = call_text(MODEL_CONFIRM, SYSTEM_CONFIRM, json.dumps(payload, ensure_ascii=False), debug=debug)
        txt = resp.output_text or ""
        if debug:
            eprint("\n[DEBUG] confirm_decision raw output_text:")
            eprint(txt)
        obj = _safe_json_load(txt)
    except Exception:
        obj = {}

    if not isinstance(obj, dict):
        obj = {}

    if not obj:
        normalized = fallback_clarification()
    else:
        normalized = enforce_confirm_schema(obj, user_reply=user_reply)

    normalized["message_to_user"] = strip_category_mentions(
        str(normalized.get("message_to_user") or ""),
        [*suggested, *valid_categories],
    )
    return normalized


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
            return (q_curr, suggested, picked_curr)

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



def confirm_once_non_interactive(
    question: str,
    *,
    picked: Dict[str, Any],
    manifest: Manifest,
    use_glimpse: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """Single non-interactive confirm pass that never asks for user input."""
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
    decision["suggested_categories"] = suggested
    return decision

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

    action = str(decision.get("action") or "REFINE").strip().upper()
    if debug:
        eprint(f"[DEBUG] non-interactive confirm action: {action}")

    cats_final = [c for c in suggested if c in set(valid)]
    if debug:
        eprint(f"[DEBUG] non-interactive chosen categories after fallback: {cats_final}")

    return (question, cats_final, picked)
