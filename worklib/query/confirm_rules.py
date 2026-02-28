from __future__ import annotations

import re
from typing import Dict, List, Tuple


_AFFIRMATIVE_SHORT = {
    "si", "sí", "ok", "dale", "listo", "de acuerdo", "correcto", "va", "yes"
}


def classify_user_reply(user_reply: str) -> str:
    text = re.sub(r"\s+", " ", (user_reply or "").strip().lower())
    if not text:
        return "first_turn"

    radical_markers = ["cambia el tema", "tema distinto", "otra pregunta", "mejor pregunta", "reformula", "cambiar a", "cámbialo a"]
    if any(m in text for m in radical_markers):
        return "rephrase"

    has_refine_marker = any(
        m in text
        for m in ["pero", "enfócate", "enfocate", "incluye", "excluye", "solo", "ajusta", "prioriza", "sin", "con"]
    )

    if text in _AFFIRMATIVE_SHORT:
        return "pass"
    if text.startswith("sí") or text.startswith("si"):
        return "refine" if has_refine_marker else "pass"
    if text.startswith("no"):
        return "refine"
    if has_refine_marker:
        return "refine"
    return "refine"


def enforce_confirm_schema(raw: Dict[str, object], *, user_reply: str) -> Dict[str, object]:
    base = {
        "message_to_user": str(raw.get("message_to_user") or "").strip(),
        "action": str(raw.get("action") or "REFINE").strip().upper(),
        "selector_instruction": str(raw.get("selector_instruction") or "").strip(),
        "rephrased_question": str(raw.get("rephrased_question") or "").strip(),
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
    }

    reply_kind = classify_user_reply(user_reply)
    if reply_kind == "first_turn":
        base["action"] = "REFINE"
    elif reply_kind == "pass":
        base["action"] = "PASS"
    elif reply_kind == "rephrase":
        base["action"] = "REPHRASE"
    else:
        base["action"] = "REFINE"

    if base["action"] == "REPHRASE" and not base["rephrased_question"]:
        base["rephrased_question"] = str(user_reply or "").strip() or "¿Puedes reformular tu pregunta en una sola oración?"
    if base["action"] == "REFINE" and not base["selector_instruction"]:
        if reply_kind == "first_turn":
            base["selector_instruction"] = "Confirma o ajusta el foco de búsqueda con una instrucción breve."
        else:
            base["selector_instruction"] = str(user_reply or "").strip() or "Ajusta el enfoque de la búsqueda con mayor precisión."

    if not base["message_to_user"]:
        if base["action"] == "PASS":
            base["message_to_user"] = "Perfecto, continúo con la consulta."
        elif base["action"] == "REPHRASE":
            base["message_to_user"] = "Entendido, reformulo la pregunta para volver a ejecutar la búsqueda."
        else:
            base["message_to_user"] = "Confirma o ajusta el enfoque que debo usar antes de continuar."

    base["confidence"] = max(0.0, min(1.0, float(base["confidence"])))
    return base


def strip_category_mentions(message: str, categories: List[str]) -> str:
    out = str(message or "")
    for cat in categories:
        c = str(cat or "").strip()
        if not c:
            continue
        out = re.sub(re.escape(c), "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,;:-")
    return out or "Confirma o ajusta el enfoque antes de continuar."


def fallback_clarification() -> Dict[str, object]:
    return {
        "message_to_user": "No pude interpretar la confirmación. Indícame si continúo, ajusto enfoque o reformulo la pregunta.",
        "action": "REFINE",
        "selector_instruction": "Pide al usuario una instrucción breve de enfoque.",
        "rephrased_question": "",
        "confidence": 0.0,
    }


def route_confirm_action(*, action: str, user_reply: str) -> str:
    action_u = str(action or "REFINE").strip().upper()
    has_reply = bool((user_reply or "").strip())
    if action_u == "PASS":
        return "CONFIRMED"
    if action_u == "REPHRASE":
        return "PARTIAL"
    if has_reply:
        return "REPICK"
    return "AWAITING_USER_REPLY"


def self_check_cases() -> List[Dict[str, object]]:
    cases = [
        "",
        "sí",
        "no, enfócate en tributario",
        "cambia el tema a retención en la fuente",
    ]
    out: List[Dict[str, object]] = []
    for reply in cases:
        normalized = enforce_confirm_schema({}, user_reply=reply)
        out.append({"user_reply": reply, "result": normalized})
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_check_cases(), ensure_ascii=False, indent=2))
