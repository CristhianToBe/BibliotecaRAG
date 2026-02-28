from __future__ import annotations

import re
from typing import Dict, List

_TRIBUTARIO_TRIGGERS = [
    "pasivos inexistentes",
    "activos omitidos",
    "omisión de activos",
    "omision de activos",
    "sanción por inexactitud",
    "sancion por inexactitud",
]


def detect_domain_hint(question: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    if any(t in q for t in _TRIBUTARIO_TRIGGERS):
        return "TRIBUTARIO"
    return ""


def apply_domain_hint_constraints(
    *,
    domain_hint: str,
    must_terms: List[str],
    avoid_terms: List[str],
    query_text: str,
) -> Dict[str, object]:
    must = [str(x).strip() for x in (must_terms or []) if str(x).strip()]
    avoid = [str(x).strip() for x in (avoid_terms or []) if str(x).strip()]
    q = str(query_text or "")

    if domain_hint == "TRIBUTARIO":
        protected = {"dian", "tributario", "estatuto tributario"}
        avoid = [x for x in avoid if x.strip().lower() not in protected]
        for t in ["DIAN", "Estatuto Tributario"]:
            if t.lower() not in {m.lower() for m in must}:
                must.append(t)
        if "dian" not in q.lower() and "estatuto tributario" not in q.lower():
            q = f"{q} DIAN Estatuto Tributario".strip()

    return {
        "must_terms": must,
        "avoid_terms": avoid,
        "query_text": q,
    }
