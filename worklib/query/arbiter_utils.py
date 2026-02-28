from __future__ import annotations

from typing import Dict, List, Tuple


def normalize_indexes(values: object) -> List[int]:
    if not isinstance(values, list):
        return []
    out: List[int] = []
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def choose_variant_name(signal_summary: Dict[str, Dict[str, float | int]], considered: List[str]) -> str:
    names = [str(x) for x in considered if str(x)]
    if not names:
        return "A1"
    ranked = sorted(
        names,
        key=lambda name: (
            float((signal_summary.get(name) or {}).get("must_terms_coverage", 0.0)),
            int((signal_summary.get(name) or {}).get("unique_docs", 0)),
            int((signal_summary.get(name) or {}).get("hits_count", 0)),
        ),
        reverse=True,
    )
    return ranked[0]


def validate_evidence_indexes(
    *,
    selected_indexes: List[int],
    also_indexes: List[int],
    evidence_len: int,
    fallback_order: List[int],
) -> Tuple[List[int], List[int], bool]:
    valid_selected = [i for i in normalize_indexes(selected_indexes) if 0 <= i < evidence_len]
    valid_also = [i for i in normalize_indexes(also_indexes) if 0 <= i < evidence_len and i not in valid_selected]
    used_fallback = False

    if not valid_selected:
        used_fallback = True
        valid_selected = [i for i in fallback_order if 0 <= i < evidence_len][:3]

    if not valid_selected and evidence_len > 0:
        used_fallback = True
        valid_selected = list(range(min(3, evidence_len)))

    return valid_selected, valid_also, used_fallback
