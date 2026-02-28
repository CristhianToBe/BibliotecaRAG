from __future__ import annotations

from typing import List, Optional


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


def derive_winner(*, considered: List[str], winner: str = "", selected_indexes: Optional[List[int]] = None) -> str:
    considered_norm = [str(x) for x in (considered or []) if str(x)]
    candidate = str(winner or "").strip()
    idxs = normalize_indexes(selected_indexes)

    if candidate and candidate in considered_norm:
        return candidate
    if candidate and candidate.upper() in [c.upper() for c in considered_norm]:
        idx = [c.upper() for c in considered_norm].index(candidate.upper())
        return considered_norm[idx]

    if idxs:
        idx0 = idxs[0]
        if 0 <= idx0 < len(considered_norm):
            return considered_norm[idx0]

    return considered_norm[0] if considered_norm else "A1"
