from __future__ import annotations

from pathlib import Path

from ..utils import safe_copy, safe_move


def stage_materialize_files(moves, mode: str) -> None:
    print("\n🚚 Materializando archivos...")
    for m in moves:
        src = Path(m["src"])
        dst = Path(m["dst"])
        if mode == "move":
            safe_move(src, dst)
        else:
            safe_copy(src, dst)
