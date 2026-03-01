from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def stage_build_final_paths(docs, doc_prefix: Dict[str, Dict[str, str]], accepted_base: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    def final_path_for(did: str) -> str:
        pr = doc_prefix.get(did, {"author_key": "DESCONOCIDO", "year": "sin_anio"})
        return f"{pr['author_key']}/{pr['year']}/{accepted_base.get(did, 'misc')}"

    accepted_final: Dict[str, str] = {d.doc_id: final_path_for(d.doc_id) for d in docs}
    final_paths = sorted(set(accepted_final.values()))
    return accepted_final, final_paths


def stage_generate_plan(version_label: str, vroot: Path, v_lib: Path, docs, accepted_final: Dict[str, str]):
    moves: List[Dict[str, str]] = []
    for d in docs:
        src = Path(d.abs_path) if d.abs_path else None
        if not src or not src.exists():
            continue
        dst = v_lib / accepted_final[d.doc_id] / src.name
        moves.append({"doc_id": d.doc_id, "src": str(src), "dst": str(dst)})

    (vroot / "plan.json").write_text(json.dumps({"version": version_label, "moves": moves}, ensure_ascii=False, indent=2), encoding="utf-8")
    return moves
