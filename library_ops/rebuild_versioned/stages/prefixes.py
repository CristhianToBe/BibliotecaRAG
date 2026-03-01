from __future__ import annotations

from typing import Any, Dict, List

from worklib.store import Doc

from ..heuristics import author_key_heuristic, extract_year_heuristic
from ..llm_steps import infer_prefixes_batch
from ..utils import norm_author_key


def stage_infer_prefixes(client, model_nano: str, docs: List[Doc], prefix_batch: int, prefix_system: str) -> Dict[str, Dict[str, Any]]:
    print(f"⚡ ({model_nano}) Prefijos AUTOR/AÑO para {len(docs)} docs...")
    doc_prefix: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(docs), prefix_batch):
        batch = docs[i : i + prefix_batch]
        try:
            out = infer_prefixes_batch(client, model_nano, batch, prefix_system=prefix_system)
        except Exception:
            out = {}
        for d in batch:
            heur_author = author_key_heuristic(d)
            heur_year = extract_year_heuristic(d)
            got = out.get(d.doc_id, {})
            ak = norm_author_key(got.get("author_key") or heur_author)
            y = got.get("year")
            if y is None:
                y = heur_year
            doc_prefix[d.doc_id] = {"author_key": ak, "year": (y if y is not None else "sin_anio")}
    return doc_prefix
