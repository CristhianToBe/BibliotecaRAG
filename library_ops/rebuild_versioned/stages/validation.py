from __future__ import annotations

import json
from typing import Any, Dict, List, Set

from worklib.store import Manifest

from library_ops.openai_utils import get_vs_file_text

from ..llm_steps import propose_base_path, validate_batch


def stage_validate_with_reproposal(
    client,
    model_mini: str,
    model_nano: str,
    manifest: Manifest,
    docs,
    proposals: Dict[str, Dict[str, Any]],
    base_paths: List[str],
    *,
    validate_system: str,
    propose_system: str,
    tmp_vs: str,
    batch_size: int,
    max_rounds: int,
):
    accepted_base: Dict[str, str] = {}
    pending: Set[str] = set([d.doc_id for d in docs])
    rejected_fb: Dict[str, Dict[str, Any]] = {}

    def batch_items(doc_ids: List[str]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for did in doc_ids:
            d = manifest.docs[did]
            p = proposals[did]
            items.append({
                "doc_id": did,
                "filename": d.filename,
                "title": d.title,
                "author": d.author,
                "tags": d.tags,
                "current_category": d.category,
                "proposed_path": p.get("proposed_path", "misc"),
                "confidence": p.get("confidence", 0.0),
                "rationale": p.get("rationale", ""),
            })
        return items

    for round_num in range(1, max_rounds + 1):
        if not pending:
            break
        print(f"🧾 ({model_mini}) Validación ronda {round_num} | pendientes={len(pending)}")
        pending_list = list(pending)
        for i in range(0, len(pending_list), batch_size):
            batch_ids = pending_list[i : i + batch_size]
            result = validate_batch(client, model_mini, base_paths, batch_items(batch_ids), validate_system=validate_system)
            for r in (result.get("results") or []):
                did = r.get("doc_id")
                if not did or did not in pending:
                    continue
                if r.get("decision") == "accept":
                    accepted_base[did] = proposals[did]["proposed_path"]
                    pending.discard(did)
                else:
                    rejected_fb[did] = {"reason": r.get("reason", ""), "alternatives": r.get("alternatives", [])}

        if pending and round_num < max_rounds:
            print(f"🔁 ({model_nano}) Re-propuesta para rechazados: {len(pending)}")
            for did in list(pending):
                d = manifest.docs[did]
                fb = rejected_fb.get(did, {})
                vsid = d.vector_store_id or tmp_vs
                doc_text = ""
                if vsid and d.openai_file_id:
                    doc_text = get_vs_file_text(client, vsid, d.openai_file_id, max_chars=4500)
                doc_text = (doc_text + "\n\n[MINI_FEEDBACK]\n" + json.dumps(fb, ensure_ascii=False))[:6000]
                proposals[did] = propose_base_path(client, model_nano, base_paths, d, doc_text, propose_system=propose_system)

    for did in list(pending):
        accepted_base[did] = "misc"

    return accepted_base
