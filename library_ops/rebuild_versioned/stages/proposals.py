from __future__ import annotations

from typing import Any, Dict, List

from worklib.store import Doc, Manifest

from library_ops.openai_utils import get_vs_file_text

from ..llm_steps import propose_base_path


def stage_propose_base_paths(client, model_nano: str, manifest: Manifest, docs: List[Doc], base_paths: List[str], *, propose_system: str) -> Dict[str, Dict[str, Any]]:
    tmp_vs = manifest.categories.get("__ingest_tmp__").vector_store_id if "__ingest_tmp__" in manifest.categories else ""
    proposals: Dict[str, Dict[str, Any]] = {}
    print(f"🔎 ({model_nano}) Proponiendo carpeta base para {len(docs)} docs...")
    for d in docs:
        vsid = d.vector_store_id or tmp_vs
        doc_text = ""
        if vsid and d.openai_file_id:
            doc_text = get_vs_file_text(client, vsid, d.openai_file_id, max_chars=6000)
        proposals[d.doc_id] = propose_base_path(client, model_nano, base_paths, d, doc_text, propose_system=propose_system)
    return proposals, tmp_vs
