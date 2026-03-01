from __future__ import annotations

import json
from typing import Any, Dict, List

from worklib.prompt_loader import load_prompt
from worklib.store import Doc, Manifest

from library_ops.openai_utils import llm_json

from .utils import norm_author_key, slugify


def get_prompts() -> Dict[str, str]:
    return {
        "prefix": load_prompt("library_ops_prefix_system"),
        "prefix_normalize": load_prompt("library_ops_prefix_normalize_system"),
        "taxonomy": load_prompt("library_ops_taxonomy_system"),
        "propose": load_prompt("library_ops_propose_system"),
        "validate": load_prompt("library_ops_validate_system"),
    }


def infer_prefixes_batch(client, model_nano: str, docs: List[Doc], *, prefix_system: str) -> Dict[str, Dict[str, Any]]:
    payload = {"docs": [{"doc_id": d.doc_id, "filename": d.filename, "title": d.title, "author": d.author, "tags": d.tags} for d in docs]}
    out = llm_json(client=client, model=model_nano, system=prefix_system, user=json.dumps(payload, ensure_ascii=False, indent=2))
    results: Dict[str, Dict[str, Any]] = {}
    for r in (out.get("results") or []):
        did = r.get("doc_id")
        if not did:
            continue
        ak = str(r.get("author_key") or "DESCONOCIDO")
        y = r.get("year", None)
        try:
            y = int(y) if y is not None else None
        except Exception:
            y = None
        if y is not None and not (1900 <= y <= 2100):
            y = None
        results[did] = {"author_key": ak, "year": y}
    return results


def build_taxonomy(client, model_tax: str, manifest: Manifest, *, taxonomy_system: str) -> Dict[str, Any]:
    cats = []
    for c in manifest.categories.values():
        if c.name == "__ingest_tmp__":
            continue
        cats.append({"name": c.name, "keywords": list((c.keywords or [])[:25])})
    payload = {"categories": cats}
    return llm_json(client=client, model=model_tax, system=taxonomy_system, user=json.dumps(payload, ensure_ascii=False, indent=2))


def propose_base_path(client, model_nano: str, taxonomy_paths: List[str], doc: Doc, doc_text: str, *, propose_system: str) -> Dict[str, Any]:
    payload = {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "title": doc.title,
        "author": doc.author,
        "tags": doc.tags,
        "current_category": doc.category,
        "taxonomy_paths": taxonomy_paths[:2000],
        "doc_text_excerpt": doc_text[:6000],
    }
    out = llm_json(client=client, model=model_nano, system=propose_system, user=json.dumps(payload, ensure_ascii=False, indent=2))
    out["doc_id"] = doc.doc_id
    p = out.get("proposed_path", "misc")
    p = "/".join([slugify(x) for x in str(p).split("/") if x.strip()]) or "misc"
    if p not in taxonomy_paths:
        cand = [t for t in taxonomy_paths if t.startswith(p + "/")]
        p = cand[0] if cand else "misc"
    out["proposed_path"] = p
    return out


def validate_batch(client, model_mini: str, taxonomy_paths: List[str], batch_items: List[Dict[str, Any]], *, validate_system: str) -> Dict[str, Any]:
    payload = {"taxonomy_paths": taxonomy_paths[:2000], "batch": batch_items}
    return llm_json(client=client, model=model_mini, system=validate_system, user=json.dumps(payload, ensure_ascii=False, indent=2))


def normalize_prefixes_batch(client, model_nano: str, raw_author_keys: List[str], *, normalize_system: str) -> Dict[str, str]:
    payload = {"author_keys": list(raw_author_keys)}
    out = llm_json(client=client, model=model_nano, system=normalize_system, user=json.dumps(payload, ensure_ascii=False, indent=2))
    mapping = out.get("mapping") or {}
    if not isinstance(mapping, dict):
        mapping = {}
    fixed: Dict[str, str] = {}
    for raw in raw_author_keys:
        cand = mapping.get(raw)
        fixed[raw] = norm_author_key(str(cand) if cand is not None else "DESCONOCIDO")
    return fixed
