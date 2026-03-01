from __future__ import annotations

from typing import Any, Dict, List

from worklib.store import Doc

from ..llm_steps import infer_prefixes_batch, normalize_prefixes_batch
from ..utils import norm_author_key


def stage_infer_prefixes(
    client,
    model_nano: str,
    docs: List[Doc],
    prefix_batch: int,
    prefix_system: str,
    normalize_system: str,
    *,
    debug: bool,
) -> Dict[str, Dict[str, Any]]:
    print(f"⚡ ({model_nano}) Prefijos AUTOR/AÑO para {len(docs)} docs...")

    raw_by_doc: Dict[str, str] = {}
    year_by_doc: Dict[str, Any] = {}

    for i in range(0, len(docs), prefix_batch):
        batch = docs[i : i + prefix_batch]
        try:
            out = infer_prefixes_batch(client, model_nano, batch, prefix_system=prefix_system)
        except Exception:
            out = {}
        for d in batch:
            got = out.get(d.doc_id, {})
            raw_ak = str(got.get("author_key") or "DESCONOCIDO")
            raw_by_doc[d.doc_id] = raw_ak
            y = got.get("year")
            year_by_doc[d.doc_id] = y if isinstance(y, int) and 1900 <= y <= 2100 else "sin_anio"

    raw_author_keys = sorted(set(raw_by_doc.values()))
    try:
        mapping = normalize_prefixes_batch(
            client,
            model_nano,
            raw_author_keys,
            normalize_system=normalize_system,
        )
    except Exception:
        mapping = {k: norm_author_key(k) for k in raw_author_keys}

    for k in raw_author_keys:
        mapping[k] = norm_author_key(mapping.get(k) or "DESCONOCIDO")

    if debug:
        print(f"[debug] raw_author_keys_distintos={len(raw_author_keys)}")
        print(f"[debug] raw_author_keys_sample={raw_author_keys[:20]}")
        print(f"[debug] raw_to_canonical_mapping={mapping}")

    doc_prefix: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        raw_ak = raw_by_doc.get(d.doc_id, "DESCONOCIDO")
        canonical_ak = mapping.get(raw_ak, "DESCONOCIDO")
        year = year_by_doc.get(d.doc_id, "sin_anio")
        doc_prefix[d.doc_id] = {"author_key": canonical_ak, "year": year}

    if debug:
        for d in docs[:10]:
            raw_ak = raw_by_doc.get(d.doc_id, "DESCONOCIDO")
            canonical_ak = doc_prefix[d.doc_id]["author_key"]
            year = doc_prefix[d.doc_id]["year"]
            print(f"[debug] doc={d.doc_id} raw_author_key={raw_ak} -> canonical_author_key={canonical_ak} year={year}")

    return doc_prefix
