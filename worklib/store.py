from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any

MANIFEST_VERSION = 3


@dataclass
class Category:
    name: str
    vector_store_id: str
    folder: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class Doc:
    doc_id: str
    filename: str
    abs_path: str
    category: str
    title: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    openai_file_id: str = ""
    vector_store_id: str = ""
    # Added in v3 for fast dedupe & auditing
    sha256: str = ""
    size_bytes: int = 0
    mtime: float = 0.0  # seconds since epoch


@dataclass
class Manifest:
    version: int = MANIFEST_VERSION
    categories: Dict[str, Category] = field(default_factory=dict)
    docs: Dict[str, Doc] = field(default_factory=dict)


def _coerce_category(name: str, raw: Any) -> Category:
    if isinstance(raw, Category):
        return raw
    if not isinstance(raw, dict):
        raw = {}
    return Category(
        name=name,
        vector_store_id=str(raw.get("vector_store_id", "") or ""),
        folder=str(raw.get("folder", "") or ""),
        keywords=list(raw.get("keywords", []) or []),
    )


def _coerce_doc(doc_id: str, raw: Any) -> Doc:
    if isinstance(raw, Doc):
        return raw
    if not isinstance(raw, dict):
        raw = {}
    return Doc(
        doc_id=str(raw.get("doc_id", doc_id) or doc_id),
        filename=str(raw.get("filename", "") or ""),
        abs_path=str(raw.get("abs_path", "") or ""),
        category=str(raw.get("category", "") or ""),
        title=str(raw.get("title", "") or ""),
        author=str(raw.get("author", "") or ""),
        tags=list(raw.get("tags", []) or []),
        openai_file_id=str(raw.get("openai_file_id", "") or ""),
        vector_store_id=str(raw.get("vector_store_id", "") or ""),
        sha256=str(raw.get("sha256", "") or ""),
        size_bytes=int(raw.get("size_bytes", 0) or 0),
        mtime=float(raw.get("mtime", 0.0) or 0.0),
    )


def load_manifest(path: Path) -> Manifest:
    """Loads the manifest from disk.

    Backward-compatible with earlier manifests where categories/docs were stored
    as raw dicts.
    """
    if not path.exists():
        return Manifest()

    data = json.loads(path.read_text(encoding="utf-8"))
    version = int(data.get("version", 2) or 2)

    cats: Dict[str, Category] = {}
    for cname, c in (data.get("categories", {}) or {}).items():
        cats[cname] = _coerce_category(cname, c)

    docs: Dict[str, Doc] = {}
    for did, d in (data.get("docs", {}) or {}).items():
        docs[did] = _coerce_doc(did, d)

    return Manifest(version=max(version, MANIFEST_VERSION), categories=cats, docs=docs)


def save_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "version": MANIFEST_VERSION,
        "categories": {k: asdict(v) for k, v in manifest.categories.items()},
        "docs": {k: asdict(v) for k, v in manifest.docs.items()},
    }
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_category(manifest: Manifest, cat: Category) -> None:
    manifest.categories[cat.name] = cat


def add_doc(manifest: Manifest, doc: Doc) -> None:
    manifest.docs[doc.doc_id] = doc


def find_doc_by_sha256(manifest: Manifest, sha256: str) -> Optional[Doc]:
    if not sha256:
        return None
    for d in manifest.docs.values():
        if d.sha256 and d.sha256 == sha256:
            return d
    return None
