from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from worklib.store import Category, Doc, Manifest


def stage_build_new_manifest(manifest: Manifest, docs, final_paths: List[str], accepted_final: Dict[str, str], v_lib: Path) -> Manifest:
    new_manifest = Manifest(version=manifest.version, categories={}, docs={})

    for fp in final_paths:
        cat_name = fp.replace("/", "__")
        new_manifest.categories[cat_name] = Category(
            name=cat_name,
            vector_store_id="",
            folder=str((v_lib / fp).resolve()),
            keywords=[],
        )

    for d in docs:
        fp = accepted_final[d.doc_id]
        cat_name = fp.replace("/", "__")
        src = Path(d.abs_path) if d.abs_path else None
        dst_abs = d.abs_path
        filename = d.filename
        if src and src.exists():
            dst = v_lib / fp / src.name
            if dst.exists():
                dst_abs = str(dst.resolve())
                filename = dst.name

        new_manifest.docs[d.doc_id] = Doc(
            doc_id=d.doc_id,
            filename=filename,
            abs_path=dst_abs or "",
            category=cat_name,
            title=d.title,
            author=d.author,
            tags=list(d.tags or []),
            openai_file_id=d.openai_file_id,
            vector_store_id=new_manifest.categories[cat_name].vector_store_id,
            sha256=d.sha256,
            size_bytes=d.size_bytes,
            mtime=d.mtime,
        )

    return new_manifest
