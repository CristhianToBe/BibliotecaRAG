from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from worklib.store import Manifest

from ..llm_steps import build_taxonomy
from ..utils import taxonomy_tree_txt


def stage_build_taxonomy(client, model_tax: str, manifest: Manifest, version_label: str, vroot: Path, *, taxonomy_system: str) -> Tuple[Dict[str, Any], List[str]]:
    print(f"🧠 ({model_tax}) Generando taxonomía base para {version_label}...")
    taxonomy = build_taxonomy(client, model_tax, manifest, taxonomy_system=taxonomy_system)
    base_paths = [t.get("path") for t in (taxonomy.get("taxonomy") or []) if isinstance(t, dict)]
    base_paths = sorted(set([p.strip("/").strip() for p in base_paths if p]))
    if "misc" not in base_paths:
        base_paths.append("misc")

    (vroot / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    (vroot / "taxonomy.txt").write_text(taxonomy_tree_txt(base_paths), encoding="utf-8")
    return taxonomy, base_paths


def stage_write_prefixed_taxonomy_view(vroot: Path, doc_prefix: Dict[str, Dict[str, Any]], base_paths: List[str]) -> List[str]:
    final_view_paths: List[str] = []
    by_author_year: Dict[str, set] = {}
    for pr in doc_prefix.values():
        by_author_year.setdefault(pr["author_key"], set()).add(str(pr["year"]))
    for a, ys in by_author_year.items():
        for y in sorted(ys):
            for bp in base_paths:
                final_view_paths.append(f"{a}/{y}/{bp}")
    (vroot / "taxonomy_prefixed.txt").write_text(taxonomy_tree_txt(final_view_paths), encoding="utf-8")
    return final_view_paths
