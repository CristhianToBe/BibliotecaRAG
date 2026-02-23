from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from worklib.prompt_loader import load_prompt

from .manifest_json import safe_json_load, safe_json_dump, backup_file, index_docs_by_category
from .openai_utils import get_client, llm_json, get_vs_file_text

KEYWORDS_SYSTEM = load_prompt("library_ops_fill_keywords")

def normalize_kw(k: str) -> str:
    k = (k or "").strip().lower()
    k = " ".join(k.split())
    return k

def uniq_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def run(
    *,
    manifest: Path,
    out: Path | None = None,
    model: str = "gpt-5-nano",
    per_category_docs: int = 4,
    max_chars_per_doc: int = 10000,
    only_empty: bool = False,
    debug: bool = False,
) -> int:
    load_dotenv()
    client = get_client()

    if not manifest.exists():
        print(f"❌ No existe: {manifest}")
        return 2

    out_path = out or manifest

    data = safe_json_load(manifest)
    cats: Dict[str, Any] = data.get("categories", {}) or {}
    docs: Dict[str, Any] = data.get("docs", {}) or {}
    docs_by_cat = index_docs_by_category(docs)

    total_docs = len(docs)
    total_cats = len(cats)
    empty = sum(1 for c in cats.values() if not (c.get("keywords") or []))
    print("=== INVENTARIO ===")
    print(f"- manifest: {manifest}")
    print(f"- categories: {total_cats} | empty_keywords: {empty}")
    print(f"- docs: {total_docs}")

    updated = 0
    for cat_name, cat in cats.items():
        existing = cat.get("keywords") or []
        if only_empty and existing:
            continue

        folder = cat.get("folder", "")
        sample_docs = docs_by_cat.get(cat_name, [])[: max(0, per_category_docs)]

        evidence_docs = []
        for d in sample_docs:
            excerpt = ""
            vsid = (d.get("vector_store_id") or "").strip()
            fid = (d.get("openai_file_id") or "").strip()
            if vsid and fid:
                excerpt = get_vs_file_text(client, vsid, fid, max_chars=max_chars_per_doc)
            evidence_docs.append(
                {
                    "title": d.get("title"),
                    "filename": d.get("filename"),
                    "tags": d.get("tags"),
                    "excerpt": excerpt[:max_chars_per_doc],
                }
            )

        payload = {
            "category_name": cat_name,
            "folder": folder,
            "category_path_hint": folder.replace("\\", "/"),
            "evidence_docs": evidence_docs,
        }

        outj = llm_json(client, model, KEYWORDS_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2))
        kws = outj.get("keywords") or []
        kws = [normalize_kw(k) for k in kws if isinstance(k, str)]
        kws = [k for k in kws if k]
        kws = uniq_keep_order(kws)[:25]

        cat["keywords"] = kws
        updated += 1

        if debug:
            print(f"✅ {cat_name}: {len(kws)} keywords")
        else:
            if updated % 10 == 0:
                print(f"... {updated} categorías actualizadas")

    if out_path == manifest:
        bk = backup_file(manifest)
        print(f"🧷 Backup: {bk}")

    safe_json_dump(out_path, data)
    print(f"✅ Guardado: {out_path}")
    return 0

def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("fill-keywords", help="Rellena keywords por categoría en el manifest")
    p.add_argument("--manifest", required=True, help="Ruta a vN/_state/library.json")
    p.add_argument("--out", default="", help="Salida (default: sobreescribe el manifest)")
    p.add_argument("--model", default=os.getenv("MODEL_KEYWORDS", "gpt-5-nano"))
    p.add_argument("--per-category-docs", type=int, default=4)
    p.add_argument("--max-chars-per-doc", type=int, default=10000)
    p.add_argument("--only-empty", action="store_true")
    p.add_argument("--debug", action="store_true")
    def _cmd(args: argparse.Namespace) -> int:
        out = Path(args.out).expanduser() if args.out else None
        return run(
            manifest=Path(args.manifest).expanduser(),
            out=out,
            model=args.model,
            per_category_docs=args.per_category_docs,
            max_chars_per_doc=args.max_chars_per_doc,
            only_empty=args.only_empty,
            debug=args.debug,
        )
    p.set_defaults(func=_cmd)
