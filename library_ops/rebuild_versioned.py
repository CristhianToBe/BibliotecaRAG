from __future__ import annotations

# This file is a lightly-refactored version of rebuild_library_versioned.py:
# - shared OpenAI helpers come from library_ops.openai_utils
# - keeps the core algorithm intact
#
# It still depends on your existing worklib.* modules for config + manifest dataclasses.

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from worklib.config import default_config
from worklib.store import Category, Doc, Manifest, load_manifest, save_manifest
from worklib.prompt_loader import load_prompt

from .openai_utils import get_client, llm_json, get_vs_file_text
from .bootstrap_manifest import run as bootstrap_manifest_run

# -----------------------------
# Utils (file ops)
# -----------------------------

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s/-]+", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "misc"

def norm_author_key(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "DESCONOCIDO"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def next_version_folder(parent: Path) -> Tuple[int, Path]:
    max_n = 0
    if parent.exists():
        for p in parent.iterdir():
            if p.is_dir():
                m = re.fullmatch(r"v(\d+)", p.name)
                if m:
                    max_n = max(max_n, int(m.group(1)))
    n = max_n + 1
    return n, parent / f"v{n}"

def safe_copy(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    final = dst
    if final.exists():
        stem, suf = final.stem, final.suffix
        k = 2
        while True:
            cand = final.with_name(f"{stem}__dup{k}{suf}")
            if not cand.exists():
                final = cand
                break
            k += 1
    shutil.copy2(src, final)
    return final

def safe_move(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    final = dst
    if final.exists():
        stem, suf = final.stem, final.suffix
        k = 2
        while True:
            cand = final.with_name(f"{stem}__dup{k}{suf}")
            if not cand.exists():
                final = cand
                break
            k += 1
    shutil.move(src, final)
    return final

def rm_empty_dirs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return 0
    dirs = sorted([d for d in root.rglob("*") if d.is_dir()], key=lambda p: len(str(p)), reverse=True)
    for d in dirs:
        try:
            if not any(d.iterdir()):
                d.rmdir()
                removed += 1
        except OSError:
            pass
    return removed

# -----------------------------
# Heuristics for author/year (fallback)
# -----------------------------

def extract_year_heuristic(doc: Doc) -> Optional[int]:
    hay = " ".join([doc.title or "", doc.filename or "", " ".join(doc.tags or [])])
    m = re.search(r"(19|20)\d{2}", hay)
    if not m:
        return None
    y = int(m.group(0))
    return y if 1900 <= y <= 2100 else None

def author_key_heuristic(doc: Doc) -> str:
    if (doc.author or "").strip():
        return norm_author_key(doc.author)
    hay = " ".join([doc.filename or "", doc.title or "", " ".join(doc.tags or [])]).upper()
    if "DIAN" in hay or "OFICIO" in hay:
        return "DIAN"
    if "CONSEJO" in hay or "SECCION" in hay or "SECCIÓN" in hay or "SENTENCIA" in hay or "AUTO" in hay:
        return "CONSEJO_DE_ESTADO"
    if "CTCP" in hay:
        return "CTCP"
    if "SUPERFINANCIERA" in hay or "SFC" in hay:
        return "SUPERFINANCIERA"
    if "SUPERSOCIEDADES" in hay:
        return "SUPERSOCIEDADES"
    return "DESCONOCIDO"

def model_names() -> Tuple[str, str, str]:
    model_tax = os.getenv("MODEL_TAXONOMY", "gpt-5")
    model_nano = os.getenv("MODEL_PROPOSE", "gpt-5-nano")
    model_mini = os.getenv("MODEL_VALIDATE", "gpt-5-mini")
    return model_tax, model_nano, model_mini

def taxonomy_tree_txt(paths: List[str]) -> str:
    paths = sorted(set([p.strip("/").strip() for p in paths if p]))
    root: Dict[str, Any] = {}
    for p in paths:
        node = root
        for part in p.split("/"):
            node = node.setdefault(part, {})
    lines: List[str] = []
    def walk(node: Dict[str, Any], pref: str = "") -> None:
        for k in sorted(node.keys()):
            lines.append(pref + k + "/")
            walk(node[k], pref + "  ")
    walk(root)
    return "\n".join(lines) + ("\n" if lines else "")

# -----------------------------
# LLM prompts (lazy load to keep CLI import-safe)
# -----------------------------

def get_prompts() -> Dict[str, str]:
    return {
        "prefix": load_prompt("library_ops_prefix_system"),
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
        ak = norm_author_key(r.get("author_key"))
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

def run(
    *,
    manifest_override: Optional[Path],
    apply: bool,
    mode: str,
    debug: bool,
    batch_size: int,
    prefix_batch: int,
    max_rounds: int,
    max_docs: int,
    cleanup_empty: bool,
    create_vector_stores: bool,
) -> int:
    load_dotenv()
    cfg = default_config()

    manifest_path = manifest_override if manifest_override else cfg.manifest_path
    if not manifest_path.exists():
        print(f"⚠️ No existe manifest: {manifest_path}")
        print("↪ Intentando reconstruirlo automáticamente desde los documentos existentes...")
        rc_boot = bootstrap_manifest_run(manifest_out=manifest_path, library_dir=cfg.library_dir, force=False)
        if rc_boot != 0:
            print("❌ No fue posible reconstruir el manifest automáticamente.")
            return rc_boot

    manifest = load_manifest(manifest_path)
    docs_all = list(manifest.docs.values())

    client = get_client()
    prompts = get_prompts()
    model_tax, model_nano, model_mini = model_names()

    docs = docs_all[: max_docs] if max_docs and max_docs > 0 else docs_all

    vnum, vroot = next_version_folder(cfg.library_dir)
    version_label = f"v{vnum}"
    v_state = vroot / "_state"
    v_lib = vroot / "biblioteca"
    ensure_dir(v_state); ensure_dir(v_lib)

    # Step 0: prefixes
    print(f"⚡ ({model_nano}) Prefijos AUTOR/AÑO para {len(docs)} docs...")
    doc_prefix: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(docs), prefix_batch):
        batch = docs[i:i+prefix_batch]
        try:
            out = infer_prefixes_batch(client, model_nano, batch, prefix_system=prompts["prefix"])
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

    # Step 1: taxonomy base
    print(f"🧠 ({model_tax}) Generando taxonomía base para {version_label}...")
    taxonomy = build_taxonomy(client, model_tax, manifest, taxonomy_system=prompts["taxonomy"])
    base_paths = [t.get("path") for t in (taxonomy.get("taxonomy") or []) if isinstance(t, dict)]
    base_paths = sorted(set([p.strip("/").strip() for p in base_paths if p]))
    if "misc" not in base_paths:
        base_paths.append("misc")

    (vroot / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    (vroot / "taxonomy.txt").write_text(taxonomy_tree_txt(base_paths), encoding="utf-8")

    # Human view with AUTOR/AÑO prefix (only observed combinations)
    final_view_paths: List[str] = []
    by_author_year: Dict[str, set] = {}
    for pr in doc_prefix.values():
        by_author_year.setdefault(pr["author_key"], set()).add(str(pr["year"]))
    for a, ys in by_author_year.items():
        for y in sorted(ys):
            for bp in base_paths:
                final_view_paths.append(f"{a}/{y}/{bp}")
    (vroot / "taxonomy_prefixed.txt").write_text(taxonomy_tree_txt(final_view_paths), encoding="utf-8")

    # Step 2: propose base path per doc
    tmp_vs = manifest.categories.get("__ingest_tmp__").vector_store_id if "__ingest_tmp__" in manifest.categories else ""
    proposals: Dict[str, Dict[str, Any]] = {}
    print(f"🔎 ({model_nano}) Proponiendo carpeta base para {len(docs)} docs...")
    for d in docs:
        vsid = d.vector_store_id or tmp_vs
        doc_text = ""
        if vsid and d.openai_file_id:
            doc_text = get_vs_file_text(client, vsid, d.openai_file_id, max_chars=6000)
        proposals[d.doc_id] = propose_base_path(client, model_nano, base_paths, d, doc_text, propose_system=prompts["propose"])

    # Step 3: validate loops
    accepted_base: Dict[str, str] = {}
    pending = set([d.doc_id for d in docs])
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
            batch_ids = pending_list[i:i+batch_size]
            result = validate_batch(client, model_mini, base_paths, batch_items(batch_ids), validate_system=prompts["validate"])
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
                proposals[did] = propose_base_path(client, model_nano, base_paths, d, doc_text, propose_system=prompts["propose"])

    for did in list(pending):
        accepted_base[did] = "misc"

    def final_path_for(did: str) -> str:
        pr = doc_prefix.get(did, {"author_key": "DESCONOCIDO", "year": "sin_anio"})
        return f"{pr['author_key']}/{pr['year']}/{accepted_base.get(did, 'misc')}"

    accepted_final: Dict[str, str] = {d.doc_id: final_path_for(d.doc_id) for d in docs}
    final_paths = sorted(set(accepted_final.values()))

    moves: List[Dict[str, str]] = []
    for d in docs:
        src = Path(d.abs_path) if d.abs_path else None
        if not src or not src.exists():
            continue
        dst = v_lib / accepted_final[d.doc_id] / src.name
        moves.append({"doc_id": d.doc_id, "src": str(src), "dst": str(dst)})

    (vroot / "plan.json").write_text(json.dumps({"version": version_label, "moves": moves}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RESUMEN ===")
    print(f"- versión nueva: {version_label} -> {vroot}")
    print(f"- docs procesados: {len(docs)}")
    print(f"- base_paths: {len(base_paths)}")
    print(f"- paths finales (autor/año/base): {len(final_paths)}")
    print(f"- movimientos: {len(moves)} (mode={mode})")
    print(f"- apply: {apply}")

    if not apply:
        print(f"📝 No se aplicó nada. Revisa {vroot/'plan.json'} y {vroot/'taxonomy_prefixed.txt'}.")

    if not apply:
        return 0

    print("\n🚚 Materializando archivos...")
    for m in moves:
        src = Path(m["src"])
        dst = Path(m["dst"])
        if mode == "move":
            safe_move(src, dst)
        else:
            safe_copy(src, dst)

    new_manifest = Manifest(version=manifest.version, categories={}, docs={})

    for fp in final_paths:
        cat_name = fp.replace("/", "__")
        new_manifest.categories[cat_name] = Category(
            name=cat_name,
            vector_store_id="",
            folder=str((v_lib / fp).resolve()),
            keywords=[],
        )

    if create_vector_stores:
        print("\n🧠 Creando vector stores por categoría final...")
        vs_by_cat: Dict[str, str] = {}
        for cat_name in new_manifest.categories.keys():
            vs = client.vector_stores.create(name=f"{version_label}:{cat_name}")
            vs_by_cat[cat_name] = vs.id
            new_manifest.categories[cat_name].vector_store_id = vs.id

        cat_to_files: Dict[str, List[str]] = {}
        for d in docs:
            fp = accepted_final[d.doc_id]
            cat_name = fp.replace("/", "__")
            if d.openai_file_id:
                cat_to_files.setdefault(cat_name, []).append(d.openai_file_id)

        for cat_name, file_ids in cat_to_files.items():
            vsid = vs_by_cat.get(cat_name)
            if not vsid:
                continue
            for i in range(0, len(file_ids), 2000):
                batch_ids = file_ids[i:i+2000]
                client.vector_stores.file_batches.create(vector_store_id=vsid, file_ids=batch_ids)

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

    out_manifest = v_state / "library.json"
    save_manifest(out_manifest, new_manifest)
    print(f"✅ Nuevo library.json: {out_manifest}")

    if mode == "move" and cleanup_empty:
        removed = rm_empty_dirs(cfg.library_dir)
        print(f"🧹 Carpetas vacías eliminadas en biblioteca vieja: {removed}")

    print("\n✅ Listo. La nueva versión quedó aislada; el humano decide si borra versiones anteriores.")
    return 0

def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("rebuild-versioned", help="Rebuild library into Biblioteca/biblioteca/vN with AUTOR/AÑO prefix + GPT taxonomy")
    p.add_argument("--manifest", default=None, help="Ruta a library.json (override)")
    p.add_argument("--apply", action="store_true", help="Aplica cambios. Si no, solo plan.")
    p.add_argument("--mode", choices=["copy","move"], default="copy")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--prefix-batch", type=int, default=20)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--cleanup-empty", action="store_true")
    p.add_argument("--create-vector-stores", action="store_true")
    def _cmd(args: argparse.Namespace) -> int:
        return run(
            manifest_override=Path(args.manifest).expanduser() if args.manifest else None,
            apply=args.apply,
            mode=args.mode,
            debug=args.debug,
            batch_size=args.batch_size,
            prefix_batch=args.prefix_batch,
            max_rounds=args.max_rounds,
            max_docs=args.max_docs,
            cleanup_empty=args.cleanup_empty,
            create_vector_stores=args.create_vector_stores,
        )
    p.set_defaults(func=_cmd)
