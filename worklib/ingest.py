from __future__ import annotations

import argparse
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

from worklib.openai_client import get_client
from worklib.settings import get_openai_settings
from worklib.logging_utils import setup_logging
from worklib.prompt_loader import load_prompt

from worklib.config import default_config, WORK_ONLY_HINT
from worklib.store import (
    load_manifest, save_manifest, upsert_category, add_doc,
    Category, Doc, find_doc_by_sha256
)
from worklib.fs import safe_copy, doc_id_for, slugify  # :contentReference[oaicite:0]{index=0}

load_dotenv()

log = logging.getLogger(__name__)

TMP_CATEGORY_NAME = "__ingest_tmp__"

SYSTEM_CONTENT_CATEGORIZER = load_prompt("ingest_system")

# -----------------------------
# Hash
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# -----------------------------
# Indexado vector store
# -----------------------------
def attach_and_wait(client, vs_id: str, file_id: str, poll: float = 2.0, timeout: float = 300.0) -> None:
    client.vector_stores.files.create(vector_store_id=vs_id, file_id=file_id)
    start = time.time()
    while True:
        cur = client.vector_stores.files.retrieve(vector_store_id=vs_id, file_id=file_id)
        if cur.status in ("completed", "failed", "cancelled"):
            if cur.status != "completed":
                raise RuntimeError(f"Indexado no completó: {cur.status}")
            return
        if time.time() - start > timeout:
            raise TimeoutError("Timeout esperando indexado.")
        time.sleep(poll)

def ensure_tmp_vector_store(client, manifest) -> str:
    cat = manifest.categories.get(TMP_CATEGORY_NAME)
    if cat and getattr(cat, "vector_store_id", ""):
        return cat.vector_store_id

    vs = client.vector_stores.create(name="worklib_ingest_tmp")
    upsert_category(manifest, Category(
        name=TMP_CATEGORY_NAME,
        vector_store_id=vs.id,
        folder="",
        keywords=[],
    ))
    return vs.id

def ensure_category_vector_store(client, manifest, category_name: str, category_folder: Path) -> str:
    cat = manifest.categories.get(category_name)
    if cat and getattr(cat, "vector_store_id", ""):
        return cat.vector_store_id

    vs = client.vector_stores.create(name=f"worklib_{category_name}")
    upsert_category(manifest, Category(
        name=category_name,
        vector_store_id=vs.id,
        folder=str(category_folder),
        keywords=[],
    ))
    return vs.id

def merge_keywords(existing: List[str], new: List[str], limit: int = 40) -> List[str]:
    seen = set()
    out: List[str] = []
    for k in (existing or []) + (new or []):
        kk = (k or "").strip()
        if not kk:
            continue
        low = kk.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(kk)
        if len(out) >= limit:
            break
    return out

def content_based_categorize(client, model: str, file_id: str, tmp_vs_id: str, filename: str, title: str, author: str,) -> Dict[str, Any]:

    # 1️⃣ Cargar prompts desde archivos externos
    system_prompt = load_prompt("ingest_system")
    user_template = load_prompt("ingest_user_template")

    # 2️⃣ Construir user prompt dinámico
    user_prompt = user_template.format(
        filename=filename,
        title=title,
        author=author,
        work_hint=", ".join(WORK_ONLY_HINT),
    )

    # 3️⃣ Definir tool file_search
    tools = [{
        "type": "file_search",
        "vector_store_ids": [tmp_vs_id],
    }]

    # 4️⃣ Llamar al modelo
    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "user",
                "content": (
                    "Primero, recupera fragmentos relevantes del documento (si necesitas). "
                    "Luego devuelve estrictamente el JSON solicitado, sin texto adicional."
                ),
            },
        ],
        tools=tools,
    )

    # 5️⃣ Parse seguro
    txt = r.output_text or ""

    try:
        obj = json.loads(txt)
    except Exception:
        obj = {
            "category_name": "miscelanea",
            "category_label": "Miscelánea",
            "keywords": [],
            "topics": [],
            "confidence": 0.0,
            "reason": "parse error",
            "work_relevance": 0.0,
        }

    # 6️⃣ Sanitización fuerte
    obj["category_name"] = slugify(obj.get("category_name", "miscelanea"))

    obj["keywords"] = [
        k.strip()
        for k in (obj.get("keywords", []) or [])
        if isinstance(k, str) and k.strip()
    ][:25]

    obj["topics"] = [
        t.strip()
        for t in (obj.get("topics", []) or [])
        if isinstance(t, str) and t.strip()
    ][:10]

    return obj


def ingest_document(
    path: str | Path,
    *,
    title: str = "",
    author: str = "",
    tags: List[str] | None = None,
    copy_to_library: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Ingesta un documento y devuelve metadata útil para CLI y API web."""
    setup_logging(debug=debug)

    client = get_client()
    settings = get_openai_settings()
    model = settings.ingest_model
    log.info("Ingest model=%s timeout=%ss retries=%s", model, settings.timeout_s, settings.max_retries)

    cfg = default_config()
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.library_dir.mkdir(parents=True, exist_ok=True)

    src = Path(path).resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    # PRECHECK: dedupe por manifest+sha256
    src_hash = sha256_file(src)
    manifest_now = load_manifest(cfg.manifest_path)
    existing_doc = find_doc_by_sha256(manifest_now, src_hash)
    if existing_doc and existing_doc.abs_path and Path(existing_doc.abs_path).exists():
        return {
            "status": "already_exists",
            "doc_id": existing_doc.doc_id,
            "abs_path": existing_doc.abs_path,
            "category": existing_doc.category,
            "manifest_path": str(cfg.manifest_path),
        }

    title_clean = title.strip() or src.stem
    author_clean = author.strip()
    tags_clean = [t.strip() for t in (tags or []) if t and t.strip()]

    manifest = manifest_now  # seguimos usando el mismo objeto

    # 1) Subir archivo
    with open(src, "rb") as fh:
        f = client.files.create(file=fh, purpose="assistants")
    file_id = f.id

    # 2) Indexarlo en VS temporal para "leerlo"
    tmp_vs_id = ensure_tmp_vector_store(client, manifest)
    attach_and_wait(client, tmp_vs_id, file_id)

    # 3) Categorizar leyendo contenido real
    cat = content_based_categorize(client, model, file_id, tmp_vs_id, src.name, title_clean, author_clean)

    if cat.get("work_relevance", 0.0) < 0.35:
        raise RuntimeError(
            f"El documento parece poco relacionado con trabajo (work_relevance={cat.get('work_relevance')}). "
            f"Si quieres forzar, lo volvemos flag/param."
        )

    category_name = cat["category_name"]
    category_folder = cfg.library_dir / category_name
    category_folder.mkdir(parents=True, exist_ok=True)

    # 4) Copiar (opcional) a carpeta final
    final_path = safe_copy(src, category_folder) if copy_to_library else src

    # 5) VS por categoría
    vs_id = ensure_category_vector_store(client, manifest, category_name, category_folder)

    # 6) Indexar en VS final (por categoría)
    attach_and_wait(client, vs_id, file_id)

    # 7) Guardar / actualizar categoría (merge keywords)
    existing_kw = (manifest.categories.get(category_name).keywords if manifest.categories.get(category_name) else [])
    merged_kw = merge_keywords(existing_kw, cat.get("keywords", []))

    upsert_category(manifest, Category(
        name=category_name,
        vector_store_id=vs_id,
        folder=str(category_folder),
        keywords=merged_kw,
    ))

    # 8) Guardar doc record
    doc_id = doc_id_for(final_path, file_id, vs_id)
    add_doc(manifest, Doc(
        doc_id=doc_id,
        filename=final_path.name,
        abs_path=str(final_path),
        category=category_name,
        title=title_clean,
        author=author_clean,
        tags=tags_clean,
        openai_file_id=file_id,
        vector_store_id=vs_id,
        sha256=src_hash,
        size_bytes=final_path.stat().st_size if final_path.exists() else src.stat().st_size,
        mtime=final_path.stat().st_mtime if final_path.exists() else src.stat().st_mtime,
    ))

    save_manifest(cfg.manifest_path, manifest)
    return {
        "status": "ingested",
        "category": category_name,
        "category_label": cat.get("category_label", ""),
        "topics": list(cat.get("topics", []) or []),
        "folder": str(category_folder),
        "vector_store_id": vs_id,
        "openai_file_id": file_id,
        "doc_id": doc_id,
        "keywords": list(cat.get("keywords", []) or []),
        "manifest_path": str(cfg.manifest_path),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Ruta del documento (PDF recomendado)")
    ap.add_argument("--title", default="", help="Título opcional")
    ap.add_argument("--author", default="", help="Autor opcional")
    ap.add_argument("--tags", default="", help="Tags separados por coma")
    ap.add_argument("--copy", action="store_true", help="Copiar a la biblioteca (recomendado)")
    ap.add_argument("--debug", action="store_true", help="Logging verbose")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    result = ingest_document(
        args.path,
        title=args.title,
        author=args.author,
        tags=tags,
        copy_to_library=args.copy,
        debug=args.debug,
    )

    if result["status"] == "already_exists":
        print(" Ya existe en la biblioteca (mismo contenido). No se re-indexa.")
        print(" - doc_id:", result["doc_id"])
        print(" - archivo:", result["abs_path"])
        print(" - categoria:", result["category"])
        return

    print(" Ingestado (content-based)")
    print(" - category:", result["category"], "|", result.get("category_label", ""))
    print(" - topics:", ", ".join(result.get("topics", [])))
    print(" - folder:", result["folder"])
    print(" - vector_store_id:", result["vector_store_id"])
    print(" - openai_file_id:", result["openai_file_id"])
    print(" - doc_id:", result["doc_id"])
    print(" - keywords:", ", ".join(result.get("keywords", [])))

if __name__ == "__main__":
    main()
