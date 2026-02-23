"""Construye un `library.json` a partir de archivos existentes en `biblioteca/`."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Tuple

from worklib.config import default_config
from worklib.store import Category, Doc, Manifest, save_manifest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # Ignora artefactos internos de versiones
        if "_state" in p.parts:
            continue
        yield p


def _infer_category(library_dir: Path, file_path: Path) -> Tuple[str, Path]:
    rel = file_path.relative_to(library_dir)
    parts = rel.parts
    if len(parts) >= 2:
        category = parts[0].strip() or "miscelanea"
    else:
        category = "miscelanea"
    return category, library_dir / category


def _doc_id_for_path(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def build_manifest_from_library(library_dir: Path) -> Manifest:
    manifest = Manifest()

    categories: Dict[str, Category] = {}
    docs: Dict[str, Doc] = {}

    for p in sorted(_iter_files(library_dir)):
        category_name, category_folder = _infer_category(library_dir, p)

        if category_name not in categories:
            categories[category_name] = Category(
                name=category_name,
                vector_store_id="",
                folder=str(category_folder),
                keywords=[],
            )

        st = p.stat()
        doc_id = _doc_id_for_path(p.resolve())
        docs[doc_id] = Doc(
            doc_id=doc_id,
            filename=p.name,
            abs_path=str(p.resolve()),
            category=category_name,
            title=p.stem,
            author="",
            tags=[],
            openai_file_id="",
            vector_store_id="",
            sha256=_sha256_file(p),
            size_bytes=int(st.st_size),
            mtime=float(st.st_mtime),
        )

    manifest.categories = categories
    manifest.docs = docs
    return manifest


def run(*, manifest_out: Path | None = None, library_dir: Path | None = None, force: bool = False) -> int:
    cfg = default_config()
    lib_dir = (library_dir or cfg.library_dir).resolve()
    manifest_path = (manifest_out or cfg.manifest_path).resolve()

    if not lib_dir.exists() or not lib_dir.is_dir():
        print(f"❌ No existe library_dir: {lib_dir}")
        return 2

    if manifest_path.exists() and not force:
        print(f"❌ El manifest ya existe: {manifest_path} (usa --force para sobrescribir)")
        return 2

    manifest = build_manifest_from_library(lib_dir)
    save_manifest(manifest_path, manifest)

    print("✅ Manifest reconstruido")
    print(f"- library_dir: {lib_dir}")
    print(f"- manifest: {manifest_path}")
    print(f"- categorías: {len(manifest.categories)}")
    print(f"- docs: {len(manifest.docs)}")
    return 0


def build_parser(sp: argparse._SubParsersAction) -> None:
    p = sp.add_parser("bootstrap-manifest", help="Reconstruye library.json escaneando documentos en biblioteca/")
    p.add_argument("--manifest", default="", help="Ruta de salida para library.json")
    p.add_argument("--library-dir", default="", help="Ruta de biblioteca a escanear")
    p.add_argument("--force", action="store_true", help="Sobrescribe el manifest si existe")

    def _cmd(args: argparse.Namespace) -> int:
        return run(
            manifest_out=Path(args.manifest).expanduser() if args.manifest else None,
            library_dir=Path(args.library_dir).expanduser() if args.library_dir else None,
            force=args.force,
        )

    p.set_defaults(func=_cmd)
