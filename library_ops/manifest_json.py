"""Helpers para lectura/escritura de manifests JSON en library_ops."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

from ._time import now_stamp

JsonDict = Dict[str, Any]


def safe_json_load(path: Path) -> JsonDict:
    """Carga un JSON UTF-8 y retorna un diccionario."""
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json_dump(path: Path, obj: JsonDict) -> None:
    """Guarda un objeto JSON con indentación y UTF-8."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def backup_file(path: Path) -> Path:
    """Crea una copia de seguridad del archivo con timestamp."""
    bk = path.with_suffix(path.suffix + f".bak_{now_stamp()}")
    bk.write_bytes(path.read_bytes())
    return bk


def persist_manifest(
    manifest_path: Path,
    out_path: Path | None,
    data: JsonDict,
) -> tuple[Path, Path | None]:
    """Persist manifest + backup opcional.

    Si `out_path` es `None` o coincide con `manifest_path`, crea backup y luego guarda.
    Si `out_path` es distinto, solo guarda en la nueva ruta.

    Returns:
        `(saved_path, backup_path)` donde `backup_path` puede ser `None`.
    """
    target = out_path or manifest_path
    backup = None
    if target == manifest_path:
        backup = backup_file(manifest_path)
    safe_json_dump(target, data)
    return target, backup


def index_docs_by_category(docs: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Indexa documentos del manifest por nombre de categoría."""
    by_cat: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in (docs or {}).values():
        c = (d.get("category") or "").strip()
        by_cat[c].append(d)
    return dict(by_cat)
