from __future__ import annotations

from pathlib import Path
from typing import Optional

from worklib.store import Manifest


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def resolve_local_path(
    manifest: Manifest,
    *,
    file_id: Optional[str],
    filename: Optional[str],
    library_root: Optional[Path] = None,  # se mantiene por compatibilidad, pero ya no es necesaria
) -> Optional[str]:
    """
    Resuelve la ruta local usando el índice de documentos del manifest (library.json).

    Estrategias:
      1) match por openai_file_id == file_id (lo más confiable)
      2) match por filename (case-insensitive)
    Devuelve doc.abs_path.
    """

    fid = (file_id or "").strip()
    fname = (filename or "").strip()

    # helper: iterar docs sin asumir si es dict o list
    docs = getattr(manifest, "docs", None)
    if isinstance(docs, dict):
        it = docs.values()
    elif isinstance(docs, list):
        it = docs
    else:
        it = []

    # 1) por openai_file_id (file_search -> file_id)
    if fid:
        for d in it:
            if str(getattr(d, "openai_file_id", "") or "").strip() == fid:
                p = getattr(d, "abs_path", None)
                return str(p) if p else None

    # 2) por filename
    if fname:
        fname_n = _norm(fname)
        for d in it:
            if _norm(getattr(d, "filename", "") or "") == fname_n:
                p = getattr(d, "abs_path", None)
                return str(p) if p else None

    return None