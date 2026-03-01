from __future__ import annotations

import re
from typing import Optional

from worklib.store import Doc

from .utils import norm_author_key


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
