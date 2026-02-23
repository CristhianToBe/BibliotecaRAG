from __future__ import annotations
import hashlib
import shutil
from pathlib import Path

def doc_id_for(path: Path, openai_file_id: str, vs_id: str) -> str:
    s = f"{path.name}|{openai_file_id}|{vs_id}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def safe_copy(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem, suffix = src.stem, src.suffix
        i = 2
        while True:
            cand = dst_dir / f"{stem} ({i}){suffix}"
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)
    return dst

def slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "/"):
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:60] or "miscelanea"
