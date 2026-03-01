from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def model_names() -> Tuple[str, str, str]:
    import os

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
