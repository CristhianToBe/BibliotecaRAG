from __future__ import annotations

from pathlib import Path

from ..utils import rm_empty_dirs


def stage_cleanup_empty_dirs(library_dir: Path) -> int:
    removed = rm_empty_dirs(library_dir)
    print(f"🧹 Carpetas vacías eliminadas en biblioteca vieja: {removed}")
    return removed
