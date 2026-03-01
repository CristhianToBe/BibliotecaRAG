from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ..utils import ensure_dir, next_version_folder


def stage_prepare_workspace(library_dir: Path) -> Tuple[int, Path, str, Path, Path]:
    vnum, vroot = next_version_folder(library_dir)
    version_label = f"v{vnum}"
    v_state = vroot / "_state"
    v_lib = vroot / "biblioteca"
    ensure_dir(v_state)
    ensure_dir(v_lib)
    return vnum, vroot, version_label, v_state, v_lib
