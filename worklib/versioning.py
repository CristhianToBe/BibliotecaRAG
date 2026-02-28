from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


_VERSION_RE = re.compile(r"^v(\d+)$")


def get_biblioteca_root() -> Path:
    env_root = os.getenv("WORKLIB_BIBLIOTECA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "Biblioteca").exists():
        return (cwd / "Biblioteca").resolve()
    if cwd.name == "Biblioteca":
        return cwd

    here = Path(__file__).resolve().parent
    if (here / "Biblioteca").exists():
        return (here / "Biblioteca").resolve()

    worklib_root = os.getenv("WORKLIB_ROOT")
    if worklib_root:
        candidate = Path(worklib_root).expanduser().resolve()
        if candidate.name == "Biblioteca":
            return candidate
        return (candidate / "Biblioteca").resolve()

    return (cwd / "Biblioteca").resolve()


def list_versions(root: Optional[Path] = None) -> list[tuple[str, int, Path]]:
    base = (root or get_biblioteca_root()).resolve()
    found: list[tuple[str, int, Path]] = []
    if not base.exists():
        return found
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        m = _VERSION_RE.fullmatch(entry.name)
        if not m:
            continue
        found.append((entry.name, int(m.group(1)), entry.resolve()))
    return sorted(found, key=lambda row: row[1])


def latest_version(root: Optional[Path] = None) -> Optional[str]:
    versions = list_versions(root)
    return versions[-1][0] if versions else None


def latest_version_folder(root: Optional[Path] = None) -> Optional[Path]:
    versions = list_versions(root)
    return versions[-1][2] if versions else None


def latest_manifest_path(root: Optional[Path] = None) -> Optional[Path]:
    folder = latest_version_folder(root)
    if not folder:
        return None
    path = folder / "_state" / "library.json"
    return path if path.exists() else None


def resolve_manifest_path(manifest_arg: Optional[str | Path]) -> Path:
    if manifest_arg:
        return Path(manifest_arg).expanduser().resolve()

    env_manifest = os.getenv("RAG_MANIFEST_PATH") or os.getenv("WORKLIB_MANIFEST_PATH")
    if env_manifest:
        return Path(env_manifest).expanduser().resolve()

    latest = latest_manifest_path()
    if latest:
        return latest.resolve()

    # Fallback legacy layout for bootstrap/first runs.
    return (get_biblioteca_root() / "_state" / "library.json").resolve()


def next_version_number(root: Optional[Path] = None) -> int:
    versions = list_versions(root)
    return (versions[-1][1] + 1) if versions else 1


def archive_version_folder(version_folder: Path, *, keep_old_folder: bool = False) -> Path:
    version_folder = version_folder.resolve()
    archives_dir = version_folder.parent / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = archives_dir / f"{version_folder.name}_{stamp}"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(version_folder.parent), base_dir=version_folder.name))

    if not keep_old_folder:
        shutil.rmtree(version_folder)

    return archive_path
