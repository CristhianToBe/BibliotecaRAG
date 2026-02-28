from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from worklib.versioning import get_biblioteca_root, resolve_manifest_path


@dataclass(frozen=True)
class Config:
    root: Path
    library_dir: Path
    state_dir: Path
    manifest_path: Path


def _env_path(key: str) -> Path | None:
    v = os.getenv(key)
    if not v:
        return None
    return Path(v).expanduser()


def default_config() -> Config:
    """Default configuration (robusto sin depender del cwd).

    Prioridad:
    1) WORKLIB_ROOT (si existe)
    2) Si existe carpeta 'Biblioteca' al lado de este archivo, úsala como root
    3) Si no, usa Path.cwd()
    """

    root = (_env_path("WORKLIB_ROOT") or get_biblioteca_root()).resolve()
    library_dir = (_env_path("WORKLIB_LIBRARY_DIR") or (root / "biblioteca")).resolve()
    state_dir = (_env_path("WORKLIB_STATE_DIR") or (root / "_state")).resolve()
    manifest_path = resolve_manifest_path(_env_path("WORKLIB_MANIFEST_PATH"))

    return Config(root=root, library_dir=library_dir, state_dir=state_dir, manifest_path=manifest_path)

# Reglas suaves: “trabajo”
WORK_ONLY_HINT = [
    "DIAN", "Estatuto Tributario", "Consejo de Estado", "Sección Cuarta",
    "CTCP", "Supersociedades", "Superfinanciera", "NIIF", "IFRS",
    "fiscalización", "retención", "IVA", "renta", "GMF", "exógena",
]
