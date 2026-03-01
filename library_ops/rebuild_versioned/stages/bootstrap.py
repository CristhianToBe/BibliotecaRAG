from __future__ import annotations

from pathlib import Path

from worklib.store import load_manifest

from library_ops.bootstrap_manifest import run as bootstrap_manifest_run


def stage_bootstrap_manifest(manifest_path: Path, library_dir: Path) -> int:
    if not manifest_path.exists():
        print(f"⚠️ No existe manifest: {manifest_path}")
        print("↪ Intentando reconstruirlo automáticamente desde los documentos existentes...")
        rc_boot = bootstrap_manifest_run(manifest_out=manifest_path, library_dir=library_dir, force=False)
        if rc_boot != 0:
            print("❌ No fue posible reconstruir el manifest automáticamente.")
            return rc_boot
    return 0


def stage_load_manifest(manifest_path: Path):
    return load_manifest(manifest_path)
