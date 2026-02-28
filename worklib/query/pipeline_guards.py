from __future__ import annotations

from typing import Any, Dict


def manifest_not_loaded_response(*, trace_id: str, manifest_path: str, manifest_error: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "error": "MANIFEST_NOT_LOADED",
        "details": "NO_VALID_CATEGORIES: verifica la ruta de manifest/library y su contenido.",
        "trace_id": trace_id,
        "manifest_path": str(manifest_path),
        "debug": {
            "manifest_error": manifest_error,
            "valid_categories_len": 0,
        },
    }
