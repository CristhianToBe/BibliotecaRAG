from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, DefaultDict
from collections import defaultdict

from ._time import now_stamp

JsonDict = Dict[str, Any]

def safe_json_load(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))

def safe_json_dump(path: Path, obj: JsonDict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def backup_file(path: Path) -> Path:
    bk = path.with_suffix(path.suffix + f".bak_{now_stamp()}")
    bk.write_bytes(path.read_bytes())
    return bk

def index_docs_by_category(docs: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    by_cat: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in (docs or {}).values():
        c = (d.get("category") or "").strip()
        by_cat[c].append(d)
    return dict(by_cat)
