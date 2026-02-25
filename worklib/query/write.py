from __future__ import annotations

import json
from typing import Any, Dict, List

from worklib.prompt_loader import load_prompt
from .llm import call_text, eprint, MODEL_ANSWER

SYSTEM_WRITER = load_prompt("query_writer_system")


def write_answer(question: str, evidence: List[Dict[str, Any]], *, debug: bool = False) -> str:
    payload = {
        "question": question,
        "evidence": [
            {
                "file_id": e.get("file_id"),
                "filename": e.get("filename"),
                "local_path": e.get("local_path"),
                "score": e.get("score"),
                "text": e.get("text"),
            }
            for e in evidence
        ],
    }
    resp = call_text(MODEL_ANSWER, SYSTEM_WRITER, json.dumps(payload, ensure_ascii=False), debug=debug)
    txt = (resp.output_text or "").strip()
    if debug:
        eprint("\n[DEBUG] writer output_text:")
        eprint(txt[:2000])
    return txt