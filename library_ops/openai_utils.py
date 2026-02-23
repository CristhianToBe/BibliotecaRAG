# openai_utils.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from worklib.openai_client import get_client as wl_get_client  # 👈 fuente única

JsonDict = Dict[str, Any]


def llm_json(
    *,
    model: str,
    system: str,
    user: str,
    client: Optional[OpenAI] = None,
    max_tries: Optional[int] = None,
    base_sleep_s: Optional[float] = None,
) -> JsonDict:
    """
    Robust JSON call: Responses API preferred, fallback to chat.completions + backoff.
    Usa el cliente singleton de worklib.openai_client.get_client() si no se pasa client.
    """
    client = client or wl_get_client()

    max_tries = int(os.getenv("LLM_MAX_TRIES", str(max_tries or 4)))
    base_sleep_s = float(os.getenv("LLM_RETRY_SLEEP_S", str(base_sleep_s or 2.0)))

    last_err: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            # 1) Responses API (preferido)
            try:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                out = getattr(resp, "output_text", None)
                if not out:
                    out = resp.output[0].content[0].text  # type: ignore
                return json.loads(out)
            except Exception:
                # 2) Fallback Chat Completions
                comp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                return json.loads(comp.choices[0].message.content)  # type: ignore

        except Exception as e:
            last_err = e
            name = e.__class__.__name__.lower()
            is_transientish = ("timeout" in name) or ("rate" in name) or ("connection" in name)

            if attempt >= max_tries or not is_transientish:
                break

            sleep_s = base_sleep_s * (2 ** (attempt - 1))
            print(
                f"⏳ Transient ({e.__class__.__name__}) modelo={model} "
                f"intento {attempt}/{max_tries}. Reintento en {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)

    assert last_err is not None
    raise last_err


def chunked(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def create_vector_store(client: OpenAI, name: str) -> str:
    vs = client.vector_stores.create(name=name)
    return vs.id


def attach_files(
    client: OpenAI,
    vector_store_id: str,
    file_ids: List[str],
    *,
    batch_size: int = 200,
    debug: bool = False,
) -> None:
    for batch in chunked(file_ids, batch_size):
        if debug:
            print(f"  - adjuntando batch: {len(batch)} archivos")
        client.vector_stores.file_batches.create(vector_store_id=vector_store_id, file_ids=batch)


def upload_file(client: OpenAI, abs_path: str, *, debug: bool = False) -> str:
    p = Path(abs_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {abs_path}")
    if debug:
        print(f"  - subiendo a OpenAI: {p.name}")
    with p.open("rb") as f:
        up = client.files.create(file=f, purpose="assistants")
    return up.id


def get_vs_file_text(
    client: OpenAI,
    vector_store_id: str,
    file_id: str,
    *,
    max_chars: int = 6000,
) -> str:
    """Best-effort: SDK differences."""
    try:
        content = client.vector_stores.files.content(vector_store_id=vector_store_id, file_id=file_id)
        if isinstance(content, str):
            return content[:max_chars]
        return json.dumps(content, ensure_ascii=False)[:max_chars]
    except Exception:
        try:
            content = client.vector_stores.files.retrieve_content(vector_store_id=vector_store_id, file_id=file_id)
            return json.dumps(content, ensure_ascii=False)[:max_chars]
        except Exception:
            return ""