from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict

from dotenv import load_dotenv
from openai import APITimeoutError, APIConnectionError, RateLimitError

from worklib.openai_client import get_client
from worklib.prompt_loader import load_prompt
from worklib.settings import get_openai_settings
from .telemetry import get_current_stage, get_telemetry, is_debug_enabled

load_dotenv()

client = get_client()
_settings = get_openai_settings()

MODEL_FAST = _settings.query_model_fast
MODEL_SMART = _settings.query_model_smart

MODEL_PICK = os.getenv("MODEL_PICK", "gpt-5-nano")
MODEL_CONFIRM = os.getenv("MODEL_CONFIRM", "gpt-5-nano")
MODEL_REFINE = os.getenv("MODEL_REFINE", "gpt-5-nano")
MODEL_ANSWER = os.getenv("MODEL_ANSWER", "gpt-5-mini")
MODEL_ARBITRATE = os.getenv("MODEL_ARBITRATE", MODEL_ANSWER)

SYSTEM_CONFIRM = load_prompt("confirm_system")


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def now_ms() -> int:
    return int(time.time() * 1000)


def clip(s: str, n: int = 400) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def safe_get(d: Any, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def call_text(
    model: str,
    system: str,
    user: str,
    *,
    tools=None,
    tool_choice=None,
    include=None,
    debug: bool = False,
) -> Any:
    telemetry = get_telemetry()
    stage_name = get_current_stage() or "unknown"
    debug_active = bool(debug or is_debug_enabled())
    if debug_active:
        eprint("[DEBUG] TRACE_CTX", {
            "trace_id": getattr(telemetry, "trace_id", "no-trace") if telemetry else "no-trace",
            "stage_name": stage_name,
            "where": "call_text_enter",
        })
    if telemetry is not None:
        telemetry.note_model_call(model)

    kwargs: Dict[str, Any] = dict(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if include is not None:
        kwargs["include"] = include

    if debug:
        eprint("\n[DEBUG] call_text kwargs (resumen):")
        eprint("  model:", model)
        eprint("  tool_choice:", tool_choice)
        eprint("  include:", include)
        if tools:
            eprint("  tools:", jdump(tools)[:1200])
        eprint("  user_len:", len(user or ""))

    attempts = 4
    base_sleep = 2.0
    for i in range(1, attempts + 1):
        call_started = time.perf_counter()
        try:
            resp = client.responses.create(**kwargs)
            if debug_active:
                eprint("[DEBUG] LLM_CALL", {
                    "trace_id": getattr(telemetry, "trace_id", "no-trace") if telemetry else "no-trace",
                    "stage_name": stage_name,
                    "model": model,
                    "elapsed_ms": round((time.perf_counter() - call_started) * 1000, 2),
                    "status": "ok",
                    "attempt": i,
                })
            return resp
        except (APITimeoutError, APIConnectionError, RateLimitError) as ex:
            if debug_active:
                eprint("[DEBUG] LLM_CALL", {
                    "trace_id": getattr(telemetry, "trace_id", "no-trace") if telemetry else "no-trace",
                    "stage_name": stage_name,
                    "model": model,
                    "elapsed_ms": round((time.perf_counter() - call_started) * 1000, 2),
                    "status": f"err:{type(ex).__name__}",
                    "attempt": i,
                })
            wait = base_sleep * (2 ** (i - 1)) + random.random()
            eprint(
                f"[WARN] OpenAI call failed ({type(ex).__name__}) attempt {i}/{attempts}. "
                f"Waiting {wait:.1f}s..."
            )
            if i == attempts:
                raise
            time.sleep(wait)
