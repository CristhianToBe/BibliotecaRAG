from __future__ import annotations

import os
from dataclasses import dataclass

@dataclass(frozen=True)
class OpenAISettings:
    api_key: str | None
    timeout_s: float
    max_retries: int

    ingest_model: str
    query_model_fast: str
    query_model_smart: str

def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)

def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)

def _get_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()

def get_openai_settings() -> OpenAISettings:
    # api_key puede venir del entorno estándar OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY")

    # Timeouts / retries
    timeout_s = _get_float("OPENAI_TIMEOUT_S", 60.0)
    max_retries = _get_int("OPENAI_MAX_RETRIES", 3)

    # Modelos (mantengo compatibilidad con los nombres que ya venías usando)
    ingest_model = _get_str("RAG_INGEST_MODEL", _get_str("INGEST_MODEL", "gpt-5-nano"))
    query_model_fast = _get_str("RAG_MODEL_FAST", "gpt-5-nano")
    query_model_smart = _get_str("RAG_MODEL_SMART", "gpt-5")

    return OpenAISettings(
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries=max_retries,
        ingest_model=ingest_model,
        query_model_fast=query_model_fast,
        query_model_smart=query_model_smart,
    )
