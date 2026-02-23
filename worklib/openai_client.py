from __future__ import annotations

from openai import OpenAI
from worklib.settings import get_openai_settings

_client: OpenAI | None = None

def get_client() -> OpenAI:
    """
    Cliente singleton con timeout/retries centralizados.
    """
    global _client
    if _client is None:
        s = get_openai_settings()
        # El SDK toma OPENAI_API_KEY del env automáticamente, pero lo dejamos listo si quieres inyectar.
        _client = OpenAI(
            api_key=s.api_key,        # puede ser None -> usa env
            timeout=s.timeout_s,
            max_retries=s.max_retries,
        )
    return _client
