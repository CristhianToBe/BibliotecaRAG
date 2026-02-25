from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RequestTelemetry:
    trace_id: str
    started_at: float = field(default_factory=time.perf_counter)
    stages_ms: Dict[str, float] = field(default_factory=dict)
    retrieval_ms_by_category: Dict[str, float] = field(default_factory=dict)
    retrieval_status_by_category: Dict[str, str] = field(default_factory=dict)
    model_calls: int = 0
    models: Dict[str, int] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def mark_stage(self, stage: str, elapsed_s: float) -> None:
        self.stages_ms[stage] = round(max(0.0, elapsed_s) * 1000, 2)

    def mark_retrieval_category(self, category: str, elapsed_s: float, status: str) -> None:
        self.retrieval_ms_by_category[category] = round(max(0.0, elapsed_s) * 1000, 2)
        self.retrieval_status_by_category[category] = status

    def add_event(self, stage: str, **extra: Any) -> None:
        evt: Dict[str, Any] = {
            "stage": stage,
            "at_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
        }
        evt.update(extra)
        self.events.append(evt)

    def note_model_call(self, model: str) -> None:
        self.model_calls += 1
        self.models[model] = self.models.get(model, 0) + 1

    def summary(self) -> Dict[str, Any]:
        total_ms = round((time.perf_counter() - self.started_at) * 1000, 2)
        return {
            "trace_id": self.trace_id,
            "timings_ms": dict(self.stages_ms),
            "retrieval_ms_by_category": dict(self.retrieval_ms_by_category),
            "retrieval_status_by_category": dict(self.retrieval_status_by_category),
            "model_calls": self.model_calls,
            "models": dict(self.models),
            "events": list(self.events),
            "total_ms": total_ms,
        }


_current_telemetry: contextvars.ContextVar[RequestTelemetry | None] = contextvars.ContextVar(
    "query_telemetry", default=None
)


def set_telemetry(telemetry: RequestTelemetry | None) -> contextvars.Token[RequestTelemetry | None]:
    return _current_telemetry.set(telemetry)


def reset_telemetry(token: contextvars.Token[RequestTelemetry | None]) -> None:
    _current_telemetry.reset(token)


def get_telemetry() -> RequestTelemetry | None:
    return _current_telemetry.get()

_current_stage: contextvars.ContextVar[str] = contextvars.ContextVar("query_stage", default="")
_debug_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar("query_debug_enabled", default=False)


def set_current_stage(stage_name: str) -> contextvars.Token[str]:
    return _current_stage.set(stage_name or "")


def reset_current_stage(token: contextvars.Token[str]) -> None:
    _current_stage.reset(token)


def get_current_stage() -> str:
    return _current_stage.get()


def set_debug_enabled(enabled: bool) -> contextvars.Token[bool]:
    return _debug_enabled.set(bool(enabled))


def reset_debug_enabled(token: contextvars.Token[bool]) -> None:
    _debug_enabled.reset(token)


def is_debug_enabled() -> bool:
    return bool(_debug_enabled.get())
