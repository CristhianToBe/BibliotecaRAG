from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from worklib.config import default_config
from worklib.ingest import ingest_document
from worklib.query.pipeline import run_query
from worklib.query.telemetry import RequestTelemetry, reset_telemetry, set_telemetry
from worklib.store import load_manifest

load_dotenv()

PENDING_CONVERSATIONS: dict[str, dict[str, Any]] = {}
_CATEGORIES_CACHE_TTL_S = 60
_categories_cache: dict[str, Any] = {"expires_at": 0.0, "categories": []}


def _allowed_upload_exts() -> set[str]:
    raw = os.getenv("WEBAPP_UPLOAD_ALLOWED_EXTS", "pdf,txt,doc,docx")
    out = {x.strip().lower().lstrip(".") for x in raw.split(",") if x.strip()}
    return out or {"pdf"}


def _max_upload_bytes() -> int:
    raw = os.getenv("WEBAPP_UPLOAD_MAX_BYTES", "26214400")
    try:
        return max(1, int(raw))
    except Exception:
        return 26214400


def _upload_dir() -> Path:
    return Path(os.getenv("WEBAPP_UPLOAD_DIR", "uploads")).resolve()


def _cors_origins() -> list[str]:
    default_origins = ["http://127.0.0.1:8000", "http://127.0.0.1:5173", "http://localhost:5173"]
    raw = os.getenv("WEBAPP_CORS_ORIGIN", "")
    if not raw:
        return default_origins
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if "*" in parts:
        return ["*"]
    seen: set[str] = set()
    merged: list[str] = []
    for item in parts + default_origins:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged

class PipelineTimeoutsRequest(BaseModel):
    enabled: bool = True
    total_s: float = Field(default=120, ge=5, le=600)
    # debug stabilization; pick may be slow when prompt is large
    pick_s: float = Field(default=30, ge=1, le=120)
    # temporary to reduce false timeouts during debugging; adjust later
    confirm_s: float = Field(default=20, ge=1, le=120)
    refine_s: float = Field(default=20, ge=1, le=120)
    # debug stabilization; retrieval with tool-calls can be slow
    retrieve_per_cat_s: float = Field(default=30, ge=1, le=180)
    write_s: float = Field(default=45, ge=1, le=180)


class PipelineRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    use_picker: bool = Field(default=True, validation_alias=AliasChoices("use_picker", "picker", "usePicker"))
    use_confirmer: bool = Field(default=True, validation_alias=AliasChoices("use_confirmer", "useConfirmer"))
    use_refiner: bool = Field(default=True, validation_alias=AliasChoices("use_refiner", "useRefiner"))
    refine_a1: bool = True
    refine_a2: bool = True
    refine_a3: bool = True
    use_arbiter: bool = True
    use_retrieve: bool = True
    use_write: bool = True
    max_categories: int = Field(default=2, ge=1, le=3)
    top_k: int = Field(default=8, ge=1, le=30)
    max_context_chars: int = Field(default=12000, ge=1000, le=30000)
    timeouts: PipelineTimeoutsRequest | None = Field(default=None)
    unbounded: bool | None = None


class ContinueFrom(BaseModel):
    stage: str
    state_token: str | None = None
    user_reply: str = ""
    selector_instruction: str = ""
    selected_categories: list[str] = Field(default_factory=list)


class APIError(BaseModel):
    error: str
    detail: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(default="", max_length=5000)
    question: str = Field(default="", max_length=5000)
    conversation_id: str | None = None
    manifest_path: str | None = None
    max_workers: int = Field(default=3, ge=1, le=16)
    debug: bool = False
    confirm_glimpse: bool = True
    continue_from: ContinueFrom | None = None
    pipeline: PipelineRequest | None = Field(default=None)
    manual_categories: str | list[str] = Field(default="", validation_alias=AliasChoices("manual_categories", "manual_categories_str", "manualCategories"))
    use_picker: bool | None = Field(default=None, validation_alias=AliasChoices("use_picker", "picker", "usePicker"))
    use_confirmer: bool | None = Field(default=None, validation_alias=AliasChoices("use_confirmer", "useConfirmer"))
    use_refiner: bool | None = Field(default=None, validation_alias=AliasChoices("use_refiner", "useRefiner"))

    @model_validator(mode="after")
    def apply_legacy_pipeline_toggles(self) -> "ChatRequest":
        self.pipeline = self.pipeline or PipelineRequest()
        if self.use_picker is not None:
            self.pipeline.use_picker = self.use_picker
        if self.use_confirmer is not None:
            self.pipeline.use_confirmer = self.use_confirmer
        if self.use_refiner is not None:
            self.pipeline.use_refiner = self.use_refiner
        return self


class UploadResponse(BaseModel):
    status: str
    detail: dict


app = FastAPI(title="BibliotecaRAG Web API", version="0.3.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": "Payload inválido", "fields": exc.errors()})


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/api/health")
def health() -> dict:
    cfg = default_config()
    max_bytes = _max_upload_bytes()
    categories_count = 0
    categories_endpoint_ok = False
    try:
        categories_payload = list_categories()
        categories_count = len(categories_payload.get("categories", []))
        categories_endpoint_ok = isinstance(categories_payload.get("categories"), list)
    except Exception:
        categories_endpoint_ok = False

    return {
        "ok": True,
        "manifest_path": str(cfg.manifest_path),
        "library_dir": str(cfg.library_dir),
        "categories_count": categories_count,
        "categories_endpoint_ok": categories_endpoint_ok,
        "upload": {
            "allowed_exts": sorted(_allowed_upload_exts()),
            "max_upload_bytes": max_bytes,
            "max_upload_mb": round(max_bytes / (1024 * 1024), 2),
            "upload_dir": str(_upload_dir()),
        },
    }



def _log_trace(summary: dict[str, Any], *, degrade_steps: list[str] | None = None, warnings: list[str] | None = None) -> None:
    print(
        "trace_summary",
        f"trace_id={summary.get('trace_id')}",
        f"total_ms={summary.get('total_ms')}",
        f"stages_ms={summary.get('timings_ms')}",
        f"degrade_steps={degrade_steps or []}",
        f"warnings={warnings or []}",
    )


def _resolve_manifest_path(manifest_path: str | None) -> Path:
    if manifest_path:
        return Path(manifest_path).expanduser().resolve()
    return Path(os.getenv("RAG_MANIFEST_PATH") or os.getenv("WORKLIB_MANIFEST_PATH") or str(default_config().manifest_path)).resolve()


def _normalize_manual_categories(raw: str | list[str] | None) -> list[str]:
    if isinstance(raw, list):
        parts = [str(item).strip() for item in raw]
    else:
        text = str(raw or "").strip()
        if not text:
            return []
        separator = "+" if "+" in text else ","
        if "+" in text and "," in text:
            parts = [chunk.strip() for chunk in text.replace(",", "+").split("+")]
        else:
            parts = [chunk.strip() for chunk in text.split(separator)]

    out: list[str] = []
    seen: set[str] = set()
    for item in parts:
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_picker_category(raw: Any) -> str:
    return str(raw or "").split("__", 1)[0].strip().upper()


@app.get("/api/categories")
def list_categories(manifest_path: str | None = None) -> dict[str, list[str]]:
    now = time.time()
    if manifest_path is None and _categories_cache["expires_at"] > now:
        return {"categories": list(_categories_cache["categories"])}

    path = _resolve_manifest_path(manifest_path)
    manifest = load_manifest(path)
    categories = sorted([str(name).strip() for name in manifest.categories.keys() if str(name).strip()], key=str.casefold)

    if manifest_path is None:
        _categories_cache["categories"] = categories
        _categories_cache["expires_at"] = now + _CATEGORIES_CACHE_TTL_S

    return {"categories": categories}


@app.post("/api/chat")
async def chat(request: Request):
    raw_body = await request.body()
    raw_body_preview = raw_body.decode("utf-8", errors="replace")[:2000]
    try:
        req = ChatRequest.model_validate_json(raw_body or b"{}")
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": "Payload inválido", "fields": exc.errors()})

    if req.debug:
        print("chat_debug_raw_body", raw_body_preview)
        print("chat_debug_parsed", {"pipeline": req.pipeline.model_dump(), "manual_categories": req.manual_categories})

    question = (req.message or req.question or "").strip()
    if not question and not req.continue_from:
        return JSONResponse(status_code=400, content={"error": "empty_question", "details": "message o question es requerido"})

    trace_id = uuid.uuid4().hex[:12]
    telemetry = RequestTelemetry(trace_id=trace_id)
    token = set_telemetry(telemetry)

    try:
        conversation_id = (req.conversation_id or uuid.uuid4().hex)

        pipeline_use_picker = req.pipeline.use_picker if req.pipeline is not None else None
        top_level_use_picker = getattr(req, "use_picker", None)
        effective_use_picker = pipeline_use_picker if pipeline_use_picker is not None else top_level_use_picker
        if effective_use_picker is None:
            effective_use_picker = True

        raw_top_level_manual_categories = getattr(req, "manual_categories", None)
        raw_pipeline_manual_categories = getattr(req.pipeline, "manual_categories", None) if req.pipeline is not None else None
        top_level_manual_categories = _normalize_manual_categories(raw_top_level_manual_categories)
        effective_manual_categories = top_level_manual_categories

        if req.debug:
            print("[DEBUG] chat payload snapshot", {
                "pipeline": req.pipeline.model_dump() if req.pipeline else None,
                "manual_categories": raw_top_level_manual_categories,
                "pipeline_manual_categories": raw_pipeline_manual_categories,
            })
            print(
                "[DEBUG] minimum_check",
                {
                    "effective_use_picker": effective_use_picker,
                    "effective_manual_categories": effective_manual_categories,
                    "raw_top_level_manual_categories": raw_top_level_manual_categories,
                    "raw_pipeline_manual_categories": raw_pipeline_manual_categories,
                },
            )

        if (not req.continue_from) and (not effective_use_picker) and (len(effective_manual_categories) == 0):
            content = {
                "error": "missing_minimum",
                "details": "Debes activar Picker o indicar categorías manuales.",
                "trace_id": trace_id,
            }
            if req.debug:
                content["debug_received"] = {
                    "effective_use_picker": effective_use_picker,
                    "effective_manual_categories": effective_manual_categories,
                    "raw_pipeline": req.pipeline.model_dump() if req.pipeline else None,
                    "raw_top_level_manual_categories": raw_top_level_manual_categories,
                    "raw_pipeline_manual_categories": raw_pipeline_manual_categories,
                    "raw_top_level_use_picker": top_level_use_picker,
                }
            return JSONResponse(status_code=400, content=content)

        continue_confirm = bool(req.continue_from and req.continue_from.stage == "confirm_refine")
        debug_manifest_keys: list[str] = []

        pipeline_payload = req.pipeline.model_dump(exclude_none=True) if req.pipeline else PipelineRequest().model_dump(exclude_none=True)
        pipeline_payload["use_picker"] = bool(effective_use_picker)

        conversation_state = PENDING_CONVERSATIONS.get(conversation_id) if continue_confirm else None
        if continue_confirm and not conversation_state:
            return JSONResponse(status_code=409, content={"error": "invalid_continuation", "details": "No hay estado pendiente para confirm_refine", "trace_id": trace_id})
        if continue_confirm and req.continue_from and req.continue_from.state_token and req.continue_from.state_token != conversation_state.get("state_token"):
            return JSONResponse(status_code=409, content={"error": "invalid_state_token", "details": "El state_token no coincide con el estado pendiente", "trace_id": trace_id})

        result = run_query(
            question,
            manifest_path=req.manifest_path,
            max_workers=req.max_workers,
            debug=req.debug,
            continue_from=req.continue_from.model_dump() if req.continue_from else None,
            conversation_state=conversation_state,
            pipeline=pipeline_payload,
            manual_categories=effective_manual_categories,
        )

        if result.get("status") == "needs_confirmation":
            state_token = uuid.uuid4().hex[:12]
            state = dict(result.get("state") or {})
            state["state_token"] = state_token
            PENDING_CONVERSATIONS[conversation_id] = state
            return JSONResponse(status_code=200, content={
                "status": "needs_confirmation",
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "state_token": state_token,
                "question": result.get("question") or question,
                "suggested_categories": result.get("suggested_categories") or [],
                "reason": str(result.get("reason") or ""),
                "picked_curr": result.get("picked_curr") or {},
                "confirm": result.get("confirm") or {},
                "confirm_prompt": str(result.get("message") or "Confirma o ajusta las categorías sugeridas antes de continuar."),
                "message": str(result.get("message") or "Confirma o ajusta las categorías sugeridas antes de continuar."),
            })

        if req.debug:
            print("[DEBUG] /api/chat parsed_pipeline_options", {
                "trace_id": trace_id,
                "pipeline": pipeline_payload,
            })
            debug_picker = result.get("debug_picker") if isinstance(result, dict) else {}
            if not isinstance(debug_picker, dict):
                debug_picker = {}
            debug_pipeline = result.get("debug_pipeline") if isinstance(result, dict) else {}
            if not isinstance(debug_pipeline, dict):
                debug_pipeline = {}
            print("[DEBUG] /api/chat picker_vs_manifest", {
                "trace_id": trace_id,
                "manifest.categories.keys()": debug_manifest_keys,
                "picked_curr": debug_picker.get("picked_curr", {}),
                "selected_categories": debug_picker.get("selected_categories", result.get("selected_categories")),
            })
            print("[DEBUG] /api/chat pipeline_state", {
                "trace_id": trace_id,
                "picked_curr": debug_pipeline.get("picked_curr", debug_picker.get("picked_curr", {})),
                "selected_categories": debug_pipeline.get("selected_categories", result.get("selected_categories")),
                "chosen_candidate": result.get("chosen_candidate"),
                "arbiter": result.get("arbiter"),
                "variants_enabled": result.get("variants_enabled"),
                "retrieval_by_variant": result.get("retrieval_by_variant"),
                "confirm_refine_executed": debug_pipeline.get("confirm_refine_executed", {}),
                "retrieval_cache_hit": result.get("retrieval_cache_hit"),
                "all_hits_len": debug_pipeline.get("all_hits_len"),
                "timings_ms": telemetry.summary().get("timings_ms", {}),
            })
            print("[DEBUG] /api/chat pipeline result", {
                "result": result,
                "answer": result.get("answer"),
                "warnings": result.get("warnings"),
                "degrade_steps": result.get("degrade_steps"),
                "references": result.get("references"),
                "retrieval_cache_hit": result.get("retrieval_cache_hit"),
                "effective_use_picker": effective_use_picker,
                "effective_manual_categories": effective_manual_categories,
                "raw_top_level_manual_categories": raw_top_level_manual_categories,
                "raw_pipeline_manual_categories": raw_pipeline_manual_categories,
                "manifest.categories.keys()": debug_manifest_keys,
                "picked_curr": debug_picker.get("picked_curr", {}),
                "selected_categories": result.get("selected_categories"),
            })

        PENDING_CONVERSATIONS.pop(conversation_id, None)
        summary = telemetry.summary()
        warnings = list(result.get("warnings", []) or [])
        degrade_steps = list(result.get("degrade_steps", []) or [])
        _log_trace(summary, degrade_steps=degrade_steps, warnings=warnings)

        answer_text = str(result.get("answer", ""))
        response_payload = {
            "status": str(result.get("status") or "ok"),
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "answer": answer_text,
            "selected_categories": list(result.get("selected_categories", []) or []),
            "references": list(result.get("references", []) or []),
            "warnings": warnings,
            "timings_ms": summary.get("timings_ms", {}),
        }
        if not answer_text.strip():
            response_payload["no_answer_reason"] = {
                "message": "No se pudo generar respuesta: revisa las categorías y la evidencia",
                "warnings": warnings,
                "degrade_steps": degrade_steps,
            }
        if req.debug:
            response_payload["debug_info"] = summary
        return JSONResponse(status_code=200, content=response_payload)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": "chat_failed", "details": str(exc), "trace_id": trace_id})
    finally:
        reset_telemetry(token)


@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    author: str = Form(default=""),
    tags: str = Form(default=""),
    copy_to_library: bool = Form(default=True),
    debug: bool = Form(default=False),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail={"error": "missing_filename", "detail": "El archivo no tiene nombre"})

    ext = Path(file.filename).suffix.lower().lstrip(".")
    allowed = _allowed_upload_exts()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail={"error": "invalid_extension", "detail": f"Extensión no permitida: .{ext}", "allowed_extensions": sorted(allowed)})

    content = await file.read()
    size = len(content)
    max_bytes = _max_upload_bytes()
    if size <= 0:
        raise HTTPException(status_code=400, detail={"error": "empty_file", "detail": "El archivo está vacío"})
    if size > max_bytes:
        raise HTTPException(status_code=413, detail={"error": "file_too_large", "detail": f"Archivo excede límite de {max_bytes} bytes", "file_size": size, "max_upload_bytes": max_bytes})

    upload_dir = _upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f"tmp_{uuid.uuid4().hex}_{Path(file.filename).name}"
    tmp_path.write_bytes(content)

    try:
        result = ingest_document(tmp_path, title=title, author=author, tags=[t.strip() for t in tags.split(",") if t.strip()], copy_to_library=copy_to_library, debug=debug)
        return UploadResponse(status="ok", detail=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "file_not_found", "detail": str(exc)}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail={"error": "ingest_rejected", "detail": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "upload_failed", "detail": f"Error ingestando documento: {exc}"}) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
