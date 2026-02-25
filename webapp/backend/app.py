from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from worklib.config import default_config
from worklib.ingest import ingest_document
from worklib.query.pipeline import run_after_confirm, run_pick_confirm

load_dotenv()

PENDING_CONVERSATIONS: dict[str, dict[str, Any]] = {}


def _allowed_upload_exts() -> set[str]:
    raw = os.getenv("WEBAPP_UPLOAD_ALLOWED_EXTS", "pdf,txt,doc,docx")
    out = {x.strip().lower().lstrip(".") for x in raw.split(",") if x.strip()}
    return out or {"pdf"}


def _max_upload_bytes() -> int:
    raw = os.getenv("WEBAPP_UPLOAD_MAX_BYTES", "26214400")  # 25 MB
    try:
        return max(1, int(raw))
    except Exception:
        return 26214400


def _upload_dir() -> Path:
    return Path(os.getenv("WEBAPP_UPLOAD_DIR", "uploads")).resolve()


def _chat_timeout_seconds() -> float:
    raw = os.getenv("WEBAPP_CHAT_TIMEOUT_SECONDS", "90")
    try:
        return max(1.0, float(raw))
    except Exception:
        return 90.0


class ContinueFrom(BaseModel):
    stage: str
    user_reply: str = ""
    selector_instruction: str = ""
    selected_categories: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=5000)
    question: str = Field(default="", max_length=5000)
    conversation_id: str | None = None
    manifest_path: str | None = None
    max_workers: int = Field(default=3, ge=1, le=16)
    debug: bool = False
    confirm_glimpse: bool = True
    continue_from: ContinueFrom | None = None


class UploadResponse(BaseModel):
    status: str
    detail: dict


app = FastAPI(title="BibliotecaRAG Web API", version="0.2.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": "Payload inválido",
            "fields": exc.errors(),
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("WEBAPP_CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/api/health")
def health() -> dict:
    cfg = default_config()
    max_bytes = _max_upload_bytes()
    return {
        "ok": True,
        "manifest_path": str(cfg.manifest_path),
        "library_dir": str(cfg.library_dir),
        "upload": {
            "allowed_exts": sorted(_allowed_upload_exts()),
            "max_upload_bytes": max_bytes,
            "max_upload_mb": round(max_bytes / (1024 * 1024), 2),
            "upload_dir": str(_upload_dir()),
        },
        "env": {
            "WORKLIB_MANIFEST_PATH": os.getenv("WORKLIB_MANIFEST_PATH", ""),
            "WORKLIB_LIBRARY_DIR": os.getenv("WORKLIB_LIBRARY_DIR", ""),
            "RAG_MANIFEST_PATH": os.getenv("RAG_MANIFEST_PATH", ""),
        },
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    question = (req.message or req.question or "").strip()
    timeout_s = _chat_timeout_seconds()
    started_at = time.perf_counter()

    if req.continue_from and req.continue_from.stage == "confirm_refine":
        conversation_id = (req.conversation_id or "").strip()
        if not conversation_id:
            return JSONResponse(status_code=400, content={"error": "missing_conversation_id", "details": "conversation_id es requerido"})

        pending = PENDING_CONVERSATIONS.get(conversation_id)
        if not pending or pending.get("stage") != "confirm_refine":
            return JSONResponse(status_code=409, content={"error": "invalid_continuation", "details": "No hay estado pendiente para confirm_refine"})

        if req.debug:
            print(f"[DEBUG] /api/chat continuation received conversation_id={conversation_id}")

        base_question = pending.get("last_question") or question
        selected_categories = req.continue_from.selected_categories or pending.get("selected_categories") or []
        selector_instruction = req.continue_from.selector_instruction or pending.get("selector_instruction") or ""
        user_reply = req.continue_from.user_reply or ""

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    run_after_confirm,
                    base_question,
                    selected_categories=selected_categories,
                    selector_instruction=selector_instruction,
                    user_reply=user_reply,
                    manifest_path=pending.get("manifest_path") or req.manifest_path,
                    max_workers=req.max_workers,
                    debug=req.debug,
                    picked=pending.get("picked"),
                )
                result = fut.result(timeout=timeout_s)
        except FuturesTimeoutError:
            return JSONResponse(status_code=504, content={"error": "chat_timeout", "details": f"chat pipeline exceeded timeout of {timeout_s} seconds"})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": "chat_failed", "details": str(exc)})
        finally:
            PENDING_CONVERSATIONS.pop(conversation_id, None)

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        payload = {
            "status": "ok",
            "conversation_id": conversation_id,
            "answer": str(result.get("answer", "")),
            "selected_categories": list(result.get("selected_categories", []) or []),
            "references": list(result.get("references", []) or []),
        }
        if req.debug:
            payload["debug_info"] = {"timings": {"total_ms": elapsed_ms, "timeout_s": timeout_s, "stage": "confirm_refine_continue"}}
        return JSONResponse(status_code=200, content=payload)

    if not question:
        return JSONResponse(status_code=400, content={"error": "invalid_question", "details": "message no puede estar vacía"})

    conversation_id = (req.conversation_id or uuid.uuid4().hex).strip()

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                run_pick_confirm,
                question,
                manifest_path=req.manifest_path,
                debug=req.debug,
                confirm_glimpse=req.confirm_glimpse,
            )
            pre = fut.result(timeout=timeout_s)
    except FuturesTimeoutError:
        return JSONResponse(status_code=504, content={"error": "chat_timeout", "details": f"chat pipeline exceeded timeout of {timeout_s} seconds"})
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"error": "manifest_not_found", "details": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": "chat_failed", "details": str(exc)})

    confirm = dict(pre.get("confirm") or {})
    picked = dict(pre.get("picked") or {})
    action = str(confirm.get("action") or "REFINE").upper()
    categories_final = list(confirm.get("categories_final") or [])
    suggested_categories = list(confirm.get("suggested_categories") or [])
    selected_categories = categories_final or suggested_categories or list((picked.get("selected") or [])[:2])

    if req.debug:
        print(f"[DEBUG] confirm.action={action}")

    if action == "REFINE":
        PENDING_CONVERSATIONS[conversation_id] = {
            "stage": "confirm_refine",
            "prompt": str(confirm.get("message_to_user") or ""),
            "selector_instruction": str(confirm.get("selector_instruction") or ""),
            "selected_categories": selected_categories,
            "last_question": question,
            "manifest_path": pre.get("manifest_path") or req.manifest_path,
            "picked": picked,
            "updated_at": time.time(),
        }
        if req.debug:
            print(f"[DEBUG] returning needs_user conversation_id={conversation_id}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "needs_user",
                "stage": "confirm_refine",
                "conversation_id": conversation_id,
                "prompt": str(confirm.get("message_to_user") or "Necesito más detalle para ajustar la búsqueda."),
                "selector_instruction": str(confirm.get("selector_instruction") or ""),
                "selected_categories": selected_categories,
                "confidence": float(confirm.get("confidence") or 0.0),
            },
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                run_after_confirm,
                question,
                selected_categories=selected_categories,
                manifest_path=pre.get("manifest_path") or req.manifest_path,
                max_workers=req.max_workers,
                debug=req.debug,
                picked=picked,
            )
            result = fut.result(timeout=timeout_s)
    except FuturesTimeoutError:
        return JSONResponse(status_code=504, content={"error": "chat_timeout", "details": f"chat pipeline exceeded timeout of {timeout_s} seconds"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": "chat_failed", "details": str(exc)})

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    payload = {
        "status": "ok",
        "conversation_id": conversation_id,
        "answer": str(result.get("answer", "")),
        "selected_categories": list(result.get("selected_categories", []) or []),
        "references": list(result.get("references", []) or []),
    }
    if req.debug:
        payload["debug_info"] = {"timings": {"total_ms": elapsed_ms, "timeout_s": timeout_s, "stage": "complete"}}

    return JSONResponse(status_code=200, content=payload)


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
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_extension",
                "detail": f"Extensión no permitida: .{ext}",
                "allowed_extensions": sorted(allowed),
            },
        )

    content = await file.read()
    size = len(content)
    max_bytes = _max_upload_bytes()
    if size <= 0:
        raise HTTPException(status_code=400, detail={"error": "empty_file", "detail": "El archivo está vacío"})
    if size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "detail": f"Archivo excede límite de {max_bytes} bytes",
                "file_size": size,
                "max_upload_bytes": max_bytes,
            },
        )

    upload_dir = _upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f"tmp_{uuid.uuid4().hex}_{Path(file.filename).name}"
    tmp_path.write_bytes(content)

    try:
        result = ingest_document(
            tmp_path,
            title=title,
            author=author,
            tags=[t.strip() for t in tags.split(",") if t.strip()],
            copy_to_library=copy_to_library,
            debug=debug,
        )
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
