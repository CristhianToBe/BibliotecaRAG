from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from worklib.config import default_config
from worklib.ingest import ingest_document
from worklib.query.pipeline import pro_query_non_interactive

load_dotenv()


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


class APIError(BaseModel):
    error: str
    detail: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    manifest_path: str | None = None
    max_workers: int = Field(default=3, ge=1, le=16)
    debug: bool = False
    confirm: bool = True
    confirm_rounds: int = Field(default=4, ge=0, le=10)
    confirm_glimpse: bool = True


class DocReference(BaseModel):
    doc_id: str
    filename: str
    abs_path: str


class ChatResponse(BaseModel):
    answer: str
    references: list[DocReference]
    selected_categories: list[str] | None = None
    debug_info: dict | None = None


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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail={"error": "invalid_question", "detail": "question no puede estar vacía"})

    try:
        started_at = time.perf_counter()
        result = pro_query_non_interactive(
            q,
            manifest_path=req.manifest_path,
            max_workers=req.max_workers,
            debug=req.debug,
            confirm=req.confirm,
            confirm_rounds=req.confirm_rounds,
            confirm_glimpse=req.confirm_glimpse,
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "manifest_not_found", "detail": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "chat_failed", "detail": f"Error consultando RAG: {exc}"}) from exc

    answer = str(result.get("answer", ""))
    references = list(result.get("references", []) or [])
    selected_categories = list(result.get("selected_categories", []) or [])

    if req.debug:
        debug_info = {
            "timings": {
                "total_ms": elapsed_ms,
            },
            "selected_categories": selected_categories,
            "references": references,
            "pipeline": {
                k: v
                for k, v in result.items()
                if k not in {"answer", "references", "selected_categories"}
            },
        }
        return ChatResponse(
            answer=answer,
            references=references,
            selected_categories=selected_categories,
            debug_info=debug_info,
        )

    return ChatResponse(answer=answer, references=references)


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
