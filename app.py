# """
# Enhanced FastAPI application with comprehensive error handling and logging.
# """

# import os
# import logging
# import time
# import traceback
# import json
# from pathlib import Path
# from dotenv import load_dotenv
# from datetime import datetime

# from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# from chat.chat import router as chat_router
# from chat.knowledge_base import router as kb_router
# from chat.knowledge_base import process_kb_entry
# from dummy_chat.dummy_kb_ingest import ingest_kb
# from dummy_chat.dummy_kb_retrieve import retrieve_kb_chunks
# from dummy_chat.dummy_kb_answer import answer_from_kb

# # ═══════════════════════════════════════════════════════════════════
# # LOAD ENVIRONMENT
# # ═══════════════════════════════════════════════════════════════════
# env_path = Path(__file__).parent / ".env"
# load_dotenv(env_path, override=True)

# # ═══════════════════════════════════════════════════════════════════
# # LOGGING SETUP
# # ═══════════════════════════════════════════════════════════════════
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# logging.basicConfig(
#     level=getattr(logging, LOG_LEVEL, logging.INFO),
#     format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# log = logging.getLogger("app")
# log.info("=" * 80)
# log.info("🚀 APPLICATION STARTUP")
# log.info("=" * 80)
# log.info("Environment file: %s", env_path)
# log.info("Log level: %s", LOG_LEVEL)

# # ═══════════════════════════════════════════════════════════════════
# # APP INITIALIZATION
# # ═══════════════════════════════════════════════════════════════════
# app = FastAPI(
#     title="AI Calling API with RAG",
#     version="2.0.0",
#     description="Enhanced chatbot with vector search knowledge base",
# )

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ═══════════════════════════════════════════════════════════════════
# # REQUEST LOGGING MIDDLEWARE (Detailed + Safe Async Response)
# # ═══════════════════════════════════════════════════════════════════


# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     """Logs all HTTP requests and responses with detailed trace output."""
#     request_id = f"{int(time.time() * 1000)}"
#     client_ip = request.client.host
#     path = request.url.path
#     method = request.method

#     log.info("→ REQUEST | id=%s | method=%s | path=%s | client=%s",
#              request_id, method, path, client_ip)

#     start_time = time.perf_counter()

#     try:
#         response = await call_next(request)
#         duration_ms = (time.perf_counter() - start_time) * 1000

#         # Capture and log the raw response body safely
#         body_text = ""
#         if hasattr(response, "body_iterator"):
#             body_content = b""
#             async for chunk in response.body_iterator:
#                 body_content += chunk
#             body_text = body_content.decode("utf-8", errors="ignore").strip()

#             # ✅ FIXED: restore async body iterator properly
#             async def new_body_iterator():
#                 yield body_content
#             response.body_iterator = new_body_iterator()
#             response.headers["content-length"] = str(len(body_content))

#         else:
#             body_content = getattr(response, "body", b"")
#             if isinstance(body_content, str):
#                 body_text = body_content
#                 body_content = body_content.encode("utf-8")
#             else:
#                 body_text = body_content.decode(
#                     "utf-8", errors="ignore").strip()
#             response.headers["content-length"] = str(len(body_content))

#         log.info("← RESPONSE | id=%s | status=%d | time=%.1fms",
#                  request_id, response.status_code, duration_ms)
#         log.info("  ▸ Raw Response Body:")
#         for line in body_text.splitlines():
#             log.info("    %s", line)

#         return response

#     except Exception as e:
#         duration_ms = (time.perf_counter() - start_time) * 1000
#         log.error("✖ REQUEST FAILED | id=%s | time=%.1fms | error=%s",
#                   request_id, duration_ms, str(e))
#         raise

# # ═══════════════════════════════════════════════════════════════════
# # GLOBAL EXCEPTION HANDLER
# # ═══════════════════════════════════════════════════════════════════


# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     """Catch-all exception handler with detailed logging."""
#     log.error("=" * 80)
#     log.error("UNHANDLED EXCEPTION")
#     log.error("=" * 80)
#     log.error("Path: %s %s", request.method, request.url.path)
#     log.error("Error: %s", str(exc))
#     log.error("Traceback:\n%s", traceback.format_exc())
#     log.error("=" * 80)

#     return JSONResponse(
#         status_code=500,
#         content={
#             "ok": False,
#             "error": "Internal server error",
#             "message": str(exc),
#             "path": str(request.url.path),
#             "timestamp": datetime.utcnow().isoformat() + "Z",
#         }
#     )

# # ═══════════════════════════════════════════════════════════════════
# # INCLUDE ROUTERS
# # ═══════════════════════════════════════════════════════════════════
# log.info("Loading routers...")

# try:
#     app.include_router(chat_router, prefix="/chat", tags=["Chat"])
#     app.include_router(kb_router)
#     log.info("✓ Chat router loaded")
# except Exception as e:
#     log.error("✗ Failed to load chat router: %s", e)
#     raise

# # Optional: Voice router (if available)
# try:
#     from voice.voice_router import router as voice_router
#     app.include_router(voice_router, tags=["Voice"])
#     log.info("✓ Voice router loaded")
# except ImportError:
#     log.warning("⚠ Voice router not available (optional)")
# except Exception as e:
#     log.error("✗ Failed to load voice router: %s", e)

# # ═══════════════════════════════════════════════════════════════════
# # MODELS
# # ═══════════════════════════════════════════════════════════════════


# class KBRequest(BaseModel):
#     format: str
#     status: str | None = None
#     data: dict | str | None = None
#     filename: str | None = None

# # ═══════════════════════════════════════════════════════════════════
# # KNOWLEDGE BASE ENDPOINT
# # ═══════════════════════════════════════════════════════════════════


# @app.post("/kb/train", tags=["Knowledge Base"])
# async def kb_train(
#     request: Request,
#     format: str | None = Form(None),
#     status: str | None = Form(None),
#     file: UploadFile | None = None,
#     user_id: str | None = Form(None),
#     workspace_id: str | None = Form(None),
# ):
#     """Upload knowledge base content in various formats."""
#     log.info("=" * 80)
#     log.info("📚 KB TRAINING REQUEST")
#     log.info("=" * 80)

#     ctype = request.headers.get("content-type", "")
#     log.info("Content-Type: %s", ctype)

#     try:
#         # Handle multipart/form-data
#         if ctype and "multipart/form-data" in ctype:
#             if not format:
#                 form = await request.form()
#                 format = (form.get("format") or "").strip()
#                 status = status or form.get("status")
#                 # data might be in form field 'data'
#                 data_field = form.get("data")
#                 # if user_id/workspace_id were not in Form(...) signature, try form
#                 user_id = user_id or form.get("user_id")
#                 workspace_id = workspace_id or form.get("workspace_id")
#             else:
#                 # signature-supplied form values exist; still read `data` from form for safety
#                 form = await request.form()
#                 data_field = form.get("data") if form else None

#             fmt = format.lower().strip()
#             log.info("Format: %s (multipart)", fmt)

#             if fmt not in {"qna", "file", "link"}:
#                 raise HTTPException(
#                     400, "Invalid format: must be qna, file, or link")

#             if not user_id:
#                 raise HTTPException(400, "user_id is required")

#             # ⚙️ Simulate role (user_id/workspace_id come from request)
#             user_role = "admin"  # will come from JWT/session in future

#             if fmt == "file":
#                 if not file:
#                     raise HTTPException(
#                         400, "No file uploaded for format=file")
#                 log.info("Processing file: %s", file.filename)
#                 result = await process_kb_entry(
#                     "file",
#                     status=status,
#                     file=file,
#                     user_id=user_id,
#                     workspace_id=workspace_id,
#                     user_role=user_role,
#                 )
#             else:
#                 log.info("Processing %s with data: %s",
#                          fmt, str(data_field)[:100])
#                 result = await process_kb_entry(
#                     fmt,
#                     data=data_field,
#                     status=status,
#                     user_id=user_id,
#                     workspace_id=workspace_id,
#                     user_role=user_role,
#                 )

#             log.info("✓ KB training completed successfully")
#             return result

#         # Handle JSON body
#         raw = await request.body()
#         log.debug("Raw request body bytes: %r", raw)

#         if not raw:
#             log.error("Empty request body for JSON branch")
#             raise HTTPException(400, "Empty request body: expected JSON")

#         try:
#             body_text = raw.decode("utf-8")
#         except Exception as e:
#             log.error("Failed to decode request body: %s", e)
#             raise HTTPException(400, "Unable to decode request body as UTF-8")

#         try:
#             body = json.loads(body_text)
#         except Exception as e:
#             log.error("Invalid JSON body: %s", e)
#             # include a short hint for Windows cmd users who may need to escape pipes
#             raise HTTPException(
#                 400, "Invalid JSON body: ensure Content-Type: application/json and valid JSON (on Windows cmd, escape '|' as ^| or use a payload file)")

#         log.info("Processing JSON body")
#         fmt = (body.get("format") or "").lower().strip()

#         if fmt not in {"qna", "file", "link"}:
#             raise HTTPException(400, "Invalid or missing 'format' in JSON")

#         status = body.get("status")
#         data = body.get("data")
#         user_id = body.get("user_id") or user_id
#         workspace_id = body.get("workspace_id") or workspace_id

#         if not user_id:
#             raise HTTPException(400, "user_id is required")

#         if fmt == "file":
#             raise HTTPException(
#                 400,
#                 "format=file requires multipart/form-data with a 'file' field"
#             )

#         log.info("Processing %s with data: %s", fmt, str(data)[:100])
#         user_role = "admin"

#         result = await process_kb_entry(
#             fmt,
#             data=data,
#             status=status,
#             user_id=user_id,
#             workspace_id=workspace_id,
#             user_role=user_role,
#         )

#         log.info("✓ KB training completed successfully")
#         return result

#     except HTTPException:
#         raise
#     except Exception as e:
#         log.error("KB training failed: %s\n%s", e, traceback.format_exc())
#         raise HTTPException(500, f"Training error: {e}")


# @app.post("/kb/v2/train")
# async def kb_v2_train(
#     workspace_id: str = Form(...),
#     source_type: str = Form(...),  # file | link | qna
#     file: UploadFile | None = File(None),
#     link: str | None = Form(None),
#     question: str | None = Form(None),
#     answer: str | None = Form(None),
# ):
#     """
#     Knowledge Base ingestion v2
#     Supports:
#     - file (pdf, txt, docx)
#     - link
#     - qna
#     """
#     return await ingest_kb(
#         workspace_id=workspace_id,
#         source_type=source_type,
#         file=file,
#         link=link,
#         question=question,
#         answer=answer,
#     )


# @app.post("/kb/v2/query")
# async def kb_v2_query(
#     workspace_id: str = Form(...),
#     query: str = Form(...),
# ):
#     chunks = retrieve_kb_chunks(
#         workspace_id=workspace_id,
#         query=query,
#     )

#     if not chunks:
#         return {
#             "ok": True,
#             "answer": "This information is not available in the knowledge base.",
#             "sources": [],
#         }

#     answer = answer_from_kb(
#         query=query,
#         chunks=[c["text"] for c in chunks],
#     )

#     return {
#         "ok": True,
#         "answer": answer,
#         "sources": [c["public_id"] for c in chunks],
#     }


# # ═══════════════════════════════════════════════════════════════════
# # ROOT & HEALTH ENDPOINTS
# # ═══════════════════════════════════════════════════════════════════


# @app.get("/", tags=["System"])
# def root():
#     return {
#         "ok": True,
#         "service": "AI Calling API with RAG",
#         "version": "2.0.0",
#         "routes": {
#             "chat": "/chat",
#             "chat_stream": "/chat/stream",
#             "chat_history": "/chat/history",
#             "chat_reset": "/chat/reset",
#             "kb_train": "/kb/train",
#             "diagnostics": {
#                 "ping": "/chat/diag/ping",
#                 "env": "/chat/diag/env",
#                 "db_check": "/chat/diag/db-check",
#                 "index_check": "/chat/diag/index-check",
#                 "search_test": "/chat/diag/search-test",
#                 "full_flow": "/chat/diag/full-test",
#             },
#             "health": "/health",
#         },
#         "docs": "/docs",
#         "timestamp": datetime.utcnow().isoformat() + "Z",
#     }


# @app.get("/health", tags=["System"])
# def health():
#     openai_key = bool(os.getenv("OPENAI_API_KEY"))
#     mongodb_uri = bool(os.getenv("MONGODB_URI"))

#     all_ok = openai_key and mongodb_uri

#     status = {
#         "ok": all_ok,
#         "status": "healthy" if all_ok else "degraded",
#         "checks": {
#             "openai_key": "✓ configured" if openai_key else "✗ missing",
#             "mongodb_uri": "✓ configured" if mongodb_uri else "✗ missing",
#         },
#         "timestamp": datetime.utcnow().isoformat() + "Z",
#     }

#     if not all_ok:
#         log.warning("Health check failed: %s", status)

#     return status

# # ═══════════════════════════════════════════════════════════════════
# # STARTUP / SHUTDOWN EVENTS
# # ═══════════════════════════════════════════════════════════════════


# @app.on_event("startup")
# async def startup_event():
#     log.info("=" * 80)
#     log.info("✓ APPLICATION READY")
#     log.info("=" * 80)
#     log.info("Service: AI Calling API with RAG v2.0.0")
#     log.info("Environment:")
#     log.info("  - OPENAI_API_KEY: %s",
#              "✓ set" if os.getenv("OPENAI_API_KEY") else "✗ MISSING")
#     log.info("  - MONGODB_URI: %s",
#              "✓ set" if os.getenv("MONGODB_URI") else "✗ MISSING")
#     log.info("  - KB_DB_NAME: %s", os.getenv("KB_DB_NAME", "skylix_kb"))
#     log.info("  - KB_COL_NAME: %s",
#              os.getenv("KB_COL_NAME", "knowledge_entries"))
#     log.info("  - KB_INDEX_NAME: %s", os.getenv("KB_INDEX_NAME", "kb_index"))
#     log.info("  - LOG_LEVEL: %s", LOG_LEVEL)
#     log.info("Routes:")
#     log.info("  - Chat: /chat")
#     log.info("  - KB Training: /kb/train")
#     log.info("  - Diagnostics: /chat/diag/*")
#     log.info("  - API Docs: /docs")
#     log.info("=" * 80)
#     log.info("Ready to accept requests!")
#     log.info("=" * 80)


# @app.on_event("shutdown")
# async def shutdown_event():
#     log.info("=" * 80)
#     log.info("🛑 APPLICATION SHUTDOWN")
#     log.info("=" * 80)
from dotenv import load_dotenv
import os
from openai import OpenAI
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dummy_chat.dummy_kb_ingest import ingest_kb
from dummy_chat.dummy_kb_retrieve import retrieve_kb_chunks, get_kb_topics
from dummy_chat.dummy_kb_answer import answer_from_kb, calculate_token_cost

# ═══════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

_openai_client = None


def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))

logging.basicConfig(
    level=logging.DEBUG if os.getenv("RAG_DEBUG") else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("app")
logging.getLogger("pymongo").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Simple RAG API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST LOGGING MIDDLEWARE (Detailed + Safe Async Response)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs all HTTP requests and responses with detailed trace output."""
    request_id = f"{int(time.time() * 1000)}"
    client_ip = request.client.host
    path = request.url.path
    method = request.method

    log.info(" REQUEST | id=%s | method=%s | path=%s | client=%s",
             request_id, method, path, client_ip)

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Capture and log the raw response body safely
        body_text = ""
        if hasattr(response, "body_iterator"):
            body_content = b""
            async for chunk in response.body_iterator:
                body_content += chunk
            body_text = body_content.decode("utf-8", errors="ignore").strip()

            # FIXED: restore async body iterator properly
            async def new_body_iterator():
                yield body_content
            response.body_iterator = new_body_iterator()
            response.headers["content-length"] = str(len(body_content))

        else:
            body_content = getattr(response, "body", b"")
            if isinstance(body_content, str):
                body_text = body_content
                body_content = body_content.encode("utf-8")
            else:
                body_text = body_content.decode(
                    "utf-8", errors="ignore").strip()
            response.headers["content-length"] = str(len(body_content))

        log.info(" RESPONSE | id=%s | status=%d | time=%.1fms",
                 request_id, response.status_code, duration_ms)
        log.info("   Raw Response Body:")
        for line in body_text.splitlines():
            log.info("    %s", line)

        return response

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log.error(" REQUEST FAILED | id=%s | time=%.1fms | error=%s",
                  request_id, duration_ms, str(e))
        raise


# GLOBAL EXCEPTION HANDLER
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler with detailed logging."""
    log.error("=" * 80)
    log.error("UNHANDLED EXCEPTION")
    log.error("=" * 80)
    log.error("Path: %s %s", request.method, request.url.path)
    log.error("Error: %s", str(exc))
    log.error("Traceback:\n%s", traceback.format_exc())
    log.error("=" * 80)

    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


# ═══════════════════════════════════════════════════════════════════
# SINGLE UPLOAD ENDPOINT (Like your original /kb/v2/train)
# ═══════════════════════════════════════════════════════════════════

@app.post("/kb/v2/train")
async def kb_train(
    workspace_id: str = Form(...),
    user_id: str = Form(...),
    status: str = Form(...),
    source_type: str = Form(...),  # file | link | qna | mixed
    file: list[UploadFile] | None = File(None),
    link: list[str] | None = Form(None),
    question: list[str] | None = Form(None),
    answer: list[str] | None = Form(None),
):
    """
    SINGLE ENDPOINT to upload ANY type of content

    Examples:

    1. FILE:
       curl -X POST http://localhost:8000/kb/v2/train \
            -F "workspace_id=ws_test" \
            -F "source_type=file" \
            -F "file=@document.pdf"

    2. LINK:
       curl -X POST http://localhost:8000/kb/v2/train \
            -F "workspace_id=ws_test" \
            -F "source_type=link" \
            -F "link=https://example.com/article"

    3. QNA:
       curl -X POST http://localhost:8000/kb/v2/train \
            -F "workspace_id=ws_test" \
            -F "source_type=qna" \
            -F "question=What is RAG?" \
            -F "answer=RAG is..."
    """
    log.info("=" * 80)
    log.info(f"📥 KB TRAIN REQUEST")
    log.info(f"   Workspace: {workspace_id}")
    log.info(f"   Type: {source_type}")
    log.info("=" * 80)

    def _ensure_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    try:
        items = []
        src_type = (source_type or "").strip().lower()
        if src_type not in {"file", "link", "qna", "mixed"}:
            raise HTTPException(
                400, "Invalid source_type. Must be 'file', 'link', 'qna', or 'mixed'")

        files = _ensure_list(file)
        links = _ensure_list(link)
        questions = _ensure_list(question)
        answers = _ensure_list(answer)

        if src_type in {"file", "mixed"}:
            for f in files:
                items.append(("file", {"file": f}))

        if src_type in {"link", "mixed"}:
            for l in links:
                if l:
                    items.append(("link", {"link": l}))

        if src_type in {"qna", "mixed"}:
            if len(questions) != len(answers):
                raise HTTPException(
                    400, "questions and answers must be the same length")
            for q, a in zip(questions, answers):
                if q and a:
                    items.append(("qna", {"question": q, "answer": a}))

        if not items:
            raise HTTPException(400, "No content provided for ingestion")

        results = []
        errors = []
        for item_type, payload in items:
            try:
                result = await ingest_kb(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    status=status,
                    source_type=item_type,
                    file=payload.get("file"),
                    link=payload.get("link"),
                    question=payload.get("question"),
                    answer=payload.get("answer"),
                )
                results.append(result)
            except HTTPException as e:
                errors.append({
                    "ok": False,
                    "source_type": item_type,
                    "detail": e.detail,
                })
            except Exception as e:
                errors.append({
                    "ok": False,
                    "source_type": item_type,
                    "detail": str(e),
                })

        log.info("OK Upload processed!")
        if len(results) == 1 and not errors:
            return {
                "ok": True,
                "workspace_id": workspace_id,
                "total_items": 1,
                "item": results[0],
                "items": results,
            }
        return {
            "ok": len(errors) == 0,
            "workspace_id": workspace_id,
            "total_items": len(results) + len(errors),
            "succeeded": len(results),
            "failed": len(errors),
            "items": results,
            "errors": errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# INCLUDE ROUTERS
log.info("Loading routers...")

# Only local endpoints in this file should handle /chat and KB APIs.
# No imports from chat/ are used for routing.
log.info(" No external chat routers loaded")

# Optional: Voice router (if available)
# Optional: Voice router (if available)
try:
    from voice.voice_router import router as voice_router
    app.include_router(voice_router, tags=["Voice"])
    log.info("✓ Voice router loaded")
except ImportError as e:
    log.warning("⚠ Voice router not available (optional): %s", str(e))
except Exception as e:
    log.error("✗ Failed to load voice router: %s", e)


# ═══════════════════════════════════════════════════════════════════
# QUERY ENDPOINT
# ═══════════════════════════════════════════════════════════════════

class ChatQueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    user_id: str | None = None
    workspace_ids: list[str] = []
    model: str | None = None
    temperature: float = 0
    max_tokens: int | None = None
    context: dict[str, Any] | str | None = None
    user_prompt: str | None = None
    status: str = "published"
    top_k: int = 5
    include_raw: bool = True
    fallback_text: bool = True


@app.post("/chat")
async def kb_query(payload: ChatQueryRequest):
    """
    Query the knowledge base

    Example:
       curl -X POST http://localhost:8000/chat \
            -H "Content-Type: application/json" \
            -d "{\"workspace_ids\":[\"ws_test\"],\"query\":\"Who is Karthik Kailash?\"}"
    """
    if not payload.workspace_ids:
        raise HTTPException(
            400, "workspace_ids is required and cannot be empty")

    workspace_id = payload.workspace_ids[0]
    user_id = payload.user_id
    conversation_id = payload.conversation_id
    query = payload.query
    user_prompt = payload.user_prompt
    status = payload.status
    model = payload.model
    temperature = payload.temperature
    max_tokens = payload.max_tokens
    context = payload.context
    top_k = payload.top_k
    include_raw = payload.include_raw
    fallback_text = payload.fallback_text
    log.info("=" * 80)
    log.info(f"🔍 KB QUERY REQUEST")
    log.info(f"   Workspace: {workspace_id}")
    log.info(f"   Query: {query}")
    log.info("=" * 80)

    try:
        status_norm = (status or 'published').strip().lower()
        if status_norm != 'published':
            return {
                'ok': True,
                'answer': 'This information is not available in the knowledge base.',
                'sources': [],
                'chunks_found': 0,
            }

        user_persona = None
        conversation_history = None
        if context:
            if isinstance(context, dict):
                user_persona = context.get('user_persona')
                conversation_history = context.get('conversation_history')
            else:
                try:
                    import json
                    context_obj = json.loads(context)
                    if isinstance(context_obj, dict):
                        user_persona = context_obj.get('user_persona')
                        conversation_history = context_obj.get('conversation_history')
                except Exception:
                    user_persona = None

        min_score = 0.3
        # Retrieve chunks
        retrieval = retrieve_kb_chunks(
            workspace_id=workspace_id,
            query=query,
            top_k=top_k,
            min_score=min_score,
            include_raw=include_raw,
            fallback_text=fallback_text,
        )
        if include_raw:
            chunks, raw_results, diagnostics = retrieval
        else:
            chunks = retrieval
            raw_results = None
            diagnostics = None

        if not chunks:
            log.info("No relevant chunks found")
            kb_topics = get_kb_topics(workspace_id, limit=3)
            answer = answer_from_kb(
                query=query,
                chunks=[],
                user_prompt=user_prompt,
                user_persona=user_persona,
                conversation_history=conversation_history,
                kb_topics=kb_topics,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            total_cost = None
            response = {
                "ok": True,
                "answer": answer,
                "sources": [],
                "chunks_found": 0,
                "total_cost": total_cost,
            }
            if include_raw:
                response["raw"] = {
                    "vector_results": raw_results,
                    "min_score": min_score,
                    "top_k": top_k,
                    "diagnostics": diagnostics,
                    "fallback_text": fallback_text,
                }
            return response

        # Generate answer
        log.info(f"Generating answer from {len(chunks)} chunks...")
        if include_raw:
            answer, raw_openai = answer_from_kb(
                query=query,
                chunks=[c["text"] for c in chunks],
                include_raw=True,
                user_prompt=user_prompt,
                user_persona=user_persona,
                conversation_history=conversation_history,
                kb_topics=None,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            answer = answer_from_kb(
                query=query,
                chunks=[c["text"] for c in chunks],
                user_prompt=user_prompt,
                user_persona=user_persona,
                conversation_history=conversation_history,
                kb_topics=None,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        log.info("✅ Query successful!")

        cost_details = None
        total_cost = None
        if include_raw and raw_openai:
            cost_details = calculate_token_cost(
                raw_openai.get("usage"), raw_openai.get("model")
            )
            if cost_details:
                total_cost = cost_details.get("total_cost")

        response = {
            "ok": True,
            "answer": answer,
            "sources": [
                {
                    "public_id": c["public_id"],
                    "source_name": c["source_name"],
                    "score": round(c["score"], 3),
                }
                for c in chunks
            ],
            "chunks_found": len(chunks),
            "total_cost": total_cost,
        }
        if include_raw:
            response["raw"] = {
                "vector_results": raw_results,
                "openai": raw_openai,
                "min_score": min_score,
                "top_k": top_k,
                "diagnostics": diagnostics,
                "fallback_text": fallback_text,
                "cost": cost_details,
            }
        return response

    except Exception as e:
        log.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

@app.get("/kb/stats/{workspace_id}")
async def get_stats(workspace_id: str):
    """Get workspace statistics"""
    from dummy_chat.dummy_kb_retrieve import collection

    try:
        total = collection.count_documents({
            "workspace_id": workspace_id,
            "status": "published"
        })

        # Count by source type
        pipeline = [
            {"$match": {"workspace_id": workspace_id, "status": "published"}},
            {"$group": {"_id": "$source_type", "count": {"$sum": 1}}}
        ]
        by_type = {doc["_id"]: doc["count"]
                   for doc in collection.aggregate(pipeline)}

        # Count unique sources
        pipeline = [
            {"$match": {"workspace_id": workspace_id, "status": "published"}},
            {"$group": {"_id": "$source_name"}},
            {"$count": "total"}
        ]
        result = list(collection.aggregate(pipeline))
        unique_sources = result[0]["total"] if result else 0

        return {
            "ok": True,
            "workspace_id": workspace_id,
            "total_chunks": total,
            "unique_sources": unique_sources,
            "by_type": by_type,
        }
    except Exception as e:
        log.error(f"Stats failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/")
def root():
    return {
        "service": "Simple RAG API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /kb/v2/train",
            "query": "POST /chat",
            "stats": "GET /kb/stats/{workspace_id}",
        },
        "database": {
            "name": os.getenv("KB_DB_NAME", "skylix_rag"),
            "collection": os.getenv("KB_COLLECTION_NAME", "kb_chunks"),
            "index": os.getenv("KB_VECTOR_INDEX", "kb_chunks_vector"),
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "openai": "✓" if os.getenv("OPENAI_API_KEY") else "✗",
        "mongodb": "✓" if os.getenv("MONGODB_URI") else "✗",
    }


@app.on_event("startup")
async def startup():
    log.info("=" * 80)
    log.info("🚀 SIMPLE RAG API READY")
    log.info("=" * 80)
    log.info(f"Database: {os.getenv('KB_DB_NAME', 'skylix_rag')}")
    log.info(f"Collection: {os.getenv('KB_COLLECTION_NAME', 'kb_chunks')}")
    log.info(f"Index: {os.getenv('KB_VECTOR_INDEX', 'kb_chunks_vector')}")
    log.info("=" * 80)
    log.info("Endpoints:")
    log.info("  POST /kb/v2/train  - Upload (file/link/qna)")
    log.info("  POST /chat        - Query knowledge base")
    log.info("  GET  /kb/stats/{workspace_id} - Get stats")
    log.info("=" * 80)

for r in app.router.routes:
    print(r.path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)