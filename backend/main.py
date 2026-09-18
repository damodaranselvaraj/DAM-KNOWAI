"""
RAG Pipeline API — FastAPI application entry point.
Run with: uvicorn backend.main:app --reload --port 8000
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.v1.router import api_router
from backend.config import settings
from backend.dependencies import create_memory_saver

# ─── Logging ──────────────────────────────────────────────────────────────────
# Apply log level from settings — never hardcode INFO
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_api")


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 RAG Pipeline API starting up…")

    # ── 1. Connectivity validation ────────────────────────────────────────────
    from backend.connectivity_check import check_connectivity
    try:
        check_connectivity(abort_on_failure=True)
    except RuntimeError as exc:
        logger.critical("STARTUP ABORTED — connectivity check failed:\n%s", exc)
        raise

    # ── 2. Memory saver ───────────────────────────────────────────────────────
    app.state.memory_saver = create_memory_saver()
    logger.info("✅ Memory saver ready: %r", app.state.memory_saver)

    # ── 2b. Periodic memory retention (prune) ─────────────────────────────────
    # Applies the MEMORY_PRUNE_DAYS / MEMORY_KEEP_LAST retention policy
    # documented in the README. Runs once at startup, then every 24h for
    # the life of the process — otherwise checkpoints accumulate forever.
    async def _prune_loop() -> None:
        saver = app.state.memory_saver
        while True:
            try:
                deleted = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: saver.prune(
                        older_than_days=settings.memory_prune_days,
                        keep_last=settings.memory_keep_last,
                    ),
                )
                logger.info("🧹 Memory prune complete — %d checkpoint(s) deleted.", deleted)
            except Exception as exc:
                logger.warning("⚠️  Memory prune failed: %s", exc)
            await asyncio.sleep(24 * 60 * 60)

    app.state.prune_task = asyncio.create_task(_prune_loop())

    # ── 3. Pre-warm reranker model (downloads on first run) ───────────────────
    try:
        from backend.retrieval.reranker import Reranker, RerankerConfig
        app.state.reranker = Reranker(RerankerConfig())
        # Share the pre-warmed instance with the chat module so first requests
        # don't pay the model-load cost
        from backend.api.v1.chat import _set_reranker_from_app_state
        _set_reranker_from_app_state(app.state.reranker)
        logger.info("✅ Reranker ready: Cohere %s", settings.reranker_model)
    except Exception as exc:
        app.state.reranker = None
        logger.warning(
            "⚠️  Reranker failed to load — reranking will be disabled: %s", exc
        )

    # ── 4. NLTK data download (required by SparseRetriever) ───────────────────
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        logger.info("✅ NLTK data ready")
    except Exception as exc:
        logger.warning("⚠️  NLTK data download failed: %s", exc)

    logger.info("✅ Startup complete — ready to serve requests.")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if hasattr(app.state, "prune_task"):
        app.state.prune_task.cancel()

    if hasattr(app.state, "memory_saver"):
        app.state.memory_saver.teardown()
        logger.info("💾 Memory saver closed.")

    logger.info("🛑 RAG Pipeline API shut down.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
    description=(
        "Production-ready Retrieval-Augmented Generation pipeline backend. "
        "Handles document ingestion, chunking, embedding, vector storage, and "
        "multi-turn RAG chat with citations."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Origins driven by CORS_ORIGINS in .env — never hardcoded
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request Logging Middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "%s %s → %s  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ─── Global Exception Handlers ────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    # Mask internal details in production
    detail = str(exc) if settings.app_env != "production" else "Internal server error"
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": detail,
            "status_code": 500,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
        },
    )


# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(api_router)


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health/status",
    }
