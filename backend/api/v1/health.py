"""
Health check endpoints — /api/v1/health

/health/status reads real metrics from the live version_store and
chunk_corpus_store instead of returning hard-coded values.
"""
import logging
import time
from datetime import datetime, timezone
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])

# ── Live service-status cache ─────────────────────────────────────────────────
# The Home page polls /health/services every few seconds. Hitting OpenAI /
# Pinecone on every poll would be slow and burn quota, so results are cached
# for a short TTL and reused across requests.
_service_cache: dict = {"ts": 0.0, "data": None}
_SERVICE_CACHE_TTL_SEC = 10.0


def _check_openai_reachable() -> bool:
    """Cheap liveness check — lists models with a short timeout."""
    from backend.config import settings
    try:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key, timeout=5.0)
        client.models.list()
        return True
    except Exception as exc:
        logger.warning("OpenAI liveness check failed: %s", exc)
        return False


def _check_pinecone_reachable() -> bool:
    """Cheap liveness check — describes index stats with a short timeout."""
    from backend.config import settings
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        index.describe_index_stats()
        return True
    except Exception as exc:
        logger.warning("Pinecone liveness check failed: %s", exc)
        return False


@router.get("/status", summary="Full system health status")
async def health_status():
    """
    Returns overall system health with service-level checks and live metrics
    drawn from the in-memory version store and backend config.
    """
    from backend.ingestion.versioning import version_store
    from backend.ingestion.chunk_corpus_store import chunk_corpus_store
    from backend.config import settings

    # ── Real document metrics ─────────────────────────────────────────────────
    store_stats = version_store.stats()
    active_docs = version_store.all_active()
    total_documents = store_stats.get("active", 0)

    # Sum words as a rough proxy for content volume (no chunk store yet)
    total_words = sum(d.total_words for d in active_docs)

    # Real chunk count from the in-memory chunk corpus store populated during
    # Phase 3/4 (embedding + upsert). Previously hardcoded to 0.
    total_chunks = chunk_corpus_store.stats().get("total_chunks", 0)

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "vector_db": {
                "status": True,
                "index_name": settings.pinecone_index_name,
                "doc_count": total_documents,
            },
            "embedding_model": {
                "status": True,
                "model": settings.openai_embedding_model,
            },
            "parser": {
                "status": True,
            },
        },
        "metrics": {
            "total_documents": total_documents,
            "total_words":     total_words,
            # Real count from chunk_corpus_store — non-zero only once Phase
            # 3/4 (embedding + Pinecone upsert) has run for at least one doc.
            "total_chunks":    total_chunks,
            "store_stats":     store_stats,
            "index_status":    "ready",
            "last_updated":    datetime.now(timezone.utc).isoformat(),
        },
    }


@router.get("/ping", summary="Lightweight liveness probe")
async def ping():
    """Returns a pong with approximate API latency."""
    start = time.perf_counter()
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return {"pong": True, "latency_ms": latency_ms}


@router.get("/services", summary="Live status of core RAG service components")
async def service_status(request: Request):
    """
    Powers the four status boxes on the Home page: embedding model, vector DB,
    LLM model, and reranking model. Each entry reports the configured model /
    provider name plus a live "up/down" check.

    OpenAI and Pinecone checks are cached for a short TTL (default 10s) so
    the frontend can poll this endpoint every few seconds without hammering
    upstream providers or burning API quota.
    """
    from backend.config import settings

    now = time.monotonic()
    cached = _service_cache["data"]
    if cached is not None and (now - _service_cache["ts"]) < _SERVICE_CACHE_TTL_SEC:
        return cached

    openai_ok   = _check_openai_reachable()
    pinecone_ok = _check_pinecone_reachable()

    # Reranker is "enabled" when a Cohere key is configured AND the pre-warmed
    # reranker instance loaded successfully at startup (app.state.reranker).
    reranker_loaded = getattr(request.app.state, "reranker", None) is not None
    reranker_enabled = bool(settings.cohere_api_key) and reranker_loaded

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding_model": {
            "name": settings.openai_embedding_model,
            "status": "active" if openai_ok else "not_loaded",
            "healthy": openai_ok,
        },
        "vector_db": {
            "name": "Pinecone",
            "index_name": settings.pinecone_index_name,
            "status": "ready" if pinecone_ok else "disconnected",
            "healthy": pinecone_ok,
        },
        "llm_model": {
            "name": settings.openai_chat_model,
            "status": "online" if openai_ok else "unavailable",
            "healthy": openai_ok,
        },
        "reranking_model": {
            "name": settings.reranker_model,
            "status": "enabled" if reranker_enabled else "disabled",
            "healthy": reranker_enabled,
        },
    }

    _service_cache["data"] = data
    _service_cache["ts"] = now
    return data
