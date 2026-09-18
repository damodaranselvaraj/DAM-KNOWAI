"""
App-level configuration read endpoint — /api/v1/config
Exposes non-sensitive runtime config for the frontend.
"""
from fastapi import APIRouter
from backend.config import settings

router = APIRouter(prefix="/config", tags=["Config"])


@router.get("/", summary="Get public runtime configuration")
async def get_app_config():
    """Returns non-sensitive application configuration values."""
    return {
        "app_env": settings.app_env,
        "embedding_model": settings.openai_embedding_model,
        "embedding_dims": settings.openai_embedding_dims,
        "chat_model": settings.openai_chat_model,
        "pinecone_index_name": settings.pinecone_index_name,
        "default_chunk_size": settings.default_chunk_size,
        "default_chunk_overlap": settings.default_chunk_overlap,
        "default_top_k": settings.default_top_k,
        "default_score_threshold": settings.default_score_threshold,
        "allowed_extensions": settings.allowed_extensions_list,
        "max_file_size_mb": settings.max_file_size_mb,
    }


@router.get("/indexes", summary="List available Pinecone indexes")
async def list_pinecone_indexes():
    """Returns all indexes available in the configured Pinecone account."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        indexes = [idx.name for idx in pc.list_indexes()]
        return {"indexes": indexes, "active": settings.pinecone_index_name}
    except Exception as exc:
        # Fall back to the configured index so the UI always has something
        return {"indexes": [settings.pinecone_index_name], "active": settings.pinecone_index_name, "error": str(exc)}
