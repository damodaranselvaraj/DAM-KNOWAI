"""
Embedders sub-package — Phase 3 of the RAG ingestion pipeline.

Public surface
──────────────
EmbeddingPipeline        — main orchestrator, call embed_chunks()
build_embedding_pipeline — factory that reads backend.config.settings
OpenAIEmbedder           — dense embedding client (async + sync)
SparseEmbedder           — BM25 / SPLADE sparse encoder
InProcessEmbeddingCache  — in-process vector cache
pre_embedding_validator  — validate_chunks() helper
"""
from backend.ingestion.embedders.openai_embedder     import OpenAIEmbedder
from backend.ingestion.embedders.sparse_embedder     import SparseEmbedder
from backend.ingestion.embedders.embedding_cache     import InProcessEmbeddingCache, build_cache
from backend.ingestion.embedders.pre_embedding_validator import validate_chunks
from backend.ingestion.embedders.embedding_pipeline  import (
    EmbeddingPipeline,
    build_embedding_pipeline,
)

__all__ = [
    "EmbeddingPipeline",
    "build_embedding_pipeline",
    "OpenAIEmbedder",
    "SparseEmbedder",
    "InProcessEmbeddingCache",
    "build_cache",
    "validate_chunks",
]
