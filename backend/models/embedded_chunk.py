"""
Pydantic v2 models for Phase 3 embedding output.

EmbeddedChunk extends Chunk with dense + sparse vectors and is the
canonical object written to Pinecone (Phase 4).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ─── Sparse vector ────────────────────────────────────────────────────────────

class SparseVector(BaseModel):
    """
    Pinecone-compatible sparse vector: parallel arrays of token indices
    and their BM25 / SPLADE scores.
    """
    indices: list[int]   = Field(default_factory=list)
    values:  list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _lengths_match(self) -> "SparseVector":
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseVector indices ({len(self.indices)}) and "
                f"values ({len(self.values)}) must have equal length."
            )
        return self

    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return len(self.indices)

    def is_empty(self) -> bool:
        return self.nnz == 0

    def to_dict(self) -> dict:
        return {"indices": self.indices, "values": self.values}


# ─── Embedded chunk ───────────────────────────────────────────────────────────

class EmbeddedChunk(BaseModel):
    """
    A Chunk augmented with dense and (optionally) sparse embedding vectors.

    This is the final object passed to the Pinecone upsert in Phase 4.
    All provenance metadata from the source Chunk is carried forward so
    Pinecone metadata filters and citation links work end-to-end.
    """
    model_config = {"protected_namespaces": ()}

    # ── Identity (mirrors Chunk) ──────────────────────────────────────────────
    chunk_id:       str
    doc_id:         str
    chunk_index:    int
    chunk_type:     str          # ChunkType.value string

    # ── Content ───────────────────────────────────────────────────────────────
    text:           str
    text_with_title: str = ""   # what was actually embedded
    token_count:    int  = 0
    char_count:     int  = 0

    # ── Embedding vectors ─────────────────────────────────────────────────────
    dense_vector:   list[float] = Field(default_factory=list)
    sparse_vector:  SparseVector | None = None
    embedding_model: str = ""   # filled by EmbeddingPipeline from settings
    embedding_dims:  int = 0    # filled by EmbeddingPipeline from settings

    # ── Source provenance ─────────────────────────────────────────────────────
    doc_name:       str
    file_type:      str
    page_numbers:   list[int]   = Field(default_factory=list)
    source_pages:   str         = ""
    doc_title:      str | None  = None
    doc_author:     str | None  = None
    doc_language:   str | None  = None
    published_at:   str | None  = None
    doc_sha256:     str | None  = None
    content_hash:   str         = ""
    strategy:       str         = "recursive"

    # ── Hierarchical links ────────────────────────────────────────────────────
    parent_chunk_id:      str | None = None
    grandparent_chunk_id: str | None = None

    # ── Embedding provenance ──────────────────────────────────────────────────
    embedded_at:    datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    cache_hit:      bool = False   # True when vector was served from Redis
    embed_time_ms:  float = 0.0

    # ── Extra metadata ────────────────────────────────────────────────────────
    extras:         dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk,            # backend.models.chunk.Chunk
        dense_vector:     list[float],
        sparse_vector:    SparseVector | None = None,
        embedding_model:  str  = "",
        embedding_dims:   int  = 0,
        cache_hit:        bool = False,
        embed_time_ms:    float = 0.0,
    ) -> "EmbeddedChunk":
        """Construct from a Chunk + vectors."""
        from backend.config import settings as _cfg
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            chunk_type=chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
            text=chunk.text,
            text_with_title=chunk.text_with_title or chunk.text,
            token_count=chunk.token_count,
            char_count=chunk.char_count,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            embedding_model=embedding_model or _cfg.openai_embedding_model,
            embedding_dims=embedding_dims   or _cfg.openai_embedding_dims,
            doc_name=chunk.doc_name,
            file_type=chunk.file_type,
            page_numbers=chunk.page_numbers,
            source_pages=chunk.source_pages,
            doc_title=chunk.doc_title,
            doc_author=chunk.doc_author,
            doc_language=chunk.doc_language,
            published_at=chunk.published_at,
            doc_sha256=chunk.doc_sha256,
            content_hash=chunk.content_hash,
            strategy=chunk.strategy.value if hasattr(chunk.strategy, "value") else str(chunk.strategy),
            parent_chunk_id=chunk.parent_chunk_id,
            grandparent_chunk_id=chunk.grandparent_chunk_id,
            cache_hit=cache_hit,
            embed_time_ms=embed_time_ms,
            extras=dict(chunk.extras),
        )

    def to_pinecone_record(self) -> dict:
        """
        Serialise to the dict format expected by pinecone.Index.upsert().

        Returns::
            {
                "id":       str,
                "values":   list[float],          # dense
                "sparse_values": {...} | None,    # sparse (hybrid search)
                "metadata": {...},
            }
        """
        metadata = {
            "doc_id":       self.doc_id,
            "chunk_index":  self.chunk_index,
            "chunk_type":   self.chunk_type,
            "doc_name":     self.doc_name,
            "file_type":    self.file_type,
            "source_pages": self.source_pages,
            "strategy":     self.strategy,
            "token_count":  self.token_count,
            "content_hash": self.content_hash,
            # text truncated — Pinecone metadata max 40 KB per record
            "text":         self.text[:1_000],
            "text_with_title": self.text_with_title[:500],
        }
        # Add optional provenance fields
        for field in ("doc_title", "doc_author", "doc_language",
                      "published_at", "doc_sha256"):
            val = getattr(self, field)
            if val is not None:
                metadata[field] = val
        if self.page_numbers:
            metadata["page_numbers"] = self.page_numbers
        if self.parent_chunk_id:
            metadata["parent_chunk_id"] = self.parent_chunk_id
        if self.grandparent_chunk_id:
            metadata["grandparent_chunk_id"] = self.grandparent_chunk_id

        record: dict = {
            "id":       self.chunk_id,
            "values":   self.dense_vector,
            "metadata": metadata,
        }
        if self.sparse_vector and not self.sparse_vector.is_empty():
            record["sparse_values"] = self.sparse_vector.to_dict()
        return record

    @property
    def has_dense(self) -> bool:
        return bool(self.dense_vector)

    @property
    def has_sparse(self) -> bool:
        return self.sparse_vector is not None and not self.sparse_vector.is_empty()


# ─── Embedding result envelope ────────────────────────────────────────────────

class EmbeddingResult(BaseModel):
    """Returned by EmbeddingPipeline.embed_chunks()."""

    doc_id:           str
    doc_name:         str
    model_used:       str  = ""   # filled by EmbeddingPipeline from settings

    embedded_chunks:  list[EmbeddedChunk] = Field(default_factory=list)

    # Counters
    total_input:      int = 0   # chunks fed in
    total_embedded:   int = 0   # successfully embedded
    total_skipped:    int = 0   # empty / language-filtered
    total_cache_hits: int = 0   # served from Redis
    total_tokens:     int = 0   # tokens sent to the API
    total_cost_usd:   float = 0.0

    # Diagnostics
    embed_time_ms:    float = 0.0
    batch_count:      int   = 0
    retry_count:      int   = 0
    errors:           list[str] = Field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_embedded / self.total_input

    def summary(self) -> dict:
        return {
            "doc_id":           self.doc_id,
            "doc_name":         self.doc_name,
            "model":            self.model_used,
            "total_input":      self.total_input,
            "total_embedded":   self.total_embedded,
            "total_skipped":    self.total_skipped,
            "cache_hits":       self.total_cache_hits,
            "total_tokens":     self.total_tokens,
            "cost_usd":         round(self.total_cost_usd, 6),
            "embed_time_ms":    round(self.embed_time_ms, 1),
            "batch_count":      self.batch_count,
            "retry_count":      self.retry_count,
            "errors":           self.errors,
        }


# ─── Per-chunk validation record ─────────────────────────────────────────────

class ChunkValidationRecord(BaseModel):
    """Result of pre-embedding validation for a single chunk."""
    chunk_id:   str
    valid:      bool
    skip_reason: str | None = None   # "empty" | "too_long" | "non_english" | "duplicate"
    token_count: int = 0
    language:    str | None = None
    truncated:   bool = False        # True when text was truncated to fit token limit
