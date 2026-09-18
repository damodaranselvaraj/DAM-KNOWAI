"""
Pydantic v2 models for Phase 2 chunking output.

These are the canonical objects consumed by Phase 3 (embedding).
Every Chunk carries the full provenance path back to its source document
and page so citations remain accurate end-to-end.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class ChunkingStrategy(str, Enum):
    FIXED_SIZE      = "fixed_size"
    FIXED_OVERLAP   = "fixed_overlap"
    SENTENCE_BASED  = "sentence_based"
    RECURSIVE       = "recursive"
    SEMANTIC        = "semantic"
    HIERARCHICAL    = "hierarchical"


class ChunkType(str, Enum):
    TEXT        = "text"
    TABLE       = "table"
    HEADING     = "heading"      # isolated heading kept as its own chunk
    # Hierarchical tiers
    PARENT      = "parent"
    CHILD       = "child"
    GRANDCHILD  = "grandchild"


# ─── Core chunk ───────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """
    A single chunk of content ready for embedding.

    Naming convention for chunk_id:
        {doc_id}:{chunk_index}  (unique within a document)
    """
    model_config = {"protected_namespaces": ()}

    # Identity
    chunk_id:       str = Field(default_factory=lambda: str(uuid4()))
    doc_id:         str
    chunk_index:    int = Field(..., ge=0, description="0-based position in the chunk list")
    chunk_type:     ChunkType = ChunkType.TEXT

    # Content
    text:           str
    token_count:    int = Field(0, ge=0)
    char_count:     int = Field(0, ge=0)

    # Context prefix prepended during retrieval (title / heading)
    title_prefix:   str = ""
    text_with_title: str = ""   # title_prefix + "\n\n" + text  (set by pipeline)

    # Source provenance
    doc_name:       str
    file_type:      str
    page_numbers:   list[int] = Field(default_factory=list)
    source_pages:   str = ""          # human-readable "pp. 3-5"

    # Document-level metadata propagated from ParsedDocument
    doc_title:      str | None = None
    doc_author:     str | None = None
    doc_language:   str | None = None
    published_at:   str | None = None
    doc_sha256:     str | None = None

    # Chunk-level fingerprint (SHA-256 of text content)
    content_hash:   str = ""

    # Strategy used to create this chunk
    strategy:       ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Hierarchical parent links (used when strategy=HIERARCHICAL)
    parent_chunk_id:      str | None = None
    grandparent_chunk_id: str | None = None

    # Extra metadata (table headers, row ranges, etc.)
    extras:         dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    created_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _derive_fields(self) -> "Chunk":
        if self.char_count == 0 and self.text:
            self.char_count = len(self.text)
        if not self.content_hash and self.text:
            from backend.utils.hashing import hash_text
            self.content_hash = hash_text(self.text)
        if not self.text_with_title:
            if self.title_prefix:
                self.text_with_title = f"{self.title_prefix}\n\n{self.text}"
            else:
                self.text_with_title = self.text
        if not self.source_pages and self.page_numbers:
            pages = sorted(set(self.page_numbers))
            if len(pages) == 1:
                self.source_pages = f"p. {pages[0]}"
            else:
                self.source_pages = f"pp. {pages[0]}-{pages[-1]}"
        return self

    def to_embed_text(self) -> str:
        """Text that gets sent to the embedding model — title prefix + content."""
        return self.text_with_title or self.text

    def summary(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "doc_id":      self.doc_id,
            "chunk_index": self.chunk_index,
            "chunk_type":  self.chunk_type.value,
            "token_count": self.token_count,
            "char_count":  self.char_count,
            "source_pages": self.source_pages,
            "strategy":    self.strategy.value,
        }


# ─── Chunking result envelope ─────────────────────────────────────────────────

class ChunkingResult(BaseModel):
    """Returned by ChunkingPipeline.chunk_document()."""

    doc_id:          str
    doc_name:        str
    strategy:        ChunkingStrategy

    chunks:          list[Chunk] = Field(default_factory=list)
    table_chunks:    list[Chunk] = Field(default_factory=list)

    total_chunks:    int = 0
    total_tokens:    int = 0
    duplicate_count: int = 0
    chunking_time_ms: float = 0.0

    @model_validator(mode="after")
    def _derive_totals(self) -> "ChunkingResult":
        if self.total_chunks == 0:
            self.total_chunks = len(self.chunks) + len(self.table_chunks)
        if self.total_tokens == 0:
            self.total_tokens = (
                sum(c.token_count for c in self.chunks)
                + sum(c.token_count for c in self.table_chunks)
            )
        return self

    @property
    def all_chunks(self) -> list[Chunk]:
        """Flat list of every chunk (text + table) in index order."""
        return self.chunks + self.table_chunks


# ─── Chunking config ──────────────────────────────────────────────────────────

class ChunkingConfig(BaseModel):
    """
    Runtime parameters forwarded from the Admin Panel / API.
    Defaults match the recommended production settings.
    """
    strategy:               ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Token limits
    chunk_size:             int = Field(512,  ge=64,  le=4096)
    overlap:                int = Field(50,   ge=0,   le=500)
    min_chunk_tokens:       int = Field(50,   ge=1)
    max_chunk_tokens:       int = Field(1024, ge=64,  le=8192)

    # Recursive splitter separators (ordered by precedence)
    separators:             list[str] = ["\n\n", "\n", " ", ""]

    # Semantic chunking
    breakpoint_percentile:  int = Field(95, ge=50, le=99)
    semantic_buffer_size:   int = Field(1,  ge=1,  le=10)

    # Hierarchical tiers
    parent_chunk_size:      int = Field(2048, ge=256, le=8192)
    child_chunk_size:       int = Field(512,  ge=64,  le=2048)
    grandchild_chunk_size:  int = Field(128,  ge=32,  le=512)

    # Table handling
    table_token_limit:      int = Field(600,  ge=100, le=4096)

    # Title / heading prepend
    prepend_title:          bool = True
    prepend_doc_title:      bool = True
