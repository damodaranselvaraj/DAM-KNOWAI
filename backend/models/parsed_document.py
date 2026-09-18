"""
Pydantic v2 models for parsed document output.

These are the canonical data structures that flow through every stage of
the ingestion pipeline: parse → chunk → embed → store.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class FileType(str, Enum):
    PDF  = "pdf"
    DOCX = "docx"
    CSV  = "csv"
    HTML = "html"
    TXT  = "txt"
    XLSX = "xlsx"


class ParserName(str, Enum):
    PYMUPDF    = "pymupdf"
    DOCLING    = "docling"
    LLAMAINDEX = "llamaindex"


class ParseStatus(str, Enum):
    SUCCESS          = "success"
    PARTIAL          = "partial"     # some pages/sections failed
    FALLBACK_USED    = "fallback_used"
    FAILED           = "failed"


class VersionAction(str, Enum):
    CREATED     = "created"      # first time we see this doc
    REPLACED    = "replaced"     # overwrote previous version
    KEPT_BOTH   = "kept_both"    # kept old + added new
    SOFT_DELETED = "soft_deleted" # old version marked inactive


# ─── Page-level content ────���──────────────────────────────────────────────────

class ParsedPage(BaseModel):
    """Content and metadata for a single page / sheet / section."""

    page_number: int = Field(..., ge=1, description="1-based page index")
    text:        str = Field(...,       description="Extracted plain text")
    char_count:  int = Field(0,  ge=0)
    word_count:  int = Field(0,  ge=0)
    has_tables:  bool = False
    has_images:  bool = False
    # Raw markdown produced by Docling (None for PyMuPDF/LlamaIndex)
    markdown:    str | None = None
    # Any parser-specific extras (bounding boxes, language, etc.)
    extras:      dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_counts(self) -> "ParsedPage":
        if self.char_count == 0 and self.text:
            self.char_count = len(self.text)
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())
        return self


# ─── Parser audit entry ───────────────────────────────────────────────────────

class ParserAttempt(BaseModel):
    """Records one attempt in the fallback chain."""

    parser:     ParserName
    succeeded:  bool
    duration_ms: float = Field(..., ge=0)
    error:      str | None = None
    pages_extracted: int = 0


# ─── Top-level parsed document ───────────────────────────────────────────────

class ParsedDocument(BaseModel):
    """
    Canonical output of Phase 1 parsing.

    Consumed directly by Phase 2 (chunking) and stored in the job store
    as the audit record for a single ingested file.
    """
    model_config = {"protected_namespaces": ()}

    # Identity
    doc_id:       str  = Field(default_factory=lambda: str(uuid4()))
    filename:     str
    file_type:    FileType
    size_bytes:   int  = Field(..., ge=0)

    # Content
    pages:        list[ParsedPage] = Field(default_factory=list)
    full_text:    str = ""          # concatenation of all page texts
    page_count:   int = Field(0, ge=0)
    total_chars:  int = Field(0, ge=0)
    total_words:  int = Field(0, ge=0)

    # Deduplication / versioning
    sha256:       str  = Field(..., min_length=64, max_length=64)
    version:      int  = Field(1, ge=1)
    version_action: VersionAction = VersionAction.CREATED
    previous_doc_id: str | None   = None   # doc_id of the version this supersedes

    # Parsing provenance
    parser_used:  ParserName
    parse_status: ParseStatus
    audit_trail:  list[ParserAttempt] = Field(default_factory=list)
    parse_time_ms: float = Field(0.0, ge=0)

    # Timestamps
    parsed_at:    datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: str | None = None   # extracted from doc metadata if present

    # Source metadata (filled by parser where available)
    title:        str | None = None
    author:       str | None = None
    subject:      str | None = None
    language:     str | None = None
    source_path:  str | None = None  # temp path used during parsing

    @model_validator(mode="after")
    def _derive_aggregates(self) -> "ParsedDocument":
        if self.pages and not self.full_text:
            self.full_text = "\n\n".join(p.text for p in self.pages if p.text)
        if self.page_count == 0:
            self.page_count = len(self.pages)
        if self.total_chars == 0:
            self.total_chars = sum(p.char_count for p in self.pages)
        if self.total_words == 0:
            self.total_words = sum(p.word_count for p in self.pages)
        return self

    @field_validator("sha256")
    @classmethod
    def _sha256_hex(cls, v: str) -> str:
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("sha256 must be a lowercase hex string")
        return v.lower()

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        return self.total_chars == 0

    @property
    def primary_parser_succeeded(self) -> bool:
        return bool(self.audit_trail and self.audit_trail[0].succeeded)

    def summary(self) -> dict:
        """Compact dict suitable for API list responses."""
        return {
            "doc_id":       self.doc_id,
            "filename":     self.filename,
            "file_type":    self.file_type.value,
            "size_bytes":   self.size_bytes,
            "page_count":   self.page_count,
            "total_words":  self.total_words,
            "parser_used":  self.parser_used.value,
            "parse_status": self.parse_status.value,
            "sha256":       self.sha256[:16] + "…",
            "version":      self.version,
            "parsed_at":    self.parsed_at.isoformat(),
        }


# ─── Parse request (used internally by the router) ───────────────────────────

class ParseRequest(BaseModel):
    """Input contract passed from the API layer to the parser router."""

    filename:         str
    content:          bytes            # raw file bytes
    version_handling: str = "replace"  # "replace" | "keep_both" | "soft_delete"
    force_parser:     ParserName | None = None  # override auto-selection


# ─── Parse result envelope ────────────────────────────────────────────────────

class ParseResult(BaseModel):
    """Envelope returned by the parser router to the API layer."""

    success:  bool
    document: ParsedDocument | None = None
    error:    str | None = None
    http_status_code: int = 200  # 415 for unsupported, 422 for parse failure, etc.
