"""
Pinecone upsert schema, namespace resolution, and metadata configuration.

Vector ID convention
─────────────────────
    {doc_id}:{version}:{page}:{chunk_index}

Namespace strategy  (per-tenant + per-corpus)
──────────────────────────────────────────────
    {tenant_id}__{corpus}        e.g.  "acme__legal"
    default__default             when neither is supplied

Metadata fields indexed in Pinecone
────────────────────────────────────
Only a subset of fields support equality / range filtering.  Fields marked
``indexed=True`` below must be declared in the Pinecone index metadata_config
at creation time (Pinecone Serverless: all metadata is auto-indexed, but we
keep the list explicit for documentation and Qdrant/Milvus parity).

NOTE on efSearch / nProbe
─────────────────────────
Pinecone Serverless does NOT expose efSearch or nProbe.  Those knobs are
specific to HNSW/IVF implementations in Milvus, Qdrant, and FAISS.
Pinecone manages its own internal graph parameters automatically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ─── Metadata field registry ─────────────────────────────────────────────────

@dataclass(frozen=True)
class MetadataField:
    name:         str
    dtype:        str          # "string" | "number" | "boolean"
    indexed:      bool = True
    description:  str  = ""


#: Canonical metadata fields written to every Pinecone vector
METADATA_FIELDS: list[MetadataField] = [
    MetadataField("doc_id",        "string",  True,  "Source document UUID"),
    MetadataField("version",       "number",  True,  "Document version integer"),
    MetadataField("doc_ref_name",  "string",  True,  "Original filename"),
    MetadataField("page",          "number",  True,  "1-based page number (first page of chunk)"),
    MetadataField("section_path",  "string",  False, "Heading breadcrumb e.g. 'Policy > Section 2'"),
    MetadataField("text",          "string",  False, "Raw chunk text (truncated to 1 000 chars)"),
    MetadataField("published_at",  "number",  True,  "Unix epoch int of document publication date"),
    MetadataField("source_url",    "string",  False, "Origin URL or file path"),
    MetadataField("mime",          "string",  True,  "MIME type e.g. application/pdf"),
    MetadataField("language",      "string",  True,  "ISO 639-1 language code"),
    MetadataField("checksum",      "string",  False, "SHA-256 of chunk text (first 16 hex chars)"),
    MetadataField("chunk_index",   "number",  True,  "0-based position in document chunk list"),
    MetadataField("chunk_type",    "string",  True,  "text | table | parent | child | grandchild"),
    MetadataField("strategy",      "string",  False, "Chunking strategy used"),
    MetadataField("created_at",    "number",  True,  "Unix epoch of ingestion timestamp"),
    MetadataField("tenant_id",     "string",  True,  "Tenant identifier for namespace isolation"),
    MetadataField("corpus",        "string",  True,  "Corpus / collection identifier"),
    MetadataField("parent_id",     "string",  False, "Parent chunk ID (hierarchical strategy)"),
]

INDEXED_FIELD_NAMES: list[str] = [f.name for f in METADATA_FIELDS if f.indexed]

# Pinecone metadata_config dict (used at index creation)
PINECONE_METADATA_CONFIG: dict = {
    "indexed": INDEXED_FIELD_NAMES,
}


# ─── Namespace resolver ───────────────────────────────────────────────────────

_NS_SEP = "__"
_DEFAULT_TENANT = "default"
_DEFAULT_CORPUS = "default"


def resolve_namespace(
    tenant_id: str | None = None,
    corpus:    str | None = None,
) -> str:
    """
    Build a Pinecone namespace string from tenant + corpus.

        acme  + legal   →  "acme__legal"
        acme  + None    →  "acme__default"
        None  + legal   →  "default__legal"
        None  + None    →  "default__default"

    Namespaces are lowercased and stripped of illegal chars
    (Pinecone allows [a-zA-Z0-9_-]).
    """
    t = _sanitise(tenant_id or _DEFAULT_TENANT)
    c = _sanitise(corpus    or _DEFAULT_CORPUS)
    return f"{t}{_NS_SEP}{c}"


def parse_namespace(namespace: str) -> tuple[str, str]:
    """
    Split a namespace string back into (tenant_id, corpus).
    Returns ("default", "default") if the string doesn't match the convention.
    """
    parts = namespace.split(_NS_SEP, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return _DEFAULT_TENANT, _DEFAULT_CORPUS


def _sanitise(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_\-]", "_", s)
    return s[:64]   # Pinecone namespace max length


# ─── Vector ID builder ────────────────────────────────────────────────────────

def build_vector_id(
    doc_id:      str,
    version:     int,
    page:        int,
    chunk_index: int,
) -> str:
    """
    Construct the canonical Pinecone vector ID.
    Format: "{doc_id}:{version}:{page}:{chunk_index}"
    """
    return f"{doc_id}:{version}:{page}:{chunk_index}"


def parse_vector_id(vector_id: str) -> dict:
    """
    Parse a vector ID into its components.  Returns empty strings for
    malformed IDs so callers don't need try/except.
    """
    parts = vector_id.split(":", 3)
    if len(parts) == 4:
        return {
            "doc_id":      parts[0],
            "version":     int(parts[1]) if parts[1].isdigit() else 0,
            "page":        int(parts[2]) if parts[2].isdigit() else 0,
            "chunk_index": int(parts[3]) if parts[3].isdigit() else 0,
        }
    return {"doc_id": vector_id, "version": 0, "page": 0, "chunk_index": 0}


# ─── Metadata builder ─────────────────────────────────────────────────────────

def build_metadata(
    embedded_chunk,                  # EmbeddedChunk
    version:     int       = 1,
    tenant_id:   str       = "default",
    corpus:      str       = "default",
    source_url:  str | None = None,
) -> dict[str, Any]:
    """
    Build the Pinecone metadata dict from an EmbeddedChunk.

    published_at and created_at are stored as Unix epoch integers so
    Pinecone numeric range filters work correctly.

    Args:
        embedded_chunk: EmbeddedChunk instance from Phase 3.
        version:        Document version integer.
        tenant_id:      Tenant identifier.
        corpus:         Corpus / collection name.
        source_url:     Optional origin URL / file path.
    """
    ec = embedded_chunk

    # Convert published_at ISO string → epoch int
    pub_epoch = _to_epoch(ec.published_at)

    # MIME type from file extension
    mime = _ext_to_mime(ec.file_type)

    # Section path from title_prefix (strip doc title prefix)
    section_path = _extract_section_path(
        getattr(ec, "title_prefix", "") or "",
        ec.doc_name,
    )

    # Page number — use first page if multiple
    page = min(ec.page_numbers) if ec.page_numbers else 0

    return {
        "doc_id":       ec.doc_id,
        "version":      version,
        "doc_ref_name": ec.doc_name,
        "page":         page,
        "section_path": section_path,
        "text":         (ec.text or "")[:1_000],   # Pinecone metadata cap
        "published_at": pub_epoch,
        "source_url":   source_url or "",
        "mime":         mime,
        "language":     ec.doc_language or "en",
        "checksum":     (ec.content_hash or "")[:16],
        "chunk_index":  ec.chunk_index,
        "chunk_type":   ec.chunk_type,
        "strategy":     ec.strategy,
        "created_at":   int(time.time()),
        "tenant_id":    tenant_id,
        "corpus":       corpus,
        "parent_id":    ec.parent_chunk_id or "",
    }


# ─── Full upsert record ───────────────────────────────────────────────────────

class PineconeRecord(BaseModel):
    """
    One record ready for ``index.upsert(vectors=[record.to_dict()])``.

    id      : "{doc_id}:{version}:{page}:{chunk_index}"
    values  : dense float32 vector (1 536 dims)
    sparse_values : BM25/SPLADE sparse vector (optional)
    metadata : full metadata dict
    """
    model_config = {"protected_namespaces": ()}

    id:            str
    values:        list[float]
    sparse_values: dict | None = None   # {"indices": [...], "values": [...]}
    metadata:      dict[str, Any] = Field(default_factory=dict)
    namespace:     str = "default__default"

    @classmethod
    def from_embedded_chunk(
        cls,
        embedded_chunk,
        version:    int        = 1,
        tenant_id:  str        = "default",
        corpus:     str        = "default",
        source_url: str | None = None,
    ) -> "PineconeRecord":
        """Build a PineconeRecord from an EmbeddedChunk."""
        ec         = embedded_chunk
        page       = min(ec.page_numbers) if ec.page_numbers else 0
        vector_id  = build_vector_id(ec.doc_id, version, page, ec.chunk_index)
        namespace  = resolve_namespace(tenant_id, corpus)
        metadata   = build_metadata(ec, version, tenant_id, corpus, source_url)

        sparse = None
        if ec.sparse_vector and not ec.sparse_vector.is_empty():
            sparse = ec.sparse_vector.to_dict()

        return cls(
            id=vector_id,
            values=ec.dense_vector,
            sparse_values=sparse,
            metadata=metadata,
            namespace=namespace,
        )

    def to_dict(self) -> dict:
        """Pinecone SDK-compatible dict for upsert."""
        d: dict = {"id": self.id, "values": self.values, "metadata": self.metadata}
        if self.sparse_values:
            d["sparse_values"] = self.sparse_values
        return d


# ─── Helpers ─────────────────────────────────────────────────────────────────

_MIME_MAP: dict[str, str] = {
    "pdf":  "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv":  "text/csv",
    "html": "text/html",
    "txt":  "text/plain",
}


def _ext_to_mime(ext: str) -> str:
    return _MIME_MAP.get((ext or "").lower().lstrip("."), "application/octet-stream")


def _to_epoch(date_str: str | None) -> int:
    """Convert an ISO date string or None to a Unix epoch integer."""
    if not date_str:
        return 0
    try:
        # Handle "2024-01-15" and full ISO-8601 strings
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def _extract_section_path(title_prefix: str, doc_name: str) -> str:
    """
    Strip the document-title portion from title_prefix to get the
    section path (heading breadcrumb).

    title_prefix format: "Doc Title | Section Name | Sub-section"
    section_path:                   "Section Name | Sub-section"
    """
    if not title_prefix:
        return ""
    stem = doc_name.rsplit(".", 1)[0].replace("_", " ")
    parts = title_prefix.split(" | ")
    # Drop parts that are just the filename stem
    filtered = [p for p in parts if p.strip().lower() != stem.lower()]
    return " | ".join(filtered)
