"""
Chunk-level corpus store used to keep the BM25 sparse index aligned with
the same chunk granularity used by dense + SPLADE retrieval.

Why this exists
────────────────
Dense and SPLADE vectors are stored in Pinecone keyed by *chunk* id
("{doc_id}:{version}:{page}:{chunk_index}"). The BM25 index used to be
rebuilt from whole-document text (`ParsedDocument.full_text`), which meant
its "documents" were entire files while dense/SPLADE results were
individual chunks — the three retrieval arms could never agree on what a
single "hit" even was.

This store keeps a flat, in-memory map of chunk_id → {text, metadata} so
the BM25 index (backend/retrieval/sparse_retriever.py) can be built from
exactly the same chunks that get embedded and upserted in Phase 3/4.

Like `version_store` (backend/ingestion/versioning.py), this is an
in-memory registry — swap for a persistent implementation if the process
needs to survive restarts without a full re-ingest.
"""
from __future__ import annotations

import logging
from threading import Lock

logger = logging.getLogger(__name__)


class ChunkCorpusStore:
    """Thread-safe in-memory registry of chunk_id → {text, metadata}."""

    def __init__(self) -> None:
        self._chunks: dict[str, dict] = {}
        self._lock = Lock()

    def add_chunks(self, embedded_chunks: list) -> None:
        """Insert/replace entries for a batch of EmbeddedChunk objects."""
        with self._lock:
            for ec in embedded_chunks:
                text = ec.text_with_title or ec.text
                if not text:
                    continue
                self._chunks[ec.chunk_id] = {
                    "text": text,
                    "metadata": {
                        "doc_id":      ec.doc_id,
                        "doc_name":    ec.doc_name,
                        "chunk_index": ec.chunk_index,
                    },
                }

    def remove_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks belonging to *doc_id*. Returns count removed."""
        with self._lock:
            keep = {
                cid: v for cid, v in self._chunks.items()
                if v["metadata"].get("doc_id") != doc_id
            }
            removed = len(self._chunks) - len(keep)
            self._chunks = keep
        return removed

    def all(self) -> list[tuple[str, dict]]:
        with self._lock:
            return list(self._chunks.items())

    def stats(self) -> dict:
        with self._lock:
            return {"total_chunks": len(self._chunks)}


# ─── Module-level singleton ───────────────────────────────────────────────────
chunk_corpus_store = ChunkCorpusStore()
