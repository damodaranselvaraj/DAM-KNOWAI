"""
VectorStorePipeline — Phase 4 orchestrator.

Receives EmbeddingResult from Phase 3 and upserts all EmbeddedChunks
into Pinecone under the correct namespace.

Flow
────
  EmbeddingResult (list[EmbeddedChunk])
       │
       ├─ Build PineconeRecord per chunk
       │    id       = "{doc_id}:{version}:{page}:{chunk_index}"
       │    values   = dense vector (1 536 floats)
       │    sparse   = BM25/SPLADE sparse vector (if hybrid_search=True)
       │    metadata = full provenance dict
       │
       ├─ Resolve namespace  (tenant_id + corpus → "acme__legal")
       │
       ├─ Batched upsert (100 vectors/call, retried on transient error)
       │
       └─ VectorStoreResult  (upserted_count, failed_count, timing)

Design decisions
────────────────
• Index creation is an operator concern — the pipeline assumes the index
  already exists and raises PineconeIndexNotFoundError if it doesn't.
• Namespace resolution is automatic from tenant_id + corpus kwargs.
• Hybrid search (sparse_values in the record) is opt-in.  Pass
  enable_sparse=False to skip BM25 vectors and save upsert payload size.
• efSearch / nProbe are NOT set here — they don't exist in Pinecone
  Serverless (see schema.py for the full note).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from backend.models.embedded_chunk import EmbeddedChunk, EmbeddingResult
from backend.vector_store.pinecone_client import PineconeClient, UpsertResult
from backend.vector_store.schema import PineconeRecord, resolve_namespace

logger = logging.getLogger(__name__)


# ─── Result envelope ─────────────────────────────────────────────────────────

@dataclass
class VectorStoreResult:
    doc_id:         str
    doc_name:       str
    namespace:      str
    upserted_count: int   = 0
    failed_count:   int   = 0
    skipped_count:  int   = 0   # empty dense vectors skipped
    batch_count:    int   = 0
    duration_ms:    float = 0.0
    errors:         list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed_count == 0 and self.upserted_count > 0

    def summary(self) -> dict:
        return {
            "doc_id":         self.doc_id,
            "doc_name":       self.doc_name,
            "namespace":      self.namespace,
            "upserted":       self.upserted_count,
            "failed":         self.failed_count,
            "skipped":        self.skipped_count,
            "batches":        self.batch_count,
            "duration_ms":    round(self.duration_ms, 1),
            "success":        self.success,
            "errors":         self.errors,
        }


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class VectorStorePipeline:
    """
    Stateless Phase 4 orchestrator.  Thread-safe after construction.

    Args:
        client:          Initialised PineconeClient.
        enable_sparse:   Include sparse_values in upsert records (hybrid search).
        score_threshold: Minimum cosine score filter (applied at query time,
                         not at upsert time). Defaults to
                         backend.config.settings.default_score_threshold.
    """

    def __init__(
        self,
        client:           PineconeClient,
        enable_sparse:    bool  = True,
        score_threshold:  float | None = None,
    ) -> None:
        from backend.config import settings as _cfg
        self._client          = client
        self.enable_sparse    = enable_sparse
        self.score_threshold  = (
            score_threshold if score_threshold is not None else _cfg.default_score_threshold
        )

    # ── Upsert from EmbeddingResult ───────────────────────────────────────────

    def store_embedding_result(
        self,
        result:     EmbeddingResult,
        tenant_id:  str        = "default",
        corpus:     str        = "default",
        version:    int        = 1,
        source_url: str | None = None,
    ) -> VectorStoreResult:
        """
        Build Pinecone records from an EmbeddingResult and upsert them.

        Args:
            result:     EmbeddingResult from Phase 3.
            tenant_id:  Tenant identifier (used for namespace).
            corpus:     Corpus / collection name (used for namespace).
            version:    Document version integer (used in vector ID).
            source_url: Optional origin URL stored in metadata.

        Returns:
            VectorStoreResult with counts and timing.
        """
        namespace = resolve_namespace(tenant_id, corpus)
        vs_result = VectorStoreResult(
            doc_id=result.doc_id,
            doc_name=result.doc_name,
            namespace=namespace,
        )

        if not result.embedded_chunks:
            logger.warning(
                "No embedded chunks in EmbeddingResult for '%s' — nothing to upsert.",
                result.doc_name,
            )
            return vs_result

        t0 = time.perf_counter()

        records, skipped = self._build_records(
            result.embedded_chunks,
            version=version,
            tenant_id=tenant_id,
            corpus=corpus,
            source_url=source_url,
        )
        vs_result.skipped_count = skipped

        if not records:
            logger.warning(
                "All %d chunks skipped (no dense vector) for '%s'.",
                len(result.embedded_chunks), result.doc_name,
            )
            vs_result.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return vs_result

        upsert_result = self._client.upsert_records(records, namespace=namespace)

        vs_result.upserted_count = upsert_result.upserted_count
        vs_result.failed_count   = upsert_result.failed_count
        vs_result.batch_count    = upsert_result.batch_count
        vs_result.errors         = upsert_result.errors
        vs_result.duration_ms    = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "VectorStorePipeline '%s': %d upserted, %d failed, %d skipped "
            "into namespace '%s' (%.0f ms)",
            result.doc_name,
            vs_result.upserted_count,
            vs_result.failed_count,
            vs_result.skipped_count,
            namespace,
            vs_result.duration_ms,
        )
        return vs_result

    # ── Upsert list of EmbeddedChunks directly ────────────────────────────────

    def store_chunks(
        self,
        chunks:     list[EmbeddedChunk],
        doc_id:     str        = "",
        doc_name:   str        = "",
        tenant_id:  str        = "default",
        corpus:     str        = "default",
        version:    int        = 1,
        source_url: str | None = None,
    ) -> VectorStoreResult:
        """
        Upsert a flat list of EmbeddedChunk objects directly.

        Use when you have individual chunks rather than a full EmbeddingResult.
        """
        namespace = resolve_namespace(tenant_id, corpus)
        doc_id    = doc_id   or (chunks[0].doc_id   if chunks else "")
        doc_name  = doc_name or (chunks[0].doc_name if chunks else "")

        vs_result = VectorStoreResult(
            doc_id=doc_id, doc_name=doc_name, namespace=namespace
        )

        if not chunks:
            return vs_result

        t0 = time.perf_counter()
        records, skipped = self._build_records(
            chunks, version=version, tenant_id=tenant_id,
            corpus=corpus, source_url=source_url,
        )
        vs_result.skipped_count = skipped

        if records:
            ur = self._client.upsert_records(records, namespace=namespace)
            vs_result.upserted_count = ur.upserted_count
            vs_result.failed_count   = ur.failed_count
            vs_result.batch_count    = ur.batch_count
            vs_result.errors         = ur.errors

        vs_result.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        return vs_result

    # ── Query convenience wrappers ────────────────────────────────────────────

    def query(
        self,
        dense_vector:  list[float],
        sparse_vector: dict | None = None,
        top_k:         int         = 10,
        tenant_id:     str         = "default",
        corpus:        str         = "default",
        filter:        dict | None = None,
        hybrid:        bool        = False,
        alpha:         float       = 0.5,
    ):
        """
        Query the vector store — dense or hybrid depending on flags.

        Args:
            dense_vector:  Query embedding.
            sparse_vector: BM25/SPLADE query vector (required when hybrid=True).
            top_k:         Number of results.
            tenant_id:     Namespace tenant.
            corpus:        Namespace corpus.
            filter:        Pinecone metadata filter dict.
            hybrid:        Use hybrid search (requires sparse_vector).
            alpha:         Dense weight (0=sparse-only, 1=dense-only).

        Returns:
            QueryResult from PineconeClient.
        """
        namespace = resolve_namespace(tenant_id, corpus)

        if hybrid and sparse_vector:
            return self._client.query_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                alpha=alpha,
                score_threshold=self.score_threshold,
            )

        return self._client.query_dense(
            vector=dense_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            score_threshold=self.score_threshold,
        )

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_document(
        self,
        doc_id:    str,
        tenant_id: str  = "default",
        corpus:    str  = "default",
    ) -> bool:
        """Delete all vectors belonging to `doc_id` in the given namespace."""
        namespace = resolve_namespace(tenant_id, corpus)
        return self._client.delete_by_filter(
            filter={"doc_id": {"$eq": doc_id}},
            namespace=namespace,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_records(
        self,
        chunks:     list[EmbeddedChunk],
        version:    int,
        tenant_id:  str,
        corpus:     str,
        source_url: str | None,
    ) -> tuple[list[PineconeRecord], int]:
        """
        Convert EmbeddedChunks to PineconeRecords.
        Returns (records, skipped_count).
        """
        records: list[PineconeRecord] = []
        skipped = 0

        for ec in chunks:
            if not ec.dense_vector or all(v == 0.0 for v in ec.dense_vector):
                skipped += 1
                logger.debug(
                    "Skipping chunk %s — zero dense vector.", ec.chunk_id
                )
                continue

            # Strip sparse if hybrid is disabled for this pipeline
            if not self.enable_sparse:
                import copy
                ec = copy.copy(ec)
                object.__setattr__(ec, "sparse_vector", None)

            rec = PineconeRecord.from_embedded_chunk(
                ec,
                version=version,
                tenant_id=tenant_id,
                corpus=corpus,
                source_url=source_url,
            )
            records.append(rec)

        return records, skipped


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_vector_store_pipeline(
    pinecone_api_key: str = "",
    index_name:       str = "",
    batch_size:       int = 0,
    enable_sparse:    bool = True,
    score_threshold:  float | None = None,
) -> VectorStorePipeline:
    """
    Convenience factory — reads ALL defaults from backend.config.settings.
    Caller-supplied kwargs override settings values when provided.
    """
    from backend.config import settings as _cfg
    api_key  = pinecone_api_key or _cfg.pinecone_api_key
    idx_name = index_name       or _cfg.pinecone_index_name
    bsize    = batch_size       or _cfg.default_batch_size
    threshold = score_threshold if score_threshold is not None else _cfg.default_score_threshold

    client = PineconeClient(
        api_key=api_key,
        index_name=idx_name,
        batch_size=bsize,
    )
    return VectorStorePipeline(
        client=client,
        enable_sparse=enable_sparse,
        score_threshold=threshold,
    )
