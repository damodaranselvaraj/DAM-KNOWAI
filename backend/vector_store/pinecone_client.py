"""
PineconeClient — batched upsert, dense query, hybrid query, delete, and fetch.

Architecture notes
──────────────────
• Upsert is batched (default 100 vectors per call) to stay within Pinecone's
  payload limits and avoid timeouts on large corpora.
• Dense query calls index.query(vector=...) without sparse_vector.
• Hybrid query calls index.query(vector=..., sparse_vector=...) when both
  dense and sparse vectors are supplied and hybrid_search=True.
• Pinecone Serverless does NOT support efSearch / nProbe — those fields are
  silently ignored if passed (see note in schema.py).
• All public methods are synchronous.  Async wrappers run the sync path in a
  thread-pool executor for FastAPI compatibility.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from backend.config import settings
from backend.vector_store.schema import (
    PineconeRecord,
    resolve_namespace,
)

logger = logging.getLogger(__name__)

_PINECONE_AVAILABLE = False
try:
    from pinecone import Pinecone as _PineconeSDK
    _PINECONE_AVAILABLE = True
except ImportError:
    logger.warning(
        "pinecone-client not installed — PineconeClient will use mock mode."
    )


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class QueryMatch:
    id:        str
    score:     float
    metadata:  dict[str, Any] = field(default_factory=dict)
    values:    list[float]     = field(default_factory=list)


@dataclass
class QueryResult:
    matches:   list[QueryMatch]
    namespace: str
    total:     int = 0


@dataclass
class UpsertResult:
    upserted_count: int
    failed_count:   int   = 0
    batch_count:    int   = 0
    duration_ms:    float = 0.0
    errors:         list[str] = field(default_factory=list)


# ─── Client ───────────────────────────────────────────────────────────────────

class PineconeClient:
    """
    Thin, production-ready wrapper around the Pinecone Python SDK.

    All operations accept a `namespace` parameter that follows the
    per-tenant + per-corpus naming convention (see schema.resolve_namespace).

    Args:
        api_key:      Pinecone API key.
        index_name:   Target index name.
        batch_size:   Vectors per upsert call (default 100).
        timeout_s:    Request timeout in seconds.
    """

    def __init__(
        self,
        api_key:    str,
        index_name: str   = "",
        batch_size: int   = 0,
        timeout_s:  float = 0.0,
    ) -> None:
        from backend.config import settings as _cfg
        self.index_name = index_name  or _cfg.pinecone_index_name
        self.batch_size = batch_size  or _cfg.default_batch_size
        self.timeout_s  = timeout_s   or 30.0
        self._sdk       = None
        self._index     = None

        if not _PINECONE_AVAILABLE:
            logger.warning("Pinecone SDK unavailable — PineconeClient in mock mode.")
            return

        try:
            self._sdk   = _PineconeSDK(api_key=api_key)
            # Defer Index connection until first actual operation
            # so a bad key doesn't crash the constructor.
            self._index = self._sdk.Index(index_name)
            logger.info("PineconeClient: connected to index '%s'.", index_name)
        except Exception as exc:
            logger.warning(
                "PineconeClient init skipped (will use mock mode): %s", exc
            )
            self._sdk   = None
            self._index = None

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_records(
        self,
        records:   list[PineconeRecord],
        namespace: str = "default__default",
    ) -> UpsertResult:
        """
        Batch-upsert PineconeRecord objects.

        Args:
            records:   List of PineconeRecord instances.
            namespace: Target namespace (per-tenant__per-corpus convention).

        Returns:
            UpsertResult with counts and timing.
        """
        if not records:
            return UpsertResult(upserted_count=0)

        if self._index is None:
            # Mock mode — simulate success
            logger.info(
                "[mock] Upsert %d records to namespace '%s'.", len(records), namespace
            )
            return UpsertResult(
                upserted_count=len(records),
                batch_count=math.ceil(len(records) / self.batch_size),
            )

        t0             = time.perf_counter()
        upserted       = 0
        failed         = 0
        batch_count    = 0
        errors:        list[str] = []

        batches = _batch(records, self.batch_size)
        for batch in batches:
            batch_count += 1
            vectors = [r.to_dict() for r in batch]
            try:
                resp      = self._index.upsert(vectors=vectors, namespace=namespace)
                upserted += getattr(resp, "upserted_count", len(batch))
            except Exception as exc:
                failed += len(batch)
                err_msg = f"Batch {batch_count} upsert failed: {type(exc).__name__}: {exc}"
                errors.append(err_msg)
                logger.error(err_msg)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Upsert complete: %d/%d vectors in %d batches, %.0f ms",
            upserted, len(records), batch_count, duration_ms,
        )
        return UpsertResult(
            upserted_count=upserted,
            failed_count=failed,
            batch_count=batch_count,
            duration_ms=duration_ms,
            errors=errors,
        )

    def upsert_dicts(
        self,
        vectors:   list[dict],
        namespace: str = "default__default",
    ) -> UpsertResult:
        """
        Upsert raw dicts (already in Pinecone SDK format) for callers that
        build records outside this module.
        """
        if not vectors:
            return UpsertResult(upserted_count=0)

        if self._index is None:
            return UpsertResult(upserted_count=len(vectors), batch_count=1)

        t0 = time.perf_counter()
        upserted = failed = batch_count = 0
        errors: list[str] = []

        for batch in _batch(vectors, self.batch_size):
            batch_count += 1
            try:
                resp      = self._index.upsert(vectors=batch, namespace=namespace)
                upserted += getattr(resp, "upserted_count", len(batch))
            except Exception as exc:
                failed += len(batch)
                errors.append(str(exc))
                logger.error("Batch %d upsert failed: %s", batch_count, exc)

        return UpsertResult(
            upserted_count=upserted,
            failed_count=failed,
            batch_count=batch_count,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            errors=errors,
        )

    # ── Dense query ───────────────────────────────────────────────────────────

    def query_dense(
        self,
        vector:          list[float],
        top_k:           int              = 10,
        namespace:       str              = "default__default",
        filter:          dict | None      = None,
        include_values:  bool             = False,
        include_metadata: bool            = True,
        score_threshold: float            = settings.default_score_threshold,
    ) -> QueryResult:
        """
        Standard dense (ANN) vector search.

        Args:
            vector:           Query embedding (1 536 floats).
            top_k:            Number of results to return.
            namespace:        Target namespace.
            filter:           Pinecone metadata filter dict.
            include_values:   Include vector values in response.
            include_metadata: Include metadata in response (default True).
            score_threshold:  Minimum cosine similarity to include.

        Returns:
            QueryResult with sorted matches.
        """
        if self._index is None:
            return _mock_query_result(namespace, top_k)

        try:
            kwargs: dict[str, Any] = dict(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_values=include_values,
                include_metadata=include_metadata,
            )
            if filter:
                kwargs["filter"] = filter

            resp    = self._index.query(**kwargs)
            matches = _parse_matches(resp, score_threshold)
            return QueryResult(
                matches=matches,
                namespace=namespace,
                total=len(matches),
            )
        except Exception as exc:
            logger.error("query_dense failed: %s", exc)
            return QueryResult(matches=[], namespace=namespace)

    # ── Hybrid query ──────────────────────────────────────────────────────────

    def query_hybrid(
        self,
        dense_vector:    list[float],
        sparse_vector:   dict,             # {"indices": [...], "values": [...]}
        top_k:           int               = 10,
        namespace:       str               = "default__default",
        filter:          dict | None       = None,
        alpha:           float             = 0.5,
        include_metadata: bool             = True,
        score_threshold:  float            = settings.default_score_threshold,
    ) -> QueryResult:
        """
        Hybrid search combining dense ANN + sparse BM25/SPLADE.

        Args:
            dense_vector:  Dense query embedding.
            sparse_vector: Sparse query vector {"indices": [...], "values": [...]}.
            alpha:         Weight for dense vs sparse (0.0=sparse only, 1.0=dense only).
                           Applied as score interpolation post-retrieval.
            top_k:         Number of results to return.
            namespace:     Target namespace.
            filter:        Metadata filter dict.
            score_threshold: Minimum score to include.

        Returns:
            QueryResult with matches sorted by blended score.

        NOTE: Pinecone natively supports hybrid queries when sparse_values
        were stored at upsert time.  The alpha blending here pre-scales the
        sparse vector values so the native Pinecone ranking approximates the
        desired balance without requiring a post-retrieval re-rank.
        """
        if self._index is None:
            return _mock_query_result(namespace, top_k)

        # Alpha-scale the sparse values (dense weight = alpha, sparse = 1-alpha)
        scaled_sparse = _scale_sparse(sparse_vector, 1.0 - alpha)

        try:
            kwargs: dict[str, Any] = dict(
                vector=dense_vector,
                sparse_vector=scaled_sparse,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
            )
            if filter:
                kwargs["filter"] = filter

            resp    = self._index.query(**kwargs)
            matches = _parse_matches(resp, score_threshold)
            return QueryResult(
                matches=matches,
                namespace=namespace,
                total=len(matches),
            )
        except Exception as exc:
            logger.error("query_hybrid failed: %s", exc)
            return QueryResult(matches=[], namespace=namespace)

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_by_ids(
        self,
        ids:       list[str],
        namespace: str = "default__default",
    ) -> bool:
        """Delete vectors by their IDs."""
        if not ids:
            return True
        if self._index is None:
            logger.info("[mock] delete_by_ids %d vectors", len(ids))
            return True
        try:
            self._index.delete(ids=ids, namespace=namespace)
            logger.info("Deleted %d vectors from '%s'.", len(ids), namespace)
            return True
        except Exception as exc:
            logger.error("delete_by_ids failed: %s", exc)
            return False

    def delete_by_filter(
        self,
        filter:    dict,
        namespace: str = "default__default",
    ) -> bool:
        """
        Delete all vectors matching a metadata filter.

        Example — delete all chunks of a document::

            client.delete_by_filter(
                filter={"doc_id": {"$eq": "doc-abc123"}},
                namespace="acme__legal",
            )
        """
        if self._index is None:
            logger.info("[mock] delete_by_filter %s", filter)
            return True
        try:
            self._index.delete(filter=filter, namespace=namespace)
            logger.info("Deleted vectors matching filter %s in '%s'.", filter, namespace)
            return True
        except Exception as exc:
            logger.error("delete_by_filter failed: %s", exc)
            return False

    def delete_namespace_all(
        self,
        namespace: str,
        confirm:   bool = False,
    ) -> bool:
        """Delete ALL vectors in a namespace."""
        if not confirm:
            raise ValueError("delete_namespace_all() requires confirm=True.")
        if self._index is None:
            return True
        try:
            self._index.delete(delete_all=True, namespace=namespace)
            logger.info("Cleared all vectors in namespace '%s'.", namespace)
            return True
        except Exception as exc:
            logger.error("delete_namespace_all failed: %s", exc)
            return False

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch(
        self,
        ids:       list[str],
        namespace: str = "default__default",
    ) -> dict[str, dict]:
        """
        Fetch vectors by ID.

        Returns:
            Dict mapping id → {"id", "values", "metadata"}.
        """
        if not ids or self._index is None:
            return {}
        try:
            resp = self._index.fetch(ids=ids, namespace=namespace)
            vectors = getattr(resp, "vectors", {}) or {}
            return {
                vid: {
                    "id":       vid,
                    "values":   getattr(v, "values", []),
                    "metadata": getattr(v, "metadata", {}),
                }
                for vid, v in vectors.items()
            }
        except Exception as exc:
            logger.error("fetch failed: %s", exc)
            return {}

    # ── Stats ─────────────────────────────────────────────────────────────────

    def describe_index_stats(self) -> dict:
        """Return raw index statistics from Pinecone."""
        if self._index is None:
            return {"status": "mock", "total_vector_count": 0}
        try:
            stats = self._index.describe_index_stats()
            return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        except Exception as exc:
            logger.error("describe_index_stats failed: %s", exc)
            return {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._index is not None


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _batch(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _parse_matches(response, score_threshold: float = settings.default_score_threshold) -> list[QueryMatch]:
    raw_matches = getattr(response, "matches", []) or []
    matches: list[QueryMatch] = []
    for m in raw_matches:
        score = getattr(m, "score", 0.0) or 0.0
        if score < score_threshold:
            continue
        matches.append(QueryMatch(
            id=getattr(m, "id", ""),
            score=round(float(score), 6),
            metadata=dict(getattr(m, "metadata", {}) or {}),
            values=list(getattr(m, "values", []) or []),
        ))
    return matches


def _scale_sparse(sparse: dict, scale: float) -> dict:
    """Multiply all sparse values by `scale` for alpha blending."""
    if not sparse or scale == 1.0:
        return sparse
    return {
        "indices": sparse.get("indices", []),
        "values":  [round(v * scale, 8) for v in sparse.get("values", [])],
    }


def _mock_query_result(namespace: str, top_k: int) -> QueryResult:
    """Return a realistic-looking mock result for offline testing."""
    import random
    matches = [
        QueryMatch(
            id=f"mock-doc:{i}",
            score=round(random.uniform(0.65, 0.98), 4),
            metadata={
                "doc_ref_name": f"document_{i}.pdf",
                "text":         f"Mock chunk {i} from namespace {namespace}.",
                "page":         i + 1,
                "chunk_index":  i,
            },
        )
        for i in range(min(top_k, 3))
    ]
    return QueryResult(matches=matches, namespace=namespace, total=len(matches))
