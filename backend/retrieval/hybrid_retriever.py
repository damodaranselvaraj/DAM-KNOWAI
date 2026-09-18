"""
Hybrid retriever supporting three retrieval modes.

Architecture — ingestion (Phase 3/4) computes and stores all three
representations for every chunk:

    DOCUMENT CHUNKS
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
  Dense SPLADE BM25
    │    │    │
    ▼    ▼    ▼
  Pinecone Pinecone  BM25 Index (local rank_bm25)

At query time, three modes are available (``HybridRetrieverConfig.mode``):

    "dense"          → DenseRetriever only (pure Pinecone ANN query).
    "hybrid_bm25"     → DenseRetriever + SparseRetriever (local BM25 index),
                        fused client-side via Reciprocal Rank Fusion (RRF).
                        [DEFAULT]
    "hybrid_splade"   → A single native Pinecone hybrid query using the
                        dense vector + a query-time SPLADE sparse vector
                        (``sparse_values``). Fusion happens server-side in
                        Pinecone; no RRF step is needed for this mode.

RRF formula (Cormack et al., 2009), used only in "hybrid_bm25" mode:
    RRF(d) = Σ  1 / (k + rank(d))
    where k is a smoothing constant (default 60).

Configuration (all via environment / .env):
    RETRIEVAL_MODE          "dense" | "hybrid_bm25" | "hybrid_splade"   (default: hybrid_bm25)
    DENSE_TOP_K             (default: 50)
    SPARSE_TOP_K            (default: 50)
    RRF_K                   (default: 60)
    DENSE_WEIGHT            (default: 0.5)   — used only for weighted-score fallback
    SPARSE_WEIGHT           (default: 0.5)   — used only for weighted-score fallback
    RETRIEVAL_TIMEOUT_SECONDS (default: 10)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from backend.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from backend.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

class HybridRetrieverConfig(BaseSettings):
    """Loaded from environment variables / .env — all defaults from backend.config.settings."""

    # NOTE: typed as `str` (not Literal) so an empty/unset env var doesn't
    # fail validation before `model_post_init` can fill it from settings.
    # The Literal constraint is enforced in `_validate_weights` below instead.
    mode: str = Field("", alias="RETRIEVAL_MODE")
    dense_top_k: int   = Field(0,   alias="DENSE_TOP_K")
    sparse_top_k: int  = Field(0,   alias="SPARSE_TOP_K")
    rrf_k: int         = Field(0,   alias="RRF_K")
    dense_weight: float  = Field(0.0, alias="DENSE_WEIGHT")
    sparse_weight: float = Field(0.0, alias="SPARSE_WEIGHT")
    timeout_seconds: float = Field(0.0, alias="RETRIEVAL_TIMEOUT_SECONDS")

    def model_post_init(self, __context) -> None:
        """Fill zero/empty fields from central settings."""
        from backend.config import settings as _cfg
        if not self.mode:
            object.__setattr__(self, "mode", _cfg.retrieval_mode)
        if not self.dense_top_k:
            object.__setattr__(self, "dense_top_k", _cfg.dense_top_k)
        if not self.sparse_top_k:
            object.__setattr__(self, "sparse_top_k", _cfg.sparse_top_k)
        if not self.rrf_k:
            object.__setattr__(self, "rrf_k", _cfg.rrf_k)
        if not self.dense_weight:
            object.__setattr__(self, "dense_weight", _cfg.dense_weight)
        if not self.sparse_weight:
            object.__setattr__(self, "sparse_weight", _cfg.sparse_weight)
        if not self.timeout_seconds:
            object.__setattr__(self, "timeout_seconds", _cfg.retrieval_timeout_seconds)

    @model_validator(mode="after")
    def _validate_weights(self) -> "HybridRetrieverConfig":
        if self.mode not in ("dense", "hybrid_bm25", "hybrid_splade"):
            raise ValueError(
                f"RETRIEVAL_MODE must be 'dense', 'hybrid_bm25' or 'hybrid_splade', got '{self.mode}'."
            )
        total = round(self.dense_weight + self.sparse_weight, 6)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"DENSE_WEIGHT ({self.dense_weight}) + "
                f"SPARSE_WEIGHT ({self.sparse_weight}) must equal 1.0, got {total}."
            )
        return self

    model_config = {
        "populate_by_name": True,
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "protected_namespaces": (),
    }

# ─── Hybrid retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Production-grade hybrid retriever.

    Runs DenseRetriever and SparseRetriever **concurrently** via a
    ThreadPoolExecutor, merges results with Reciprocal Rank Fusion, and
    returns a single ranked list.

    Usage::

        config   = HybridRetrieverConfig()
        hybrid   = HybridRetriever(config, dense_retriever, sparse_retriever)
        results  = hybrid.retrieve("What is the claims process?", top_k=10)
    """

    def __init__(
        self,
        config: HybridRetrieverConfig,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        sparse_embedder=None,   # backend.ingestion.embedders.sparse_embedder.SparseEmbedder | None
    ) -> None:
        self.config = config
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        # Lazily constructed on first "hybrid_splade" query if not injected —
        # avoids loading the SPLADE model for deployments that never use it.
        self._sparse_embedder = sparse_embedder

        logger.info(
            json.dumps({
                "event": "hybrid_retriever_init",
                "mode": config.mode,
                "dense_top_k": config.dense_top_k,
                "sparse_top_k": config.sparse_top_k,
                "rrf_k": config.rrf_k,
                "timeout_seconds": config.timeout_seconds,
            })
        )

    def _get_sparse_embedder(self):
        if self._sparse_embedder is None:
            from backend.ingestion.embedders.sparse_embedder import SparseEmbedder
            self._sparse_embedder = SparseEmbedder()
        return self._sparse_embedder

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def rrf_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        k: int | None = None,
    ) -> List[RetrievalResult]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        For every unique document ID across both lists:
            rrf_score(d) = 1/(k + dense_rank) + 1/(k + sparse_rank)

        Documents absent from a list are penalised with
        ``rank = len(list) + 1``.

        Parameters
        ----------
        dense_results:
            Ranked list from the dense retriever (index 0 = rank 1).
        sparse_results:
            Ranked list from the sparse retriever (index 0 = rank 1).
        k:
            RRF smoothing constant. Defaults to ``config.rrf_k``.

        Returns
        -------
        list[RetrievalResult]
            Merged list sorted descending by ``rrf_score``.
            Each result carries ``dense_rank``, ``sparse_rank``,
            ``dense_score``, ``sparse_score``, and ``rrf_score``.
        """
        rrf_k = k if k is not None else self.config.rrf_k

        # Build lookup tables: id → (rank 1-indexed, result)
        dense_map: Dict[str, Tuple[int, RetrievalResult]] = {
            r.id: (i + 1, r) for i, r in enumerate(dense_results)
        }
        sparse_map: Dict[str, Tuple[int, RetrievalResult]] = {
            r.id: (i + 1, r) for i, r in enumerate(sparse_results)
        }

        # Penalty ranks for absent documents
        dense_penalty  = len(dense_results)  + 1
        sparse_penalty = len(sparse_results) + 1

        all_ids = set(dense_map) | set(sparse_map)
        fused: List[RetrievalResult] = []

        for doc_id in all_ids:
            d_rank, d_result = dense_map.get(doc_id,  (dense_penalty,  None))
            s_rank, s_result = sparse_map.get(doc_id, (sparse_penalty, None))

            rrf_score = 1.0 / (rrf_k + d_rank) + 1.0 / (rrf_k + s_rank)

            # Prefer dense result as the base; fall back to sparse
            base: RetrievalResult = (d_result or s_result).model_copy()  # type: ignore[union-attr]

            fused.append(
                base.model_copy(
                    update={
                        "source": "hybrid",
                        "score": rrf_score,
                        "rrf_score": rrf_score,
                        "dense_rank": d_rank if d_result else None,
                        "sparse_rank": s_rank if s_result else None,
                        "dense_score": d_result.score if d_result else None,
                        "sparse_score": s_result.score if s_result else None,
                        "metadata": {
                            **(d_result.metadata if d_result else {}),
                            **(s_result.metadata if s_result else {}),
                            "_dense_rank": d_rank if d_result else None,
                            "_sparse_rank": s_rank if s_result else None,
                            "_dense_score": d_result.score if d_result else None,
                            "_sparse_score": s_result.score if s_result else None,
                            "_rrf_score": rrf_score,
                        },
                    }
                )
            )

        fused.sort(key=lambda r: r.rrf_score or 0.0, reverse=True)
        return fused

    # ── Concurrent execution ─────────────────────────────────────────────────

    def _run_concurrent(
        self,
        query: str,
        namespace: str | None,
        metadata_filter: dict | None,
        dense_top_k: int,
        sparse_top_k: int,
        query_id: str,
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        Execute dense and sparse retrieval concurrently.

        If one retriever times out or raises, a warning is logged and an
        empty list is returned for that arm — the pipeline degrades
        gracefully to a single-arm result.

        Returns
        -------
        (dense_results, sparse_results)
        """
        dense_results:  List[RetrievalResult] = []
        sparse_results: List[RetrievalResult] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(
                self._dense.retrieve,
                query,
                dense_top_k,
                namespace,
                metadata_filter,
                query_id,
            )
            sparse_future = pool.submit(
                self._sparse.retrieve,
                query,
                sparse_top_k,
                None,   # filter_ids handled externally
                query_id,
            )

            # Dense
            try:
                dense_results = dense_future.result(timeout=self.config.timeout_seconds)
            except FuturesTimeout:
                logger.warning(
                    json.dumps({
                        "event": "dense_timeout",
                        "query_id": query_id,
                        "timeout_seconds": self.config.timeout_seconds,
                        "fallback": "sparse only",
                    })
                )
            except Exception as exc:
                logger.warning(
                    json.dumps({
                        "event": "dense_error",
                        "query_id": query_id,
                        "error": str(exc),
                        "fallback": "sparse only",
                    })
                )

            # Sparse
            try:
                sparse_results = sparse_future.result(timeout=self.config.timeout_seconds)
            except FuturesTimeout:
                logger.warning(
                    json.dumps({
                        "event": "sparse_timeout",
                        "query_id": query_id,
                        "timeout_seconds": self.config.timeout_seconds,
                        "fallback": "dense only",
                    })
                )
            except Exception as exc:
                logger.warning(
                    json.dumps({
                        "event": "sparse_error",
                        "query_id": query_id,
                        "error": str(exc),
                        "fallback": "dense only",
                    })
                )

        return dense_results, sparse_results

    # ── Main retrieve ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
        query_id: str | None = None,
        mode: str | None = None,
    ) -> List[RetrievalResult]:
        """
        Execute the retrieval pipeline for *query*.

        Routing (``mode`` param, falls back to ``config.mode``):
        - ``"dense"``          → dense retriever only (pure Pinecone ANN).
        - ``"hybrid_bm25"``     → dense (Pinecone) + BM25 (local rank_bm25
                                 index) run concurrently, fused client-side
                                 via Reciprocal Rank Fusion.  [DEFAULT]
        - ``"hybrid_splade"``   → single native Pinecone hybrid query using
                                 the dense vector + a query-time SPLADE
                                 sparse vector.  Server-side fusion; no RRF.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Number of results to return after fusion. Falls back to the
            larger of ``config.dense_top_k`` / ``config.sparse_top_k``.
        namespace:
            Pinecone namespace forwarded to the dense retriever.
        metadata_filter:
            Pinecone metadata filter forwarded to the dense retriever.
        query_id:
            Caller-supplied trace ID; auto-generated when omitted.
        mode:
            Per-call override of the retrieval mode. Falls back to
            ``self.config.mode`` (default "hybrid_bm25") when omitted.

        Returns
        -------
        list[RetrievalResult]
            Ranked list (descending score) trimmed to *top_k*.
        """
        query_id = query_id or str(uuid.uuid4())
        effective_top_k = top_k or max(
            self.config.dense_top_k, self.config.sparse_top_k
        )
        t0 = time.perf_counter()

        dense_results:  List[RetrievalResult] = []
        sparse_results: List[RetrievalResult] = []
        fusion_ms = 0.0

        mode = mode or self.config.mode

        # ── Route ─────────────────────────────────────────────────────────────
        if mode == "dense":
            dense_results = self._dense.retrieve(
                query,
                self.config.dense_top_k,
                namespace,
                metadata_filter,
                query_id,
            )
            results = dense_results[:effective_top_k]

        elif mode == "hybrid_splade":
            # Single native Pinecone hybrid query: dense vector + query-time
            # SPLADE sparse vector. Fusion happens server-side in Pinecone.
            splade_vec = None
            try:
                sparse_result = self._get_sparse_embedder().encode(query)
                if sparse_result and not sparse_result.is_empty():
                    splade_vec = sparse_result.to_dict()
            except Exception as exc:
                logger.warning(
                    json.dumps({
                        "event": "splade_query_encode_failed",
                        "query_id": query_id,
                        "error": str(exc),
                        "fallback": "dense only",
                    })
                )

            dense_results = self._dense.retrieve(
                query,
                self.config.dense_top_k,
                namespace,
                metadata_filter,
                query_id,
                sparse_vector=splade_vec,
            )
            results = dense_results[:effective_top_k]

        else:  # hybrid_bm25 (default)
            dense_results, sparse_results = self._run_concurrent(
                query,
                namespace,
                metadata_filter,
                self.config.dense_top_k,
                self.config.sparse_top_k,
                query_id,
            )

            fusion_start = time.perf_counter()
            fused = self.rrf_fusion(dense_results, sparse_results)
            fusion_ms = (time.perf_counter() - fusion_start) * 1_000

            results = fused[:effective_top_k]

        total_ms = (time.perf_counter() - t0) * 1_000

        # ── Structured log ────────────────────────────────────────────────────
        logger.info(
            json.dumps({
                "event": "hybrid_retrieve",
                "query_id": query_id,
                "mode": mode,
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "fused_count": len(results),
                "fusion_ms": round(fusion_ms, 1),
                "total_latency_ms": round(total_ms, 1),
            })
        )

        return results
