"""
Dense retriever using Pinecone + OpenAI embeddings.

Responsibilities:
- Embed queries with text-embedding-3-small (or configured model)
- Query Pinecone with metadata filters and namespace routing
- Return ranked RetrievalResult list with structured latency logs

Configuration (all via environment / .env):
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT
    DENSE_TOP_K              (default: 50)
    EMBEDDING_MODEL          (default: text-embedding-3-small)
    EMBEDDING_DIMENSION      (default: 1536)
    DENSE_TIMEOUT_SECONDS    (default: 10)
    DENSE_MAX_RETRIES        (default: 3)
"""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from typing import Any, List, Literal, Optional

import openai
from openai import OpenAI
from pinecone import Pinecone, PineconeException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

class DenseRetrieverConfig(BaseSettings):
    """Loaded from environment variables / .env — all defaults from backend.config.settings."""

    # Pinecone
    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("", alias="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field("", alias="PINECONE_ENVIRONMENT")

    # Retrieval
    top_k: int = Field(0, alias="DENSE_TOP_K")
    timeout_seconds: int = Field(0, alias="DENSE_TIMEOUT_SECONDS")
    max_retries: int = Field(0, alias="DENSE_MAX_RETRIES")

    # Embedding
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    embedding_model: str = Field("", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(0, alias="EMBEDDING_DIMENSION")

    # Result filtering — must default to settings.default_score_threshold
    # (0.75) to match backend.vector_store.pinecone_client's query defaults.
    # Leaving this at 0.0 silently disables similarity filtering and lets
    # low-relevance chunks reach the LLM context.
    score_threshold: float = Field(0.0, alias="DEFAULT_SCORE_THRESHOLD")

    model_config = {
        "populate_by_name": True,
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "protected_namespaces": (),
    }

    def model_post_init(self, __context) -> None:
        """Fill zero/empty fields from central settings."""
        from backend.config import settings as _cfg
        if not self.pinecone_index_name:
            object.__setattr__(self, "pinecone_index_name", _cfg.pinecone_index_name)
        if not self.pinecone_environment:
            object.__setattr__(self, "pinecone_environment", _cfg.pinecone_environment)
        if not self.top_k:
            object.__setattr__(self, "top_k", _cfg.dense_top_k)
        if not self.timeout_seconds:
            object.__setattr__(self, "timeout_seconds", _cfg.dense_timeout_seconds)
        if not self.max_retries:
            object.__setattr__(self, "max_retries", _cfg.dense_max_retries)
        if not self.embedding_model:
            object.__setattr__(self, "embedding_model", _cfg.embedding_model)
        if not self.embedding_dimension:
            object.__setattr__(self, "embedding_dimension", _cfg.embedding_dimension)
        if not self.score_threshold:
            object.__setattr__(self, "score_threshold", _cfg.default_score_threshold)


# ─── Data models ──────────────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    """A single retrieved document with its score and provenance."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: Literal["dense", "sparse", "hybrid"] = "dense"

    # Extended fields populated by hybrid retriever / reranker
    rrf_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_retrieval_score: Optional[float] = None
    rerank_position_change: Optional[int] = None


# ─── Custom exceptions ────────────────────────────────────────────────────────

class EmbeddingError(Exception):
    """Raised when query embedding fails after all retries."""


class PineconeConnectionError(Exception):
    """Raised on unrecoverable Pinecone connectivity failures."""


# ─── Dense retriever ──────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Production-grade dense retriever backed by Pinecone (serverless / pod).

    Usage::

        config  = DenseRetrieverConfig()
        ret     = DenseRetriever(config)
        results = ret.retrieve("What is the refund policy?", top_k=10)
    """

    def __init__(self, config: DenseRetrieverConfig) -> None:
        self.config = config

        # ── OpenAI client ─────────────────────────────────────────────────────
        self._oai = OpenAI(api_key=config.openai_api_key)

        # ── Pinecone client ───────────────────────────────────────────────────
        self._pc = Pinecone(api_key=config.pinecone_api_key)
        self._index = self._connect_index()

        logger.info(
            json.dumps({
                "event": "dense_retriever_init",
                "index": config.pinecone_index_name,
                "embedding_model": config.embedding_model,
                "top_k": config.top_k,
            })
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _connect_index(self):
        """Connect to an existing Pinecone index with retry."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                index = self._pc.Index(self.config.pinecone_index_name)
                # Lightweight stats call to verify connectivity
                index.describe_index_stats()
                return index
            except PineconeException as exc:
                wait = 2 ** attempt
                logger.warning(
                    json.dumps({
                        "event": "pinecone_connect_retry",
                        "attempt": attempt,
                        "wait_seconds": wait,
                        "error": str(exc),
                    })
                )
                if attempt == self.config.max_retries:
                    raise PineconeConnectionError(
                        f"Cannot connect to Pinecone index "
                        f"'{self.config.pinecone_index_name}' after "
                        f"{self.config.max_retries} attempts."
                    ) from exc
                time.sleep(wait)

    @staticmethod
    def _l2_normalize(vector: List[float]) -> List[float]:
        """Return L2-normalised copy of the vector."""
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> List[float]:
        """
        Embed *query* using the configured OpenAI model.

        Retries up to ``max_retries`` times with exponential backoff on
        rate-limit (429) and transient server errors (5xx).

        Returns
        -------
        list[float]
            L2-normalised dense vector of length ``embedding_dimension``.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._oai.embeddings.create(
                    input=query,
                    model=self.config.embedding_model,
                    dimensions=self.config.embedding_dimension,
                )
                vector = response.data[0].embedding
                return self._l2_normalize(vector)

            except openai.RateLimitError as exc:
                wait = 2 ** attempt
                logger.warning(
                    json.dumps({
                        "event": "embed_rate_limit",
                        "attempt": attempt,
                        "wait_seconds": wait,
                    })
                )
                time.sleep(wait)
                last_exc = exc

            except openai.APIStatusError as exc:
                if exc.status_code and exc.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        json.dumps({
                            "event": "embed_server_error",
                            "status_code": exc.status_code,
                            "attempt": attempt,
                            "wait_seconds": wait,
                        })
                    )
                    time.sleep(wait)
                    last_exc = exc
                else:
                    raise EmbeddingError(
                        f"OpenAI embedding failed with status {exc.status_code}: {exc}"
                    ) from exc

            except Exception as exc:
                raise EmbeddingError(
                    f"Unexpected error during query embedding: {exc}"
                ) from exc

        raise EmbeddingError(
            f"Query embedding failed after {self.config.max_retries} attempts. "
            f"Last error: {last_exc}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        filter: dict | None = None,
        query_id: str | None = None,
        sparse_vector: dict | None = None,
    ) -> List[RetrievalResult]:
        """
        Embed *query*, query Pinecone, and return ranked results.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of results to return. Falls back to ``config.top_k``.
        namespace:
            Pinecone namespace to query (multi-tenant routing).
        filter:
            Pinecone metadata filter dict, e.g. ``{"doc_id": {"$eq": "abc"}}``.
        query_id:
            Caller-supplied trace ID; auto-generated (UUID4) when omitted.
        sparse_vector:
            Optional ``{"indices": [...], "values": [...]}`` SPLADE sparse
            vector for the query text.  When supplied, this performs a
            single native Pinecone hybrid query (dense + sparse combined
            server-side) instead of a pure dense ANN query.  Requires the
            target index to have been created with sparse support (e.g.
            ``metric="dotproduct"``) and sparse_values to have been stored
            at upsert time.

        Returns
        -------
        list[RetrievalResult]
            Sorted descending by similarity score.
        """
        query_id = query_id or str(uuid.uuid4())
        k = top_k if top_k is not None else self.config.top_k
        t0 = time.perf_counter()

        # ── 1. Embed query ────────────────────────────────────────────────────
        embed_start = time.perf_counter()
        try:
            query_vector = self.embed_query(query)
        except EmbeddingError:
            raise
        embed_ms = (time.perf_counter() - embed_start) * 1_000

        # ── 2. Query Pinecone ─────────────────────────────────────────────────
        pinecone_start = time.perf_counter()
        query_kwargs: dict[str, Any] = {
            "vector": query_vector,
            "top_k": k,
            "include_metadata": True,
        }
        if namespace:
            query_kwargs["namespace"] = namespace
        if filter:
            query_kwargs["filter"] = filter
        if sparse_vector and sparse_vector.get("indices"):
            query_kwargs["sparse_vector"] = sparse_vector

        last_exc = None
        response = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._index.query(**query_kwargs)
                break
            except PineconeException as exc:
                wait = 2 ** attempt
                logger.warning(
                    json.dumps({
                        "event": "pinecone_query_retry",
                        "query_id": query_id,
                        "attempt": attempt,
                        "wait_seconds": wait,
                        "error": str(exc),
                    })
                )
                time.sleep(wait)
                last_exc = exc

        if response is None:
            raise PineconeConnectionError(
                f"Pinecone query failed after {self.config.max_retries} "
                f"attempts. Last error: {last_exc}"
            )

        pinecone_ms = (time.perf_counter() - pinecone_start) * 1_000
        total_ms = (time.perf_counter() - t0) * 1_000

        # ── 3. Deserialise matches ────────────────────────────────────────────
        used_sparse = bool(sparse_vector and sparse_vector.get("indices"))
        source_label = "hybrid" if used_sparse else "dense"
        results: List[RetrievalResult] = []
        for match in response.matches:
            score = float(match.score)
            if score < self.config.score_threshold:
                continue
            meta = match.metadata or {}
            results.append(
                RetrievalResult(
                    id=match.id,
                    score=score,
                    text=meta.get("text", meta.get("text_with_title", "")),
                    metadata=meta,
                    source=source_label,
                    dense_score=score,
                )
            )

        # ── 4. Structured log ─────────────────────────────────────────────────
        logger.info(
            json.dumps({
                "event": "dense_retrieve",
                "query_id": query_id,
                "top_k": k,
                "result_count": len(results),
                "embed_ms": round(embed_ms, 1),
                "pinecone_ms": round(pinecone_ms, 1),
                "total_latency_ms": round(total_ms, 1),
                "namespace": namespace,
                "has_filter": filter is not None,
                "used_sparse_vector": used_sparse,
            })
        )

        return results
