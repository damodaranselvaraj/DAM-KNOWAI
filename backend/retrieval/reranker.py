"""
Cohere reranker.

Uses the Cohere Rerank API to reorder retrieval candidates by relevance.
The public interface (RerankerConfig, Reranker, Reranker.rerank) is
identical to the previous CrossEncoder implementation so no other module
needs to change.

Configuration (all via environment / .env):
    COHERE_API_KEY          (required — shared with backend.config.settings)
    RERANKER_MODEL          (default: rerank-v3.5)
    RERANKER_TOP_K          (default: 10)   — candidates sent to Cohere
    FINAL_TOP_K             (default: 5)    — returned after reranking
    RERANKER_MAX_RETRIES    (default: 3)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

try:
    import cohere  # type: ignore
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "cohere is required for the Cohere reranker. "
        "Install it with: pip install cohere"
    ) from _exc

from backend.retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

class RerankerConfig(BaseSettings):
    """Loaded from environment variables / .env."""

    cohere_api_key: str  = Field("",  alias="COHERE_API_KEY")
    model_name: str      = Field("",  alias="RERANKER_MODEL")
    top_k: int           = Field(0,   alias="RERANKER_TOP_K")
    final_top_k: int     = Field(0,   alias="FINAL_TOP_K")
    max_retries: int     = Field(0,   alias="RERANKER_MAX_RETRIES")

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
        if not self.cohere_api_key:
            object.__setattr__(self, "cohere_api_key", _cfg.cohere_api_key)
        if not self.model_name:
            object.__setattr__(self, "model_name", _cfg.reranker_model)
        if not self.top_k:
            object.__setattr__(self, "top_k", _cfg.reranker_top_k)
        if not self.final_top_k:
            object.__setattr__(self, "final_top_k", _cfg.final_top_k)
        if not self.max_retries:
            object.__setattr__(self, "max_retries", _cfg.default_max_retries)


# ─── Reranker ─────────────────────────────────────────────────────────────────

class Reranker:
    """
    Production-grade reranker backed by the Cohere Rerank API.

    Takes the top ``RERANKER_TOP_K`` candidates from the retrieval stage,
    sends them to Cohere's rerank endpoint, and returns the top
    ``FINAL_TOP_K`` results ordered by relevance score.

    Original retrieval scores are preserved in
    ``result.original_retrieval_score`` and
    ``result.metadata["_retrieval_score"]`` for downstream analysis.

    Usage::

        config   = RerankerConfig()
        reranker = Reranker(config)
        results  = reranker.rerank(query, retrieval_results)
    """

    def __init__(self, config: RerankerConfig) -> None:
        if not config.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY is not set. "
                "Add it to your .env file or environment."
            )

        self.config = config
        self._client = cohere.Client(
            api_key=config.cohere_api_key,
        )

        logger.info(
            json.dumps({
                "event": "reranker_init",
                "provider": "cohere",
                "model": config.model_name,
                "top_k": config.top_k,
                "final_top_k": config.final_top_k,
            })
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
        query_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank *candidates* against *query* using the Cohere Rerank API.

        Steps:
        1. Trim candidates to ``RERANKER_TOP_K``.
        2. Send (query, documents) to Cohere rerank endpoint.
        3. Map returned relevance scores back to ``RetrievalResult`` objects.
        4. Return top ``FINAL_TOP_K`` (or *top_k* if supplied).

        Preserves original retrieval score in:
        - ``result.original_retrieval_score``
        - ``result.metadata["_retrieval_score"]``

        Sets ``rerank_position_change`` = original_rank − new_rank
        (positive = document moved up).

        Parameters
        ----------
        query:
            Natural-language query string.
        candidates:
            Retrieval results from dense / sparse / hybrid stage.
        top_k:
            Override ``FINAL_TOP_K``.
        query_id:
            Caller-supplied trace ID; auto-generated when omitted.

        Returns
        -------
        list[RetrievalResult]
            Reranked and trimmed to *top_k*.
        """
        query_id = query_id or str(uuid.uuid4())
        final_k  = top_k if top_k is not None else self.config.final_top_k

        if not candidates:
            return []

        # 1. Trim to RERANKER_TOP_K
        pool = candidates[: self.config.top_k]

        # Build plain-text documents for Cohere
        documents = [r.text or "" for r in pool]

        t0 = time.perf_counter()

        # 2. Call Cohere rerank
        response = self._client.rerank(
            model=self.config.model_name,
            query=query,
            documents=documents,
            top_n=final_k,
            return_documents=False,
        )

        latency_ms = (time.perf_counter() - t0) * 1_000

        # 3. Map scores back to RetrievalResult objects
        # response.results is ordered by relevance (highest first)
        original_rank_map = {i: r for i, r in enumerate(pool)}

        reranked: List[RetrievalResult] = []
        for new_rank, hit in enumerate(response.results):
            orig_idx   = hit.index
            orig_rank  = orig_idx   # 0-based position in pool before reranking
            rerank_score = float(hit.relevance_score)
            orig_result  = original_rank_map[orig_idx]

            updated_meta = {
                **orig_result.metadata,
                "_retrieval_score": orig_result.score,
                "_rerank_score": rerank_score,
            }

            reranked.append(
                orig_result.model_copy(
                    update={
                        "rerank_score": rerank_score,
                        "original_retrieval_score": orig_result.score,
                        "score": rerank_score,
                        "metadata": updated_meta,
                        "rerank_position_change": orig_rank - new_rank,
                    }
                )
            )

        scores = [r.rerank_score for r in reranked]

        logger.info(
            json.dumps({
                "event": "rerank",
                "query_id": query_id,
                "provider": "cohere",
                "model": self.config.model_name,
                "candidates_in": len(pool),
                "results_out": len(reranked),
                "latency_ms": round(latency_ms, 1),
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "min_score": round(min(scores), 4) if scores else 0.0,
                "max_score": round(max(scores), 4) if scores else 0.0,
            })
        )

        return reranked
