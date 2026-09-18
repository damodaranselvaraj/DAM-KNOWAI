"""
EmbeddingPipeline — Phase 3 orchestrator.

Flow
────
  Chunks
    │
    ├─ Pre-embedding validation (empty / token / language / dedup)
    │
    ├─ In-process cache lookup    ← skip API call on hit
    │
    ├─ OpenAI dense embedding         (async batched, retry, rate-limit)
    │
    ├─ BM25 / SPLADE sparse encoding  (optional, for hybrid search)
    │
    └─ EmbeddedChunk assembly → EmbeddingResult
"""
from __future__ import annotations

import asyncio
import logging
import time

from backend.ingestion.embedders.embedding_cache import InProcessEmbeddingCache, build_cache
from backend.ingestion.embedders.openai_embedder import OpenAIEmbedder
from backend.ingestion.embedders.pre_embedding_validator import validate_chunks
from backend.ingestion.embedders.sparse_embedder import SparseEmbedder
from backend.models.chunk import Chunk
from backend.models.embedded_chunk import EmbeddedChunk, EmbeddingResult, SparseVector
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Stateless (after construction) Phase 3 orchestrator.

    Args:
        openai_api_key:    OpenAI API key.
        model:             Embedding model name.
        dimensions:        Vector dimensions.
        batch_size:        Texts per OpenAI API call.
        max_concurrency:   Max concurrent in-flight batches.
        max_retries:       Per-batch retry limit.
        retry_on_fail:     Propagate vs. zero-vector on persistent failure.
        enable_sparse:     Compute BM25/SPLADE sparse vectors.
        enable_cache:      Use in-process embedding cache.
        truncate_long:     Truncate (vs. skip) chunks > max_tokens.
        warn_non_english:  Log warning for non-English chunks.
    """

    def __init__(
        self,
        openai_api_key:    str,
        model:             str  = "",
        dimensions:        int  = 0,
        batch_size:        int  = 0,
        max_concurrency:   int  = 10,
        max_retries:       int  = 0,
        retry_on_fail:     bool = True,
        enable_sparse:     bool = True,
        enable_cache:      bool = True,
        truncate_long:     bool = True,
        warn_non_english:  bool = True,
    ) -> None:
        from backend.config import settings as _cfg
        self.model      = model      or _cfg.openai_embedding_model
        self.dimensions = dimensions or _cfg.openai_embedding_dims
        self.enable_sparse = enable_sparse
        self.enable_cache  = enable_cache
        self.truncate_long = truncate_long
        self.warn_non_english = warn_non_english

        self._dense_embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model=self.model,
            dimensions=self.dimensions,
            batch_size=batch_size or _cfg.default_batch_size,
            max_concurrency=max_concurrency,
            max_retries=max_retries or _cfg.default_max_retries,
            retry_on_fail=retry_on_fail,
        )
        self._sparse_embedder = SparseEmbedder() if enable_sparse else None
        self._cache: InProcessEmbeddingCache | None = (
            build_cache() if enable_cache else None
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_chunks(
        self,
        chunks:         list[Chunk],
        doc_id:         str   = "",
        doc_name:       str   = "",
        known_hashes:   set[str] | None = None,
    ) -> EmbeddingResult:
        """
        Synchronous entry point.  Runs the full Phase 3 pipeline.

        Args:
            chunks:       Chunks from Phase 2 (ChunkingResult.all_chunks).
            doc_id:       Source document ID (for the result envelope).
            doc_name:     Source filename.
            known_hashes: Content hashes already in the vector store
                          (used for cross-document deduplication).

        Returns:
            EmbeddingResult with embedded_chunks ready for Pinecone upsert.
        """
        t0 = time.perf_counter()
        result = EmbeddingResult(
            doc_id=doc_id or (chunks[0].doc_id if chunks else ""),
            doc_name=doc_name or (chunks[0].doc_name if chunks else ""),
            model_used=self.model,
            total_input=len(chunks),
        )

        if not chunks:
            return result

        # ── 1. Pre-embedding validation ───────────────────────────────────────
        valid_chunks, val_summary = validate_chunks(
            chunks,
            truncate_long=self.truncate_long,
            warn_non_english=self.warn_non_english,
            known_hashes=known_hashes,
        )
        result.total_skipped = (
            val_summary.skipped_empty
            + val_summary.skipped_long
            + val_summary.skipped_dupes
        )

        if not valid_chunks:
            logger.warning("No valid chunks after validation for '%s'.", doc_name)
            result.embed_time_ms = round((time.perf_counter() - t0) * 1000, 2)
            return result

        # ── 2. Cache lookup ───────────────────────────────────────────────────
        texts_to_embed:  list[str]  = []
        cache_map:       dict[str, list[float]] = {}   # content_hash → vector
        cache_hits       = 0

        for chunk in valid_chunks:
            key = chunk.content_hash
            if self._cache and key:
                cached = self._cache.get(key)
                if cached is not None:
                    cache_map[key] = cached
                    cache_hits    += 1
                    continue
            texts_to_embed.append(chunk.text_with_title or chunk.text)

        result.total_cache_hits = cache_hits
        logger.info(
            "Cache: %d hits, %d misses for '%s'",
            cache_hits, len(texts_to_embed), doc_name,
        )

        # ── 3. Dense embedding ────────────────────────────────────────────────
        cache_miss_chunks: list[Chunk] = []  # defined here for scope safety
        if texts_to_embed:
            t_embed = time.perf_counter()
            dense_vectors = self._dense_embedder.embed_texts(texts_to_embed)
            embed_ms      = (time.perf_counter() - t_embed) * 1000

            # Map vectors back to chunks that needed embedding
            cache_miss_chunks = [
                c for c in valid_chunks
                if (c.content_hash not in cache_map)
            ]
            for chunk, vector in zip(cache_miss_chunks, dense_vectors):
                cache_map[chunk.content_hash] = vector
                if self._cache and chunk.content_hash:
                    self._cache.set(chunk.content_hash, vector)

            # Token + cost accounting
            total_tokens = sum(count_tokens(t) for t in texts_to_embed)
            result.total_tokens   = total_tokens
            result.total_cost_usd = OpenAIEmbedder.cost_usd(total_tokens, self.model)
            result.batch_count    = math.ceil(
                len(texts_to_embed) / self._dense_embedder.batch_size
            )
            result.retry_count    = self._dense_embedder.total_retries
        else:
            embed_ms = 0.0

        # ── 4. Assemble EmbeddedChunks ────────────────────────────────────────
        miss_hashes = {c.content_hash for c in cache_miss_chunks}
        embedded: list[EmbeddedChunk] = []
        for chunk in valid_chunks:
            dense  = cache_map.get(chunk.content_hash, [0.0] * self.dimensions)
            is_hit = chunk.content_hash not in miss_hashes

            # Sparse encoding
            sparse: SparseVector | None = None
            if self._sparse_embedder:
                try:
                    sparse = self._sparse_embedder.encode(
                        chunk.text_with_title or chunk.text
                    )
                except Exception as exc:
                    logger.warning("Sparse encode failed for %s: %s", chunk.chunk_id, exc)

            ec = EmbeddedChunk.from_chunk(
                chunk=chunk,
                dense_vector=dense,
                sparse_vector=sparse,
                embedding_model=self.model,
                embedding_dims=self.dimensions,
                cache_hit=is_hit,
                embed_time_ms=round(embed_ms, 2),
            )
            embedded.append(ec)

        result.embedded_chunks = embedded
        result.total_embedded  = len(embedded)
        result.embed_time_ms   = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "EmbeddingPipeline '%s': %d→%d embedded (%d cached, %d skipped) "
            "| %d tokens | $%.5f | %.0f ms",
            doc_name,
            result.total_input,
            result.total_embedded,
            result.total_cache_hits,
            result.total_skipped,
            result.total_tokens,
            result.total_cost_usd,
            result.embed_time_ms,
        )
        return result

    # ── Async entry point ─────────────────────────────────────────────────────

    async def embed_chunks_async(
        self,
        chunks:       list[Chunk],
        doc_id:       str = "",
        doc_name:     str = "",
        known_hashes: set[str] | None = None,
    ) -> EmbeddingResult:
        """
        Async version — runs validation + cache checks synchronously,
        then awaits the OpenAI async calls directly.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.embed_chunks(chunks, doc_id, doc_name, known_hashes),
        )

    # ── Cache management ──────────────────────────────────────────────────────

    def cache_stats(self) -> dict:
        if self._cache:
            return self._cache.stats()
        return {"backend": "disabled"}

    def clear_cache(self) -> None:
        if self._cache:
            self._cache.clear()

    def fit_sparse_vocabulary(self, documents: list[str]) -> None:
        """Pre-train BM25 vocabulary from a corpus of document strings."""
        if self._sparse_embedder:
            self._sparse_embedder.fit_vocabulary(documents)


# ─── Module-level factory ─────────────────────────────────────────────────────

import math  # noqa: E402 (used in embed_chunks above)


def build_embedding_pipeline(
    openai_api_key: str = "",
    **kwargs,
) -> EmbeddingPipeline:
    """
    Convenience factory — ALL defaults come from backend.config.settings.
    Caller-supplied kwargs override settings values when provided.
    """
    from backend.config import settings as _cfg
    api_key     = openai_api_key            or _cfg.openai_api_key
    model       = kwargs.pop("model",       _cfg.openai_embedding_model)
    dimensions  = kwargs.pop("dimensions",  _cfg.openai_embedding_dims)
    batch_size  = kwargs.pop("batch_size",  _cfg.default_batch_size)
    max_retries = kwargs.pop("max_retries", _cfg.default_max_retries)

    return EmbeddingPipeline(
        openai_api_key=api_key,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size,
        max_retries=max_retries,
        **kwargs,
    )
