"""
OpenAI dense embedder with async batching, semaphore concurrency control,
exponential-backoff retry, and token-rate-limit handling.

Model      : text-embedding-3-small (default)
Dimensions : 1536
Max tokens : 8 191 per input
Rate limit : ~1 M tokens / minute (Tier-1; doubles at higher tiers)

Async design
────────────
embed_chunks_async() batches the input into groups of `batch_size` (default 100),
then fires up to `max_concurrency` (default 10) batches concurrently using an
asyncio.Semaphore.  Each batch is independently retried with exponential backoff.

Sync wrapper
────────────
embed_chunks() wraps embed_chunks_async() via asyncio.run() / nest_asyncio
for callers that are not in an async context.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time

logger = logging.getLogger(__name__)

_OPENAI_AVAILABLE = False
try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    logger.warning(
        "openai not installed — OpenAIEmbedder will return zero vectors. "
        "Install with: pip install openai"
    )

# Cost per 1 000 tokens (USD) — update when OpenAI changes pricing
_COST_PER_1K: dict[str, float] = {
    "text-embedding-3-small": 0.000_020,
    "text-embedding-3-large": 0.000_130,
    "text-embedding-ada-002": 0.000_100,
}

# OpenAI rate-limit constants
_TOKENS_PER_MINUTE = 1_000_000   # conservative Tier-1 assumption
_RATE_WINDOW_SEC   = 60.0


class EmbeddingDimensionMismatchError(RuntimeError):
    """
    Raised when the OpenAI API returns vectors whose length doesn't match
    the configured `dimensions`. Indicates a misconfigured model/dimension
    combo (e.g. a model that ignores the `dimensions` kwarg) — a config
    bug, not a transient failure, so it is never retried or swallowed into
    a zero vector.
    """


class OpenAIEmbedder:
    """
    Async-capable OpenAI embedding client.

    Args:
        api_key:         OpenAI API key.
        model:           Embedding model name.
        dimensions:      Output vector dimensions.
        batch_size:      Number of texts per API call (max 2048 per OpenAI).
        max_concurrency: Max concurrent in-flight API requests.
        max_retries:     Retry attempts on transient errors.
        retry_on_fail:   If False, propagate errors immediately.
    """

    def __init__(
        self,
        api_key:         str,
        model:           str  = "",
        dimensions:      int  = 0,
        batch_size:      int  = 0,
        max_concurrency: int  = 10,
        max_retries:     int  = 0,
        retry_on_fail:   bool = True,
    ) -> None:
        from backend.config import settings as _cfg
        self.model           = model      or _cfg.openai_embedding_model
        self.dimensions      = dimensions or _cfg.openai_embedding_dims
        self.batch_size      = min(batch_size or _cfg.default_batch_size, 2048)
        self.max_concurrency = max_concurrency
        self.max_retries     = max_retries or _cfg.default_max_retries
        self.retry_on_fail   = retry_on_fail

        if _OPENAI_AVAILABLE:
            self._async_client = _openai.AsyncOpenAI(api_key=api_key)
            self._sync_client  = _openai.OpenAI(api_key=api_key)
        else:
            self._async_client = None
            self._sync_client  = None

        # Token-rate tracking (approximate)
        self._tokens_this_window: int   = 0
        self._window_start:       float = time.monotonic()

        self._total_retries = 0

    # ── Async entry point ─────────────────────────────────────────────────────

    async def embed_texts_async(
        self,
        texts:       list[str],
        token_counts: list[int] | None = None,
    ) -> list[list[float]]:
        """
        Embed a list of texts asynchronously.

        Args:
            texts:        List of strings to embed.
            token_counts: Pre-computed per-text token counts, in the same
                          order as `texts`. When supplied, the rate-limit
                          guard uses these instead of re-tokenising each
                          text. Ignored (with a warning) if its length
                          doesn't match `texts`.

        Returns:
            list[list[float]] — one dense vector per input text.
            Zero-vectors are returned for any text that fails all retries.
        """
        if not texts:
            return []

        if not _OPENAI_AVAILABLE or self._async_client is None:
            logger.warning("OpenAI unavailable — returning zero vectors.")
            return [self._zero_vector() for _ in texts]

        if token_counts is not None and len(token_counts) != len(texts):
            logger.warning(
                "token_counts length (%d) != texts length (%d) — ignoring "
                "pre-computed counts and re-counting.",
                len(token_counts), len(texts),
            )
            token_counts = None

        # Split into batches (texts and their matching pre-computed token
        # counts, if supplied, must be batched together so the rate limiter
        # can use them instead of re-counting).
        batches = _batch(texts, self.batch_size)
        count_batches: list[list[int] | None] = (
            _batch(token_counts, self.batch_size) if token_counts is not None
            else [None] * len(batches)
        )
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def embed_batch(
            batch: list[str], counts: list[int] | None
        ) -> list[list[float]]:
            async with semaphore:
                return await self._embed_batch_with_retry(batch, counts)

        tasks   = [
            embed_batch(b, c) for b, c in zip(batches, count_batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        vectors: list[list[float]] = []
        for r in results:
            if isinstance(r, EmbeddingDimensionMismatchError):
                # Config bug affecting every batch identically — propagate
                # immediately instead of masking it as per-batch zero vectors.
                raise r
            if isinstance(r, Exception):
                logger.error("Batch embedding failed: %s", r)
                # Fall back to zero vectors for the failed batch
                # (we can't know which batch without more bookkeeping — use
                # the batch size as a safe upper bound)
                vectors.extend([self._zero_vector()] * self.batch_size)
            else:
                vectors.extend(r)

        return vectors[: len(texts)]  # trim any overflow from zero-pad

    # ── Sync convenience wrapper ──────────────────────────────────────────────

    def embed_texts(
        self,
        texts:       list[str],
        token_counts: list[int] | None = None,
    ) -> list[list[float]]:
        """
        Synchronous wrapper around embed_texts_async().
        Handles both nested-loop and top-level contexts.
        """
        if not texts:
            return []

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Already inside an event loop (FastAPI / Jupyter)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, self.embed_texts_async(texts, token_counts))
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.embed_texts_async(texts, token_counts)
                )
        except RuntimeError:
            return asyncio.run(self.embed_texts_async(texts, token_counts))

    # ── Batch + retry ─────────────────────────────────────────────────────────

    async def _embed_batch_with_retry(
        self, texts: list[str], token_counts: list[int] | None = None
    ) -> list[list[float]]:
        """Send one batch to the API with exponential-backoff retry."""
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                await self._rate_limit_wait(texts, token_counts)
                response = await self._async_client.embeddings.create(  # type: ignore[union-attr]
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions,
                )
                # Sort by index to guarantee order
                vectors = [
                    e.embedding
                    for e in sorted(response.data, key=lambda e: e.index)
                ]
                logger.debug(
                    "Batch embedded: %d texts, %d tokens used",
                    len(texts),
                    response.usage.total_tokens if response.usage else -1,
                )

                # ── Dimension sanity check ────────────────────────────────────
                # A misconfigured model/dimensions combo (e.g. requesting 3072
                # dims from a model that ignores the `dimensions` kwarg) would
                # otherwise fail silently here and only surface much later as
                # a cryptic Pinecone dimension-mismatch rejection at upsert
                # time. Fail fast instead. This is a config error, not a
                # transient failure, so it's raised immediately rather than
                # retried or masked behind zero vectors.
                if vectors:
                    actual_dim = len(vectors[0])
                    if actual_dim != self.dimensions:
                        raise EmbeddingDimensionMismatchError(
                            f"Embedding dimension mismatch: configured "
                            f"dimensions={self.dimensions} but model "
                            f"'{self.model}' returned vectors of length "
                            f"{actual_dim}. Check openai_embedding_dims / "
                            f"the model's supported dimensions."
                        )

                return vectors

            except EmbeddingDimensionMismatchError:
                raise

            except Exception as exc:
                last_exc = exc
                self._total_retries += 1

                if not self.retry_on_fail:
                    raise

                if attempt == self.max_retries:
                    break

                wait = _backoff_seconds(attempt, exc)
                logger.warning(
                    "Embedding batch attempt %d/%d failed (%s: %s) — "
                    "retrying in %.1f s",
                    attempt + 1, self.max_retries + 1,
                    type(exc).__name__, exc, wait,
                )
                await asyncio.sleep(wait)

        logger.error(
            "All %d embedding attempts failed for batch of %d texts. "
            "Last error: %s",
            self.max_retries + 1, len(texts), last_exc,
        )
        return [self._zero_vector() for _ in texts]

    # ── Rate-limit guard ──────────────────────────────────────────────────────

    async def _rate_limit_wait(
        self, texts: list[str], token_counts: list[int] | None = None
    ) -> None:
        """
        Approximate rate-limit enforcement.
        If adding this batch would exceed _TOKENS_PER_MINUTE, sleep until
        the current window resets.

        Uses pre-computed `token_counts` when supplied (avoids re-tokenising
        text that the caller already counted, e.g. during pre-embedding
        validation) and falls back to counting `texts` directly otherwise.
        """
        if token_counts is not None:
            tokens = sum(token_counts)
        else:
            from backend.utils.token_counter import count_tokens_batch
            tokens = sum(count_tokens_batch(texts))

        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed >= _RATE_WINDOW_SEC:
            # New window
            self._window_start        = now
            self._tokens_this_window  = 0

        if self._tokens_this_window + tokens > _TOKENS_PER_MINUTE:
            sleep_for = _RATE_WINDOW_SEC - elapsed + 0.5
            if sleep_for > 0:
                logger.info(
                    "Rate limit guard: sleeping %.1f s to reset token window.", sleep_for
                )
                await asyncio.sleep(sleep_for)
            self._window_start        = time.monotonic()
            self._tokens_this_window  = 0

        self._tokens_this_window += tokens

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _zero_vector(self) -> list[float]:
        return [0.0] * self.dimensions

    @staticmethod
    def cost_usd(tokens: int, model: str) -> float:
        rate = _COST_PER_1K.get(model, 0.000_020)
        return (tokens / 1_000) * rate

    @property
    def total_retries(self) -> int:
        return self._total_retries


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _batch(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _backoff_seconds(attempt: int, exc: Exception) -> float:
    """
    Exponential backoff with jitter.
    RateLimitError gets a longer base wait.
    """
    is_rate_limit = (
        hasattr(exc, "status_code") and exc.status_code == 429  # type: ignore[union-attr]
    ) or "rate" in str(exc).lower()

    base  = 60.0 if is_rate_limit else 2.0
    delay = base * (2 ** attempt) + random.uniform(0, 1)
    return min(delay, 120.0)   # cap at 2 minutes
