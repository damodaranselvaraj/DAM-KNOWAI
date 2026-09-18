"""
tiktoken-based token counting utilities.

Uses cl100k_base encoding, which is the correct encoding for:
  - text-embedding-3-small
  - text-embedding-3-large
  - text-embedding-ada-002
  - gpt-4o / gpt-4 / gpt-3.5-turbo

The module keeps one shared Encoding object per process (thread-safe)
to avoid re-loading the vocabulary file on every call.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

logger = logging.getLogger(__name__)

_ENCODING_NAME = "cl100k_base"

_TIKTOKEN_AVAILABLE = False
try:
    import tiktoken as _tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.warning(
        "tiktoken not installed — token counting will fall back to word-count "
        "approximation.  Install with: pip install tiktoken"
    )


# ─── Encoding singleton ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_encoding():
    """Return the shared cl100k_base Encoding (cached after first call)."""
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        return _tiktoken.get_encoding(_ENCODING_NAME)
    except Exception as exc:
        logger.error("Failed to load tiktoken encoding '%s': %s", _ENCODING_NAME, exc)
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Return the number of tokens in `text` using cl100k_base.

    Falls back to len(text.split()) * 1.3 (word-based approximation)
    when tiktoken is unavailable.
    """
    if not text:
        return 0
    enc = _get_encoding()
    if enc is None:
        # Approximation: average English word ≈ 1.3 tokens
        return max(1, round(len(text.split()) * 1.3))
    try:
        return len(enc.encode(text))
    except Exception as exc:
        logger.warning("Token counting failed: %s — using word approximation", exc)
        return max(1, round(len(text.split()) * 1.3))


def count_tokens_batch(texts: Sequence[str]) -> list[int]:
    """Count tokens for a list of strings in one batch (faster than calling count_tokens repeatedly)."""
    enc = _get_encoding()
    if enc is None:
        return [count_tokens(t) for t in texts]
    try:
        encoded = enc.encode_batch(list(texts))
        return [len(ids) for ids in encoded]
    except Exception as exc:
        logger.warning("Batch token counting failed: %s — falling back to single", exc)
        return [count_tokens(t) for t in texts]


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate `text` so it fits within `max_tokens`.
    Returns the truncated string (decoded back from token IDs).
    Falls back to character-ratio truncation when tiktoken is absent.
    """
    if not text or max_tokens <= 0:
        return ""
    enc = _get_encoding()
    if enc is None:
        # Character-ratio approximation
        ratio = max_tokens / max(count_tokens(text), 1)
        cutoff = int(len(text) * ratio)
        return text[:cutoff].rstrip()
    try:
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        return enc.decode(ids[:max_tokens])
    except Exception as exc:
        logger.warning("Truncation failed: %s", exc)
        return text


def fits_in_tokens(text: str, max_tokens: int) -> bool:
    """Return True if `text` fits within `max_tokens`."""
    return count_tokens(text) <= max_tokens


def token_overlap_chars(text: str, overlap_tokens: int) -> int:
    """
    Return the number of characters corresponding to the last
    `overlap_tokens` tokens of `text`.  Used to compute character-based
    sliding windows when splitting without a tokeniser.
    """
    if not text or overlap_tokens <= 0:
        return 0
    enc = _get_encoding()
    if enc is None:
        # Approximate: 1 token ≈ 4 chars
        return overlap_tokens * 4
    try:
        ids     = enc.encode(text)
        tail    = ids[-overlap_tokens:] if overlap_tokens < len(ids) else ids
        decoded = enc.decode(tail)
        return len(decoded)
    except Exception:
        return overlap_tokens * 4


def encoding_name() -> str:
    """Return the encoding name in use."""
    return _ENCODING_NAME
