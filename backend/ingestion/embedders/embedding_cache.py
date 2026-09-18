"""
In-process embedding cache.

Cache key  : content_hash (str)
Cache value: list[float]  (dense vector)

A lightweight thread-safe dict cache used by EmbeddingPipeline to avoid
re-embedding chunks whose text hasn't changed within a single process
lifetime.  Vectors are not persisted across restarts.
"""
from __future__ import annotations

import logging
from threading import Lock
from typing import Protocol

logger = logging.getLogger(__name__)


# ─── Abstract cache protocol ──────────────────────────────────────────────────

class AbstractEmbeddingCache(Protocol):
    def get(self, content_hash: str) -> list[float] | None: ...
    def set(self, content_hash: str, vector: list[float]) -> None: ...
    def delete(self, content_hash: str) -> None: ...
    def clear(self) -> None: ...
    def stats(self) -> dict: ...


# ─── In-process cache ─────────────────────────────────────────────────────────

class InProcessEmbeddingCache:
    """Thread-safe in-memory dict cache (no expiry, no persistence)."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}
        self._lock   = Lock()
        self._hits   = 0
        self._misses = 0

    def get(self, content_hash: str) -> list[float] | None:
        with self._lock:
            v = self._store.get(content_hash)
            if v is None:
                self._misses += 1
            else:
                self._hits += 1
            return v

    def set(self, content_hash: str, vector: list[float]) -> None:
        with self._lock:
            self._store[content_hash] = vector

    def delete(self, content_hash: str) -> None:
        with self._lock:
            self._store.pop(content_hash, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "backend":  "in_process",
                "size":     len(self._store),
                "hits":     self._hits,
                "misses":   self._misses,
                "hit_rate": round(self._hits / max(total, 1), 3),
            }


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_cache() -> InProcessEmbeddingCache:
    """Return a new InProcessEmbeddingCache instance."""
    return InProcessEmbeddingCache()
