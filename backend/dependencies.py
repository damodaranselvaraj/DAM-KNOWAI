"""
Shared FastAPI dependencies.

Memory saver factory
────────────────────
``get_memory_saver()`` is the single place that reads ``MEMORY_BACKEND``
and constructs the correct saver.  It is called once at application
startup (inside the lifespan hook in main.py) and the resulting instance
is stored in ``app.state.memory_saver`` so every request handler can
retrieve it via the ``memory_saver()`` dependency without paying the
construction cost on every request.

Usage in a route::

    from fastapi import Depends
    from backend.dependencies import memory_saver

    @router.post("/chat/query")
    async def query(saver: BaseMemorySaver = Depends(memory_saver)):
        state = saver.get({"configurable": {"thread_id": session_id}})
        ...

Backend selection (MEMORY_BACKEND in .env):
    "sqlite"  → SQLiteSaver  (DEFAULT — durable, survives restarts)
    "memory"  → InMemorySaver (ephemeral, dev/test)
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request

from backend.config import settings
from backend.memory.base import BaseMemorySaver, MemoryBackend
from backend.memory.in_memory_saver import InMemorySaver
from backend.memory.sqlite_saver import SQLiteSaver, SQLiteSaverConfig

logger = logging.getLogger(__name__)

# Module-level singleton — set by create_memory_saver() at startup
_memory_saver: Optional[BaseMemorySaver] = None


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_memory_saver() -> BaseMemorySaver:
    """
    Instantiate and initialise the memory saver selected by ``MEMORY_BACKEND``.

    Called once during application startup (lifespan).  The returned
    instance is fully initialised (``setup()`` already called).

    Returns
    -------
    BaseMemorySaver
        A ready-to-use saver (either SQLiteSaver or InMemorySaver).

    Raises
    ------
    ValueError
        When ``MEMORY_BACKEND`` contains an unrecognised value.
    """
    global _memory_saver

    backend = settings.memory_backend.lower()

    if backend == MemoryBackend.SQLITE:
        saver_config = SQLiteSaverConfig(
            db_path=settings.sqlite_db_path,
            wal_mode=settings.sqlite_wal_mode,
            pool_timeout=settings.sqlite_pool_timeout,
            max_checkpoints=settings.memory_max_checkpoints,
            prune_days=settings.memory_prune_days,
            keep_last=settings.memory_keep_last,
        )
        saver: BaseMemorySaver = SQLiteSaver(saver_config)
        logger.info(
            "Memory backend: SQLiteSaver  db='%s'",
            settings.sqlite_db_path,
        )

    elif backend == MemoryBackend.MEMORY:
        saver = InMemorySaver(
            max_checkpoints_per_thread=settings.memory_max_per_thread
        )
        logger.info(
            "Memory backend: InMemorySaver  max_per_thread=%d",
            settings.memory_max_per_thread,
        )

    else:
        raise ValueError(
            f"Unknown MEMORY_BACKEND='{backend}'. "
            f"Valid values: {[b.value for b in MemoryBackend]}"
        )

    saver.setup()
    _memory_saver = saver
    return saver


def get_memory_saver() -> BaseMemorySaver:
    """
    Return the module-level saver singleton.

    Raises ``RuntimeError`` when ``create_memory_saver()`` has not been
    called yet (i.e. the app lifespan hook hasn't run).
    """
    if _memory_saver is None:
        raise RuntimeError(
            "Memory saver has not been initialised. "
            "Ensure create_memory_saver() is called in the FastAPI lifespan hook."
        )
    return _memory_saver


# ─── FastAPI dependency ───────────────────────────────────────────────────────

def memory_saver(request: Request) -> BaseMemorySaver:
    """
    FastAPI dependency — injects the memory saver into route handlers.

    Retrieves the saver from ``app.state.memory_saver`` (set during lifespan).
    Falls back to the module-level singleton for compatibility.

    Usage::

        @router.post("/chat/query")
        async def query(saver: BaseMemorySaver = Depends(memory_saver)):
            ...
    """
    # Prefer app.state (set in lifespan) so the dep works with TestClient too
    saver = getattr(request.app.state, "memory_saver", None)
    if saver is not None:
        return saver
    return get_memory_saver()
