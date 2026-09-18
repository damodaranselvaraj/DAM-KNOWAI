"""
Abstract base class for all memory / checkpoint savers.

Every concrete saver must implement the full interface defined here.
The interface is intentionally LangGraph-compatible — each method maps
directly to the LangGraph BaseCheckpointSaver contract while adding
project-specific helpers (session management, pruning, health-checks).

LangGraph checkpoint model
──────────────────────────
A *checkpoint* is the serialised state of a graph at a given step.
Each checkpoint belongs to a *thread* (= session_id in this project)
and carries:

    thread_id   — unique conversation / session identifier
    checkpoint  — dict containing the full graph state
    metadata    — arbitrary dict (step number, token counts, etc.)
    parent_id   — id of the checkpoint this one was created from
                  (None for the first checkpoint in a thread)

Design notes
────────────
• All public methods are *sync* by default.  Async variants (_aput, _aget,
  etc.) can be added in subclasses — LangGraph will call async variants
  when available inside async runnables.

• Subclasses should be thread-safe where the backend requires it
  (e.g. SQLiteSaver uses a threading.Lock around DB writes).

• The ``prune`` and ``clear_session`` operations are not part of the
  official LangGraph API but are needed for production memory hygiene.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────

class MemoryBackend(str, Enum):
    """Supported persistence backends, keyed by MEMORY_BACKEND env var."""
    MEMORY = "memory"
    SQLITE = "sqlite"


# ─── Data models ──────────────────────────────────────────────────────────────

class CheckpointRecord(BaseModel):
    """
    A single persisted checkpoint for a thread.

    Maps directly to the LangGraph checkpoint tuple::

        (config, checkpoint, metadata)

    where ``config`` carries thread_id + checkpoint_id.
    """
    thread_id:     str
    checkpoint_id: str                            # UUID of this checkpoint step
    parent_id:     Optional[str] = None           # None = first checkpoint
    checkpoint:    Dict[str, Any] = Field(default_factory=dict)   # graph state
    metadata:      Dict[str, Any] = Field(default_factory=dict)   # step metadata
    created_at:    datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SessionInfo(BaseModel):
    """Lightweight summary of a thread / session."""
    thread_id:        str
    checkpoint_count: int = 0
    created_at:       Optional[datetime] = None
    last_updated:     Optional[datetime] = None
    metadata:         Dict[str, Any] = Field(default_factory=dict)


# ─── Abstract base ────────────────────────────────────────────────────────────

class BaseMemorySaver(ABC):
    """
    Abstract base for LangGraph-compatible checkpoint savers.

    Subclasses must implement every ``@abstractmethod``.  Optional
    overrides (``setup``, ``teardown``, ``health_check``) have
    sensible no-op defaults so they don't have to be re-implemented
    for simple backends.

    Public interface quick-reference
    ─────────────────────────────────
    LangGraph core:
        put(config, checkpoint, metadata) → config
        get(config)                        → checkpoint | None
        list(config, *, limit, before)     → Iterator[CheckpointRecord]

    Session helpers (project-specific):
        get_session_info(thread_id)        → SessionInfo | None
        list_sessions(limit)               → list[SessionInfo]
        clear_session(thread_id)           → int   (rows deleted)
        prune(older_than_days, keep_last)  → int   (rows deleted)

    Lifecycle:
        setup()        — called once on startup
        teardown()     — called on shutdown / context exit
        health_check() → bool
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        Initialise backend resources (create tables, open connections…).

        Called once during application startup.  Idempotent.
        """

    def teardown(self) -> None:
        """
        Release backend resources (close DB connections, flush caches…).

        Called during application shutdown.
        """

    def health_check(self) -> bool:
        """
        Return ``True`` when the backend is reachable and operational.

        Override to add a real ping (e.g. a SELECT 1 for SQLite).
        Default returns ``True`` (always healthy) for non-networked backends.
        """
        return True

    # ── LangGraph core interface ───────────────────────────────────────────────

    @abstractmethod
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Persist a checkpoint for the thread described by *config*.

        Parameters
        ----------
        config:
            Must contain ``configurable.thread_id`` and optionally
            ``configurable.checkpoint_id``.  A new ``checkpoint_id``
            (UUID) is generated when not supplied.
        checkpoint:
            Full serialisable graph state dict.
        metadata:
            Arbitrary metadata (step index, token counts, model name, …).

        Returns
        -------
        dict
            Updated config with the assigned ``checkpoint_id`` stored at
            ``config["configurable"]["checkpoint_id"]``.
        """

    @abstractmethod
    def get(
        self,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the **latest** checkpoint for the thread in *config*.

        When ``config["configurable"]["checkpoint_id"]`` is present the
        exact checkpoint with that ID is returned; otherwise the most
        recent one is returned.

        Returns ``None`` when no checkpoint exists.
        """

    @abstractmethod
    def list(
        self,
        config: Dict[str, Any],
        *,
        limit: Optional[int] = None,
        before: Optional[str] = None,
    ) -> Iterator[CheckpointRecord]:
        """
        Iterate checkpoints for the thread in *config*, newest first.

        Parameters
        ----------
        config:
            Must contain ``configurable.thread_id``.
        limit:
            Maximum number of records to yield.
        before:
            Only yield checkpoints created before this checkpoint_id
            (exclusive cursor for pagination).
        """

    # ── Session helpers ───────────────────────────────────────────────────────

    @abstractmethod
    def get_session_info(self, thread_id: str) -> Optional[SessionInfo]:
        """
        Return a lightweight summary for *thread_id*, or ``None`` if not found.
        """

    @abstractmethod
    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SessionInfo]:
        """
        Return up to *limit* sessions ordered by last-updated descending.
        """

    @abstractmethod
    def clear_session(self, thread_id: str) -> int:
        """
        Delete all checkpoints for *thread_id*.

        Returns the number of records removed.
        """

    @abstractmethod
    def prune(
        self,
        older_than_days: int = 30,
        keep_last: int = 1,
    ) -> int:
        """
        Remove old checkpoints to bound storage growth.

        Rules applied **per thread**:
        - Always keep the ``keep_last`` most recent checkpoints.
        - Delete any remaining checkpoints older than ``older_than_days``.

        Returns total number of records deleted across all threads.
        """

    # ── Helpers available to subclasses ──────────────────────────────────────

    @staticmethod
    def _extract_thread_id(config: Dict[str, Any]) -> str:
        """Pull ``thread_id`` from ``config["configurable"]``, raise if missing."""
        try:
            thread_id = config["configurable"]["thread_id"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "config must contain configurable.thread_id — "
                f"got: {json.dumps(config, default=str)}"
            ) from exc
        if not thread_id:
            raise ValueError("configurable.thread_id must be a non-empty string.")
        return str(thread_id)

    @staticmethod
    def _extract_checkpoint_id(config: Dict[str, Any]) -> Optional[str]:
        """Pull optional ``checkpoint_id`` from ``config["configurable"]``."""
        try:
            return config["configurable"].get("checkpoint_id")
        except (KeyError, AttributeError):
            return None

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.__class__.__name__})"
