"""
InMemorySaver — ephemeral, process-local checkpoint backend.

Stores all checkpoints in a plain Python dict protected by a
threading.RLock.  State is lost when the process exits.

Best for:
    - Local development and unit tests
    - Stateless / serverless deployments where persistence is handled
      externally (e.g. the caller serialises state between invocations)

NOT suitable for:
    - Multi-process deployments (state is NOT shared across workers)
    - Production workloads that require durability across restarts

LangGraph integration
─────────────────────
LangGraph's own ``MemorySaver`` is a thin dict wrapper with no pruning,
session listing, or health-check.  This class extends the project's
``BaseMemorySaver`` contract so the same interface works regardless of
which backend is active.

Thread safety
─────────────
All public methods acquire a reentrant lock before touching shared state,
making the saver safe to use from multiple threads in a single process
(e.g. FastAPI with a thread-pool executor).
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional

from backend.memory.base import BaseMemorySaver, CheckpointRecord, SessionInfo

logger = logging.getLogger(__name__)


class InMemorySaver(BaseMemorySaver):
    """
    Ephemeral in-process checkpoint saver.

    Internal storage layout::

        _store: {
            thread_id: [CheckpointRecord, ...]   # ordered oldest → newest
        }

    All mutation is guarded by ``_lock`` (threading.RLock).

    Usage::

        saver = InMemorySaver(max_checkpoints_per_thread=50)
        saver.setup()

        config  = {"configurable": {"thread_id": "session-abc"}}
        config  = saver.put(config, checkpoint={"messages": [...]}, metadata={})
        state   = saver.get(config)
    """

    def __init__(self, max_checkpoints_per_thread: int = 100) -> None:
        """
        Parameters
        ----------
        max_checkpoints_per_thread:
            Hard cap on stored checkpoints per thread.  When exceeded the
            oldest checkpoint is evicted automatically (FIFO).  Set to 0
            for no cap.
        """
        self._max_per_thread = max_checkpoints_per_thread
        # thread_id → list[CheckpointRecord] (oldest first)
        self._store: Dict[str, List[CheckpointRecord]] = defaultdict(list)
        self._lock = threading.RLock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        logger.info(
            json.dumps({
                "event": "memory_saver_setup",
                "backend": "memory",
                "max_per_thread": self._max_per_thread,
            })
        )

    def teardown(self) -> None:
        with self._lock:
            count = sum(len(v) for v in self._store.values())
            self._store.clear()
        logger.info(
            json.dumps({
                "event": "memory_saver_teardown",
                "backend": "memory",
                "checkpoints_discarded": count,
            })
        )

    def health_check(self) -> bool:
        # In-memory store is always healthy while the process is alive
        return True

    # ── LangGraph core ────────────────────────────────────────────────────────

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Persist *checkpoint* for the thread in *config*.

        A new ``checkpoint_id`` (UUID4) is minted when not already present
        in ``config["configurable"]``.

        Returns an updated config dict with ``checkpoint_id`` filled in.
        """
        thread_id     = self._extract_thread_id(config)
        checkpoint_id = (
            self._extract_checkpoint_id(config) or str(uuid.uuid4())
        )

        # Determine parent: the most recent existing checkpoint_id for this thread
        with self._lock:
            existing = self._store[thread_id]
            parent_id = existing[-1].checkpoint_id if existing else None

            record = CheckpointRecord(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                parent_id=parent_id,
                checkpoint=checkpoint,
                metadata=metadata,
                created_at=self._utcnow(),
            )
            existing.append(record)

            # Enforce per-thread cap (evict oldest)
            if self._max_per_thread and len(existing) > self._max_per_thread:
                evicted = existing.pop(0)
                logger.debug(
                    json.dumps({
                        "event": "memory_saver_evict",
                        "thread_id": thread_id,
                        "evicted_checkpoint_id": evicted.checkpoint_id,
                    })
                )

        logger.debug(
            json.dumps({
                "event": "memory_saver_put",
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "parent_id": parent_id,
            })
        )

        updated_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": checkpoint_id,
            },
        }
        return updated_config

    def get(
        self,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Return the latest checkpoint state dict for the thread, or ``None``.

        If ``config["configurable"]["checkpoint_id"]`` is set, that exact
        checkpoint is retrieved; otherwise the most-recent one is returned.
        """
        thread_id     = self._extract_thread_id(config)
        checkpoint_id = self._extract_checkpoint_id(config)

        with self._lock:
            records = self._store.get(thread_id, [])
            if not records:
                return None

            if checkpoint_id:
                # Exact lookup
                for r in reversed(records):
                    if r.checkpoint_id == checkpoint_id:
                        return r.checkpoint
                return None

            # Most recent
            return records[-1].checkpoint

    def list(
        self,
        config: Dict[str, Any],
        *,
        limit: Optional[int] = None,
        before: Optional[str] = None,
    ) -> Iterator[CheckpointRecord]:
        """
        Yield CheckpointRecords for the thread, newest first.

        Parameters
        ----------
        limit:
            Maximum records to yield.
        before:
            Cursor: only yield records created before the checkpoint with
            this ID (exclusive).
        """
        thread_id = self._extract_thread_id(config)

        with self._lock:
            # Copy to avoid mutation during iteration
            records = list(reversed(self._store.get(thread_id, [])))

        # Apply `before` cursor
        if before:
            cutoff_idx = next(
                (i for i, r in enumerate(records) if r.checkpoint_id == before),
                None,
            )
            if cutoff_idx is not None:
                records = records[cutoff_idx + 1 :]

        # Apply limit
        if limit:
            records = records[:limit]

        yield from records

    # ── Session helpers ───────────────────────────────────────────────────────

    def get_session_info(self, thread_id: str) -> Optional[SessionInfo]:
        with self._lock:
            records = self._store.get(thread_id)
            if not records:
                return None
            return SessionInfo(
                thread_id=thread_id,
                checkpoint_count=len(records),
                created_at=records[0].created_at,
                last_updated=records[-1].created_at,
                metadata=records[-1].metadata,
            )

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SessionInfo]:
        with self._lock:
            # Build and sort by last_updated descending
            sessions: List[SessionInfo] = []
            for thread_id, records in self._store.items():
                if records:
                    sessions.append(
                        SessionInfo(
                            thread_id=thread_id,
                            checkpoint_count=len(records),
                            created_at=records[0].created_at,
                            last_updated=records[-1].created_at,
                            metadata=records[-1].metadata,
                        )
                    )
            sessions.sort(
                key=lambda s: s.last_updated or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            return sessions[offset : offset + limit]

    def clear_session(self, thread_id: str) -> int:
        """Delete all checkpoints for *thread_id*. Returns count removed."""
        with self._lock:
            records = self._store.pop(thread_id, [])
            count = len(records)

        logger.info(
            json.dumps({
                "event": "memory_saver_clear_session",
                "thread_id": thread_id,
                "deleted": count,
            })
        )
        return count

    def prune(
        self,
        older_than_days: int = 30,
        keep_last: int = 1,
    ) -> int:
        """
        Prune old checkpoints across all threads.

        Per thread: always keep ``keep_last`` newest; delete the rest if
        they are older than ``older_than_days``.

        Returns total checkpoints deleted.
        """
        cutoff = self._utcnow() - timedelta(days=older_than_days)
        total_deleted = 0

        with self._lock:
            for thread_id, records in self._store.items():
                if len(records) <= keep_last:
                    continue
                # Keep the newest `keep_last`; prune the rest if old enough
                to_keep   = records[-keep_last:]
                candidates = records[:-keep_last]
                survivors  = [r for r in candidates if r.created_at >= cutoff]
                deleted    = len(candidates) - len(survivors)
                self._store[thread_id] = survivors + to_keep
                total_deleted += deleted

        logger.info(
            json.dumps({
                "event": "memory_saver_prune",
                "older_than_days": older_than_days,
                "keep_last": keep_last,
                "total_deleted": total_deleted,
            })
        )
        return total_deleted

    # ── Extra helpers ─────────────────────────────────────────────────────────

    @property
    def total_checkpoints(self) -> int:
        """Total number of checkpoints currently held in memory."""
        with self._lock:
            return sum(len(v) for v in self._store.values())

    @property
    def total_sessions(self) -> int:
        """Number of distinct threads / sessions currently tracked."""
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return (
            f"InMemorySaver("
            f"sessions={self.total_sessions}, "
            f"checkpoints={self.total_checkpoints}, "
            f"max_per_thread={self._max_per_thread})"
        )
