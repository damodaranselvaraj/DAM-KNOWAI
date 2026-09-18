"""
SQLiteSaver — durable, file-backed checkpoint backend.  DEFAULT.

Persists every checkpoint to a local SQLite database.  State survives
process restarts and is safe for single-server, multi-thread deployments.

Schema
──────
Table: checkpoints
    thread_id       TEXT  NOT NULL
    checkpoint_id   TEXT  NOT NULL  (UUID4, PRIMARY KEY with thread_id)
    parent_id       TEXT            (NULL for first checkpoint)
    checkpoint_json TEXT  NOT NULL  (JSON blob of graph state)
    metadata_json   TEXT  NOT NULL  (JSON blob of step metadata)
    created_at      TEXT  NOT NULL  (ISO-8601 UTC)

Indexes:
    idx_thread_created  ON checkpoints(thread_id, created_at DESC)

Configuration (all via environment / .env):
    MEMORY_BACKEND          (default: sqlite)
    SQLITE_DB_PATH          (default: .data/memory/checkpoints.db)
    SQLITE_POOL_TIMEOUT     (default: 30 seconds)
    SQLITE_WAL_MODE         (default: true — enables WAL journal for concurrency)
    MEMORY_MAX_CHECKPOINTS  (default: 0 — unlimited)
    MEMORY_PRUNE_DAYS       (default: 30)
    MEMORY_KEEP_LAST        (default: 5)

Thread safety
─────────────
SQLite in WAL mode supports concurrent readers + one writer.  All write
operations (put, clear_session, prune) acquire a threading.Lock to
serialise writes without blocking reads.  The connection is opened in
``check_same_thread=False`` mode — safe because we handle locking ourselves.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field

from backend.memory.base import BaseMemorySaver, CheckpointRecord, SessionInfo

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

class SQLiteSaverConfig(BaseSettings):
    """Loaded from environment / .env."""

    db_path: str = Field(".data/memory/checkpoints.db", alias="SQLITE_DB_PATH")
    pool_timeout: int = Field(30, alias="SQLITE_POOL_TIMEOUT")
    wal_mode: bool = Field(True, alias="SQLITE_WAL_MODE")
    max_checkpoints: int = Field(0, alias="MEMORY_MAX_CHECKPOINTS")  # 0 = unlimited
    prune_days: int = Field(30, alias="MEMORY_PRUNE_DAYS")
    keep_last: int = Field(5, alias="MEMORY_KEEP_LAST")

    model_config = {
        "populate_by_name": True,
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


# ─── SQLiteSaver ─────────────────────────────────────────────────────────────

class SQLiteSaver(BaseMemorySaver):
    """
    Durable SQLite checkpoint saver.  The DEFAULT backend for this project.

    Usage::

        config_  = SQLiteSaverConfig()           # reads .env
        saver    = SQLiteSaver(config_)
        saver.setup()                            # creates DB + table

        thread   = {"configurable": {"thread_id": "session-abc"}}
        thread   = saver.put(thread, {"messages": [...]}, {"step": 1})
        state    = saver.get(thread)             # latest checkpoint dict

        for record in saver.list(thread, limit=10):
            print(record.checkpoint_id, record.created_at)

        saver.prune()                            # honour .env retention policy
        saver.teardown()                         # close connection

    Connection lifecycle
    ────────────────────
    A single ``sqlite3.Connection`` is opened in ``setup()`` and closed in
    ``teardown()``.  For FastAPI, call ``setup()`` in the lifespan startup
    hook and ``teardown()`` in the shutdown hook.
    """

    def __init__(self, config: SQLiteSaverConfig) -> None:
        self.config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.Lock()   # serialise writes

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        Open the SQLite connection, enable WAL mode, and create the
        ``checkpoints`` table + indexes if they don't already exist.
        """
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,     # we manage locking ourselves
            timeout=self.config.pool_timeout,
            isolation_level=None,        # autocommit — we manage transactions
        )
        self._conn.row_factory = sqlite3.Row

        if self.config.wal_mode:
            self._conn.execute("PRAGMA journal_mode=WAL;")

        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")  # WAL-safe performance

        self._create_schema()

        logger.info(
            json.dumps({
                "event": "sqlite_saver_setup",
                "db_path": str(db_path),
                "wal_mode": self.config.wal_mode,
                "max_checkpoints": self.config.max_checkpoints,
            })
        )

    def _create_schema(self) -> None:
        """Create the checkpoints table and indexes if not present."""
        assert self._conn is not None, "Call setup() first."
        with self._write_lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id       TEXT NOT NULL,
                    checkpoint_id   TEXT NOT NULL,
                    parent_id       TEXT,
                    checkpoint_json TEXT NOT NULL,
                    metadata_json   TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_id)
                );

                CREATE INDEX IF NOT EXISTS idx_thread_created
                    ON checkpoints (thread_id, created_at DESC);
                """
            )

    def teardown(self) -> None:
        """Close the SQLite connection cleanly."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info(json.dumps({"event": "sqlite_saver_teardown"}))

    def health_check(self) -> bool:
        """Return ``True`` when the DB file is reachable and readable."""
        if self._conn is None:
            return False
        try:
            self._conn.execute("SELECT 1;")
            return True
        except Exception as exc:
            logger.warning(
                json.dumps({"event": "sqlite_saver_health_fail", "error": str(exc)})
            )
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_ready(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "SQLiteSaver is not initialised. Call setup() before use."
            )
        return self._conn

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            thread_id=row["thread_id"],
            checkpoint_id=row["checkpoint_id"],
            parent_id=row["parent_id"],
            checkpoint=json.loads(row["checkpoint_json"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ── LangGraph core ────────────────────────────────────────────────────────

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Persist *checkpoint* for the thread in *config*.

        Behaviour:
        - Generates a new UUID4 ``checkpoint_id`` when not supplied.
        - Looks up the current latest checkpoint to set ``parent_id``.
        - Enforces ``MEMORY_MAX_CHECKPOINTS`` per thread (evicts oldest).
        - Returns an updated config with ``checkpoint_id`` filled in.
        """
        conn = self._assert_ready()
        thread_id     = self._extract_thread_id(config)
        checkpoint_id = self._extract_checkpoint_id(config) or str(uuid.uuid4())
        created_at    = self._utcnow().isoformat()

        with self._write_lock:
            # Parent = current latest checkpoint for this thread
            row = conn.execute(
                """
                SELECT checkpoint_id FROM checkpoints
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
            parent_id = row["checkpoint_id"] if row else None

            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                    (thread_id, checkpoint_id, parent_id,
                     checkpoint_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_id,
                    parent_id,
                    json.dumps(checkpoint, default=str),
                    json.dumps(metadata, default=str),
                    created_at,
                ),
            )

            # Enforce per-thread cap
            max_cp = self.config.max_checkpoints
            if max_cp > 0:
                overflow = conn.execute(
                    """
                    SELECT checkpoint_id FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (thread_id, max_cp),
                ).fetchall()
                if overflow:
                    ids_to_delete = [r["checkpoint_id"] for r in overflow]
                    placeholders = ",".join("?" * len(ids_to_delete))
                    conn.execute(
                        f"DELETE FROM checkpoints "
                        f"WHERE thread_id = ? AND checkpoint_id IN ({placeholders})",
                        (thread_id, *ids_to_delete),
                    )
                    logger.debug(
                        json.dumps({
                            "event": "sqlite_saver_evict",
                            "thread_id": thread_id,
                            "evicted_count": len(ids_to_delete),
                        })
                    )

        logger.debug(
            json.dumps({
                "event": "sqlite_saver_put",
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "parent_id": parent_id,
            })
        )

        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": checkpoint_id,
            },
        }

    def get(
        self,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Return the checkpoint state dict for the thread.

        - When ``checkpoint_id`` is in config → exact lookup.
        - Otherwise → most recent checkpoint.
        - Returns ``None`` when nothing is found.
        """
        conn = self._assert_ready()
        thread_id     = self._extract_thread_id(config)
        checkpoint_id = self._extract_checkpoint_id(config)

        if checkpoint_id:
            row = conn.execute(
                "SELECT checkpoint_json FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT checkpoint_json FROM checkpoints "
                "WHERE thread_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (thread_id,),
            ).fetchone()

        if row is None:
            return None
        return json.loads(row["checkpoint_json"])

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
            Cursor: only yield checkpoints whose ``created_at`` is strictly
            earlier than that of the checkpoint with this ID.
        """
        conn = self._assert_ready()
        thread_id = self._extract_thread_id(config)

        params: list = [thread_id]
        where  = "WHERE thread_id = ?"

        if before:
            # Resolve the created_at of the cursor checkpoint
            cursor_row = conn.execute(
                "SELECT created_at FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, before),
            ).fetchone()
            if cursor_row:
                where += " AND created_at < ?"
                params.append(cursor_row["created_at"])

        sql = f"SELECT * FROM checkpoints {where} ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        rows = conn.execute(sql, params).fetchall()
        for row in rows:
            yield self._row_to_record(row)

    # ── Session helpers ───────────────────────────────────────────────────────

    def get_session_info(self, thread_id: str) -> Optional[SessionInfo]:
        conn = self._assert_ready()
        agg = conn.execute(
            """
            SELECT
                COUNT(*)          AS cnt,
                MIN(created_at)   AS first_at,
                MAX(created_at)   AS last_at,
                MAX(metadata_json) AS latest_meta
            FROM checkpoints
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()

        if not agg or agg["cnt"] == 0:
            return None

        return SessionInfo(
            thread_id=thread_id,
            checkpoint_count=agg["cnt"],
            created_at=datetime.fromisoformat(agg["first_at"]),
            last_updated=datetime.fromisoformat(agg["last_at"]),
            metadata=json.loads(agg["latest_meta"] or "{}"),
        )

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SessionInfo]:
        conn = self._assert_ready()
        rows = conn.execute(
            """
            SELECT
                thread_id,
                COUNT(*)            AS cnt,
                MIN(created_at)     AS first_at,
                MAX(created_at)     AS last_at,
                MAX(metadata_json)  AS latest_meta
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        return [
            SessionInfo(
                thread_id=row["thread_id"],
                checkpoint_count=row["cnt"],
                created_at=datetime.fromisoformat(row["first_at"]),
                last_updated=datetime.fromisoformat(row["last_at"]),
                metadata=json.loads(row["latest_meta"] or "{}"),
            )
            for row in rows
        ]

    def clear_session(self, thread_id: str) -> int:
        """Delete all checkpoints for *thread_id*. Returns rows deleted."""
        conn = self._assert_ready()
        with self._write_lock:
            cur = conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            deleted = cur.rowcount

        logger.info(
            json.dumps({
                "event": "sqlite_saver_clear_session",
                "thread_id": thread_id,
                "deleted": deleted,
            })
        )
        return deleted

    def prune(
        self,
        older_than_days: Optional[int] = None,
        keep_last: Optional[int] = None,
    ) -> int:
        """
        Remove old checkpoints to bound database growth.

        Uses ``MEMORY_PRUNE_DAYS`` and ``MEMORY_KEEP_LAST`` from config
        when the parameters are not explicitly supplied.

        Per thread: always keep ``keep_last`` most-recent checkpoints;
        delete remaining ones older than ``older_than_days``.

        Returns total rows deleted across all threads.
        """
        conn = self._assert_ready()
        days    = older_than_days if older_than_days is not None else self.config.prune_days
        k_last  = keep_last       if keep_last       is not None else self.config.keep_last
        cutoff  = (self._utcnow() - timedelta(days=days)).isoformat()

        # For each thread: find checkpoint_ids beyond keep_last that are old
        thread_ids = [
            r["thread_id"]
            for r in conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            ).fetchall()
        ]

        total_deleted = 0
        with self._write_lock:
            for tid in thread_ids:
                # IDs of the k_last most recent — these are always spared
                keep_rows = conn.execute(
                    """
                    SELECT checkpoint_id FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (tid, k_last),
                ).fetchall()
                keep_ids = {r["checkpoint_id"] for r in keep_rows}

                if not keep_ids:
                    continue

                placeholders = ",".join("?" * len(keep_ids))
                cur = conn.execute(
                    f"""
                    DELETE FROM checkpoints
                    WHERE thread_id = ?
                      AND checkpoint_id NOT IN ({placeholders})
                      AND created_at < ?
                    """,
                    (tid, *keep_ids, cutoff),
                )
                total_deleted += cur.rowcount

        logger.info(
            json.dumps({
                "event": "sqlite_saver_prune",
                "older_than_days": days,
                "keep_last": k_last,
                "total_deleted": total_deleted,
            })
        )
        return total_deleted

    # ── Extra helpers ─────────────────────────────────────────────────────────

    def vacuum(self) -> None:
        """
        Run ``VACUUM`` to compact the database file after heavy pruning.

        This is an exclusive write operation — avoid calling it during
        peak traffic.  Safe to schedule as an off-peak maintenance task.
        """
        conn = self._assert_ready()
        with self._write_lock:
            conn.execute("VACUUM;")
        logger.info(json.dumps({"event": "sqlite_saver_vacuum"}))

    @property
    def db_size_bytes(self) -> int:
        """Return the size of the SQLite database file in bytes."""
        path = Path(self.config.db_path)
        return path.stat().st_size if path.exists() else 0

    def __repr__(self) -> str:
        size_kb = self.db_size_bytes / 1024
        return (
            f"SQLiteSaver("
            f"db='{self.config.db_path}', "
            f"size={size_kb:.1f} KB, "
            f"wal={self.config.wal_mode})"
        )
