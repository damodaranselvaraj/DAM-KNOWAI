"""
In-memory staging store for uploaded-but-not-yet-parsed files.

Phase 1 parsing is deliberately deferred until the pipeline is explicitly
triggered (the "Run Pipeline" button / POST /pipeline/run). The upload
endpoint (POST /documents/upload) only validates the file and stages the
raw bytes here — it does not parse, chunk, embed, or store anything.

/pipeline/run drains this store first, parsing each staged file through
the normal Phase 1 pipeline (ingest_document) before continuing on to the
existing Phase 2-4 (chunk → embed → store) loop.

Replace with a persistent/DB-backed store for multi-process deployments.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class PendingUpload:
    """A validated-but-unparsed file waiting for the pipeline to run."""
    upload_id:        str
    filename:         str
    content:          bytes
    content_type:     str
    size_bytes:       int
    version_handling: str = "replace"
    force_parser:     str | None = None
    staged_at:        str = ""


class PendingUploadStore:
    """Thread-safe in-memory registry of files waiting to be parsed."""

    def __init__(self) -> None:
        self._items: dict[str, PendingUpload] = {}
        self._lock = Lock()

    def add(
        self,
        filename:         str,
        content:          bytes,
        content_type:     str,
        version_handling: str        = "replace",
        force_parser:     str | None = None,
    ) -> PendingUpload:
        item = PendingUpload(
            upload_id=str(uuid4()),
            filename=filename,
            content=content,
            content_type=content_type,
            size_bytes=len(content),
            version_handling=version_handling,
            force_parser=force_parser,
            staged_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._items[item.upload_id] = item
        logger.debug(
            "PendingUploadStore: staged '%s' (upload_id=%s, %d bytes)",
            filename, item.upload_id, item.size_bytes,
        )
        return item

    def all(self) -> list[PendingUpload]:
        """Return all currently staged uploads without removing them."""
        with self._lock:
            return list(self._items.values())

    def pop_all(self) -> list[PendingUpload]:
        """Atomically remove and return every staged upload (drained by a pipeline run)."""
        with self._lock:
            items = list(self._items.values())
            self._items.clear()
        return items

    def count(self) -> int:
        with self._lock:
            return len(self._items)


# ─── Module-level singleton ───────────────────────────────────────────────────
pending_upload_store = PendingUploadStore()
