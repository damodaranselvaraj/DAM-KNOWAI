"""
Document versioning and SHA-256 deduplication for Phase 1 ingestion.

Version strategies
──────────────────
replace      Overwrite the previous record; previous doc is marked inactive.
keep_both    Store both old and new as separate active records with incremented version.
soft_delete  Mark previous record as deleted; new record is version N+1.

The VersionStore is an in-memory registry.  In production, replace the
dict-backed store with a database-backed implementation that conforms to
the same AbstractVersionStore interface.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext as _null_ctx
from datetime import datetime, timezone
from threading import Lock

from backend.models.parsed_document import ParsedDocument, VersionAction
from backend.utils.hashing import hashes_equal

logger = logging.getLogger(__name__)


# ─── Abstract store interface ─────────────────────────────────────────────────

class AbstractVersionStore(ABC):
    """Minimum interface a concrete version store must satisfy."""

    @abstractmethod
    def find_by_sha256(self, sha256: str) -> ParsedDocument | None:
        """Return the most recent active document with this hash, or None."""

    @abstractmethod
    def find_by_filename(self, filename: str) -> list[ParsedDocument]:
        """Return all (active + inactive) records for this filename, newest first."""

    @abstractmethod
    def save(self, doc: ParsedDocument) -> None:
        """Persist a document record."""

    @abstractmethod
    def mark_inactive(self, doc_id: str) -> None:
        """Flag a document as superseded / soft-deleted."""

    @abstractmethod
    def all_active(self) -> list[ParsedDocument]:
        """Return all currently active documents."""


# ─── In-memory implementation ─────────────────────────────────────────────────

class InMemoryVersionStore(AbstractVersionStore):
    """
    Thread-safe in-memory store.

    Suitable for development, testing, and single-process deployments.
    Replace with a database-backed store for production.
    """

    def __init__(self) -> None:
        self._docs:     dict[str, ParsedDocument] = {}   # doc_id → doc
        self._inactive: set[str]                  = set() # doc_ids marked inactive
        self._lock      = Lock()

    def find_by_sha256(self, sha256: str) -> ParsedDocument | None:
        """
        Search ALL stored documents (active and inactive) for a hash match.
        Deduplication must fire even if the previous version was superseded —
        otherwise a re-upload of an old file would be ingested again.
        Returns the most recently parsed match, or None.
        """
        with self._lock:
            matches = [
                doc for doc in self._docs.values()
                if hashes_equal(doc.sha256, sha256)
            ]
        if not matches:
            return None
        # Return the newest match
        return max(matches, key=lambda d: d.parsed_at)

    def find_by_filename(self, filename: str) -> list[ParsedDocument]:
        with self._lock:
            matches = [
                d for d in self._docs.values()
                if d.filename == filename
            ]
        # Newest first
        return sorted(matches, key=lambda d: d.parsed_at, reverse=True)

    def save(self, doc: ParsedDocument) -> None:
        with self._lock:
            self._docs[doc.doc_id] = doc
        logger.debug("VersionStore: saved doc_id=%s  filename=%s  v%d",
                     doc.doc_id, doc.filename, doc.version)

    def mark_inactive(self, doc_id: str) -> None:
        with self._lock:
            self._inactive.add(doc_id)
        logger.debug("VersionStore: marked inactive doc_id=%s", doc_id)

    def all_active(self) -> list[ParsedDocument]:
        with self._lock:
            return [
                d for d in self._docs.values()
                if d.doc_id not in self._inactive
            ]

    def stats(self) -> dict:
        with self._lock:
            return {
                "total":    len(self._docs),
                "active":   len(self._docs) - len(self._inactive),
                "inactive": len(self._inactive),
            }


# ─── Version handler ──────────────────────────────────────────────────────────

class VersionHandler:
    """
    Applies a version strategy to an incoming ParsedDocument,
    updating the store and the document's version metadata in-place.

    Usage
    ─────
        handler = VersionHandler(store)
        doc = handler.apply(incoming_doc, strategy="replace")
    """

    def __init__(self, store: AbstractVersionStore) -> None:
        self._store = store

    def apply(
        self,
        doc:      ParsedDocument,
        strategy: str = "replace",
    ) -> ParsedDocument:
        """
        Resolve the version action for `doc` and persist it.

        Args:
            doc:      Freshly parsed document (mutated in-place).
            strategy: "replace" | "keep_both" | "soft_delete"

        Returns:
            The (potentially mutated) ParsedDocument after versioning.
        """
        strategy = strategy.lower().strip()

        # ── Exact duplicate check (same SHA-256) ──────────────────────────────
        existing_hash_match = self._store.find_by_sha256(doc.sha256)
        if existing_hash_match:
            logger.info(
                "Duplicate detected: '%s' matches doc_id=%s (SHA-256 collision). "
                "Skipping re-ingest.",
                doc.filename, existing_hash_match.doc_id,
            )
            # Return the already-stored document unchanged
            return existing_hash_match

        # ── Find previous version by filename ────────────────────────────────
        previous_versions = self._store.find_by_filename(doc.filename)
        # Take the most recent record that is still active
        with getattr(self._store, "_lock", _null_ctx()):
            inactive = getattr(self._store, "_inactive", set())
            previous = next(
                (d for d in previous_versions if d.doc_id not in inactive),
                None,
            )

        if previous is None:
            # First time we see this filename — simply store it
            doc.version        = 1
            doc.version_action = VersionAction.CREATED
            doc.previous_doc_id = None
            self._store.save(doc)
            logger.info("VersionHandler: CREATED '%s' as v1 (doc_id=%s)",
                        doc.filename, doc.doc_id)
            return doc

        # ── Apply strategy ────────────────────────────────────────────────────
        next_version = previous.version + 1

        if strategy == "replace":
            doc.version         = next_version
            doc.version_action  = VersionAction.REPLACED
            doc.previous_doc_id = previous.doc_id
            self._store.mark_inactive(previous.doc_id)
            self._store.save(doc)
            logger.info(
                "VersionHandler: REPLACED '%s' v%d→v%d "
                "(old doc_id=%s, new doc_id=%s)",
                doc.filename, previous.version, next_version,
                previous.doc_id, doc.doc_id,
            )

        elif strategy == "keep_both":
            doc.version         = next_version
            doc.version_action  = VersionAction.KEPT_BOTH
            doc.previous_doc_id = previous.doc_id
            # Do NOT mark previous inactive — both stay active
            self._store.save(doc)
            logger.info(
                "VersionHandler: KEPT_BOTH '%s' v%d + v%d "
                "(doc_ids=%s, %s)",
                doc.filename, previous.version, next_version,
                previous.doc_id, doc.doc_id,
            )

        elif strategy == "soft_delete":
            doc.version         = next_version
            doc.version_action  = VersionAction.SOFT_DELETED
            doc.previous_doc_id = previous.doc_id
            self._store.mark_inactive(previous.doc_id)
            self._store.save(doc)
            logger.info(
                "VersionHandler: SOFT_DELETED old '%s' v%d, stored new v%d "
                "(old doc_id=%s, new doc_id=%s)",
                doc.filename, previous.version, next_version,
                previous.doc_id, doc.doc_id,
            )

        else:
            # Unknown strategy — treat as replace and warn
            logger.warning(
                "Unknown version strategy '%s'; falling back to 'replace'.", strategy
            )
            return self.apply(doc, "replace")

        return doc


# ─── Module-level singletons ─────────────────────────────────────────────────
# Swap `version_store` for a DB-backed implementation at startup.

version_store   = InMemoryVersionStore()
version_handler = VersionHandler(version_store)
