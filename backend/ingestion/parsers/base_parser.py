"""
Abstract base class for all document parsers.

Every concrete parser must implement `parse_bytes()`.  The base class
provides the common timing harness, logging, and the `_make_attempt()`
audit helper so individual parsers stay focused on extraction logic.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import final

from backend.models.parsed_document import (
    FileType,
    ParsedDocument,
    ParsedPage,
    ParserAttempt,
    ParserName,
)
from backend.utils.hashing import hash_bytes

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """
    Contract every parser must satisfy.

    Subclasses implement:
        parse_bytes(content, filename, file_type) -> ParsedDocument

    The public entry-point `parse()` wraps the implementation with
    timing and a structured audit entry that the router accumulates.
    """

    # ── Must be overridden by subclasses ──────────────────────────────────────
    @property
    @abstractmethod
    def name(self) -> ParserName:
        """Unique parser identifier used in audit trails."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> frozenset[FileType]:
        """File types this parser can handle."""
        ...

    @abstractmethod
    def parse_bytes(
        self,
        content:   bytes,
        filename:  str,
        file_type: FileType,
    ) -> ParsedDocument:
        """
        Extract text and metadata from raw file bytes.

        Args:
            content:   Raw bytes of the uploaded file.
            filename:  Original filename (used for metadata / logging).
            file_type: Validated FileType enum value.

        Returns:
            A fully populated ParsedDocument.

        Raises:
            Any exception — the router will catch it, record the failure
            in the audit trail, and attempt the next parser in the chain.
        """
        ...

    # ── Public timed entry-point (do not override) ───────────────────────────
    @final
    def parse(
        self,
        content:   bytes,
        filename:  str,
        file_type: FileType,
    ) -> tuple[ParsedDocument | None, ParserAttempt]:
        """
        Timed wrapper around `parse_bytes()`.

        Returns:
            (ParsedDocument, ParserAttempt) on success — document is populated,
                attempt.succeeded = True.
            (None, ParserAttempt) on failure — attempt.succeeded = False,
                attempt.error contains the exception message.
        """
        if file_type not in self.supported_types:
            attempt = ParserAttempt(
                parser=self.name,
                succeeded=False,
                duration_ms=0.0,
                error=f"{self.name.value} does not support file type '{file_type.value}'",
                pages_extracted=0,
            )
            logger.debug(
                "[%s] Skipped — unsupported type '%s'",
                self.name.value, file_type.value,
            )
            return None, attempt

        logger.info(
            "[%s] Parsing '%s' (%d bytes) …",
            self.name.value, filename, len(content),
        )
        t0 = time.perf_counter()
        try:
            doc = self.parse_bytes(content, filename, file_type)
            duration_ms = (time.perf_counter() - t0) * 1000
            attempt = ParserAttempt(
                parser=self.name,
                succeeded=True,
                duration_ms=round(duration_ms, 2),
                pages_extracted=len(doc.pages),
            )
            logger.info(
                "[%s] ✓ Parsed '%s' in %.0f ms — %d page(s), %d words",
                self.name.value, filename, duration_ms,
                len(doc.pages), doc.total_words,
            )
            return doc, attempt

        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            attempt = ParserAttempt(
                parser=self.name,
                succeeded=False,
                duration_ms=round(duration_ms, 2),
                error=f"{type(exc).__name__}: {exc}",
                pages_extracted=0,
            )
            logger.warning(
                "[%s] ✗ Failed on '%s' after %.0f ms — %s: %s",
                self.name.value, filename, duration_ms,
                type(exc).__name__, exc,
            )
            return None, attempt

    # ── Shared helpers available to all subclasses ────────────────────────────

    def _base_doc(
        self,
        content:   bytes,
        filename:  str,
        file_type: FileType,
    ) -> dict:
        """
        Return a dict of fields common to every ParsedDocument so
        subclasses don't repeat boilerplate.
        """
        return dict(
            filename=filename,
            file_type=file_type,
            size_bytes=len(content),
            sha256=hash_bytes(content),
            parser_used=self.name,
        )

    @staticmethod
    def _make_page(
        page_number: int,
        text:        str,
        *,
        has_tables: bool = False,
        has_images: bool = False,
        markdown:   str | None = None,
        extras:     dict | None = None,
    ) -> ParsedPage:
        """Convenience constructor for ParsedPage."""
        return ParsedPage(
            page_number=page_number,
            text=text.strip(),
            has_tables=has_tables,
            has_images=has_images,
            markdown=markdown,
            extras=extras or {},
        )

    def can_handle(self, file_type: FileType) -> bool:
        """Quick check used by the router before calling parse()."""
        return file_type in self.supported_types
