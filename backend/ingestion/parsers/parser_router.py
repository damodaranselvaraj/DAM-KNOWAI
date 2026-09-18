"""
ParserRouter — fallback chain orchestrator for Phase 1 document parsing.

Routing rules
─────────────
PDF
  1. PyMuPDF   (primary  — fast, reliable for text-based PDFs)
  2. Docling   (secondary — complex layouts, mixed content, tables)
  3. LlamaIndex (last resort)

DOCX
  1. Docling   (primary  — preserves tables, formatting)
  2. LlamaIndex (fallback)

CSV / HTML / TXT / XLSX
  1. Docling   (primary)
  2. LlamaIndex (fallback)

A `force_parser` override in ParseRequest bypasses the chain and uses
only the requested parser, raising 422 if it fails.

The full audit trail (every attempt, success or failure) is attached to
the returned ParsedDocument so downstream stages have complete provenance.
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

from backend.ingestion.parsers.base_parser   import BaseParser
from backend.ingestion.parsers.pymupdf_parser  import PyMuPDFParser
from backend.ingestion.parsers.docling_parser  import DoclingParser
from backend.ingestion.parsers.llamaindex_parser import LlamaIndexParser
from backend.models.parsed_document import (
    FileType,
    ParsedDocument,
    ParserAttempt,
    ParserName,
    ParseRequest,
    ParseResult,
    ParseStatus,
    VersionAction,
)
from backend.ingestion.file_validator import validate_upload

logger = logging.getLogger(__name__)

# ─── Fallback chains per file type ───────────────────────────────────────────
# Order matters: first parser in the list is tried first.

_PYMUPDF    = PyMuPDFParser()
_DOCLING    = DoclingParser()
_LLAMAINDEX = LlamaIndexParser()

_CHAIN: dict[FileType, list[BaseParser]] = {
    FileType.PDF:  [_PYMUPDF,  _DOCLING,    _LLAMAINDEX],
    FileType.DOCX: [_DOCLING,               _LLAMAINDEX],
    FileType.CSV:  [_DOCLING,               _LLAMAINDEX],
    FileType.HTML: [_DOCLING,               _LLAMAINDEX],
    FileType.TXT:  [_DOCLING,               _LLAMAINDEX],
    FileType.XLSX: [_DOCLING,               _LLAMAINDEX],
}

# Parser-name → instance (used for force_parser override)
_BY_NAME: dict[ParserName, BaseParser] = {
    ParserName.PYMUPDF:    _PYMUPDF,
    ParserName.DOCLING:    _DOCLING,
    ParserName.LLAMAINDEX: _LLAMAINDEX,
}


# ─── Router ───────────────────────────────────────────────────────────────────

class ParserRouter:
    """
    Stateless orchestrator.  Call `route()` once per upload.

    Thread-safe: all state lives in the ParseRequest / return values.
    """

    def route(self, request: ParseRequest) -> ParseResult:
        """
        Validate the file, select the parser chain, run with fallback,
        attach the audit trail, and return a ParseResult.

        Args:
            request: ParseRequest carrying filename, raw bytes, and options.

        Returns:
            ParseResult(success=True,  document=ParsedDocument) on success.
            ParseResult(success=False, error=…, http_status_code=…) on failure.
              • 415 — file type not supported
              • 422 — all parsers in the chain failed
        """
        t_total = time.perf_counter()

        # ── 1. Validate ───────────────────────────────────────────────────────
        try:
            validation = validate_upload(request.filename, request.content)
        except Exception as exc:
            # validate_upload raises HTTPException(415) — re-wrap for ParseResult
            code = getattr(exc, "status_code", 415)
            detail = getattr(exc, "detail", str(exc))
            logger.warning("Validation failed for '%s': %s", request.filename, detail)
            return ParseResult(success=False, error=detail, http_status_code=code)

        file_type = FileType(validation.extension)

        # ── 2. Build parser sequence ─────────────────────────────────────────
        if request.force_parser:
            chain = self._forced_chain(request.force_parser, file_type)
            if chain is None:
                msg = (
                    f"Parser '{request.force_parser.value}' does not support "
                    f"file type '{file_type.value}'"
                )
                return ParseResult(success=False, error=msg, http_status_code=415)
        else:
            chain = _CHAIN.get(file_type, [_LLAMAINDEX])

        # ── 3. Run chain with fallback ────────────────────────────────────────
        audit_trail: list[ParserAttempt] = []
        document:    ParsedDocument | None = None

        for parser in chain:
            doc, attempt = parser.parse(request.content, request.filename, file_type)
            audit_trail.append(attempt)

            if attempt.succeeded and doc is not None:
                document = doc
                break
            else:
                logger.info(
                    "Parser '%s' failed for '%s' — trying next in chain.",
                    parser.name.value, request.filename,
                )

        total_ms = round((time.perf_counter() - t_total) * 1000, 2)

        # ── 4. All parsers failed ─────────────────────────────────────────────
        if document is None:
            errors = "; ".join(
                f"{a.parser.value}: {a.error}" for a in audit_trail if a.error
            )
            logger.error(
                "All parsers failed for '%s' in %.0f ms — %s",
                request.filename, total_ms, errors,
            )
            return ParseResult(
                success=False,
                error=f"All parsers failed for '{request.filename}': {errors}",
                http_status_code=422,
            )

        # ── 5. Attach audit trail & timing ────────────────────────────────────
        document.audit_trail   = audit_trail
        document.parse_time_ms = total_ms

        # Mark status as FALLBACK_USED when the first parser in the chain failed
        if len(audit_trail) > 1 and not audit_trail[0].succeeded:
            document.parse_status = ParseStatus.FALLBACK_USED

        logger.info(
            "Parsed '%s' via '%s' in %.0f ms total — %d page(s), %d words. "
            "Chain: [%s]",
            request.filename,
            document.parser_used.value,
            total_ms,
            document.page_count,
            document.total_words,
            " → ".join(a.parser.value + ("✓" if a.succeeded else "✗")
                        for a in audit_trail),
        )

        return ParseResult(success=True, document=document)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _forced_chain(
        force: ParserName,
        file_type: FileType,
    ) -> list[BaseParser] | None:
        """Return a single-parser chain for the forced parser, or None if incompatible."""
        parser = _BY_NAME.get(force)
        if parser is None or not parser.can_handle(file_type):
            return None
        return [parser]

    # ── Introspection helpers (useful for /config endpoint) ──────────────────

    @staticmethod
    def describe_chains() -> dict[str, list[str]]:
        """Return the routing table as a plain dict for API documentation."""
        return {
            ft.value: [p.name.value for p in parsers]
            for ft, parsers in _CHAIN.items()
        }

    @staticmethod
    def supported_file_types() -> list[str]:
        return [ft.value for ft in _CHAIN]


# ─── Module-level singleton ───────────────────────────────────────────────────
router = ParserRouter()
