"""
PyMuPDF (fitz) parser — primary handler for text-based PDF files.

Extracts per-page text, detects tables/images, and pulls document
metadata from the PDF info dict.  Falls back gracefully if a page
cannot be rendered.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

from backend.ingestion.parsers.base_parser import BaseParser
from backend.models.parsed_document import (
    FileType,
    ParsedDocument,
    ParsedPage,
    ParserName,
    ParseStatus,
)

logger = logging.getLogger(__name__)

_PYMUPDF_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    logger.warning("PyMuPDF (fitz) not installed — PyMuPDFParser will always fail.")


class PyMuPDFParser(BaseParser):
    """
    Fast, lightweight PDF parser using PyMuPDF (fitz).

    Best for:  Clean, text-based PDFs without complex layouts.
    Limitation: Scanned PDFs (image-only) yield empty text — Docling
                with OCR should be the fallback in that case.
    """

    @property
    def name(self) -> ParserName:
        return ParserName.PYMUPDF

    @property
    def supported_types(self) -> frozenset[FileType]:
        return frozenset({FileType.PDF})

    # ── Core implementation ───────────────────────────────────────────────────

    def parse_bytes(
        self,
        content:   bytes,
        filename:  str,
        file_type: FileType,
    ) -> ParsedDocument:
        if not _PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF is not installed (pip install pymupdf)")

        base = self._base_doc(content, filename, file_type)

        with fitz.open(stream=io.BytesIO(content), filetype="pdf") as pdf:
            meta         = pdf.metadata or {}
            pages        = self._extract_pages(pdf)
            parse_status = self._assess_status(pages, pdf.page_count)

        doc = ParsedDocument(
            **base,
            pages=pages,
            parse_status=parse_status,
            # Metadata from PDF info dict
            title=meta.get("title") or None,
            author=meta.get("author") or None,
            subject=meta.get("subject") or None,
            published_at=self._parse_pdf_date(meta.get("creationDate")),
        )
        return doc

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_pages(self, pdf: "fitz.Document") -> list[ParsedPage]:
        pages: list[ParsedPage] = []
        failed_pages: list[int] = []

        for page_num in range(pdf.page_count):
            try:
                page = pdf[page_num]
                text = page.get_text("text")  # plain-text extraction

                # Detect images embedded in the page
                has_images = bool(page.get_images(full=False))

                # Detect tables: look for drawing blocks that resemble lines/rects
                has_tables = self._has_table_heuristic(page)

                pages.append(
                    self._make_page(
                        page_number=page_num + 1,
                        text=text,
                        has_images=has_images,
                        has_tables=has_tables,
                        extras={
                            "width":  page.rect.width,
                            "height": page.rect.height,
                            "rotation": page.rotation,
                        },
                    )
                )
            except Exception as exc:
                logger.warning(
                    "[PyMuPDF] Page %d extraction failed: %s", page_num + 1, exc
                )
                failed_pages.append(page_num + 1)

        if failed_pages:
            logger.warning(
                "[PyMuPDF] %d page(s) failed extraction: %s",
                len(failed_pages), failed_pages,
            )
        return pages

    @staticmethod
    def _has_table_heuristic(page: "fitz.Page") -> bool:
        """
        Lightweight heuristic: a page likely contains a table if it has
        ≥ 4 drawing rectangles (typical for table cell borders).
        """
        try:
            drawings = page.get_drawings()
            rect_count = sum(1 for d in drawings if d.get("type") == "rect")
            return rect_count >= 4
        except Exception:
            return False

    @staticmethod
    def _assess_status(pages: list[ParsedPage], total_pages: int) -> ParseStatus:
        if not pages:
            return ParseStatus.FAILED
        extracted = sum(1 for p in pages if p.char_count > 0)
        if extracted == 0:
            # PDF parsed structurally but all pages are empty (scanned)
            return ParseStatus.PARTIAL
        if len(pages) < total_pages:
            return ParseStatus.PARTIAL
        return ParseStatus.SUCCESS

    @staticmethod
    def _parse_pdf_date(raw: str | None) -> str | None:
        """
        Convert a PDF date string like "D:20240115103045+00'00'" to
        ISO-8601 date "2024-01-15".  Returns None on any failure.
        """
        if not raw:
            return None
        try:
            # Strip the "D:" prefix
            s = raw.lstrip("D:").lstrip("d:")
            # Take only the first 8 chars: YYYYMMDD
            date_part = s[:8]
            dt = datetime.strptime(date_part, "%Y%m%d")
            return dt.date().isoformat()
        except Exception:
            return None
