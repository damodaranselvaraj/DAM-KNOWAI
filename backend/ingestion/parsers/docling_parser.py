"""
Docling parser — primary handler for DOCX, CSV, HTML, TXT, XLSX
and secondary fallback for complex / table-heavy PDFs.

Docling produces rich Markdown with table structure preserved.
When Docling is unavailable the class raises RuntimeError so the
router can fall through to LlamaIndex.
"""
from __future__ import annotations

import io
import logging
import tempfile
import os
from pathlib import Path

from backend.ingestion.parsers.base_parser import BaseParser
from backend.models.parsed_document import (
    FileType,
    ParsedDocument,
    ParsedPage,
    ParserName,
    ParseStatus,
)

logger = logging.getLogger(__name__)

_DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    _DOCLING_AVAILABLE = True
except ImportError:
    logger.warning("Docling not installed — DoclingParser will always fail.")


# Map our FileType enum → Docling InputFormat
_FORMAT_MAP: dict[FileType, str] = {
    FileType.PDF:  "PDF",
    FileType.DOCX: "DOCX",
    FileType.CSV:  "CSV",
    FileType.HTML: "HTML",
    FileType.TXT:  "MD",    # Docling treats plain text as Markdown-compatible
    FileType.XLSX: "XLSX",
}

# Extension used when writing the temp file
_EXT_MAP: dict[FileType, str] = {
    FileType.PDF:  ".pdf",
    FileType.DOCX: ".docx",
    FileType.CSV:  ".csv",
    FileType.HTML: ".html",
    FileType.TXT:  ".txt",
    FileType.XLSX: ".xlsx",
}


class DoclingParser(BaseParser):
    """
    Docling-based parser.

    Docling operates on file paths (not byte streams), so we write content
    to a secure temp file, parse, then clean up immediately.

    Best for:
      - DOCX / XLSX: full fidelity including tables and formatting.
      - Complex PDFs with mixed layout, tables, and captions.
      - HTML: strips navigation/ads and extracts main content.
      - CSV: converts rows to prose for embedding.
    """

    @property
    def name(self) -> ParserName:
        return ParserName.DOCLING

    @property
    def supported_types(self) -> frozenset[FileType]:
        return frozenset({
            FileType.PDF,
            FileType.DOCX,
            FileType.CSV,
            FileType.HTML,
            FileType.TXT,
            FileType.XLSX,
        })

    # ── Core implementation ───────────────────────────────────────────────────

    def parse_bytes(
        self,
        content:   bytes,
        filename:  str,
        file_type: FileType,
    ) -> ParsedDocument:
        if not _DOCLING_AVAILABLE:
            raise RuntimeError("Docling is not installed (pip install docling)")

        base     = self._base_doc(content, filename, file_type)
        tmp_path = self._write_temp_file(content, file_type)

        try:
            converter = DocumentConverter()
            result    = converter.convert(tmp_path)
            doc_obj   = result.document

            pages        = self._extract_pages(doc_obj, file_type)
            parse_status = ParseStatus.SUCCESS if pages else ParseStatus.PARTIAL

            # Pull metadata where Docling exposes it
            meta = getattr(doc_obj, "metadata", None) or {}

            return ParsedDocument(
                **base,
                pages=pages,
                parse_status=parse_status,
                source_path=str(tmp_path),
                title=meta.get("title") or Path(filename).stem or None,
                author=meta.get("author") or None,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _write_temp_file(content: bytes, file_type: FileType) -> str:
        """Write bytes to a NamedTemporaryFile and return its path."""
        suffix = _EXT_MAP.get(file_type, ".tmp")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            return tmp.name

    def _extract_pages(self, doc_obj, file_type: FileType) -> list[ParsedPage]:
        """
        Convert Docling's document object into our ParsedPage list.

        Docling v1.x exposes text via doc_obj.export_to_markdown() and
        individual pages through doc_obj.pages (if the format has pages).
        We normalise both cases.
        """
        pages: list[ParsedPage] = []

        # ── Try page-level extraction first (PDFs, DOCX) ─────────────────────
        raw_pages = getattr(doc_obj, "pages", None)
        if raw_pages:
            for page in raw_pages:
                page_num = getattr(page, "page_no", None) or (len(pages) + 1)
                # Docling pages expose text via .export_to_markdown() or .text
                md = ""
                try:
                    md = page.export_to_markdown()
                except Exception:
                    pass
                text = _md_to_plain(md) if md else getattr(page, "text", "") or ""

                has_tables = _contains_table_md(md)
                has_images = bool(getattr(page, "images", None))

                pages.append(
                    self._make_page(
                        page_number=int(page_num),
                        text=text,
                        markdown=md or None,
                        has_tables=has_tables,
                        has_images=has_images,
                    )
                )
            if pages:
                return pages

        # ── Fallback: whole-document markdown as a single "page" ─────────────
        try:
            md = doc_obj.export_to_markdown()
        except Exception:
            md = ""

        if not md:
            try:
                md = str(doc_obj)
            except Exception:
                md = ""

        text = _md_to_plain(md)
        if text or md:
            pages.append(
                self._make_page(
                    page_number=1,
                    text=text,
                    markdown=md or None,
                    has_tables=_contains_table_md(md),
                )
            )

        return pages


# ─── Markdown helpers (no extra deps) ────────────────────────────────────────

def _md_to_plain(md: str) -> str:
    """
    Minimal Markdown → plain-text conversion without pandoc/markdown deps.
    Strips headings, bold, italic, links, and table pipes.
    """
    import re
    text = md
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)          # images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text) # links
    text = re.sub(r"#{1,6}\s+", "", text)                  # headings
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold/italic
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)    # underscore emphasis
    text = re.sub(r"`{1,3}[^`]+`{1,3}", "", text)         # code
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)  # list bullets
    text = re.sub(r"^\|.*\|$", lambda m: " ".join(
        c.strip() for c in m.group().split("|") if c.strip()
    ), text, flags=re.M)                                   # table rows → space-joined
    text = re.sub(r"^\s*[-|:]+\s*$", "", text, flags=re.M) # table separators
    text = re.sub(r"\n{3,}", "\n\n", text)                 # collapse blank lines
    return text.strip()


def _contains_table_md(md: str) -> bool:
    """Return True if the markdown string contains at least one table row."""
    import re
    return bool(re.search(r"^\|.+\|$", md, re.MULTILINE))
