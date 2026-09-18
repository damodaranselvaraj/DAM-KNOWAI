"""
LlamaIndex SimpleDirectoryReader — universal last-resort fallback parser.

Handles every supported file type when both PyMuPDF and Docling have
failed or are unavailable.  Also acts as the primary path for file types
that neither upstream parser handles well.
"""
from __future__ import annotations

import logging
import os
import tempfile
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

# LlamaIndex is lazy-imported inside parse_bytes() to avoid module-level
# Pydantic v1/v2 compat crashes that affect some llama-index builds.
# _LLAMA_AVAILABLE is set on first successful import attempt.
_LLAMA_AVAILABLE: bool | None = None   # None = not yet probed


def _get_simple_directory_reader():
    """Lazy import — returns the class or raises RuntimeError."""
    global _LLAMA_AVAILABLE
    if _LLAMA_AVAILABLE is False:
        raise RuntimeError("LlamaIndex is not available (previous import failed).")
    try:
        from llama_index.core import SimpleDirectoryReader as SDR
        _LLAMA_AVAILABLE = True
        return SDR
    except Exception:
        pass
    try:
        from llama_index import SimpleDirectoryReader as SDR  # type: ignore[no-redef]
        _LLAMA_AVAILABLE = True
        return SDR
    except Exception as exc:
        _LLAMA_AVAILABLE = False
        raise RuntimeError(
            f"LlamaIndex is not installed or has a compat issue: {exc}"
        ) from exc

# Extension used when writing the temp file
_EXT_MAP: dict[FileType, str] = {
    FileType.PDF:  ".pdf",
    FileType.DOCX: ".docx",
    FileType.CSV:  ".csv",
    FileType.HTML: ".html",
    FileType.TXT:  ".txt",
    FileType.XLSX: ".xlsx",
}


class LlamaIndexParser(BaseParser):
    """
    LlamaIndex SimpleDirectoryReader fallback parser.

    Writes content to a temporary single-file directory, runs
    SimpleDirectoryReader, then aggregates LlamaIndex Document objects
    into our ParsedDocument schema.

    Best for:  Situations where Docling and PyMuPDF are unavailable or
               have failed.  Handles every supported file type.
    """

    @property
    def name(self) -> ParserName:
        return ParserName.LLAMAINDEX

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
        SimpleDirectoryReader = _get_simple_directory_reader()  # raises if unavailable

        base     = self._base_doc(content, filename, file_type)
        tmp_dir  = tempfile.mkdtemp(prefix="rag_llama_")
        tmp_path = os.path.join(tmp_dir, filename)

        try:
            # Write file into an isolated directory
            with open(tmp_path, "wb") as fh:
                fh.write(content)

            reader = SimpleDirectoryReader(
                input_dir=tmp_dir,
                recursive=False,
            )
            llama_docs = reader.load_data()

            pages        = self._convert_to_pages(llama_docs, filename)
            parse_status = ParseStatus.SUCCESS if pages else ParseStatus.PARTIAL

            return ParsedDocument(
                **base,
                pages=pages,
                parse_status=parse_status,
                source_path=tmp_path,
                title=Path(filename).stem or None,
            )

        finally:
            # Clean up temp dir and file
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _convert_to_pages(
        self,
        llama_docs: list,
        filename:   str,
    ) -> list[ParsedPage]:
        """
        Convert a list of LlamaIndex Document objects to ParsedPage list.

        LlamaIndex may split a file into multiple Documents (one per page
        for PDFs, one per row for CSVs, etc.).  We normalise them to our
        page model, combining single-page formats into one page.
        """
        if not llama_docs:
            logger.warning("[LlamaIndex] No documents extracted from '%s'", filename)
            return []

        pages: list[ParsedPage] = []

        # Check if docs carry page_label metadata (PDFs split per page)
        first_meta = getattr(llama_docs[0], "metadata", {}) or {}
        has_page_labels = "page_label" in first_meta or "page_number" in first_meta

        if has_page_labels:
            for doc in llama_docs:
                meta     = getattr(doc, "metadata", {}) or {}
                page_num = int(
                    meta.get("page_label")
                    or meta.get("page_number")
                    or (len(pages) + 1)
                )
                text = getattr(doc, "text", "") or ""
                pages.append(
                    self._make_page(
                        page_number=page_num,
                        text=text,
                        extras={k: str(v) for k, v in meta.items()},
                    )
                )
        else:
            # Non-paginated formats: combine all docs into a single page
            combined = "\n\n".join(
                getattr(d, "text", "") for d in llama_docs if getattr(d, "text", "")
            )
            if combined:
                pages.append(
                    self._make_page(page_number=1, text=combined)
                )

        # Ensure pages are sorted by page_number
        pages.sort(key=lambda p: p.page_number)
        return pages
