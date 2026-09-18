"""
Table chunker — handles Markdown and plain-text tables separately from
prose so table structure is preserved in the embedding.

Rules
─────
≤ table_token_limit (default 600 tokens):
    → Single chunk containing the entire table (headers + all rows)

> table_token_limit:
    → Row-batched chunks: each chunk = headers row + N data rows such
      that each chunk fits within table_token_limit tokens.

All table chunks are tagged with ChunkType.TABLE and carry the table's
headers in extras["table_headers"] for downstream context.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator

from backend.ingestion.chunkers.base_chunker import BaseChunker
from backend.models.chunk import Chunk, ChunkingConfig, ChunkingStrategy, ChunkType
from backend.models.parsed_document import ParsedDocument, ParsedPage
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class TableChunker:
    """
    Standalone chunker for Markdown table strings.

    Not a subclass of BaseChunker because it operates on individual table
    strings extracted by the pipeline, not on full document text.
    """

    def chunk_tables(
        self,
        doc:    ParsedDocument,
        config: ChunkingConfig,
        base_index_offset: int = 0,
    ) -> list[Chunk]:
        """
        Extract and chunk all Markdown tables found in the document pages.

        Args:
            doc:               Source ParsedDocument.
            config:            Chunking configuration.
            base_index_offset: Starting chunk_index (so table chunks don't
                               collide with text chunk indices).

        Returns:
            List of TABLE-typed Chunk objects.
        """
        all_chunks:  list[Chunk] = []
        title_prefix = self._build_title_prefix(doc, config)
        chunk_index  = base_index_offset

        for page in doc.pages:
            if not page.has_tables:
                continue

            # Try to extract tables from the page markdown or plain text
            source = page.markdown or page.text
            tables = list(_extract_markdown_tables(source))

            if not tables:
                # No Markdown table found — treat the whole page as one table
                # chunk if the page itself is tabular (e.g. CSV)
                if doc.file_type.value in ("csv", "xlsx"):
                    tables = [page.text]

            for table_md in tables:
                if not table_md.strip():
                    continue

                sub_chunks = self._split_table(
                    table_md=table_md,
                    doc=doc,
                    config=config,
                    page_number=page.page_number,
                    title_prefix=title_prefix,
                    start_index=chunk_index,
                )
                all_chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        logger.debug(
            "[TableChunker] '%s': %d table chunk(s) across %d page(s)",
            doc.filename, len(all_chunks), len(doc.pages),
        )
        return all_chunks

    # ── Internal splitting logic ──────────────────────────────────────────────

    @staticmethod
    def _split_table(
        table_md:    str,
        doc:         ParsedDocument,
        config:      ChunkingConfig,
        page_number: int,
        title_prefix: str,
        start_index: int,
    ) -> list[Chunk]:
        token_limit = config.table_token_limit
        total_tokens = count_tokens(table_md)

        if total_tokens <= token_limit:
            # Single chunk
            return [_make_table_chunk(
                text=table_md,
                doc=doc,
                chunk_index=start_index,
                page_number=page_number,
                title_prefix=title_prefix,
                row_range=(0, -1),
            )]

        # Row-batched splitting
        rows    = _parse_table_rows(table_md)
        if not rows:
            return [_make_table_chunk(
                text=table_md, doc=doc, chunk_index=start_index,
                page_number=page_number, title_prefix=title_prefix, row_range=(0, -1),
            )]

        header     = rows[0]              # first row = column headers
        separator  = rows[1] if len(rows) > 1 and _is_separator(rows[1]) else None
        data_rows  = rows[2:] if separator else rows[1:]
        header_txt = header + ("\n" + separator if separator else "")
        header_tok = count_tokens(header_txt)

        chunks:    list[Chunk] = []
        batch:     list[str]   = []
        batch_tok              = header_tok
        batch_start            = 0

        for row_idx, row in enumerate(data_rows):
            row_tok = count_tokens(row)
            if batch_tok + row_tok > token_limit and batch:
                text = header_txt + "\n" + "\n".join(batch)
                chunks.append(_make_table_chunk(
                    text=text, doc=doc, chunk_index=start_index + len(chunks),
                    page_number=page_number, title_prefix=title_prefix,
                    row_range=(batch_start, batch_start + len(batch) - 1),
                    headers=header,
                ))
                batch_start = row_idx
                batch       = []
                batch_tok   = header_tok

            batch.append(row)
            batch_tok += row_tok

        if batch:
            text = header_txt + "\n" + "\n".join(batch)
            chunks.append(_make_table_chunk(
                text=text, doc=doc, chunk_index=start_index + len(chunks),
                page_number=page_number, title_prefix=title_prefix,
                row_range=(batch_start, batch_start + len(batch) - 1),
                headers=header,
            ))

        return chunks

    @staticmethod
    def _build_title_prefix(doc: ParsedDocument, config: ChunkingConfig) -> str:
        if not config.prepend_title:
            return ""
        parts: list[str] = []
        if config.prepend_doc_title and doc.title:
            parts.append(doc.title.strip())
        stem = doc.filename.rsplit(".", 1)[0].replace("_", " ")
        if stem and (not parts or stem.lower() != parts[0].lower()):
            parts.append(stem)
        return " | ".join(parts)


# ─── Helpers ──────────────────────────────────────────────────────────────────

_TABLE_ROW_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
_SEP_ROW_RE   = re.compile(r"^\|[\s\-:|]+\|$")


def _extract_markdown_tables(text: str) -> Iterator[str]:
    """
    Yield each contiguous block of Markdown table lines as a single string.
    """
    if not text:
        return
    lines          = text.splitlines(keepends=True)
    in_table       = False
    current: list[str] = []

    for line in lines:
        is_table_row = bool(_TABLE_ROW_RE.match(line.rstrip()))
        if is_table_row:
            in_table = True
            current.append(line.rstrip())
        else:
            if in_table and current:
                yield "\n".join(current)
                current = []
            in_table = False

    if current:
        yield "\n".join(current)


def _parse_table_rows(table_md: str) -> list[str]:
    return [line.strip() for line in table_md.splitlines() if line.strip()]


def _is_separator(row: str) -> bool:
    return bool(_SEP_ROW_RE.match(row.strip()))


def _make_table_chunk(
    text:         str,
    doc:          ParsedDocument,
    chunk_index:  int,
    page_number:  int,
    title_prefix: str,
    row_range:    tuple[int, int],
    headers:      str = "",
) -> Chunk:
    from backend.utils.hashing import hash_text

    text_with_title = f"{title_prefix}\n\n{text}" if title_prefix else text
    return Chunk(
        chunk_id=f"{doc.doc_id}:T{chunk_index}",
        doc_id=doc.doc_id,
        chunk_index=chunk_index,
        chunk_type=ChunkType.TABLE,
        text=text,
        token_count=count_tokens(text),
        char_count=len(text),
        title_prefix=title_prefix,
        text_with_title=text_with_title,
        doc_name=doc.filename,
        file_type=doc.file_type.value,
        page_numbers=[page_number],
        doc_title=doc.title,
        doc_author=doc.author,
        doc_language=doc.language,
        published_at=doc.published_at,
        doc_sha256=doc.sha256,
        content_hash=hash_text(text),
        strategy=ChunkingStrategy.FIXED_SIZE,   # tables use fixed strategy internally
        extras={
            "table_headers": headers,
            "row_start":     row_range[0],
            "row_end":       row_range[1],
        },
    )
