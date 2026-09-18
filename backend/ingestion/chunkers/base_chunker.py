"""
Abstract base class for all chunking strategies.

Every concrete chunker must implement `chunk_text()`.  The base class
provides:
  - Shared Chunk factory (_make_chunk / _make_chunks_from_texts)
  - Title-prefix logic (heading + optional doc title)
  - Token counting integration
  - Timed public entry-point that returns a list[Chunk]
"""
from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import final

from backend.models.chunk import Chunk, ChunkingConfig, ChunkingStrategy, ChunkType
from backend.models.parsed_document import ParsedDocument, ParsedPage
from backend.utils.hashing import hash_text
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """
    Contract every chunking strategy must satisfy.

    Subclasses implement:
        chunk_text(text, config, doc, pages) -> list[str]

    The public entry-point `chunk()` wraps the implementation with
    metadata hydration, title-prefix injection, token counting, and
    timing.
    """

    # ── Must be overridden ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def strategy(self) -> ChunkingStrategy:
        ...

    @abstractmethod
    def chunk_text(
        self,
        text:   str,
        config: ChunkingConfig,
        doc:    ParsedDocument,
        pages:  list[ParsedPage],
    ) -> list[str]:
        """
        Split `text` into raw string segments.

        Args:
            text:   Concatenated page text to split.
            config: Runtime chunking parameters.
            doc:    Source ParsedDocument (for metadata / title).
            pages:  Ordered list of source pages contributing to `text`.

        Returns:
            Ordered list of chunk text strings (no empty strings).
        """
        ...

    # ── Public timed entry-point ──────────────────────────────────────────────

    @final
    def chunk(
        self,
        doc:    ParsedDocument,
        config: ChunkingConfig,
        pages:  list[ParsedPage] | None = None,
    ) -> list[Chunk]:
        """
        Timed, metadata-hydrated chunking call.

        Args:
            doc:    Source ParsedDocument.
            config: Chunking configuration.
            pages:  Pages to use (defaults to all doc.pages).

        Returns:
            Ordered list of fully populated Chunk objects.
        """
        if pages is None:
            pages = doc.pages

        # Build the text to split from the given pages
        text = self._pages_to_text(pages)
        if not text.strip():
            logger.warning(
                "[%s] Empty text for doc '%s' — returning no chunks.",
                self.strategy.value, doc.filename,
            )
            return []

        t0 = time.perf_counter()
        raw_segments = self.chunk_text(text, config, doc, pages)
        elapsed_ms   = (time.perf_counter() - t0) * 1000

        # Filter empties and below-minimum size
        segments = [
            s for s in raw_segments
            if s and s.strip() and count_tokens(s) >= max(config.min_chunk_tokens, 1)
        ]

        if not segments:
            logger.warning(
                "[%s] All segments empty/below minimum for '%s'.",
                self.strategy.value, doc.filename,
            )
            return []

        chunks = self._make_chunks_from_texts(segments, doc, config, pages)

        logger.info(
            "[%s] '%s' → %d chunks in %.0f ms (tokens: min=%d, max=%d, avg=%d)",
            self.strategy.value, doc.filename, len(chunks), elapsed_ms,
            min(c.token_count for c in chunks),
            max(c.token_count for c in chunks),
            sum(c.token_count for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    # ── Chunk factory helpers ─────────────────────────────────────────────────

    def _make_chunks_from_texts(
        self,
        texts:  list[str],
        doc:    ParsedDocument,
        config: ChunkingConfig,
        pages:  list[ParsedPage],
    ) -> list[Chunk]:
        title_prefix  = self._build_title_prefix(doc, config)
        page_numbers  = [p.page_number for p in pages]

        chunks: list[Chunk] = []
        for idx, text in enumerate(texts):
            chunk = self._make_chunk(
                text=text,
                doc=doc,
                config=config,
                chunk_index=idx,
                page_numbers=page_numbers,
                title_prefix=title_prefix,
            )
            chunks.append(chunk)
        return chunks

    def _make_chunk(
        self,
        text:         str,
        doc:          ParsedDocument,
        config:       ChunkingConfig,
        chunk_index:  int,
        page_numbers: list[int],
        title_prefix: str = "",
        chunk_type:   ChunkType = ChunkType.TEXT,
        extras:       dict | None = None,
        parent_chunk_id: str | None = None,
        grandparent_chunk_id: str | None = None,
    ) -> Chunk:
        token_count = count_tokens(text)
        text_with_title = (
            f"{title_prefix}\n\n{text}" if title_prefix else text
        )

        return Chunk(
            chunk_id=f"{doc.doc_id}:{chunk_index}",
            doc_id=doc.doc_id,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            text=text,
            token_count=token_count,
            char_count=len(text),
            title_prefix=title_prefix,
            text_with_title=text_with_title,
            doc_name=doc.filename,
            file_type=doc.file_type.value,
            page_numbers=page_numbers,
            doc_title=doc.title,
            doc_author=doc.author,
            doc_language=doc.language,
            published_at=doc.published_at,
            doc_sha256=doc.sha256,
            content_hash=hash_text(text),
            strategy=self.strategy,
            parent_chunk_id=parent_chunk_id,
            grandparent_chunk_id=grandparent_chunk_id,
            extras=extras or {},
        )

    # ── Text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pages_to_text(pages: list[ParsedPage]) -> str:
        """Join page texts with double newline separator."""
        return "\n\n".join(p.text for p in pages if p.text and p.text.strip())

    @staticmethod
    def _build_title_prefix(doc: ParsedDocument, config: ChunkingConfig) -> str:
        """
        Construct the title prefix prepended to every chunk.
        Format: "{doc_title} | {filename_stem}" or just the filename stem.
        """
        if not config.prepend_title:
            return ""
        parts: list[str] = []
        if config.prepend_doc_title and doc.title:
            parts.append(doc.title.strip())
        # Always include filename stem as fallback context
        stem = doc.filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
        if stem and (not parts or stem.lower() != parts[0].lower()):
            parts.append(stem)
        return " | ".join(parts)

    @staticmethod
    def _extract_headings(text: str) -> list[tuple[int, str]]:
        """
        Return (position, heading_text) tuples for Markdown headings
        and ALL-CAPS lines that look like section titles.
        """
        headings: list[tuple[int, str]] = []
        for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
            headings.append((m.start(), m.group(2).strip()))
        for m in re.finditer(r"^([A-Z][A-Z\s]{4,})$", text, re.MULTILINE):
            candidate = m.group(1).strip()
            if 3 <= len(candidate.split()) <= 10:
                headings.append((m.start(), candidate))
        headings.sort(key=lambda h: h[0])
        return headings

    @staticmethod
    def _find_nearest_heading(pos: int, headings: list[tuple[int, str]]) -> str:
        """Return the heading text that appears immediately before `pos`."""
        result = ""
        for h_pos, h_text in headings:
            if h_pos <= pos:
                result = h_text
            else:
                break
        return result

    @staticmethod
    def _split_by_tokens(
        text:       str,
        chunk_size: int,
        overlap:    int = 0,
    ) -> list[str]:
        """
        Pure-Python token-aware splitter used by fixed-size strategies.
        Splits on whitespace boundaries, respects chunk_size and overlap.
        """
        words  = text.split()
        if not words:
            return []

        # Approximate tokens per word ≈ 1.3 for English
        words_per_chunk = max(1, int(chunk_size / 1.3))
        words_overlap   = max(0, int(overlap / 1.3))

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end    = min(start + words_per_chunk, len(words))
            chunk  = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end >= len(words):
                break
            start  = end - words_overlap if words_overlap else end
        return chunks
