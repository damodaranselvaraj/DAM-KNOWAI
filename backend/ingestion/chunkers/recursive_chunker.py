"""
Strategy 4 — Recursive chunker  ★ DEFAULT ★
Hierarchically splits on a priority-ordered list of separators:
  ["\n\n", "\n", " ", ""]

Target: 300–450 tokens per chunk, 15 % overlap (~45–67 tokens).
Title / heading detected from the nearest preceding heading and prepended
to every chunk for contextual retrieval.

Falls back to LlamaIndex SentenceSplitter if available; otherwise uses
the built-in recursive implementation.
"""
from __future__ import annotations

import logging
import re

from backend.config import settings
from backend.ingestion.chunkers.base_chunker import BaseChunker
from backend.models.chunk import Chunk, ChunkingConfig, ChunkingStrategy
from backend.models.parsed_document import ParsedDocument, ParsedPage
from backend.utils.hashing import hash_text
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# Recommended defaults for RecursiveChunker.
# NOTE: config.chunk_size always has a Pydantic default (see
# backend.models.chunk.ChunkingConfig), so this fallback is effectively
# unreachable in practice — kept only as a defensive guard. Sourced from
# backend.config.settings so it can't drift from the app-wide default.
_DEFAULT_CHUNK_SIZE:    int   = settings.default_chunk_size
_DEFAULT_OVERLAP_RATIO: float = 0.15  # 15 %


class RecursiveChunker(BaseChunker):
    """
    Recursive character-level text splitter.

    Works top-down through the separator list:
      1. Try to split on "\n\n"  (paragraph)
      2. If pieces still exceed budget → split on "\n"  (line)
      3. If still too large → split on " "  (word)
      4. Last resort → split on ""  (character)

    Each split respects `chunk_size` and `overlap` token budgets.
    Heading context is attached to every chunk as `title_prefix`.
    """

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.RECURSIVE

    # ── chunk_text overrides BaseChunker.chunk() for heading injection ────────

    def chunk_text(
        self,
        text:   str,
        config: ChunkingConfig,
        doc:    ParsedDocument,
        pages:  list[ParsedPage],
    ) -> list[str]:
        chunk_size = config.chunk_size or _DEFAULT_CHUNK_SIZE
        overlap    = config.overlap or max(1, int(chunk_size * _DEFAULT_OVERLAP_RATIO))
        separators = config.separators or ["\n\n", "\n", " ", ""]

        raw = _recursive_split(text, separators, chunk_size, overlap)
        logger.debug(
            "[Recursive] '%s': %d tokens → %d segments (size=%d, overlap=%d, seps=%s)",
            doc.filename, count_tokens(text), len(raw),
            chunk_size, overlap, separators,
        )
        return raw

    # ── Override chunk() to inject per-chunk heading prefixes ─────────────────

    def chunk(
        self,
        doc:    ParsedDocument,
        config: ChunkingConfig,
        pages:  list[ParsedPage] | None = None,
    ) -> list[Chunk]:
        if pages is None:
            pages = doc.pages

        text = self._pages_to_text(pages)
        if not text.strip():
            return []

        doc_title_prefix = self._build_title_prefix(doc, config)
        headings         = self._extract_headings(text)
        page_numbers     = [p.page_number for p in pages]

        raw_segments = self.chunk_text(text, config, doc, pages)
        segments     = [
            s for s in raw_segments
            if s and s.strip() and count_tokens(s) >= max(config.min_chunk_tokens, 1)
        ]

        chunks: list[Chunk] = []
        # Track character offset into the full text for heading lookup
        search_start = 0

        for idx, seg in enumerate(segments):
            # Find the position of this segment in the full text
            pos = text.find(seg[:60].strip(), search_start)
            if pos == -1:
                pos = search_start
            search_start = max(0, pos + len(seg) - 50)

            nearest_heading = self._find_nearest_heading(pos, headings)
            title_prefix    = (
                f"{doc_title_prefix} | {nearest_heading}"
                if nearest_heading and doc_title_prefix
                else nearest_heading or doc_title_prefix
            )

            chunk = self._make_chunk(
                text=seg,
                doc=doc,
                config=config,
                chunk_index=idx,
                page_numbers=page_numbers,
                title_prefix=title_prefix,
            )
            chunks.append(chunk)

        logger.info(
            "[Recursive] '%s' → %d chunks (tokens: min=%d, max=%d, avg=%d)",
            doc.filename, len(chunks),
            min((c.token_count for c in chunks), default=0),
            max((c.token_count for c in chunks), default=0),
            sum(c.token_count for c in chunks) // max(len(chunks), 1),
        )
        return chunks


# ─── Core recursive splitting algorithm ──────────────────────────────────────

def _recursive_split(
    text:       str,
    separators: list[str],
    chunk_size: int,
    overlap:    int,
) -> list[str]:
    """
    Recursively split `text` using the first separator that produces
    pieces small enough to fit in `chunk_size` tokens.
    """
    if count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # No more separators — hard character split
        return _char_split(text, chunk_size, overlap)

    sep      = separators[0]
    rest     = separators[1:]
    splits   = text.split(sep) if sep else list(text)

    chunks:  list[str] = []
    current: list[str] = []
    current_tokens     = 0

    for piece in splits:
        if not piece:
            continue
        piece_tokens = count_tokens(piece)

        if piece_tokens > chunk_size:
            # Recurse: this piece is still too large
            if current:
                merged = sep.join(current)
                if merged.strip():
                    chunks.append(merged)
                current, current_tokens = _trim_to_overlap(current, sep, overlap)

            sub = _recursive_split(piece, rest, chunk_size, overlap)
            chunks.extend(sub)
            continue

        if current_tokens + piece_tokens + (len(sep) > 0) > chunk_size and current:
            merged = sep.join(current)
            if merged.strip():
                chunks.append(merged)
            current, current_tokens = _trim_to_overlap(current, sep, overlap)

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        merged = sep.join(current)
        if merged.strip():
            chunks.append(merged)

    return [c for c in chunks if c.strip()]


def _trim_to_overlap(
    pieces:  list[str],
    sep:     str,
    overlap: int,
) -> tuple[list[str], int]:
    """Keep only the tail of `pieces` that fits within `overlap` tokens."""
    if overlap <= 0:
        return [], 0
    tail:   list[str] = []
    tokens = 0
    for p in reversed(pieces):
        t = count_tokens(p)
        if tokens + t > overlap:
            break
        tail.insert(0, p)
        tokens += t
    return tail, tokens


def _char_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard character-level split respecting approximate token budget."""
    # ~4 chars per token approximation
    chars_per_chunk   = chunk_size  * 4
    chars_per_overlap = overlap     * 4
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end   = min(start + chars_per_chunk, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chars_per_overlap if chars_per_overlap else end
    return chunks
