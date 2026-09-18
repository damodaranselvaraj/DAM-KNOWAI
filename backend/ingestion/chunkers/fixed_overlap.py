"""
Strategy 2 — Fixed-size with overlap chunker.
Splits text into windows of `chunk_size` tokens with `overlap` tokens
carried over from the previous chunk to preserve context across boundaries.
"""
from __future__ import annotations

import logging

from backend.ingestion.chunkers.base_chunker import BaseChunker
from backend.models.chunk import ChunkingConfig, ChunkingStrategy
from backend.models.parsed_document import ParsedDocument, ParsedPage
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class FixedOverlapChunker(BaseChunker):
    """
    Sliding-window chunker: each chunk overlaps the previous one by
    `overlap` tokens.  Reduces information loss at chunk boundaries
    while keeping chunk sizes predictable.

    Default params: chunk_size=512, overlap=50.
    """

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.FIXED_OVERLAP

    def chunk_text(
        self,
        text:   str,
        config: ChunkingConfig,
        doc:    ParsedDocument,
        pages:  list[ParsedPage],
    ) -> list[str]:
        overlap  = min(config.overlap, config.chunk_size // 2)
        segments = self._split_by_tokens(text, config.chunk_size, overlap=overlap)
        logger.debug(
            "[FixedOverlap] '%s': %d tokens → %d segments (size=%d, overlap=%d)",
            doc.filename, count_tokens(text), len(segments),
            config.chunk_size, overlap,
        )
        return segments
