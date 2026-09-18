"""
Strategy 1 — Fixed-size chunker.
Splits text into equal-sized segments of exactly `chunk_size` tokens with
no overlap between consecutive chunks.
"""
from __future__ import annotations

import logging

from backend.ingestion.chunkers.base_chunker import BaseChunker
from backend.models.chunk import ChunkingConfig, ChunkingStrategy
from backend.models.parsed_document import ParsedDocument, ParsedPage
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size windows of `chunk_size` tokens.
    No overlap — each token appears in exactly one chunk.

    Best for: tabular / CSV content where semantic boundaries
              don't matter and uniform chunk sizes are preferred.
    """

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.FIXED_SIZE

    def chunk_text(
        self,
        text:   str,
        config: ChunkingConfig,
        doc:    ParsedDocument,
        pages:  list[ParsedPage],
    ) -> list[str]:
        segments = self._split_by_tokens(text, config.chunk_size, overlap=0)
        logger.debug(
            "[FixedSize] '%s': %d tokens → %d segments (size=%d)",
            doc.filename, count_tokens(text), len(segments), config.chunk_size,
        )
        return segments
