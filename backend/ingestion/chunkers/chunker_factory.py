"""
ChunkDeduplicator — removes duplicate chunks by SHA-256 content hash.
ChunkerFactory    — resolves a ChunkingStrategy to a concrete chunker instance.

Both are stateless utilities imported by the ChunkingPipeline.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.models.chunk import Chunk, ChunkingStrategy

if TYPE_CHECKING:
    from backend.ingestion.chunkers.base_chunker import BaseChunker

logger = logging.getLogger(__name__)


# ─── Deduplicator ─────────────────────────────────────────────────────────────

class ChunkDeduplicator:
    """
    Removes exact-duplicate chunks within a single document's chunk list.

    Two chunks are duplicates when their SHA-256 content_hash is identical.
    The first occurrence is kept; subsequent duplicates are discarded and
    counted in the ChunkingResult.duplicate_count.

    Usage::

        dedup = ChunkDeduplicator()
        unique, n_dups = dedup.deduplicate(chunks)
    """

    def deduplicate(self, chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        """
        Args:
            chunks: Ordered list of Chunk objects to deduplicate.

        Returns:
            (unique_chunks, duplicate_count)
            unique_chunks is in the same relative order as the input.
        """
        seen:       set[str]   = set()
        unique:     list[Chunk] = []
        duplicates: int         = 0

        for chunk in chunks:
            key = chunk.content_hash or chunk.text  # fallback to raw text
            if key in seen:
                duplicates += 1
                logger.debug(
                    "Duplicate chunk dropped: doc_id=%s idx=%d hash=%s…",
                    chunk.doc_id, chunk.chunk_index, key[:12],
                )
            else:
                seen.add(key)
                unique.append(chunk)

        if duplicates:
            logger.info(
                "Deduplication: removed %d duplicate chunk(s) from %d total.",
                duplicates, len(chunks),
            )

        return unique, duplicates


# ─── Factory ──────────────────────────────────────────────────────────────────

class ChunkerFactory:
    """
    Returns a concrete BaseChunker instance for a given ChunkingStrategy.

    All chunker instances are created lazily and cached (one instance per
    strategy per factory lifetime).  Since chunkers are stateless, sharing
    instances across documents is safe.
    """

    def __init__(self) -> None:
        self._cache: dict[ChunkingStrategy, "BaseChunker"] = {}

    def get(self, strategy: ChunkingStrategy) -> "BaseChunker":
        """
        Return the chunker for `strategy`, creating it if not yet cached.

        Raises:
            ValueError: If `strategy` is not a known ChunkingStrategy value.
        """
        if strategy not in self._cache:
            self._cache[strategy] = self._create(strategy)
        return self._cache[strategy]

    @staticmethod
    def _create(strategy: ChunkingStrategy) -> "BaseChunker":
        # Import here to avoid circular imports at module load time
        from backend.ingestion.chunkers.fixed_size        import FixedSizeChunker
        from backend.ingestion.chunkers.fixed_overlap     import FixedOverlapChunker
        from backend.ingestion.chunkers.sentence_based    import SentenceBasedChunker
        from backend.ingestion.chunkers.recursive_chunker import RecursiveChunker
        from backend.ingestion.chunkers.semantic_chunker  import SemanticChunker
        from backend.ingestion.chunkers.hierarchical_chunker import HierarchicalChunker

        mapping: dict[ChunkingStrategy, type] = {
            ChunkingStrategy.FIXED_SIZE:    FixedSizeChunker,
            ChunkingStrategy.FIXED_OVERLAP: FixedOverlapChunker,
            ChunkingStrategy.SENTENCE_BASED: SentenceBasedChunker,
            ChunkingStrategy.RECURSIVE:     RecursiveChunker,
            ChunkingStrategy.SEMANTIC:      SemanticChunker,
            ChunkingStrategy.HIERARCHICAL:  HierarchicalChunker,
        }

        cls = mapping.get(strategy)
        if cls is None:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Valid options: {[s.value for s in ChunkingStrategy]}"
            )

        instance = cls()
        logger.debug("ChunkerFactory: created %s", cls.__name__)
        return instance

    def available_strategies(self) -> list[str]:
        return [s.value for s in ChunkingStrategy]


# ─── Module-level singleton ───────────────────────────────────────────────────

chunker_factory = ChunkerFactory()
chunk_deduplicator = ChunkDeduplicator()
