"""
Chunkers sub-package — Phase 2 of the RAG ingestion pipeline.

Public surface
──────────────
ChunkingPipeline   — main orchestrator, call chunk_document()
ChunkerFactory     — get a chunker by ChunkingStrategy
ChunkDeduplicator  — deduplicate chunks by content hash
TableChunker       — standalone table splitter

All strategy classes are also re-exported for direct use / testing.
"""
from backend.ingestion.chunkers.chunker_factory     import ChunkerFactory, ChunkDeduplicator, chunker_factory, chunk_deduplicator
from backend.ingestion.chunkers.table_chunker       import TableChunker
from backend.ingestion.chunkers.fixed_size          import FixedSizeChunker
from backend.ingestion.chunkers.fixed_overlap       import FixedOverlapChunker
from backend.ingestion.chunkers.sentence_based      import SentenceBasedChunker
from backend.ingestion.chunkers.recursive_chunker   import RecursiveChunker
from backend.ingestion.chunkers.semantic_chunker    import SemanticChunker
from backend.ingestion.chunkers.hierarchical_chunker import HierarchicalChunker

__all__ = [
    "ChunkerFactory",
    "ChunkDeduplicator",
    "chunker_factory",
    "chunk_deduplicator",
    "TableChunker",
    "FixedSizeChunker",
    "FixedOverlapChunker",
    "SentenceBasedChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "HierarchicalChunker",
]
