"""
ChunkingPipeline — Phase 2 orchestrator.

Receives a ParsedDocument from Phase 1 and returns a ChunkingResult
containing text chunks + table chunks ready for Phase 3 embedding.

Auto-Strategy Selection
───────────────────────
CSV / XLSX          → FIXED_SIZE   (uniform rows, no semantic structure)
TXT                 → SENTENCE_BASED
Policy / legal doc  → HIERARCHICAL (detected via title + content keywords)
All others          → RECURSIVE    ★ default

Force-override:
    pipeline.chunk_document(doc, config)
    where config.strategy is set explicitly by the caller.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Final

from backend.ingestion.chunkers.chunker_factory import chunk_deduplicator, chunker_factory
from backend.ingestion.chunkers.table_chunker import TableChunker
from backend.models.chunk import (
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkingStrategy,
)
from backend.models.parsed_document import FileType, ParsedDocument

logger = logging.getLogger(__name__)

# ─── Policy / legal keyword detector ─────────────────────────────────────────

_POLICY_KEYWORDS: Final[frozenset[str]] = frozenset({
    "policy", "policies", "regulation", "regulations", "regulatory",
    "compliance", "compliant", "legislation", "legislative", "statute",
    "statutes", "act ", "acts ", "bylaw", "bye-law", "ordinance",
    "directive", "directives", "procedure", "procedures", "protocol",
    "protocols", "guideline", "guidelines", "framework", "frameworks",
    "clause", "clauses", "section", "sections", "article ", "articles ",
    "schedule", "schedules", "appendix", "annex", "terms and conditions",
    "terms of service", "privacy policy", "data protection",
    "contract", "agreement", "obligation", "obligations", "liability",
    "indemnity", "warranty", "warranties",
})

_POLICY_TITLE_RE = re.compile(
    r"\b(policy|regulation|compliance|directive|bylaw|statute|"
    r"ordinance|agreement|contract|procedure|guideline)\b",
    re.IGNORECASE,
)


def _is_policy_document(doc: ParsedDocument) -> bool:
    """
    Return True if the document looks like a policy or legal text.
    Checks document title, filename, and first 500 chars of body text.
    """
    signals: list[str] = []
    if doc.title:
        signals.append(doc.title.lower())
    signals.append(doc.filename.lower())
    if doc.full_text:
        signals.append(doc.full_text[:500].lower())

    combined = " ".join(signals)
    if _POLICY_TITLE_RE.search(combined):
        return True

    # Count keyword hits in the first 2 000 chars of full text
    sample = doc.full_text[:2_000].lower() if doc.full_text else ""
    hits = sum(1 for kw in _POLICY_KEYWORDS if kw in sample)
    return hits >= 3


# ─── Strategy auto-selector ───────────────────────────────────────────────────

def auto_select_strategy(doc: ParsedDocument) -> ChunkingStrategy:
    """
    Infer the best chunking strategy from document type and content signals.

    Priority:
      1. Tabular formats (csv, xlsx) → FIXED_SIZE
      2. Plain text                  → SENTENCE_BASED
      3. Policy / legal keywords     → HIERARCHICAL
      4. Default                     → RECURSIVE
    """
    ft = doc.file_type

    if ft in (FileType.CSV, FileType.XLSX):
        logger.debug("Auto-select: FIXED_SIZE for tabular format '%s'", ft.value)
        return ChunkingStrategy.FIXED_SIZE

    if ft == FileType.TXT:
        logger.debug("Auto-select: SENTENCE_BASED for plain-text '%s'", doc.filename)
        return ChunkingStrategy.SENTENCE_BASED

    if _is_policy_document(doc):
        logger.debug("Auto-select: HIERARCHICAL (policy signals) for '%s'", doc.filename)
        return ChunkingStrategy.HIERARCHICAL

    logger.debug("Auto-select: RECURSIVE (default) for '%s'", doc.filename)
    return ChunkingStrategy.RECURSIVE


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class ChunkingPipeline:
    """
    Stateless orchestrator — thread-safe, one instance per process.

    Usage::

        pipeline = ChunkingPipeline()
        result   = pipeline.chunk_document(parsed_doc)          # auto strategy
        result   = pipeline.chunk_document(parsed_doc, config)  # explicit strategy
    """

    def __init__(self) -> None:
        self._table_chunker = TableChunker()

    def chunk_document(
        self,
        doc:    ParsedDocument,
        config: ChunkingConfig | None = None,
    ) -> ChunkingResult:
        """
        Run Phase 2 chunking on a ParsedDocument.

        Args:
            doc:    Output of Phase 1 parsing.
            config: Optional ChunkingConfig.  If None, defaults are applied
                    and the strategy is auto-selected from the document type.

        Returns:
            ChunkingResult with deduplicated text + table chunks.
        """
        t0 = time.perf_counter()

        auto_mode = config is None
        if config is None:
            config = ChunkingConfig()

        # ── Strategy selection ────────────────────────────────────────────────
        # Auto-select only when the caller passed no config at all.
        # An explicit config (even with strategy=RECURSIVE) is always honoured.
        if auto_mode:
            effective_strategy = auto_select_strategy(doc)
            if effective_strategy != config.strategy:
                config = config.model_copy(update={"strategy": effective_strategy})
        else:
            effective_strategy = config.strategy

        logger.info(
            "ChunkingPipeline: '%s' [%s] → strategy=%s",
            doc.filename, doc.file_type.value, effective_strategy.value,
        )

        # ── Guard: empty document ─────────────────────────────────────────────
        if doc.is_empty or not doc.full_text.strip():
            logger.warning("ChunkingPipeline: '%s' has no extractable text.", doc.filename)
            return ChunkingResult(
                doc_id=doc.doc_id,
                doc_name=doc.filename,
                strategy=effective_strategy,
                chunking_time_ms=0.0,
            )

        # ── Guard: extremely large document ───────────────────────────────────
        # Cap the amount of text handed to the chunker so a pathologically
        # large upload can't blow up chunking time/memory unboundedly.
        # If full_text exceeds the ceiling, chunk only the leading slice and
        # log a warning — the caller still gets a usable (partial) result
        # instead of a multi-minute hang or OOM.
        from backend.config import settings as _cfg
        max_chars = _cfg.max_chunking_input_chars
        if len(doc.full_text) > max_chars:
            logger.warning(
                "ChunkingPipeline: '%s' full_text is %d chars, exceeds the "
                "%d char ceiling — truncating input for chunking.",
                doc.filename, len(doc.full_text), max_chars,
            )
            import copy
            doc = copy.copy(doc)
            object.__setattr__(doc, "full_text", doc.full_text[:max_chars])

        # ── Text chunking ─────────────────────────────────────────────────────
        chunker      = chunker_factory.get(effective_strategy)
        raw_chunks   = chunker.chunk(doc, config)

        # ── Table chunking ────────────────────────────────────────────────────
        has_tables = any(p.has_tables for p in doc.pages)
        table_chunks: list[Chunk] = []
        if has_tables:
            table_chunks = self._table_chunker.chunk_tables(
                doc=doc,
                config=config,
                base_index_offset=len(raw_chunks),
            )

        # ── Deduplication ─────────────────────────────────────────────────────
        unique_text, text_dups   = chunk_deduplicator.deduplicate(raw_chunks)
        unique_table, table_dups = chunk_deduplicator.deduplicate(table_chunks)
        total_dups               = text_dups + table_dups

        # ── Re-index chunks so indices are sequential across both lists ────────
        unique_text  = _reindex(unique_text,  start=0)
        unique_table = _reindex(unique_table, start=len(unique_text))

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        result = ChunkingResult(
            doc_id=doc.doc_id,
            doc_name=doc.filename,
            strategy=effective_strategy,
            chunks=unique_text,
            table_chunks=unique_table,
            duplicate_count=total_dups,
            chunking_time_ms=elapsed_ms,
        )

        logger.info(
            "ChunkingPipeline: '%s' → %d text chunk(s), %d table chunk(s), "
            "%d duplicate(s) removed, %d total tokens, %.0f ms",
            doc.filename,
            len(unique_text),
            len(unique_table),
            total_dups,
            result.total_tokens,
            elapsed_ms,
        )
        return result

    def chunk_pages(
        self,
        doc:        ParsedDocument,
        page_nums:  list[int],
        config:     ChunkingConfig | None = None,
    ) -> ChunkingResult:
        """
        Chunk only a specific subset of pages (e.g. for incremental re-ingestion).

        Args:
            doc:       Full ParsedDocument.
            page_nums: 1-based page numbers to include.
            config:    Optional chunking config.
        """
        selected = [p for p in doc.pages if p.page_number in set(page_nums)]
        if not selected:
            logger.warning(
                "chunk_pages: none of pages %s found in '%s'.",
                page_nums, doc.filename,
            )
            return ChunkingResult(
                doc_id=doc.doc_id,
                doc_name=doc.filename,
                strategy=(config or ChunkingConfig()).strategy,
            )

        # Build a synthetic doc view with only the selected pages
        import copy
        partial_doc = copy.copy(doc)
        object.__setattr__(partial_doc, "pages",    selected)
        object.__setattr__(partial_doc, "full_text", "\n\n".join(p.text for p in selected))
        return self.chunk_document(partial_doc, config)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _reindex(chunks: list[Chunk], start: int = 0) -> list[Chunk]:
    """
    Return a new list with chunk_index reassigned starting from `start`.
    chunk_id is updated to match the new index using the existing doc_id prefix.
    """
    reindexed: list[Chunk] = []
    for new_idx, chunk in enumerate(chunks, start=start):
        updated = chunk.model_copy(update={"chunk_index": new_idx})
        reindexed.append(updated)
    return reindexed


# ─── Module-level singleton ───────────────────────────────────────────────────

chunking_pipeline = ChunkingPipeline()
