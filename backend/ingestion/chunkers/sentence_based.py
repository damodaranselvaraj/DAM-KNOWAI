"""
Strategy 3 — Sentence-based chunker.
Uses LlamaIndex SentenceSplitter to split at sentence boundaries while
respecting the token budget.  Falls back to the recursive splitter when
LlamaIndex is unavailable.
"""
from __future__ import annotations

import logging
import re

from backend.ingestion.chunkers.base_chunker import BaseChunker
from backend.models.chunk import ChunkingConfig, ChunkingStrategy
from backend.models.parsed_document import ParsedDocument, ParsedPage
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

_LLAMA_AVAILABLE = False
try:
    from llama_index.core.node_parser import SentenceSplitter as _SentenceSplitter
    _LLAMA_AVAILABLE = True
except Exception:
    try:
        from llama_index.node_parser import SentenceSplitter as _SentenceSplitter  # type: ignore
        _LLAMA_AVAILABLE = True
    except Exception:
        logger.warning(
            "LlamaIndex SentenceSplitter unavailable — "
            "SentenceBasedChunker will use built-in sentence splitter."
        )


class SentenceBasedChunker(BaseChunker):
    """
    Chunks text by grouping complete sentences up to `chunk_size` tokens.

    Preserves sentence integrity — no sentence is cut in the middle.
    Sentences that exceed the token budget are split at punctuation.

    Best for: plain-text documents (TXT, e-mails, narrative prose).
    """

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.SENTENCE_BASED

    def chunk_text(
        self,
        text:   str,
        config: ChunkingConfig,
        doc:    ParsedDocument,
        pages:  list[ParsedPage],
    ) -> list[str]:
        if _LLAMA_AVAILABLE:
            return self._llama_sentence_split(text, config)
        return self._builtin_sentence_split(text, config)

    # ── LlamaIndex path ───────────────────────────────────────────────────────

    @staticmethod
    def _llama_sentence_split(text: str, config: ChunkingConfig) -> list[str]:
        try:
            from llama_index.core.schema import Document as LlamaDoc
            splitter = _SentenceSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.overlap,
            )
            nodes = splitter.get_nodes_from_documents([LlamaDoc(text=text)])
            return [n.get_content() for n in nodes if n.get_content().strip()]
        except Exception as exc:
            logger.warning(
                "LlamaIndex SentenceSplitter failed (%s) — falling back.", exc
            )
            return SentenceBasedChunker._builtin_sentence_split(text, config)

    # ── Built-in path ─────────────────────────────────────────────────────────

    @staticmethod
    def _builtin_sentence_split(text: str, config: ChunkingConfig) -> list[str]:
        """
        Sentence tokeniser using regex boundaries.
        Groups sentences greedily until the token budget is full, then
        starts a new chunk with `overlap` tokens carried over.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks:  list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens(sent)

            # Single sentence exceeds budget — force-split it
            if sent_tokens > config.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                # Split the oversized sentence by word windows
                parts = BaseChunker._split_by_tokens(sent, config.chunk_size, 0)
                chunks.extend(parts[:-1])
                # Keep the last part in current buffer for overlap
                if parts:
                    current       = [parts[-1]]
                    current_tokens = count_tokens(parts[-1])
                continue

            if current_tokens + sent_tokens > config.chunk_size and current:
                chunks.append(" ".join(current))
                # Overlap: carry last N tokens worth of sentences
                current, current_tokens = _carry_overlap(current, config.overlap)

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if c.strip()]


# ─── Helpers ──────────────────────────────────────────────────────────────────

_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"])|"   # sentence ending followed by capital
    r"(?<=\n)\s*\n"                  # blank line
)


def _split_sentences(text: str) -> list[str]:
    """Simple regex-based sentence splitter."""
    parts = _SENT_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _carry_overlap(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Return the tail of `sentences` that fits within `overlap_tokens`."""
    if overlap_tokens <= 0:
        return [], 0
    tail:   list[str] = []
    tokens = 0
    for sent in reversed(sentences):
        t = count_tokens(sent)
        if tokens + t > overlap_tokens:
            break
        tail.insert(0, sent)
        tokens += t
    return tail, tokens
