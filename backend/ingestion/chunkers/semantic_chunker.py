"""
Strategy 5 — Semantic chunker.
Uses embedding-based cosine similarity to detect topic-shift boundaries.

Primary path  : LlamaIndex SemanticSplitterNodeParser (requires OpenAI embeddings).
Fallback path : Statistical sentence-similarity via dot product on TF-IDF vectors
               (zero extra dependencies) when OpenAI / LlamaIndex unavailable.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from backend.ingestion.chunkers.base_chunker import BaseChunker
from backend.models.chunk import ChunkingConfig, ChunkingStrategy
from backend.models.parsed_document import ParsedDocument, ParsedPage
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

_LLAMA_SEMANTIC_AVAILABLE = False
try:
    from llama_index.core.node_parser import SemanticSplitterNodeParser as _SemanticParser
    _LLAMA_SEMANTIC_AVAILABLE = True
except Exception:
    try:
        from llama_index.node_parser import SemanticSplitterNodeParser as _SemanticParser  # type: ignore
        _LLAMA_SEMANTIC_AVAILABLE = True
    except Exception:
        logger.warning(
            "SemanticSplitterNodeParser unavailable — "
            "SemanticChunker will use statistical fallback."
        )


class SemanticChunker(BaseChunker):
    """
    Detects topic-boundary sentences using embedding cosine similarity.

    Primary:  LlamaIndex SemanticSplitterNodeParser with OpenAI embeddings.
    Fallback: TF-IDF cosine similarity between consecutive sentence windows.

    Config params used:
        breakpoint_percentile — cosine-drop threshold (default 95)
        semantic_buffer_size  — sentences to compare at boundaries (default 1)
        chunk_size            — max token cap per chunk
    """

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.SEMANTIC

    def chunk_text(
        self,
        text:   str,
        config: ChunkingConfig,
        doc:    ParsedDocument,
        pages:  list[ParsedPage],
    ) -> list[str]:
        if _LLAMA_SEMANTIC_AVAILABLE:
            result = self._llama_semantic_split(text, config)
            if result:
                return result
            logger.warning("[Semantic] LlamaIndex path returned empty — using fallback")

        return self._statistical_split(text, config)

    # ── LlamaIndex path ───────────────────────────────────────────────────────

    @staticmethod
    def _llama_semantic_split(text: str, config: ChunkingConfig) -> list[str]:
        """Use SemanticSplitterNodeParser with OpenAI text-embedding-3-small."""
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.core.schema import Document as LlamaDoc
            from backend.config import settings

            embed_model = OpenAIEmbedding(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )
            parser = _SemanticParser(
                buffer_size=config.semantic_buffer_size,
                breakpoint_percentile_threshold=config.breakpoint_percentile,
                embed_model=embed_model,
            )
            nodes = parser.get_nodes_from_documents([LlamaDoc(text=text)])
            segments = [n.get_content() for n in nodes if n.get_content().strip()]
            logger.debug(
                "[Semantic/LlamaIndex] %d segments, percentile=%d",
                len(segments), config.breakpoint_percentile,
            )
            return segments
        except Exception as exc:
            logger.warning("[Semantic] LlamaIndex path failed: %s", exc)
            return []

    # ── Statistical TF-IDF fallback ───────────────────────────────────────────

    @staticmethod
    def _statistical_split(text: str, config: ChunkingConfig) -> list[str]:
        """
        Split on sentences whose TF-IDF cosine similarity to the previous
        buffer falls below the `breakpoint_percentile` threshold.
        """
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        buf       = config.semantic_buffer_size
        threshold = _cosine_threshold(sentences, config.breakpoint_percentile, buf)

        segments:    list[str]  = []
        current:     list[str]  = []
        cur_tokens               = 0

        for i, sent in enumerate(sentences):
            sent_tokens = count_tokens(sent)

            # Force-break if over token cap
            if cur_tokens + sent_tokens > config.chunk_size and current:
                segments.append(" ".join(current))
                current, cur_tokens = _overlap_tail(
                    current, int(config.chunk_size * 0.15)
                )

            # Semantic boundary detection
            if i >= buf and current:
                prev_window = " ".join(sentences[max(0, i - buf):i])
                curr_window = " ".join(sentences[i:i + buf])
                sim = _cosine_similarity(_tfidf(prev_window), _tfidf(curr_window))
                if sim < threshold:
                    if current:
                        segments.append(" ".join(current))
                    current, cur_tokens = _overlap_tail(
                        current, int(config.chunk_size * 0.15)
                    )

            current.append(sent)
            cur_tokens += sent_tokens

        if current:
            segments.append(" ".join(current))

        logger.debug(
            "[Semantic/Statistical] %d sentences → %d segments (threshold=%.3f)",
            len(sentences), len(segments), threshold,
        )
        return [s for s in segments if s.strip()]


# ─── TF-IDF helpers (zero external deps) ─────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def _tfidf(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    total  = len(tokens)
    return {w: c / total for w, c in counts.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot    = sum(a.get(w, 0) * b.get(w, 0) for w in a)
    norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_threshold(
    sentences: list[str],
    percentile: int,
    buffer: int,
) -> float:
    """
    Compute the similarity threshold as the (100-percentile) quantile
    of pairwise consecutive-window cosine similarities.
    """
    sims: list[float] = []
    for i in range(buffer, len(sentences)):
        prev = " ".join(sentences[max(0, i - buffer):i])
        curr = " ".join(sentences[i:i + buffer])
        sims.append(_cosine_similarity(_tfidf(prev), _tfidf(curr)))
    if not sims:
        return 0.3
    sims.sort()
    idx = max(0, int(len(sims) * (100 - percentile) / 100) - 1)
    return sims[idx]


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])|(?<=\n)\s*\n")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _overlap_tail(sents: list[str], max_tokens: int) -> tuple[list[str], int]:
    if max_tokens <= 0:
        return [], 0
    tail:   list[str] = []
    tokens = 0
    for s in reversed(sents):
        t = count_tokens(s)
        if tokens + t > max_tokens:
            break
        tail.insert(0, s)
        tokens += t
    return tail, tokens
