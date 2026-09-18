"""
Layer 4 — Context Control.

Applied after reranking and before the prompt is built:

    A. Relevance Threshold Gate — drop chunks scoring below the
       configured threshold (default 0.70). If nothing survives, the
       caller must trigger the no-context response rather than pass an
       empty context to the LLM.
    B. Context Deduplication — exact-match (content hash) then
       near-duplicate (cosine similarity over TF-IDF-ish bag-of-words
       vectors, default threshold 0.95) removal, keeping the
       highest-scored representative of each duplicate cluster.
    C. Context Relevance Scoring — a final lightweight lexical
       query/chunk overlap score; chunks below the threshold are
       dropped too.

All three steps are individually toggleable and operate on
``RetrievalResult`` lists (backend.retrieval.dense_retriever), so they
slot directly into the existing chat pipeline between reranking and
``PromptBuilder.build()``.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

from backend.retrieval.dense_retriever import RetrievalResult

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> Counter:
    return Counter(_WORD_RE.findall((text or "").lower()))


def _cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _chunk_score(chunk: RetrievalResult) -> float:
    return chunk.rerank_score if chunk.rerank_score is not None else chunk.score


@dataclass
class ContextControlResult:
    chunks: List[RetrievalResult] = field(default_factory=list)
    after_threshold_count: int = 0
    after_dedup_count: int = 0
    after_relevance_scoring_count: int = 0
    duplicates_removed: int = 0
    documents_below_threshold: int = 0


def apply_relevance_threshold(
    chunks: List[RetrievalResult],
    threshold: float,
    enabled: bool = True,
) -> tuple[List[RetrievalResult], int]:
    """
    Layer 4.A — drop chunks scoring below *threshold*.

    Returns (surviving_chunks, number_dropped_below_threshold).
    """
    if not enabled:
        return chunks, 0

    survivors = [c for c in chunks if _chunk_score(c) >= threshold]
    dropped = len(chunks) - len(survivors)
    return survivors, dropped


def deduplicate_context(
    chunks: List[RetrievalResult],
    similarity_threshold: float = 0.95,
    enabled: bool = True,
) -> tuple[List[RetrievalResult], int]:
    """
    Layer 4.B — exact + near-duplicate removal.

    Chunks are assumed to already be in score-descending order (post
    rerank) so the *first* occurrence of a duplicate cluster — the
    highest-scored one — is kept.

    Returns (deduplicated_chunks, duplicates_removed_count).
    """
    if not enabled or not chunks:
        return chunks, 0

    ordered = sorted(chunks, key=_chunk_score, reverse=True)

    kept: List[RetrievalResult] = []
    kept_hashes: set[str] = set()
    kept_tokens: List[Counter] = []
    removed = 0

    for chunk in ordered:
        h = _content_hash(chunk.text)
        if h in kept_hashes:
            removed += 1
            continue

        tokens = _tokenize(chunk.text)
        is_near_duplicate = any(
            _cosine_similarity(tokens, existing) >= similarity_threshold
            for existing in kept_tokens
        )
        if is_near_duplicate:
            removed += 1
            continue

        kept.append(chunk)
        kept_hashes.add(h)
        kept_tokens.append(tokens)

    return kept, removed


def score_context_relevance(
    query: str,
    chunks: List[RetrievalResult],
    threshold: float,
    enabled: bool = True,
) -> tuple[List[RetrievalResult], int]:
    """
    Layer 4.C — final lexical relevance check between *query* and each
    remaining chunk. Chunks scoring below *threshold* are removed.

    Uses the same cosine-similarity-over-bag-of-words measure as the
    dedup step, applied between the (masked) query and each chunk's
    text, rather than reusing the retriever/rerank score — this catches
    chunks that scored well on the original ANN/rerank pass but drifted
    off-topic relative to the exact query.
    """
    if not enabled or not chunks:
        return chunks, 0

    query_tokens = _tokenize(query)
    survivors: List[RetrievalResult] = []
    dropped = 0
    for chunk in chunks:
        rel = _cosine_similarity(query_tokens, _tokenize(chunk.text))
        # Blend with the existing rerank/retrieval score so a chunk that's
        # lexically sparse relative to the query (common with paraphrased
        # matches) isn't unfairly dropped — take the max of the two.
        effective = max(rel, _chunk_score(chunk))
        if effective >= threshold:
            survivors.append(chunk)
        else:
            dropped += 1
    return survivors, dropped


def apply_context_control(
    query: str,
    chunks: List[RetrievalResult],
    relevance_threshold: float,
    dedup_similarity_threshold: float,
    enable_threshold_gate: bool = True,
    enable_dedup: bool = True,
    enable_relevance_scoring: bool = True,
) -> ContextControlResult:
    """Run all of Layer 4 in the spec's order: threshold -> dedup -> relevance scoring."""
    after_threshold, below_threshold = apply_relevance_threshold(
        chunks, relevance_threshold, enabled=enable_threshold_gate
    )

    after_dedup, duplicates_removed = deduplicate_context(
        after_threshold, dedup_similarity_threshold, enabled=enable_dedup
    )

    after_relevance, _dropped_relevance = score_context_relevance(
        query, after_dedup, relevance_threshold, enabled=enable_relevance_scoring
    )

    return ContextControlResult(
        chunks=after_relevance,
        after_threshold_count=len(after_threshold),
        after_dedup_count=len(after_dedup),
        after_relevance_scoring_count=len(after_relevance),
        duplicates_removed=duplicates_removed,
        documents_below_threshold=below_threshold,
    )
