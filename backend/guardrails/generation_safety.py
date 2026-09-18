"""
Layer 5 — Generation Safety.

Runs after the LLM has produced an answer, before it is shown to the
user (Layer 6 output checks run after this):

    A. Groundedness Check     — every factual claim (sentence) in the
       answer should be lexically supported by the retrieved context.
    B. Hallucination Detection — cross-references key entities (numbers,
       capitalised proper nouns, dates) mentioned in the answer against
       the retrieved context; unsupported entities count as
       hallucination signals.
    C. Citation Validation    — every ``[Source N]`` reference in the
       answer must point at a citation that was actually included in
       the context sent to the LLM.

These are heuristic, lexical-overlap based checks (no external NLI
model dependency) — sufficient to catch obviously fabricated or
uncited content while adding minimal latency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from backend.retrieval.dense_retriever import RetrievalResult

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*|\d[\d,.]*%?)\b"
)
_CITATION_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to",
    "and", "or", "for", "with", "this", "that", "it", "as", "by", "at",
    "be", "has", "have", "had", "not", "but", "from", "into", "your",
    "you", "i", "we", "they", "he", "she",
}


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS}


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]


@dataclass
class GroundednessResult:
    score: float
    ungrounded_sentences: List[str] = field(default_factory=list)


def check_groundedness(answer: str, context_chunks: List[RetrievalResult]) -> GroundednessResult:
    """
    Estimate what fraction of the answer's sentences have meaningful
    lexical overlap with the retrieved context.

    Returns a 0.0-1.0 groundedness score (1.0 = every sentence is
    supported) plus the list of sentences that appear unsupported.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return GroundednessResult(score=1.0)

    context_tokens: set[str] = set()
    for chunk in context_chunks:
        context_tokens |= _tokenize(chunk.text)

    if not context_tokens:
        # No context at all — nothing to ground against.
        return GroundednessResult(score=0.0, ungrounded_sentences=sentences)

    ungrounded: List[str] = []
    grounded_count = 0
    for sentence in sentences:
        # Skip the "Sources" footer / very short connective sentences.
        if sentence.lower().startswith("sources") or len(sentence.split()) < 4:
            grounded_count += 1
            continue
        sent_tokens = _tokenize(sentence)
        if not sent_tokens:
            grounded_count += 1
            continue
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        if overlap >= 0.35:
            grounded_count += 1
        else:
            ungrounded.append(sentence)

    score = grounded_count / len(sentences)
    return GroundednessResult(score=score, ungrounded_sentences=ungrounded)


@dataclass
class HallucinationResult:
    detected: bool
    score: float  # fraction of extracted entities NOT found in context
    unsupported_entities: List[str] = field(default_factory=list)


def check_hallucination(
    answer: str,
    context_chunks: List[RetrievalResult],
    threshold: float = 0.5,
) -> HallucinationResult:
    """
    Cross-reference proper nouns / numbers mentioned in the answer
    against the retrieved context. A high proportion of unsupported
    entities indicates fabricated facts, names, or figures.
    """
    entities = list(dict.fromkeys(_ENTITY_RE.findall(answer or "")))
    # Filter out very common short words the regex may catch (e.g. "I").
    entities = [e for e in entities if len(e) > 2]

    if not entities:
        return HallucinationResult(detected=False, score=0.0)

    context_text = " ".join(chunk.text or "" for chunk in context_chunks).lower()

    unsupported = [e for e in entities if e.lower() not in context_text]
    score = len(unsupported) / len(entities)

    return HallucinationResult(
        detected=score >= threshold,
        score=score,
        unsupported_entities=unsupported,
    )


@dataclass
class CitationValidationResult:
    valid_citations: List[int] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    cleaned_answer: str = ""


def validate_citations(answer: str, citation_meta: List[Dict]) -> CitationValidationResult:
    """
    Validate every ``[Source N]`` reference in *answer* against the
    citation metadata actually sent to the LLM (PromptBuilder's
    ``citation_meta``). Invalid references (pointing at a source number
    that doesn't exist) are stripped from the answer text.
    """
    valid_ns = {cm["n"] for cm in citation_meta}
    referenced_ns = {int(m.group(1)) for m in _CITATION_RE.finditer(answer or "")}

    valid = sorted(n for n in referenced_ns if n in valid_ns)
    invalid = sorted(n for n in referenced_ns if n not in valid_ns)

    cleaned_answer = answer
    if invalid:
        def _strip_invalid(match: re.Match) -> str:
            n = int(match.group(1))
            return match.group(0) if n in valid_ns else ""
        cleaned_answer = _CITATION_RE.sub(_strip_invalid, answer or "")

    return CitationValidationResult(
        valid_citations=valid,
        invalid_citations=invalid,
        cleaned_answer=cleaned_answer,
    )
