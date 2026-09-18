"""
Layer 2 — Query Control.

Classifies the (already input-guardrailed, PII-masked) query into a
``QueryIntent`` and decides the routing action, before any embedding or
retrieval happens.

The classifier is a lightweight, dependency-free heuristic (keyword /
pattern based) so it adds effectively no latency — swap in an LLM-based
or small-model classifier later behind the same ``classify_intent()``
signature without touching call sites.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.guardrails import messages


class QueryIntent(str, Enum):
    GREETING = "greeting"
    CASUAL = "casual"
    KNOWLEDGE = "knowledge"
    DOCUMENT = "document"
    OUT_OF_SCOPE = "out_of_scope"


_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings|yo)\b[\s!.,]*$",
    re.IGNORECASE,
)

_CASUAL_PATTERNS = [
    re.compile(r"\bhow\s+are\s+you\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s\s+up\b", re.IGNORECASE),
    re.compile(r"\bthank\s*(you|s)\b", re.IGNORECASE),
    re.compile(r"\bwho\s+are\s+you\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+can\s+you\s+do\b", re.IGNORECASE),
    re.compile(r"\bgood\s*bye\b|\bbye\b", re.IGNORECASE),
    re.compile(r"\bhow'?s\s+it\s+going\b", re.IGNORECASE),
]

# Signals the query is about the enterprise knowledge base / documents.
_KNOWLEDGE_PATTERNS = [
    re.compile(r"\bpolicy|policies\b", re.IGNORECASE),
    re.compile(r"\bcompliance\b", re.IGNORECASE),
    re.compile(r"\bdocument|report|guideline|procedure|sop\b", re.IGNORECASE),
    re.compile(r"\baccording to\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(does|is|are)\b.*\b(policy|process|procedure|rule)\b", re.IGNORECASE),
    re.compile(r"\bsection\s+\d", re.IGNORECASE),
    re.compile(r"\brisk\s+assessment\b", re.IGNORECASE),
]

_DOCUMENT_PATTERNS = [
    re.compile(r"\bin\s+the\s+(document|file|report|pdf)\b", re.IGNORECASE),
    re.compile(r"\baccording\s+to\s+the\s+(document|file|report)\b", re.IGNORECASE),
    re.compile(r"\bsummarize\b.*\b(document|file|report)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(document|file|pdf)\b", re.IGNORECASE),
]

# Clearly out-of-scope: general trivia, creative writing, personal advice
# unrelated to enterprise/work content.
_OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\bwrite\s+(me\s+)?a\s+(poem|story|song|joke)\b", re.IGNORECASE),
    re.compile(r"\bwho\s+(won|is\s+the\s+president|is\s+the\s+prime\s+minister)\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s\s+the\s+weather\b", re.IGNORECASE),
    re.compile(r"\bgive\s+me\s+(dating|relationship|dietary|medical|legal)\s+advice\b", re.IGNORECASE),
    re.compile(r"\bhow\s+do\s+i\s+(lose\s+weight|cook|bake)\b", re.IGNORECASE),
    re.compile(r"\brecommend\s+a\s+(movie|book|restaurant)\b", re.IGNORECASE),
    re.compile(r"\btrivia\b", re.IGNORECASE),
]


@dataclass
class IntentClassification:
    intent: QueryIntent
    should_block: bool
    block_message: Optional[str] = None
    route_to_rag: bool = False


def classify_intent(query: str) -> QueryIntent:
    """Classify *query* into one of the QueryIntent categories."""
    if not query or not query.strip():
        return QueryIntent.OUT_OF_SCOPE

    if _GREETING_RE.match(query.strip()):
        return QueryIntent.GREETING

    for pattern in _CASUAL_PATTERNS:
        if pattern.search(query):
            return QueryIntent.CASUAL

    for pattern in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(query):
            return QueryIntent.OUT_OF_SCOPE

    for pattern in _DOCUMENT_PATTERNS:
        if pattern.search(query):
            return QueryIntent.DOCUMENT

    for pattern in _KNOWLEDGE_PATTERNS:
        if pattern.search(query):
            return QueryIntent.KNOWLEDGE

    # Default: treat substantive questions as knowledge-seeking so the RAG
    # pipeline gets a chance to answer from the indexed corpus rather than
    # over-aggressively rejecting ambiguous queries as out-of-scope.
    if len(query.split()) >= 3:
        return QueryIntent.KNOWLEDGE

    return QueryIntent.CASUAL


def route_query(query: str, enabled: bool = True) -> IntentClassification:
    """
    Classify *query* and decide the routing action.

    When ``enabled`` is False (intent classification toggled off in
    settings) every query routes straight to the RAG pipeline as
    ``KNOWLEDGE``, preserving prior behaviour.
    """
    if not enabled:
        return IntentClassification(intent=QueryIntent.KNOWLEDGE, should_block=False, route_to_rag=True)

    intent = classify_intent(query)

    if intent == QueryIntent.OUT_OF_SCOPE:
        return IntentClassification(
            intent=intent,
            should_block=True,
            block_message=messages.OUT_OF_SCOPE,
            route_to_rag=False,
        )

    if intent in (QueryIntent.GREETING, QueryIntent.CASUAL):
        return IntentClassification(intent=intent, should_block=False, route_to_rag=False)

    # KNOWLEDGE / DOCUMENT
    return IntentClassification(intent=intent, should_block=False, route_to_rag=True)
