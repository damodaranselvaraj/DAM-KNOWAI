"""
Lightweight lexicon-based toxicity / abuse detector.

This is a heuristic, dependency-free classifier: it flags hateful,
abusive, profane, or violent language via keyword/pattern matching. It
intentionally trades recall for zero added latency and no external
service dependency — swap in a hosted moderation model later by
replacing ``is_toxic()`` without touching call sites.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Deliberately coarse-grained categories. Real deployments should back
# this with a proper moderation model/API; this keeps the pipeline
# functional and testable without external dependencies.
_PROFANITY_TERMS = [
    "fuck", "shit", "asshole", "bastard", "bitch", "cunt", "dick",
    "piss off", "motherfucker",
]

_HATE_TERMS = [
    "kill all", "subhuman", "go back to your country",
    "racial slur", "ethnic cleansing",
]

_VIOLENCE_TERMS = [
    "kill you", "i will kill", "bomb the", "shoot up", "murder you",
    "beat you up", "burn it down",
]

_ABUSE_TERMS = [
    "you are worthless", "you are stupid", "shut up idiot", "you idiot",
]

_ALL_TERMS = _PROFANITY_TERMS + _HATE_TERMS + _VIOLENCE_TERMS + _ABUSE_TERMS

_TERM_PATTERNS = [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in _ALL_TERMS]


@dataclass
class ToxicityResult:
    is_toxic: bool
    matched_categories: List[str] = field(default_factory=list)


def _categorize(term: str) -> str:
    if term in _PROFANITY_TERMS:
        return "profanity"
    if term in _HATE_TERMS:
        return "hate_speech"
    if term in _VIOLENCE_TERMS:
        return "violence"
    return "abuse"


def check_toxicity(text: str) -> ToxicityResult:
    """Return whether *text* contains toxic/abusive/violent language."""
    if not text:
        return ToxicityResult(is_toxic=False)

    categories: List[str] = []
    for term, pattern in zip(_ALL_TERMS, _TERM_PATTERNS):
        if pattern.search(text):
            cat = _categorize(term)
            if cat not in categories:
                categories.append(cat)

    return ToxicityResult(is_toxic=bool(categories), matched_categories=categories)
