"""
Language detection utility.

Primary:  langdetect (probabilistic, fast, offline)
Fallback: ASCII-ratio heuristic (no extra deps)

Returns ISO 639-1 language codes ("en", "fr", "de", …).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect as _ld_detect
    from langdetect import DetectorFactory, LangDetectException
    # Seed for reproducible results
    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except ImportError:
    logger.warning(
        "langdetect not installed — language detection will use ASCII heuristic. "
        "Install with: pip install langdetect"
    )

# Minimum text length for reliable detection
_MIN_CHARS = 20
_SAMPLE_CHARS = 500   # only inspect first N chars for speed


def detect_language(text: str) -> str | None:
    """
    Detect the primary language of `text`.

    Args:
        text: Arbitrary text string.

    Returns:
        ISO 639-1 code (e.g. "en", "fr") or None if detection fails
        or the text is too short.
    """
    if not text or len(text.strip()) < _MIN_CHARS:
        return None

    sample = text.strip()[:_SAMPLE_CHARS]

    if _LANGDETECT_AVAILABLE:
        try:
            return _ld_detect(sample)
        except Exception as exc:
            logger.debug("langdetect failed: %s — using fallback", exc)

    return _ascii_heuristic(sample)


def is_english(text: str, threshold: float = 0.85) -> bool:
    """
    Return True if `text` is likely English.

    Uses langdetect when available; otherwise falls back to the
    ASCII-ratio heuristic (English text is overwhelmingly ASCII).

    Args:
        text:      Input text.
        threshold: ASCII-char ratio required for the fallback heuristic.
    """
    lang = detect_language(text)
    if lang is not None:
        return lang == "en"
    # When detection is inconclusive, apply the ratio heuristic directly
    return _ascii_ratio(text) >= threshold


def detect_language_batch(texts: list[str]) -> list[str | None]:
    """Detect language for each text in a list."""
    return [detect_language(t) for t in texts]


# ─── Fallback heuristic ───────────────────────────────────────────────────────

def _ascii_heuristic(text: str) -> str | None:
    """
    Rough language guess based on character set:
      - High ASCII ratio + no CJK → "en" (or other Latin language)
      - CJK block presence         → "zh" / "ja" / "ko"
      - Cyrillic                   → "ru"
      - Arabic                     → "ar"
    Returns None if inconclusive.
    """
    ratio = _ascii_ratio(text)

    if re.search(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]", text):
        return "zh"   # broad CJK guess
    if re.search(r"[\u0400-\u04FF]", text):
        return "ru"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if ratio >= 0.85:
        return "en"
    return None


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text)
