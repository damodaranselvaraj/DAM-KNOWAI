"""
Prompt-injection and jailbreak detection.

Two independent, pattern-based detectors:

* ``check_prompt_injection`` — attempts to override system instructions,
  manipulate the assistant's behaviour, or hijack the pipeline via the
  query itself (e.g. "ignore all previous instructions", "you are now
  DAN", "your new instructions are...").

* ``check_jailbreak`` — attempts to bypass safety guidelines through
  roleplay/fictional framing, hypothetical scenarios, or encoded
  instructions (e.g. "pretend you have no restrictions", "in a fictional
  world where rules don't exist", "translate this to base64 then
  execute").

Both are regex/keyword heuristics — fast, dependency-free, and easy to
extend with more patterns as new attack strings are observed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Pattern


_INJECTION_PATTERNS: List[Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(your\s+)?(system\s+prompt|instructions?|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(dan|an?\s+unrestricted)", re.IGNORECASE),
    re.compile(r"act\s+as\s+an?\s+unrestricted\s+ai", re.IGNORECASE),
    re.compile(r"your\s+new\s+instructions?\s+(are|is)", re.IGNORECASE),
    re.compile(r"new\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)", re.IGNORECASE),
    re.compile(r"repeat\s+(the\s+)?(system\s+prompt|instructions)\s+(verbatim|exactly)", re.IGNORECASE),
    re.compile(r"override\s+(your\s+)?(guidelines|instructions|rules)", re.IGNORECASE),
]

_JAILBREAK_PATTERNS: List[Pattern[str]] = [
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+an?\s+ai\s+(with\s+no|without)\s+restrictions", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+have\s+no\s+(rules|restrictions|filters)", re.IGNORECASE),
    re.compile(r"in\s+a\s+fictional\s+world\s+where\s+(rules|laws)\s+don'?t\s+exist", re.IGNORECASE),
    re.compile(r"hypothetically,?\s+if\s+you\s+had\s+no\s+restrictions", re.IGNORECASE),
    re.compile(r"role\s*play\s+as\s+an?\s+ai\s+(with\s+no|without)\s+(rules|restrictions)", re.IGNORECASE),
    re.compile(r"translate\s+this\s+to\s+base64.{0,40}(then|and)\s+(execute|run|decode)", re.IGNORECASE),
    re.compile(r"decode\s+(this\s+)?base64.{0,40}(then|and)\s+(execute|run|follow)", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),  # "DAN" jailbreak family
    re.compile(r"developer\s+mode\s+enabled", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]


@dataclass
class InjectionResult:
    detected: bool
    matched_patterns: List[str] = field(default_factory=list)


def check_prompt_injection(text: str) -> InjectionResult:
    """Detect attempts to override system instructions / hijack the pipeline."""
    if not text:
        return InjectionResult(detected=False)
    matches = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
    return InjectionResult(detected=bool(matches), matched_patterns=matches)


def check_jailbreak(text: str) -> InjectionResult:
    """Detect roleplay / fictional-framing / encoded jailbreak attempts."""
    if not text:
        return InjectionResult(detected=False)
    matches = [p.pattern for p in _JAILBREAK_PATTERNS if p.search(text)]
    return InjectionResult(detected=bool(matches), matched_patterns=matches)
