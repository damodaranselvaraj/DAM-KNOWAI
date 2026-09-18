"""
PII detection and masking — shared by Layer 1 (input) and Layer 6 (output).

Regex-based detector for the PII types called out in the spec:
    PAN, Aadhaar, Credit Card, Bank Account, Email, Phone, Physical
    Address, Employee ID, Customer ID.

Masking strategy: replace each detected span with a typed placeholder,
e.g. ``john@example.com`` -> ``[EMAIL_REDACTED]``. The function never
returns the raw matched value to the caller — only the masked text and
the list of PII *types* found (for audit metadata), matching the
"never store raw PII values" requirement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Pattern, Tuple

# ─── Pattern registry ──────────────────────────────────────────────────────
# Order matters: more specific patterns (credit card, Aadhaar) are checked
# before the generic bank-account pattern so a 16-digit card number isn't
# double-masked as a bank account.

_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

_AADHAAR_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")

_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ -]?){13,16}\b"
)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
)

# Employee / Customer IDs — configurable org-specific patterns. Defaults
# cover common conventions (e.g. EMP-00123, CUST-00123). Extend via
# register_custom_pattern() if the organisation uses a different scheme.
_EMPLOYEE_ID_RE = re.compile(r"\bEMP[-_]?\d{3,10}\b", re.IGNORECASE)
_CUSTOMER_ID_RE = re.compile(r"\bCUST[-_]?\d{3,10}\b", re.IGNORECASE)

# Physical address heuristic: street/city/pincode style combination.
# Deliberately conservative (line must contain a pincode-like 5-6 digit
# group plus a street-ish keyword) to avoid false positives on ordinary
# numeric text.
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.,'\s]{3,40}"
    r"(?:street|st\.|road|rd\.|avenue|ave\.|lane|ln\.|drive|dr\.|block|sector)"
    r"[A-Za-z0-9.,'\s]{0,40}\b\d{5,6}\b",
    re.IGNORECASE,
)

# Bank account numbers — generic 9-18 digit sequences NOT already matched
# by the more specific patterns above. Applied last.
_BANK_ACCOUNT_RE = re.compile(r"\b\d{9,18}\b")


@dataclass
class PiiEntry:
    pii_type: str
    pattern: Pattern[str]


# Ordered list — first match wins for a given span.
_PII_PATTERNS: List[PiiEntry] = [
    PiiEntry("PAN", _PAN_RE),
    PiiEntry("EMAIL", _EMAIL_RE),
    PiiEntry("EMPLOYEE_ID", _EMPLOYEE_ID_RE),
    PiiEntry("CUSTOMER_ID", _CUSTOMER_ID_RE),
    PiiEntry("CREDIT_CARD", _CREDIT_CARD_RE),
    PiiEntry("AADHAAR", _AADHAAR_RE),
    PiiEntry("ADDRESS", _ADDRESS_RE),
    PiiEntry("PHONE", _PHONE_RE),
    PiiEntry("BANK_ACCOUNT", _BANK_ACCOUNT_RE),
]


@dataclass
class PiiScanResult:
    masked_text: str
    pii_types_found: List[str] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.pii_types_found)


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum — used to reduce false positives on credit-card-shaped numbers."""
    total = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def scan_and_mask(text: str) -> PiiScanResult:
    """
    Detect and mask every supported PII type in *text*.

    Returns the masked text plus the list of PII type labels found (never
    the raw matched values). Matching is applied left-to-right; each
    character span is masked at most once (first matching pattern wins),
    so overlapping patterns (e.g. a 16-digit card vs. a bank-account
    catch-all) don't double-mask the same digits.
    """
    if not text:
        return PiiScanResult(masked_text=text, pii_types_found=[])

    # Collect all raw candidate matches first, then resolve overlaps by
    # preferring earlier-registered (more specific) patterns and, on tie,
    # the longer match.
    candidates: List[Tuple[int, int, str, str]] = []  # (start, end, type, matched_text)

    for entry in _PII_PATTERNS:
        for m in entry.pattern.finditer(text):
            matched = m.group(0)
            digits_only = re.sub(r"\D", "", matched)

            if entry.pii_type == "CREDIT_CARD":
                if len(digits_only) not in (13, 14, 15, 16):
                    continue
                if not _luhn_valid(digits_only):
                    continue
            if entry.pii_type == "PHONE":
                if len(digits_only) < 7 or len(digits_only) > 13:
                    continue
            if entry.pii_type == "BANK_ACCOUNT":
                if len(digits_only) < 9 or len(digits_only) > 18:
                    continue

            candidates.append((m.start(), m.end(), entry.pii_type, matched))

    if not candidates:
        return PiiScanResult(masked_text=text, pii_types_found=[])

    # Sort by start, then by longer span first, so wider/more-specific
    # matches win when spans overlap.
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))

    selected: List[Tuple[int, int, str]] = []
    last_end = -1
    for start, end, pii_type, _matched in candidates:
        if start >= last_end:
            selected.append((start, end, pii_type))
            last_end = end

    # Build masked text right-to-left so earlier offsets stay valid.
    masked_text = text
    types_found: List[str] = []
    for start, end, pii_type in reversed(selected):
        placeholder = f"[{pii_type}_REDACTED]"
        masked_text = masked_text[:start] + placeholder + masked_text[end:]
        types_found.append(pii_type)

    # Preserve left-to-right discovery order, de-duplicated.
    types_found = list(dict.fromkeys(reversed(types_found)))

    return PiiScanResult(masked_text=masked_text, pii_types_found=types_found)
