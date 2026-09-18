"""
Layer 6 — Output Safety.

Runs on the final LLM-generated answer, right before it is returned to
the user:

    A. PII Detection + Masking (Output) — reuses backend.guardrails.pii
       so the same typed-placeholder masking strategy is applied
       consistently to both input and output.
    B. Output Toxicity / Abuse Detection — reuses
       backend.guardrails.toxicity to scan the generated text; a hit
       blocks the response entirely (never shown to the user) rather
       than masking it, since a toxic *generated* response has no safe
       partial form to return.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from backend.guardrails import messages
from backend.guardrails.pii import scan_and_mask
from backend.guardrails.toxicity import check_toxicity


@dataclass
class OutputSafetyResult:
    blocked: bool
    safe_text: str
    pii_types_found: List[str] = field(default_factory=list)
    block_reason: Optional[str] = None
    user_message: Optional[str] = None


def apply_output_safety(
    answer: str,
    enable_pii_masking: bool = True,
    enable_toxicity_check: bool = True,
) -> OutputSafetyResult:
    """Run Layer 6 against the final generated *answer*."""

    if enable_toxicity_check:
        tox = check_toxicity(answer)
        if tox.is_toxic:
            return OutputSafetyResult(
                blocked=True,
                safe_text="",
                block_reason="output_toxicity_detected",
                user_message=messages.OUTPUT_BLOCKED_SAFETY,
            )

    safe_text = answer
    pii_types_found: List[str] = []
    if enable_pii_masking:
        result = scan_and_mask(answer)
        safe_text = result.masked_text
        pii_types_found = result.pii_types_found

    return OutputSafetyResult(
        blocked=False,
        safe_text=safe_text,
        pii_types_found=pii_types_found,
    )
