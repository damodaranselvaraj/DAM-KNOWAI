"""
Layer 1 — Input Guardrails.

Runs, in order, against the raw user query:

    A. Input length / token validation
    B. Toxicity / abuse detection
    C. Prompt injection detection
    D. Jailbreak detection
    E. PII detection + masking

Each check is individually toggleable via ``GuardrailToggles`` so the
Guardrails settings page can turn any of them off. The length check
never silently truncates — it rejects and returns a specific reason.

The result of running this layer is a ``InputGuardrailResult``: either
``blocked=True`` with a user-facing message and the guardrail(s) that
fired, or ``blocked=False`` with the (possibly PII-masked) sanitized
query that downstream stages (intent classification, retrieval, LLM,
logging) must use instead of the original raw text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from backend.guardrails import messages
from backend.guardrails.pii import scan_and_mask
from backend.guardrails.prompt_injection import check_jailbreak, check_prompt_injection
from backend.guardrails.toxicity import check_toxicity
from backend.models.guardrail_config import GuardrailConfig

# Rough chars-per-token heuristic, consistent with prompt_builder.py's
# _approx_tokens so token-budget checks stay in the same ballpark without
# adding a tiktoken dependency.
_CHARS_PER_TOKEN = 3.8


@dataclass
class InputGuardrailResult:
    blocked: bool
    block_reason: Optional[str] = None
    user_message: Optional[str] = None
    triggered: List[str] = field(default_factory=list)
    sanitized_query: str = ""
    pii_types_found: List[str] = field(default_factory=list)


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def run_input_guardrails(
    raw_query: str,
    config: GuardrailConfig,
    document_count: int = 0,
) -> InputGuardrailResult:
    """
    Execute Layer 1 against *raw_query*.

    Parameters
    ----------
    raw_query:
        The user's query text (already Unicode/whitespace normalised by
        ``sanitize_query`` upstream, but NOT yet PII-masked).
    config:
        Active GuardrailConfig (toggles + thresholds).
    document_count:
        Number of documents the user has attached/referenced in this
        request, checked against ``max_documents``.
    """
    triggered: List[str] = []
    toggles = config.toggles
    thresholds = config.thresholds

    # ── A. Length / token / document-count validation ──────────────────────
    if toggles.enable_input_length_check:
        if len(raw_query) > thresholds.max_characters:
            return InputGuardrailResult(
                blocked=True,
                block_reason="length_exceeded_characters",
                user_message=messages.INPUT_TOO_LONG,
                triggered=["length_exceeded"],
            )

        token_estimate = _approx_tokens(raw_query)
        if token_estimate > thresholds.max_tokens:
            return InputGuardrailResult(
                blocked=True,
                block_reason="length_exceeded_tokens",
                user_message=messages.INPUT_TOO_LONG,
                triggered=["length_exceeded"],
            )

        word_count = len(raw_query.split())
        if word_count > thresholds.max_query_length:
            return InputGuardrailResult(
                blocked=True,
                block_reason="length_exceeded_words",
                user_message=messages.INPUT_TOO_LONG,
                triggered=["length_exceeded"],
            )

        if document_count > thresholds.max_documents:
            return InputGuardrailResult(
                blocked=True,
                block_reason="max_documents_exceeded",
                user_message=messages.INPUT_TOO_LONG,
                triggered=["max_documents_exceeded"],
            )

    # ── B. Toxicity / abuse detection ───────────────────────────────────────
    if toggles.enable_toxicity_check:
        tox = check_toxicity(raw_query)
        if tox.is_toxic:
            triggered.append("toxicity_detected")
            return InputGuardrailResult(
                blocked=True,
                block_reason="toxicity_detected",
                user_message=messages.TOXIC_INPUT,
                triggered=triggered,
            )

    # ── C. Prompt injection detection ───────────────────────────────────────
    if toggles.enable_prompt_injection_check:
        inj = check_prompt_injection(raw_query)
        if inj.detected:
            triggered.append("prompt_injection_detected")
            return InputGuardrailResult(
                blocked=True,
                block_reason="prompt_injection_detected",
                user_message=messages.PROMPT_INJECTION_DETECTED,
                triggered=triggered,
            )

    # ── D. Jailbreak detection ──────────────────────────────────────────────
    if toggles.enable_jailbreak_check:
        jb = check_jailbreak(raw_query)
        if jb.detected:
            triggered.append("jailbreak_detected")
            return InputGuardrailResult(
                blocked=True,
                block_reason="jailbreak_detected",
                user_message=messages.JAILBREAK_DETECTED,
                triggered=triggered,
            )

    # ── E. PII detection + masking ──────────────────────────────────────────
    sanitized_query = raw_query
    pii_types_found: List[str] = []
    if toggles.enable_pii_masking_input:
        pii_result = scan_and_mask(raw_query)
        sanitized_query = pii_result.masked_text
        pii_types_found = pii_result.pii_types_found
        if pii_result.has_pii:
            triggered.append("pii_detected")

    return InputGuardrailResult(
        blocked=False,
        triggered=triggered,
        sanitized_query=sanitized_query,
        pii_types_found=pii_types_found,
    )
