"""
Guardrail pipeline orchestrator.

Wires together all six layers for a single chat turn and produces the
structured ``GuardrailAuditRecord``. This is the single entry point
``backend.api.v1.chat`` calls into — it does not know about individual
layer modules directly.

Flow
────
    run_pre_retrieval_guardrails()
        Layer 1 (input) -> Layer 2 (query control)
        Returns either a block decision (with a fixed safe message) or
        the sanitized/masked query + routing decision for the caller to
        act on (skip retrieval for greeting/casual, run RAG otherwise).

    run_context_guardrails()
        Layer 4 (context control) — called by the caller after it has
        performed retrieval (Layer 3, using the existing hybrid
        retriever) and reranking.

    run_post_generation_guardrails()
        Layer 5 (generation safety) -> Layer 6 (output safety) — called
        by the caller after the LLM has produced an answer.

Each stage returns a small dataclass; the caller (chat.py) is
responsible for short-circuiting the HTTP response when ``blocked`` is
True and for calling ``finalize_audit()`` exactly once per request.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.guardrails import messages
from backend.guardrails.context_control import apply_context_control
from backend.guardrails.generation_safety import (
    check_groundedness,
    check_hallucination,
    validate_citations,
)
from backend.guardrails.input_guardrails import run_input_guardrails
from backend.guardrails.output_safety import apply_output_safety
from backend.guardrails.query_control import QueryIntent, route_query
from backend.models.guardrail_audit import GuardrailAuditRecord
from backend.models.guardrail_config import GuardrailConfig
from backend.retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditAccumulator:
    """Mutable scratch pad passed through the pipeline stages, finalised into a GuardrailAuditRecord."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    t_start: float = field(default_factory=time.perf_counter)

    intent_classified: Optional[str] = None
    input_guardrails_triggered: List[str] = field(default_factory=list)
    pii_types_detected_input: List[str] = field(default_factory=list)
    pii_types_detected_output: List[str] = field(default_factory=list)

    documents_retrieved: int = 0
    documents_after_reranking: int = 0
    documents_after_threshold: int = 0
    documents_after_dedup: int = 0
    relevance_threshold_used: float = 0.0

    response_blocked: bool = False
    block_reason: Optional[str] = None

    groundedness_score: Optional[float] = None
    hallucination_detected: bool = False

    def finalize(self) -> GuardrailAuditRecord:
        latency_ms = (time.perf_counter() - self.t_start) * 1_000
        return GuardrailAuditRecord(
            request_id=self.request_id,
            timestamp=_now_iso(),
            intent_classified=self.intent_classified,
            input_guardrails_triggered=self.input_guardrails_triggered,
            pii_types_detected_input=self.pii_types_detected_input,
            pii_types_detected_output=self.pii_types_detected_output,
            documents_retrieved=self.documents_retrieved,
            documents_after_reranking=self.documents_after_reranking,
            documents_after_threshold=self.documents_after_threshold,
            documents_after_dedup=self.documents_after_dedup,
            relevance_threshold_used=self.relevance_threshold_used,
            response_blocked=self.response_blocked,
            block_reason=self.block_reason,
            groundedness_score=self.groundedness_score,
            hallucination_detected=self.hallucination_detected,
            latency_ms=round(latency_ms, 1),
        )


@dataclass
class PreRetrievalResult:
    blocked: bool
    user_message: Optional[str] = None
    sanitized_query: str = ""
    intent: Optional[QueryIntent] = None
    route_to_rag: bool = False


def run_pre_retrieval_guardrails(
    raw_query: str,
    config: GuardrailConfig,
    audit: AuditAccumulator,
    document_count: int = 0,
) -> PreRetrievalResult:
    """Layer 1 (input) -> Layer 2 (query control)."""

    input_result = run_input_guardrails(raw_query, config, document_count=document_count)
    audit.input_guardrails_triggered.extend(input_result.triggered)
    audit.pii_types_detected_input.extend(input_result.pii_types_found)

    if input_result.blocked:
        audit.response_blocked = True
        audit.block_reason = input_result.block_reason
        return PreRetrievalResult(blocked=True, user_message=input_result.user_message)

    routing = route_query(
        input_result.sanitized_query,
        enabled=config.toggles.enable_intent_classification,
    )
    audit.intent_classified = routing.intent.value

    if routing.should_block:
        audit.response_blocked = True
        audit.block_reason = "out_of_scope"
        return PreRetrievalResult(
            blocked=True,
            user_message=routing.block_message or messages.OUT_OF_SCOPE,
            sanitized_query=input_result.sanitized_query,
            intent=routing.intent,
        )

    return PreRetrievalResult(
        blocked=False,
        sanitized_query=input_result.sanitized_query,
        intent=routing.intent,
        route_to_rag=routing.route_to_rag,
    )


@dataclass
class ContextGuardrailResult:
    blocked: bool
    chunks: List[RetrievalResult] = field(default_factory=list)
    user_message: Optional[str] = None


def run_context_guardrails(
    query: str,
    retrieved_chunks: List[RetrievalResult],
    reranked_chunks: List[RetrievalResult],
    config: GuardrailConfig,
    audit: AuditAccumulator,
) -> ContextGuardrailResult:
    """Layer 4 — context control, applied after retrieval + reranking (Layer 3)."""

    audit.documents_retrieved = len(retrieved_chunks)
    audit.documents_after_reranking = len(reranked_chunks)
    audit.relevance_threshold_used = config.thresholds.relevance_threshold

    result = apply_context_control(
        query=query,
        chunks=reranked_chunks,
        relevance_threshold=config.thresholds.relevance_threshold,
        dedup_similarity_threshold=config.thresholds.dedup_similarity_threshold,
        enable_threshold_gate=config.toggles.enable_relevance_threshold_gate,
        enable_dedup=config.toggles.enable_context_dedup,
        enable_relevance_scoring=config.toggles.enable_context_relevance_scoring,
    )

    audit.documents_after_threshold = result.after_threshold_count
    audit.documents_after_dedup = result.after_dedup_count

    if not result.chunks:
        audit.response_blocked = True
        audit.block_reason = "no_relevant_documents"
        return ContextGuardrailResult(
            blocked=True,
            chunks=[],
            user_message=messages.NO_RELEVANT_DOCUMENTS,
        )

    return ContextGuardrailResult(blocked=False, chunks=result.chunks)


@dataclass
class PostGenerationResult:
    blocked: bool
    safe_answer: str
    user_message: Optional[str] = None
    cleaned_citation_ns: Optional[List[int]] = None


def run_post_generation_guardrails(
    answer: str,
    context_chunks: List[RetrievalResult],
    citation_meta: List[Dict],
    config: GuardrailConfig,
    audit: AuditAccumulator,
) -> PostGenerationResult:
    """Layer 5 (generation safety) -> Layer 6 (output safety)."""

    working_answer = answer

    # ── Layer 5.A — Groundedness ────────────────────────────────────────────
    if config.toggles.enable_groundedness_check:
        grounded = check_groundedness(working_answer, context_chunks)
        audit.groundedness_score = round(grounded.score, 3)
    else:
        grounded = None

    # ── Layer 5.B — Hallucination detection ─────────────────────────────────
    if config.toggles.enable_hallucination_check:
        halluc = check_hallucination(
            working_answer, context_chunks, threshold=config.thresholds.hallucination_threshold
        )
        audit.hallucination_detected = halluc.detected
        if halluc.detected:
            audit.response_blocked = True
            audit.block_reason = "hallucination_detected"
            return PostGenerationResult(
                blocked=True,
                safe_answer="",
                user_message=messages.HALLUCINATION_DETECTED,
            )

    if (
        grounded is not None
        and grounded.score < config.thresholds.groundedness_threshold
        and not working_answer.rstrip().endswith(
            "may not be fully grounded in the knowledge base."
        )
    ):
        working_answer = (
            f"{working_answer}\n\n"
            "_Disclaimer: parts of this answer may not be fully grounded "
            "in the knowledge base._"
        )

    # ── Layer 5.C — Citation validation ─────────────────────────────────────
    cleaned_ns: Optional[List[int]] = None
    if config.toggles.enable_citation_validation and citation_meta:
        cit_result = validate_citations(working_answer, citation_meta)
        working_answer = cit_result.cleaned_answer
        cleaned_ns = cit_result.valid_citations

    # ── Layer 6 — Output safety (PII masking + toxicity) ────────────────────
    output_result = apply_output_safety(
        working_answer,
        enable_pii_masking=config.toggles.enable_pii_masking_output,
        enable_toxicity_check=config.toggles.enable_output_toxicity_check,
    )
    audit.pii_types_detected_output.extend(output_result.pii_types_found)

    if output_result.blocked:
        audit.response_blocked = True
        audit.block_reason = output_result.block_reason
        return PostGenerationResult(
            blocked=True,
            safe_answer="",
            user_message=output_result.user_message,
        )

    return PostGenerationResult(
        blocked=False,
        safe_answer=output_result.safe_text,
        cleaned_citation_ns=cleaned_ns,
    )
