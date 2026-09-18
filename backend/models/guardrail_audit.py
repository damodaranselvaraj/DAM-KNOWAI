"""
Structured audit-log model for the guardrail pipeline.

Every request that passes through the guardrail pipeline produces one
``GuardrailAuditRecord``. Per spec, these records must NEVER contain raw
PII, the user's query text, or the LLM-generated response text — only
counts, flags, and enum/category labels.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GuardrailAuditRecord(BaseModel):
    request_id: str
    timestamp: str

    intent_classified: Optional[str] = None

    input_guardrails_triggered: List[str] = Field(default_factory=list)
    pii_types_detected_input: List[str] = Field(default_factory=list)
    pii_types_detected_output: List[str] = Field(default_factory=list)

    documents_retrieved: int = 0
    documents_after_reranking: int = 0
    documents_after_threshold: int = 0
    documents_after_dedup: int = 0
    relevance_threshold_used: float = 0.0

    response_blocked: bool = False
    block_reason: Optional[str] = None

    groundedness_score: Optional[float] = None
    hallucination_detected: bool = False

    latency_ms: float = 0.0
