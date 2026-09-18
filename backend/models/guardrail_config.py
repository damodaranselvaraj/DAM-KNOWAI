"""
Pydantic models for the Guardrails configuration.

Single source of truth for every threshold / toggle exposed on the
Guardrails settings page (frontend/pages/06_guardrails.py) and consumed
by the guardrail pipeline (backend/guardrails/*).

All defaults mirror the specification: relevance_threshold=0.70,
max_characters=2000, max_tokens=500, max_documents=10,
max_query_length=300 words, top_n_retrieval=20,
dedup_similarity_threshold=0.95, groundedness_threshold=0.75, and every
enable_* toggle defaults to True.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class GuardrailToggles(BaseModel):
    """Enable/disable switches for each guardrail layer/check."""

    # Layer 1 — Input Guardrails
    enable_input_length_check: bool = True
    enable_toxicity_check: bool = True
    enable_prompt_injection_check: bool = True
    enable_jailbreak_check: bool = True
    enable_pii_masking_input: bool = True

    # Layer 2 — Query Control
    enable_intent_classification: bool = True
    enable_out_of_scope_check: bool = True

    # Layer 3 — Retrieval Safety
    enable_metadata_filtering: bool = True
    enable_hybrid_retrieval: bool = True

    # Layer 4 — Context Control
    enable_relevance_threshold_gate: bool = True
    enable_context_dedup: bool = True
    enable_context_relevance_scoring: bool = True

    # Layer 5 — Generation Safety
    enable_groundedness_check: bool = True
    enable_hallucination_check: bool = True
    enable_citation_validation: bool = True

    # Layer 6 — Output Safety
    enable_pii_masking_output: bool = True
    enable_output_toxicity_check: bool = True


class GuardrailThresholds(BaseModel):
    """Numeric thresholds / limits configurable from the settings page."""

    relevance_threshold: float = Field(0.70, ge=0.0, le=1.0)
    max_characters: int = Field(2000, ge=1)
    max_tokens: int = Field(500, ge=1)
    max_documents: int = Field(10, ge=1)
    max_query_length: int = Field(300, ge=1, description="Max words in a query.")
    top_n_retrieval: int = Field(20, ge=1, le=200)
    dedup_similarity_threshold: float = Field(0.95, ge=0.0, le=1.0)
    groundedness_threshold: float = Field(0.75, ge=0.0, le=1.0)
    hallucination_threshold: float = Field(0.5, ge=0.0, le=1.0)


class GuardrailConfig(BaseModel):
    """Full persisted guardrail configuration: toggles + thresholds."""

    toggles: GuardrailToggles = Field(default_factory=GuardrailToggles)
    thresholds: GuardrailThresholds = Field(default_factory=GuardrailThresholds)
