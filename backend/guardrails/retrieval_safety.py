"""
Layer 3 — Retrieval Safety.

The platform's hybrid retriever (backend.retrieval.hybrid_retriever) and
Pinecone filter validation (backend.retrieval.query_validator) already
implement metadata filtering and dense+sparse hybrid search. This module
adds the guardrail-level knobs the spec calls for on top of that
existing infrastructure:

    * ``resolve_top_n`` — the configurable Top-N retrieved before rerank
      (default 20), sourced from GuardrailThresholds instead of a
      hardcoded constant.
    * ``build_metadata_filter`` — merges caller-supplied filters with any
      access-control metadata (department, doc type, access level, date
      range) so retrieval never returns documents outside the caller's
      permission scope. Delegates the actual allow-list enforcement to
      the existing ``validate_filters`` so nothing new is introduced that
      could bypass it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from backend.models.guardrail_config import GuardrailConfig
from backend.retrieval.query_validator import validate_filters


def resolve_top_n(config: GuardrailConfig) -> int:
    """Return the configured Top-N retrieval count (pre-rerank)."""
    return config.thresholds.top_n_retrieval


def build_metadata_filter(
    base_filters: Optional[Dict[str, Any]],
    *,
    department: Optional[str] = None,
    document_type: Optional[str] = None,
    access_level: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    enabled: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Merge ACL-style metadata constraints into the caller-supplied filter
    dict, then re-validate the merged result against the existing
    Pinecone filter allow-list (backend.retrieval.query_validator).

    When ``enabled`` is False, only the caller's own filters are
    validated and returned unchanged (metadata filtering guardrail
    toggled off in settings).
    """
    clauses: list[Dict[str, Any]] = []
    if base_filters:
        clauses.append(base_filters)

    if enabled:
        if department:
            clauses.append({"department": {"$eq": department}})
        if document_type:
            clauses.append({"document_type": {"$eq": document_type}})
        if access_level:
            clauses.append({"access_level": {"$eq": access_level}})
        if date_from or date_to:
            date_clause: Dict[str, Any] = {}
            if date_from:
                date_clause["$gte"] = date_from
            if date_to:
                date_clause["$lte"] = date_to
            clauses.append({"date": date_clause})

    if not clauses:
        return None
    merged = clauses[0] if len(clauses) == 1 else {"$and": clauses}
    return validate_filters(merged)
