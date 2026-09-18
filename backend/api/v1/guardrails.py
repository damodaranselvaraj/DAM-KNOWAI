"""
Guardrails configuration + audit endpoints — /api/v1/guardrails

Mirrors the in-memory config pattern used by /pipeline/config
(backend.api.v1.pipeline): a module-level singleton holds the active
GuardrailConfig, defaulting to the spec's defaults until explicitly saved.

/guardrails/audit exposes the most recent structured audit records
(no raw PII / query / response content — see GuardrailAuditRecord) so
the platform's observability requirements are satisfiable from the API
without a dedicated log store.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query

from backend.models.guardrail_config import GuardrailConfig
from backend.models.guardrail_audit import GuardrailAuditRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/guardrails", tags=["Guardrails"])

# ── In-memory config singleton (replace with DB in production) ───────────────
_current_config: GuardrailConfig = GuardrailConfig()

# ── In-memory audit ring buffer ───────────────────────────────────────────────
_audit_log: list[dict] = []
_MAX_AUDIT_RECORDS = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_current_config() -> GuardrailConfig:
    """Accessor used by the chat pipeline to read the active guardrail config."""
    return _current_config


def record_audit(record: GuardrailAuditRecord) -> None:
    """Append a structured audit record, called by the guardrail pipeline."""
    _audit_log.append(record.model_dump())
    if len(_audit_log) > _MAX_AUDIT_RECORDS:
        _audit_log.pop(0)


@router.get("/config", response_model=GuardrailConfig, summary="Get current guardrail configuration")
async def get_guardrail_config():
    """Returns the active guardrail configuration (toggles + thresholds)."""
    return _current_config


@router.post("/config", response_model=GuardrailConfig, summary="Save guardrail configuration")
async def save_guardrail_config(config: GuardrailConfig):
    """Persists a full guardrail configuration (toggles + thresholds)."""
    global _current_config
    _current_config = config
    logger.info("Guardrail configuration updated.")
    return _current_config


@router.post("/config/reset", response_model=GuardrailConfig, summary="Reset guardrail configuration to defaults")
async def reset_guardrail_config():
    """Resets the active configuration back to spec defaults (all guardrails enabled)."""
    global _current_config
    _current_config = GuardrailConfig()
    return _current_config


@router.get("/audit", summary="List recent guardrail audit records")
async def list_audit_records(limit: int = Query(50, ge=1, le=500)):
    """
    Returns the most recent structured audit records (newest first).

    Records never contain raw PII values, query text, or response text —
    only counts, flags, and category labels.
    """
    recent = list(reversed(_audit_log))[:limit]
    return {"records": recent, "total": len(_audit_log)}
