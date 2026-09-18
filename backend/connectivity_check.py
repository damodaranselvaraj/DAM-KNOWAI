"""
Connectivity validation — run once at startup before serving requests.

Checks:
  1. OpenAI API key is valid  (cheap models.list() call)
  2. Pinecone index is reachable  (describe_index_stats)

There is currently no Redis dependency in this codebase, so no Redis check
is performed here.

Raises RuntimeError listing all failures so the operator sees every problem
in a single restart rather than discovering them one by one.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_connectivity(abort_on_failure: bool = True) -> dict[str, bool]:
    """
    Validate all external service connections.

    Args:
        abort_on_failure: When True (default), raises RuntimeError if any
                          REQUIRED service (OpenAI, Pinecone) is unreachable.

    Returns:
        Dict mapping service name → True (ok) / False (failed).
    """
    from backend.config import settings

    results: dict[str, bool] = {}
    errors:  list[str] = []

    # ── 1. OpenAI ─────────────────────────────────────────────────────────────
    try:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key, timeout=10.0)
        # Cheapest possible call — list models returns immediately
        client.models.list()
        results["openai"] = True
        logger.info("✅ OpenAI connectivity OK (model=%s)", settings.openai_chat_model)
    except Exception as exc:
        results["openai"] = False
        msg = f"OpenAI unreachable: {exc}"
        errors.append(msg)
        logger.error("❌ %s", msg)

    # ── 2. Pinecone ───────────────────────────────────────────────────────────
    try:
        from pinecone import Pinecone as _PC
        pc    = _PC(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        index.describe_index_stats()
        results["pinecone"] = True
        logger.info(
            "✅ Pinecone connectivity OK (index=%s)", settings.pinecone_index_name
        )
    except Exception as exc:
        results["pinecone"] = False
        msg = (
            f"Pinecone unreachable (index='{settings.pinecone_index_name}'): {exc}. "
            "Ensure the index exists and PINECONE_API_KEY / PINECONE_INDEX_NAME are correct."
        )
        errors.append(msg)
        logger.error("❌ %s", msg)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if errors and abort_on_failure:
        raise RuntimeError(
            "Startup connectivity check failed:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    return results
