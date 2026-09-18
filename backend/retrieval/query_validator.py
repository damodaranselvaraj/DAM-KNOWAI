"""
Query validation and Pinecone filter sanitization.

Two responsibilities:

1. ``sanitize_query`` — normalises and validates free-text user queries
   before they reach the OpenAI embedding call or the local BM25 index:
   Unicode normalisation, control-character stripping, whitespace
   collapsing, and length enforcement.

2. ``validate_filters`` — validates the ``filters`` dict a caller supplies
   on ``ChatQuery`` before it is forwarded to Pinecone's ``filter=``.
   Enforces an allow-list of metadata fields (sourced from the canonical
   Pinecone metadata registry in ``backend.vector_store.schema``) and a
   fixed set of Pinecone comparison / logical operators, so a caller
   cannot filter on arbitrary metadata fields (e.g. bypassing tenant
   isolation) or inject unsupported operator payloads.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from backend.vector_store.schema import INDEXED_FIELD_NAMES

# ─── Query text validation ─────────────────────────────────────────────────

MIN_QUERY_LENGTH = 1
MAX_QUERY_LENGTH = 4_000  # keep in sync with ChatQuery.query max_length

# Strip C0/C1 control characters (NUL, ESC, etc.) but leave normal
# printable text and standard whitespace alone — whitespace runs are
# collapsed separately below. Chat queries are single-line free text, so
# tabs/newlines are folded into single spaces rather than rejected.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE_RE = re.compile(r"\s+")


class QueryValidationError(ValueError):
    """Raised when a user-supplied query fails validation."""


def sanitize_query(raw_query: str) -> str:
    """
    Normalise and validate a free-text query before it is embedded or
    passed to BM25 / Pinecone.

    Steps:
        1. Unicode NFKC normalisation (collapses homoglyph / width tricks
           sometimes used to smuggle characters past naive filters).
        2. Strip control characters (NUL bytes, escape sequences, etc.).
        3. Collapse repeated whitespace (including tabs/newlines) to a
           single space, then strip leading/trailing whitespace.
        4. Enforce length bounds.

    Raises
    ------
    QueryValidationError
        If the query is empty after sanitisation or exceeds the max length.
    """
    if raw_query is None:
        raise QueryValidationError("Query must not be empty.")

    normalized = unicodedata.normalize("NFKC", raw_query)
    stripped = _CONTROL_CHAR_RE.sub("", normalized)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()

    if len(collapsed) < MIN_QUERY_LENGTH:
        raise QueryValidationError("Query must not be empty.")
    if len(collapsed) > MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters."
        )

    return collapsed


# ─── Pinecone filter allow-list ─────────────────────────────────────────────

# Fields callers are permitted to filter on. Sourced from the canonical
# Pinecone metadata field registry (backend.vector_store.schema) so the
# allow-list can never drift from what is actually indexed/upserted.
ALLOWED_FILTER_FIELDS: frozenset[str] = frozenset(INDEXED_FIELD_NAMES)

# Pinecone comparison / logical operators we permit. This is purely an
# allow-list of Pinecone's own filter DSL — nothing beyond what Pinecone
# supports is enabled.
_COMPARISON_OPS: frozenset[str] = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
)
_LOGICAL_OPS: frozenset[str] = frozenset({"$and", "$or"})

_MAX_FILTER_DEPTH = 6


class FilterValidationError(ValueError):
    """Raised when a caller-supplied Pinecone filter uses a disallowed field/operator."""


def validate_filters(
    filters: dict[str, Any] | None,
    *,
    _depth: int = 0,
) -> dict[str, Any] | None:
    """
    Recursively validate a Pinecone ``filter=`` dict against the metadata
    field allow-list and the supported operator set.

    Rejects:
      - any top-level or nested key that isn't a known metadata field or
        a ``$and`` / ``$or`` logical operator
      - any comparison operator outside ``_COMPARISON_OPS``
      - filters nested deeper than ``_MAX_FILTER_DEPTH`` levels (defends
        against pathological payloads)

    Returns the filter unchanged if valid (validation only — this function
    does not mutate or rewrite the filter).

    Raises
    ------
    FilterValidationError
        On any disallowed field, operator, or malformed shape.
    """
    if filters is None:
        return None
    if _depth > _MAX_FILTER_DEPTH:
        raise FilterValidationError("Filter nesting too deep.")
    if not isinstance(filters, dict):
        raise FilterValidationError("Filter must be a JSON object.")

    for key, value in filters.items():
        if key in _LOGICAL_OPS:
            if not isinstance(value, list):
                raise FilterValidationError(f"'{key}' must be a list of filter clauses.")
            for clause in value:
                validate_filters(clause, _depth=_depth + 1)
            continue

        if key not in ALLOWED_FILTER_FIELDS:
            raise FilterValidationError(
                f"Filtering on field '{key}' is not permitted. "
                f"Allowed fields: {sorted(ALLOWED_FILTER_FIELDS)}"
            )

        if isinstance(value, dict):
            for op in value.keys():
                if op not in _COMPARISON_OPS:
                    raise FilterValidationError(
                        f"Operator '{op}' is not permitted on field '{key}'. "
                        f"Allowed operators: {sorted(_COMPARISON_OPS)}"
                    )
        # Bare equality shorthand, e.g. {"doc_id": "abc"}, is permitted as-is.

    return filters
