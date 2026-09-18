"""
Pre-embedding validation pipeline.

Four sequential checks applied to every chunk before it is sent to the
OpenAI embedding API:

  1. Empty-chunk filter       — skip chunks with no usable text
  2. Token-count validation   — truncate or skip chunks exceeding 8 191 tokens
  3. Language detection       — warn (not skip) for non-English content
  4. Deduplication            — skip chunks whose content_hash was already seen
                                in this embedding run

Returns a (valid_chunks, skipped_records) tuple.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.config import settings
from backend.models.chunk import Chunk
from backend.models.embedded_chunk import ChunkValidationRecord
from backend.utils.language_detector import detect_language
from backend.utils.token_counter import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)

# Hard limit imposed by text-embedding-3-small / ada-002 — sourced from
# backend.config.settings (openai_max_tokens) as the single source of truth.
_MAX_TOKENS: int = settings.openai_max_tokens
# Minimum non-whitespace characters to be considered non-empty
_MIN_CHARS:  int = 10


@dataclass
class ValidationSummary:
    total:         int = 0
    valid:         int = 0
    skipped_empty: int = 0
    skipped_long:  int = 0
    skipped_dupes: int = 0
    warned_lang:   int = 0
    truncated:     int = 0
    records: list[ChunkValidationRecord] = field(default_factory=list)


def validate_chunks(
    chunks:            list[Chunk],
    max_tokens:        int  = _MAX_TOKENS,
    truncate_long:     bool = True,
    warn_non_english:  bool = True,
    deduplicate:       bool = True,
    known_hashes:      set[str] | None = None,
) -> tuple[list[Chunk], ValidationSummary]:
    """
    Run all four pre-embedding checks and return only valid chunks.

    Args:
        chunks:           Input chunk list.
        max_tokens:       Token ceiling per chunk (default 8 191).
        truncate_long:    When True, oversized chunks are truncated instead
                          of skipped.  When False they are dropped.
        warn_non_english: Emit a WARNING for non-English chunks (never skips).
        deduplicate:      Drop chunks whose content_hash already appears in
                          `known_hashes` or earlier in this same batch.
        known_hashes:     Pre-existing content hashes to check against
                          (extends across multiple batch calls).

    Returns:
        (valid_chunks, ValidationSummary)
        valid_chunks are in the same relative order as input, possibly
        with their `text` and `token_count` mutated by truncation.
    """
    summary      = ValidationSummary(total=len(chunks))
    seen_hashes: set[str] = set(known_hashes or [])
    valid:       list[Chunk] = []

    for chunk in chunks:
        record = ChunkValidationRecord(
            chunk_id=chunk.chunk_id,
            valid=True,
            token_count=chunk.token_count or count_tokens(chunk.text),
        )

        # ── 1. Empty filter ───────────────────────────────────────────────────
        embed_text = (chunk.text_with_title or chunk.text or "").strip()
        if len(embed_text) < _MIN_CHARS or not embed_text:
            record.valid       = False
            record.skip_reason = "empty"
            summary.skipped_empty += 1
            summary.records.append(record)
            logger.debug("Skipping empty chunk %s", chunk.chunk_id)
            continue

        # ── 2. Token count ────────────────────────────────────────────────────
        token_count = count_tokens(embed_text)
        record.token_count = token_count

        if token_count > max_tokens:
            if truncate_long:
                embed_text  = truncate_to_tokens(embed_text, max_tokens)
                token_count = count_tokens(embed_text)
                record.truncated   = True
                record.token_count = token_count
                summary.truncated += 1
                # Mutate the chunk in-place so the embedder gets the trimmed text
                object.__setattr__(chunk, "text_with_title", embed_text)
                object.__setattr__(chunk, "token_count",     token_count)
                logger.warning(
                    "Chunk %s truncated from >%d to %d tokens",
                    chunk.chunk_id, max_tokens, token_count,
                )
            else:
                record.valid       = False
                record.skip_reason = "too_long"
                summary.skipped_long += 1
                summary.records.append(record)
                logger.warning(
                    "Skipping chunk %s: %d tokens exceeds limit %d",
                    chunk.chunk_id, token_count, max_tokens,
                )
                continue

        # ── 3. Language detection ─────────────────────────────────────────────
        if warn_non_english:
            lang = detect_language(embed_text[:500])
            record.language = lang
            if lang and lang != "en":
                summary.warned_lang += 1
                logger.warning(
                    "Chunk %s detected as '%s' (not English). "
                    "Embedding quality may be degraded.",
                    chunk.chunk_id, lang,
                )

        # ── 4. Deduplication ──────────────────────────────────────────────────
        if deduplicate:
            key = chunk.content_hash or embed_text
            if key in seen_hashes:
                record.valid       = False
                record.skip_reason = "duplicate"
                summary.skipped_dupes += 1
                summary.records.append(record)
                logger.debug("Skipping duplicate chunk %s", chunk.chunk_id)
                continue
            seen_hashes.add(key)

        # ── Valid ─────────────────────────────────────────────────────────────
        summary.valid += 1
        summary.records.append(record)
        valid.append(chunk)

    logger.info(
        "Pre-embedding validation: %d in → %d valid, %d empty, "
        "%d too-long, %d dupes, %d truncated, %d non-English warned",
        summary.total, summary.valid, summary.skipped_empty,
        summary.skipped_long, summary.skipped_dupes,
        summary.truncated, summary.warned_lang,
    )
    return valid, summary
