"""
SHA-256 hashing utilities for deduplication and chunk fingerprinting.

All hashes are lowercase hex strings (64 chars).  The module is stateless
and importable anywhere without side-effects.
"""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


# ─── File / byte hashing ─────────────────────────────────────────────────────

def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of a raw byte sequence."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    """
    Stream-hash a file on disk without loading it fully into memory.
    Reads in 8 MB blocks — safe for large PDFs.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ─── Text / chunk hashing ─────────────────────────────────────────────────────

def hash_text(text: str, encoding: str = "utf-8") -> str:
    """Return the SHA-256 hex digest of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode(encoding)).hexdigest()


def hash_chunk(text: str, doc_id: str, chunk_index: int) -> str:
    """
    Deterministic chunk ID: SHA-256 over the concatenation of
    doc_id + chunk_index + chunk_text.

    Using doc_id + index ensures two identical text chunks from
    different documents (or different positions) receive different IDs.
    """
    payload = f"{doc_id}:{chunk_index}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ─── Short fingerprints ───────────────────────────────────────────────────────

def short_hash(data: bytes | str, length: int = 12) -> str:
    """
    Return the first `length` hex chars of a SHA-256 digest.
    Useful for human-readable IDs in logs / UI.  Not collision-safe
    for large corpora — use full hashes for deduplication.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


# ─── Equality helpers ─────────────────────────────────────────────────────────

def hashes_equal(a: str, b: str) -> bool:
    """
    Constant-time comparison of two hex digest strings.
    Avoids timing attacks in security-sensitive contexts.
    """
    return hmac.compare_digest(a.lower(), b.lower())


def is_duplicate(incoming_hash: str, known_hashes: set[str]) -> bool:
    """
    Return True if `incoming_hash` is already present in `known_hashes`.
    Comparison is case-insensitive and constant-time per element.
    """
    ih = incoming_hash.lower()
    return any(hashes_equal(ih, kh) for kh in known_hashes)
