"""
File validation: MIME type detection, extension allow-listing,
and size guard.  Raises FastAPI HTTPException(415) for unsupported types.

Size limits are driven by settings.max_file_size_mb (MAX_FILE_SIZE_MB in .env)
so no hardcoded byte limits exist in this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _max_bytes() -> int:
    """Return the global max file size in bytes from settings."""
    try:
        from backend.config import settings
        return settings.max_file_size_mb * 1024 * 1024
    except Exception:
        return 50 * 1024 * 1024  # safe fallback if settings not yet loaded


# ─── Supported types registry ────────────────────────────────────────────────

@dataclass(frozen=True)
class FileSpec:
    extension:   str          # lowercase, no leading dot
    mime_types:  tuple[str, ...]
    description: str

    @property
    def max_bytes(self) -> int:
        """Always reflect the current settings value — never cached."""
        return _max_bytes()


_SUPPORTED: dict[str, FileSpec] = {
    "pdf": FileSpec(
        extension="pdf",
        mime_types=("application/pdf",),
        description="PDF document (text-based)",
    ),
    "docx": FileSpec(
        extension="docx",
        mime_types=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ),
        description="Microsoft Word document",
    ),
    "csv": FileSpec(
        extension="csv",
        mime_types=("text/csv", "application/csv", "text/plain"),
        description="Comma-separated values",
    ),
    "html": FileSpec(
        extension="html",
        mime_types=("text/html",),
        description="HTML document",
    ),
    "txt": FileSpec(
        extension="txt",
        mime_types=("text/plain",),
        description="Plain text",
    ),
    "xlsx": FileSpec(
        extension="xlsx",
        mime_types=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ),
        description="Microsoft Excel spreadsheet",
    ),
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_SUPPORTED.keys())


# ─── Magic-byte MIME sniffing (no python-magic dependency) ───────────────────

# Signatures: (offset, bytes_prefix) → mime_type
_MAGIC: list[tuple[int, bytes, str]] = [
    (0, b"%PDF",                         "application/pdf"),
    (0, b"PK\x03\x04",                   "application/zip"),  # docx / xlsx are ZIP
    (0, b"\xd0\xcf\x11\xe0",             "application/msword"),
    (0, b"<html",                         "text/html"),
    (0, b"<HTML",                         "text/html"),
    (0, b"<!DOCTYPE",                     "text/html"),
    (0, b"<!doctype",                     "text/html"),
]


def _sniff_mime(header: bytes) -> str | None:
    """Return a MIME type from the first 512 bytes, or None if unknown."""
    for offset, prefix, mime in _MAGIC:
        end = offset + len(prefix)
        if header[offset:end] == prefix:
            return mime
    return None


def _mime_from_zip(header: bytes, extension: str) -> str:
    """Disambiguate ZIP-based formats (docx vs xlsx) by extension."""
    if extension == "docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if extension == "xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/zip"


# ─── Public API ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    extension:   str
    mime_type:   str
    size_bytes:  int
    spec:        FileSpec


def validate_upload(
    filename:   str,
    content:    bytes,
    *,
    strict_mime: bool = True,
) -> ValidationResult:
    """
    Validate an uploaded file's extension, MIME type, and size.

    Args:
        filename:    Original filename from the upload (used for extension check).
        content:     Raw file bytes.
        strict_mime: When True (default), a sniffed MIME type that doesn't match
                     the spec's allowed list raises HTTPException(415). Set False
                     to only log a warning on mismatch instead of rejecting
                     (content-type/magic-byte sniffing isn't perfectly reliable
                     for every real-world file).

    Returns:
        ValidationResult with resolved extension, MIME type, and matched FileSpec.

    Raises:
        HTTPException(415): Unsupported Media Type — extension not in allow-list,
                            MIME mismatch, or file exceeds size limit.
        HTTPException(400): Empty file.
    """
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Extension check ───────────────────────────────────────────────────────
    suffix = Path(filename).suffix.lstrip(".").lower()
    if not suffix:
        raise HTTPException(
            status_code=415,
            detail=f"File '{filename}' has no extension. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    if suffix not in _SUPPORTED:
        raise HTTPException(
            status_code=415,
            detail=(
                f"File type '.{suffix}' is not supported. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    spec = _SUPPORTED[suffix]

    # ── Size check ────────────────────────────────────────────────────────────
    size = len(content)
    if size > spec.max_bytes:
        limit_mb = spec.max_bytes / (1024 * 1024)
        actual_mb = size / (1024 * 1024)
        raise HTTPException(
            status_code=415,
            detail=(
                f"File size {actual_mb:.1f} MB exceeds the {limit_mb:.0f} MB limit "
                f"for .{suffix} files."
            ),
        )

    # ── MIME sniff ────────────────────────────────────────────────────────────
    header = content[:512]
    sniffed = _sniff_mime(header)

    if sniffed == "application/zip":
        sniffed = _mime_from_zip(header, suffix)

    # For plain-text formats (csv, txt) sniffing is unreliable — trust extension
    if suffix in ("csv", "txt"):
        sniffed = sniffed or "text/plain"

    mime_type = sniffed or f"application/{suffix}"

    if sniffed and sniffed not in spec.mime_types:
        if strict_mime:
            logger.warning(
                "Rejecting '%s': sniffed MIME %s does not match expected %s "
                "(strict_mime=True).",
                filename, sniffed, spec.mime_types,
            )
            raise HTTPException(
                status_code=415,
                detail=(
                    f"File '{filename}' content does not match its extension. "
                    f"Detected MIME type '{sniffed}', expected one of "
                    f"{list(spec.mime_types)} for .{suffix} files."
                ),
            )
        logger.warning(
            "MIME mismatch for '%s': sniffed=%s, expected one of %s "
            "(strict_mime=False — not rejecting).",
            filename, sniffed, spec.mime_types,
        )

    logger.info(
        "Validated '%s': ext=%s mime=%s size=%d bytes",
        filename, suffix, mime_type, size,
    )

    return ValidationResult(
        extension=suffix,
        mime_type=mime_type,
        size_bytes=size,
        spec=spec,
    )


def get_supported_formats() -> list[dict]:
    """Return a list of supported format descriptors for the API /config endpoint."""
    limit_mb = _max_bytes() // (1024 * 1024)
    return [
        {
            "extension":   ext,
            "mime_types":  list(spec.mime_types),
            "max_mb":      limit_mb,
            "description": spec.description,
        }
        for ext, spec in _SUPPORTED.items()
    ]
