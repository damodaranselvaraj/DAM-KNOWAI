"""
Document management endpoints — /api/v1/documents

Upload/parse split: POST /documents/upload only validates the file and
stages the raw bytes in pending_upload_store — it does NOT parse the
document. Parsing (Phase 1) now happens when the pipeline is triggered
via POST /pipeline/run, immediately before Phase 2 (chunk) begins. This
keeps the "Upload Files" action fast and side-effect free until the user
explicitly clicks "Run Pipeline".

The in-memory _documents dict bridges the parsed results to the
list/stats/delete endpoints until a database layer is added.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from backend.ingestion.file_validator import validate_upload
from backend.ingestion.parsers.parser_router import router as _parser_router
from backend.ingestion.pending_store import pending_upload_store
from backend.ingestion.versioning import version_store
from backend.models.parsed_document import ParsedDocument

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

# Activity log: each entry is {"icon", "title", "detail", "timestamp"}
_activity_log: list[dict] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_activity(icon: str, title: str, detail: str) -> None:
    """Append an event to the in-memory activity log (capped at 50 entries)."""
    _activity_log.append({
        "icon":      icon,
        "title":     title,
        "detail":    detail,
        "timestamp": _now(),
    })
    if len(_activity_log) > 50:
        _activity_log.pop(0)


# ─── Upload ───────────────────────────────────────────────────────────────────

@router.post("/upload", summary="Validate and stage a document for later parsing")
async def upload_document(
    file:             UploadFile = File(...),
    version_handling: str        = Query("replace",
        description="replace | keep_both | soft_delete"),
    force_parser:     str | None = Query(None,
        description="pymupdf | docling | llamaindex  (overrides auto-routing)"),
):
    """
    Accepts a file upload and validates its type/size, then stages the raw
    bytes in the pending-upload store. Parsing does NOT happen here — it is
    deferred until the pipeline is run (POST /pipeline/run), so clicking
    "Upload" only loads the file; clicking "Run Pipeline" is what triggers
    parse → chunk → embed → store.

    Raises HTTP 415 for unsupported file types or size violations.
    Raises HTTP 400 for an empty file.
    """
    content  = await file.read()
    filename = file.filename or "upload"

    logger.info(
        "Received upload: filename=%s size=%d bytes version_handling=%s force=%s",
        filename, len(content), version_handling, force_parser,
    )

    # Validate only — raises HTTPException(415/400) on failure. No parsing.
    validation = validate_upload(filename, content)

    staged = pending_upload_store.add(
        filename=filename,
        content=content,
        content_type=validation.mime_type,
        version_handling=version_handling,
        force_parser=force_parser,
    )

    size_kb = round(staged.size_bytes / 1024, 1)
    _log_activity(
        "📥",
        "Document staged",
        f"{staged.filename} ({size_kb} KB) — waiting for pipeline run to parse",
    )

    return {
        "upload_id":   staged.upload_id,
        "filename":    staged.filename,
        "file_type":   validation.extension,
        "size_bytes":  staged.size_bytes,
        "staged_at":   staged.staged_at,
        "status":      "staged",
        "message": (
            "File loaded and staged. Click 'Run Pipeline' to parse, chunk, "
            "embed, and index it."
        ),
    }


# ─── List ─────────────────────────────────────────────────────────────────────

@router.get("/list", summary="List all active documents")
async def list_documents(
    file_type: str | None = Query(None, description="Filter by file type (pdf, docx, …)"),
    search:    str | None = Query(None, description="Substring match on filename"),
):
    """Returns all active (non-deleted) documents from the version store."""
    docs: list[ParsedDocument] = version_store.all_active()

    # Apply optional filters
    if file_type:
        docs = [d for d in docs if d.file_type.value == file_type.lower()]
    if search:
        docs = [d for d in docs if search.lower() in d.filename.lower()]

    # Sort newest first
    docs.sort(key=lambda d: d.parsed_at, reverse=True)

    return {
        "documents": [d.summary() for d in docs],
        "total":     len(docs),
    }


# ─── Delete ───────────────────────────────────────────────────────────────────

@router.delete("/{doc_id}", summary="Soft-delete a document")
async def delete_document(doc_id: str):
    """Marks the document as inactive without purging its parsed content."""
    active_ids = {d.doc_id: d for d in version_store.all_active()}
    if doc_id not in active_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Active document '{doc_id}' not found.",
        )
    doc = active_ids[doc_id]
    version_store.mark_inactive(doc_id)

    # Purge this document's chunks from the BM25 corpus store and rebuild
    # the sparse index — otherwise a "deleted" document keeps showing up
    # in hybrid_bm25 retrieval results indefinitely (dense/SPLADE results
    # via Pinecone are unaffected here; only the local BM25 corpus needs
    # this manual purge).
    try:
        from backend.ingestion.chunk_corpus_store import chunk_corpus_store
        removed = chunk_corpus_store.remove_by_doc_id(doc_id)
        if removed:
            from backend.ingestion import _rebuild_bm25_index
            _rebuild_bm25_index()
            logger.info("Purged %d chunk(s) for doc_id=%s from BM25 corpus.", removed, doc_id)
    except Exception as exc:
        logger.warning("Failed to purge BM25 corpus for doc_id=%s: %s", doc_id, exc)

    _log_activity("🗑️", "Document deleted", doc.filename)
    logger.info("Soft-deleted doc_id=%s", doc_id)
    return {"status": "deleted", "doc_id": doc_id, "timestamp": _now()}


# ─── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats", summary="Aggregate document statistics")
async def get_document_stats():
    """Returns counts and size breakdown for all active documents."""
    docs = version_store.all_active()
    by_type:     dict[str, int] = {}
    total_size   = 0
    total_words  = 0

    for doc in docs:
        ft = doc.file_type.value
        by_type[ft]  = by_type.get(ft, 0) + 1
        total_size  += doc.size_bytes
        total_words += doc.total_words

    return {
        "total_documents": len(docs),
        "total_words":     total_words,
        "total_size_bytes": total_size,
        "by_type":         by_type,
        "store_stats":     version_store.stats(),
        "last_updated":    _now(),
    }


# ─── Formats ──────────────────────────────────────────────────────────────────

@router.get("/formats", summary="List supported file formats and parser routing")
async def get_supported_formats():
    """Returns the list of supported file types and the parser fallback chains."""
    from backend.ingestion.file_validator import get_supported_formats
    return {
        "formats":       get_supported_formats(),
        "parser_chains": _parser_router.describe_chains(),
    }


# ─── Document detail ──────────────────────────────────────────────────────────

@router.get("/{doc_id}", summary="Get full parsed document detail")
async def get_document(doc_id: str):
    """Returns the full ParsedDocument including per-page content and audit trail."""
    all_docs = version_store.all_active()
    doc = next((d for d in all_docs if d.doc_id == doc_id), None)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    return {
        **doc.summary(),
        "title":       doc.title,
        "author":      doc.author,
        "language":    doc.language,
        "published_at": doc.published_at,
        "full_text_preview": doc.full_text[:500] + "…" if len(doc.full_text) > 500 else doc.full_text,
        "pages": [
            {
                "page_number": p.page_number,
                "word_count":  p.word_count,
                "char_count":  p.char_count,
                "has_tables":  p.has_tables,
                "has_images":  p.has_images,
                "text_preview": p.text[:200] + "…" if len(p.text) > 200 else p.text,
            }
            for p in doc.pages
        ],
        "audit_trail": [
            {
                "parser":          a.parser.value,
                "succeeded":       a.succeeded,
                "duration_ms":     a.duration_ms,
                "pages_extracted": a.pages_extracted,
                "error":           a.error,
            }
            for a in doc.audit_trail
        ],
        "parse_time_ms": doc.parse_time_ms,
        "version_action": doc.version_action.value,
        "previous_doc_id": doc.previous_doc_id,
    }


# ─── Activity log ─────────────────────────────────────────────────────────────

@router.get("/activity", summary="Recent document and pipeline activity")
async def get_activity(limit: int = Query(20, ge=1, le=50)):
    """
    Returns the most recent document upload, delete, and pipeline events
    in reverse-chronological order (newest first).
    """
    recent = list(reversed(_activity_log))[:limit]
    return {"activities": recent, "total": len(_activity_log)}


def log_pipeline_activity(icon: str, title: str, detail: str) -> None:
    """
    Public hook called by pipeline.py to record pipeline-level events
    (job started, job completed, job failed) into the shared activity log.
    """
    _log_activity(icon, title, detail)
