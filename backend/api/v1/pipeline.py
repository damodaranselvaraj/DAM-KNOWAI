"""
Pipeline management endpoints — /api/v1/pipeline

/pipeline/run now performs the FULL Phase 1-4 flow in a background thread:
it first drains pending_upload_store (files staged by /documents/upload
but not yet parsed) and parses each one, then pulls all active documents
from version_store, translates the saved PipelineConfig into the internal
chunking/embedding config, and executes chunk → embed → store for each
document. Progress is tracked per-phase so the frontend stepper reflects
actual work, not a fake timer.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.ingestion.versioning import version_store
from backend.models.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# ── In-memory stores (replace with Redis/DB in production) ────────────────────
_current_config: dict = {}
_jobs: dict[str, dict] = {}

# Single shared executor for background pipeline work
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline_worker")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_phases() -> dict:
    return {
        "parse": {"name": "Parse",  "status": "pending", "progress": 0, "message": "Waiting to start"},
        "chunk": {"name": "Chunk",  "status": "pending", "progress": 0, "message": "Waiting to start"},
        "embed": {"name": "Embed",  "status": "pending", "progress": 0, "message": "Waiting to start"},
        "store": {"name": "Store",  "status": "pending", "progress": 0, "message": "Waiting to start"},
    }


# ─── Config translation ───────────────────────────────────────────────────────

def _translate_chunking_config(api_cfg: dict) -> "ChunkingConfig":
    """
    Translate the API-facing pipeline_config.ChunkingApiConfig dict into the
    internal chunk.ChunkingConfig that ChunkingPipeline.chunk_document() expects.

    Key differences:
      - strategy:   Literal string  →  ChunkingStrategy enum
      - min_chunk / max_chunk  →  min_chunk_tokens / max_chunk_tokens
      - sentence_based vs sentence naming discrepancy is handled here
    """
    from backend.models.chunk import ChunkingConfig, ChunkingStrategy
    from backend.config import settings as _cfg

    raw_strategy = api_cfg.get("strategy", "recursive")
    # The API uses "sentence"; the enum value is "sentence_based" — normalise
    if raw_strategy == "sentence":
        raw_strategy = "sentence_based"

    try:
        strategy = ChunkingStrategy(raw_strategy)
    except ValueError:
        logger.warning("Unknown chunking strategy '%s', defaulting to recursive.", raw_strategy)
        strategy = ChunkingStrategy.RECURSIVE

    return ChunkingConfig(
        strategy=strategy,
        chunk_size=api_cfg.get("chunk_size", _cfg.default_chunk_size),
        overlap=api_cfg.get("overlap", _cfg.default_chunk_overlap),
        min_chunk_tokens=api_cfg.get("min_chunk", 50),
        max_chunk_tokens=api_cfg.get("max_chunk", 1024),
        separators=api_cfg.get("separators", ["\n\n", "\n", " ", ""]),
        breakpoint_percentile=api_cfg.get("breakpoint_percentile", 95),
    )


# ─── Background pipeline worker ───────────────────────────────────────────────

def _run_pipeline_worker(job_id: str, cfg: dict) -> None:
    """
    Runs in a background thread.  First drains every staged upload from
    pending_upload_store and runs Phase 1 (parse) on it, then pulls every
    active document from version_store and executes the full Phase 2→4
    pipeline on each one.  Updates _jobs[job_id] as work progresses so
    polling reflects real state.
    """
    from backend.ingestion import ingest_document
    from backend.ingestion.embedders.embedding_pipeline import build_embedding_pipeline
    from backend.ingestion.pending_store import pending_upload_store
    from backend.vector_store.vector_store_pipeline import build_vector_store_pipeline

    job = _jobs[job_id]
    phases = job["phases"]

    def _set_phase(name: str, status: str, pct: int, msg: str) -> None:
        phases[name]["status"]   = status
        phases[name]["progress"] = pct
        phases[name]["message"]  = msg
        if status in ("complete", "failed"):
            phases[name]["completed_at"] = _now()
        job["updated_at"] = _now()

    from backend.api.v1.documents import log_pipeline_activity

    try:
        # ── Phase 1: parse every staged upload ────────────────────────────────
        # Parsing is deferred until the pipeline actually runs — uploads only
        # validate + stage the raw bytes (see /documents/upload).
        pending = pending_upload_store.pop_all()
        if pending:
            total_pending = len(pending)
            logger.info("Pipeline job %s: parsing %d staged upload(s).", job_id, total_pending)
            for idx, item in enumerate(pending, start=1):
                pct_done = int((idx - 1) / total_pending * 100)
                pct_next = int(idx / total_pending * 100)
                _set_phase("parse", "in_progress", pct_done, f"Parsing {item.filename} ({idx}/{total_pending})")
                try:
                    parse_result = ingest_document(
                        filename=item.filename,
                        content=item.content,
                        version_handling=item.version_handling,
                        force_parser=item.force_parser,
                    )
                    if parse_result.success and parse_result.document:
                        doc = parse_result.document
                        log_pipeline_activity(
                            "📤", "Document parsed",
                            f"{doc.filename} — {doc.page_count} page(s), parsed via {doc.parser_used.value}",
                        )
                    else:
                        logger.error(
                            "Job %s: failed to parse staged file '%s': %s",
                            job_id, item.filename, parse_result.error,
                        )
                        log_pipeline_activity("❌", "Parse failed", f"{item.filename}: {parse_result.error}")
                except Exception as parse_exc:
                    logger.exception(
                        "Job %s: exception parsing staged file '%s': %s",
                        job_id, item.filename, parse_exc,
                    )
                    log_pipeline_activity("❌", "Parse failed", f"{item.filename}: {parse_exc}")
                _set_phase("parse", "in_progress", pct_next, f"Parsed {idx}/{total_pending}")
            _set_phase("parse", "complete", 100, f"Parsed {total_pending} file(s)")
        else:
            _set_phase("parse", "complete", 100, "No staged files to parse")

        docs = version_store.all_active()
        if not docs:
            for p in ["chunk", "embed", "store"]:
                _set_phase(p, "complete", 100, "No documents to process")
            job["status"] = "complete"
            job["updated_at"] = _now()
            log_pipeline_activity("ℹ️", "Pipeline completed", "No documents to process")
            logger.info("Pipeline job %s: no active documents, marked complete.", job_id)
            return

        total = len(docs)
        logger.info("Pipeline job %s: processing %d document(s).", job_id, total)

        # Build pipeline objects once — reused across all documents
        chunking_cfg   = _translate_chunking_config(cfg.get("chunking", {}))
        emb_pipeline   = build_embedding_pipeline()
        vs_pipeline    = build_vector_store_pipeline()
        parser_cfg     = cfg.get("parser", {})
        force_parser   = parser_cfg.get("parser_type") if parser_cfg.get("parser_type") != "auto" else None
        version_handling = parser_cfg.get("version_handling", "replace")
        from backend.config import settings as _cfg
        tenant_id      = _cfg.pipeline_default_tenant
        corpus         = _cfg.pipeline_default_corpus

        for idx, doc in enumerate(docs, start=1):
            pct_done = int((idx - 1) / total * 100)
            pct_next = int(idx / total * 100)

            # Read the stored raw bytes — we use the already-parsed document
            # directly by calling chunk → embed → store individually so we
            # avoid re-parsing content we already have.
            try:
                # Phase 2 — chunk
                _set_phase("chunk",  "in_progress", pct_done, f"Chunking {doc.filename} ({idx}/{total})")

                from backend.ingestion import chunk_document, embed_chunks, store_embeddings

                chunking_result = chunk_document(doc, chunking_cfg)
                _set_phase("chunk", "in_progress", pct_next, f"Chunked {idx}/{total}: {chunking_result.total_chunks} chunks")

                if not chunking_result.all_chunks:
                    logger.warning("Job %s: no chunks for '%s', skipping embed/store.", job_id, doc.filename)
                    continue

                # Phase 3 — embed
                _set_phase("embed", "in_progress", pct_done, f"Embedding {doc.filename} ({idx}/{total})")
                embedding_result = embed_chunks(
                    chunks=chunking_result.all_chunks,
                    doc_id=doc.doc_id,
                    doc_name=doc.filename,
                    pipeline=emb_pipeline,
                )
                _set_phase("embed", "in_progress", pct_next,
                           f"Embedded {idx}/{total}: {embedding_result.total_embedded} vectors")

                if not embedding_result.embedded_chunks:
                    logger.warning("Job %s: no embeddings for '%s', skipping store.", job_id, doc.filename)
                    continue

                # Phase 4 — store
                _set_phase("store", "in_progress", pct_done, f"Storing {doc.filename} ({idx}/{total})")
                store_result = store_embeddings(
                    embedding_result=embedding_result,
                    tenant_id=tenant_id,
                    corpus=corpus,
                    version=doc.version,
                    vs_pipeline=vs_pipeline,
                )
                _set_phase("store", "in_progress", pct_next,
                           f"Stored {idx}/{total}: {store_result.upserted_count} vectors")

            except Exception as doc_exc:
                logger.error("Job %s: error processing '%s': %s", job_id, doc.filename, doc_exc)
                # Don't abort the whole job — log and continue with next doc

        # Mark all phases complete
        for p in ["parse", "chunk", "embed", "store"]:
            _set_phase(p, "complete", 100, "Done")
        job["status"] = "complete"
        log_pipeline_activity(
            "✅",
            "Pipeline completed",
            f"{total} document(s) processed — chunk, embed, store complete",
        )
        logger.info("Pipeline job %s completed successfully.", job_id)

    except Exception as exc:
        logger.exception("Pipeline job %s failed: %s", job_id, exc)
        for p in ["parse", "chunk", "embed", "store"]:
            if phases[p]["status"] == "in_progress":
                _set_phase(p, "failed", phases[p]["progress"], f"Error: {exc}")
        job["status"] = "failed"
        job["error"]  = str(exc)
        job["updated_at"] = _now()
        log_pipeline_activity("❌", "Pipeline failed", str(exc))


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/config", summary="Get current pipeline configuration")
async def get_pipeline_config():
    """Returns the current pipeline configuration. Returns defaults if none saved."""
    if not _current_config:
        return PipelineConfig().model_dump()
    return _current_config


@router.post("/config", summary="Save pipeline configuration")
async def save_pipeline_config(config: PipelineConfig):
    """Saves and validates a full pipeline configuration."""
    global _current_config
    _current_config = config.model_dump()
    return {"status": "saved", "timestamp": _now(), "config": _current_config}


@router.post("/run", summary="Trigger a full pipeline run")
async def run_pipeline():
    """
    Queues a new pipeline job and kicks off the real Phase 1-4 processing:
    parses every file staged by /documents/upload, then runs chunk → embed →
    store for every active document in the version store, using the latest
    saved configuration.

    Returns job_id for polling via GET /pipeline/status/{job_id}.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id":     job_id,
        "status":     "running",
        "phases":     _default_phases(),
        "created_at": _now(),
        "updated_at": _now(),
        "error":      None,
    }

    # Kick off first phase marker immediately so the UI shows activity
    _jobs[job_id]["phases"]["parse"]["status"]  = "in_progress"
    _jobs[job_id]["phases"]["parse"]["progress"] = 5
    _jobs[job_id]["phases"]["parse"]["message"]  = "Preparing documents…"

    # Run the real pipeline in a background thread so we don't block FastAPI
    cfg_snapshot = dict(_current_config) if _current_config else PipelineConfig().model_dump()
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_pipeline_worker, job_id, cfg_snapshot)

    from backend.ingestion.pending_store import pending_upload_store
    pending_count = pending_upload_store.count()
    doc_count = len(version_store.all_active()) + pending_count
    return {
        "job_id":       job_id,
        "status":       "running",
        "message":      f"Pipeline started — parsing {pending_count} staged file(s), processing {doc_count} document(s)",
        "doc_count":    doc_count,
        "pending_count": pending_count,
    }


@router.get("/status/{job_id}", summary="Poll pipeline job status")
async def get_pipeline_status(job_id: str):
    """Returns current status and per-phase progress for the given job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _jobs[job_id]


@router.get("/phases", summary="Get all phase statuses (latest job)")
async def get_phase_statuses():
    """Returns phase breakdown of the most recently created job."""
    if not _jobs:
        return {"phases": _default_phases(), "job_id": None}
    latest = sorted(_jobs.values(), key=lambda j: j["created_at"])[-1]
    return {"phases": latest["phases"], "job_id": latest["job_id"]}
