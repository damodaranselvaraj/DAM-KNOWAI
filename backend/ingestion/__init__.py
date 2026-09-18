"""
Ingestion package — Phase 1 (Parsing) + Phase 2 (Chunking) + Phase 3 (Embedding) + Phase 4 (Vector Store).

Public surface
──────────────
ingest_document(filename, content, ...)                → ParseResult          # Phase 1
chunk_document(parsed_doc, config)                     → ChunkingResult       # Phase 2
embed_chunks(chunks, doc_id, doc_name, pipeline)       → EmbeddingResult      # Phase 3
store_embeddings(embedding_result, ...)                → VectorStoreResult    # Phase 4
ingest_and_chunk(filename, content, ...)               → (ParseResult, ChunkingResult|None)
ingest_chunk_embed(filename, content, ...)             → (ParseResult, ChunkingResult|None, EmbeddingResult|None)
ingest_full_pipeline(filename, content, ...)           → FullPipelineResult   # Phase 1+2+3+4

All other symbols are implementation details.
"""
from __future__ import annotations

import logging

from backend.ingestion.parsers.parser_router import router as _parser_router
from backend.ingestion.versioning import version_handler as _version_handler
from backend.models.parsed_document import ParseRequest, ParseResult, ParserName
from backend.models.chunk import ChunkingConfig, ChunkingResult
from backend.models.embedded_chunk import EmbeddingResult

logger = logging.getLogger(__name__)

# ─── Phase 1: Parse ───────────────────────────────────────────────────────────

def ingest_document(
    filename:         str,
    content:          bytes,
    version_handling: str        = "replace",
    force_parser:     str | None = None,
) -> ParseResult:
    """
    Full Phase 1 ingestion pipeline for a single file.

    Args:
        filename:         Original filename — drives extension detection
                          and version matching.
        content:          Raw file bytes from the upload.
        version_handling: "replace" | "keep_both" | "soft_delete"
        force_parser:     Optional override: "pymupdf" | "docling" |
                          "llamaindex".  Bypasses the fallback chain.

    Returns:
        ParseResult(success=True, document=ParsedDocument) on success.
        ParseResult(success=False, error=…, http_status_code=…) on failure.
          • 415 — unsupported file type or size exceeded
          • 422 — all parsers in the chain failed
    """
    forced: ParserName | None = None
    if force_parser:
        try:
            forced = ParserName(force_parser.lower())
        except ValueError:
            logger.warning(
                "Unknown force_parser '%s' — using auto chain.", force_parser
            )

    request = ParseRequest(
        filename=filename,
        content=content,
        version_handling=version_handling,
        force_parser=forced,
    )

    result = _parser_router.route(request)
    if not result.success or result.document is None:
        return result

    try:
        doc    = _version_handler.apply(result.document, strategy=version_handling)
        result = ParseResult(success=True, document=doc)
    except Exception as exc:
        logger.exception("Versioning failed for '%s': %s", filename, exc)
        result = ParseResult(
            success=False,
            error=f"Versioning failed: {exc}",
            http_status_code=500,
        )

    return result


# ─── Phase 2: Chunk ───────────────────────────────────────────────────────────

def chunk_document(
    parsed_doc: "ParsedDocument",     # noqa: F821  (resolved at runtime)
    config:     ChunkingConfig | None = None,
) -> ChunkingResult:
    """
    Run Phase 2 chunking on an already-parsed document.

    Args:
        parsed_doc: Output of Phase 1 (ingest_document).
        config:     Optional ChunkingConfig.  When None, strategy is
                    auto-selected and all other params use defaults.

    Returns:
        ChunkingResult with deduplicated text + table chunks.
    """
    from backend.ingestion.chunking_pipeline import chunking_pipeline
    return chunking_pipeline.chunk_document(parsed_doc, config)


# ─── Combined Phase 1 + 2 convenience function ───────────────────────────────

def ingest_and_chunk(
    filename:         str,
    content:          bytes,
    chunking_config:  ChunkingConfig | None = None,
    version_handling: str               = "replace",
    force_parser:     str | None        = None,
) -> tuple[ParseResult, ChunkingResult | None]:
    """
    Run Phase 1 (parse) immediately followed by Phase 2 (chunk).

    Args:
        filename:         Original filename.
        content:          Raw file bytes.
        chunking_config:  Optional Phase 2 config (auto-selected if None).
        version_handling: Phase 1 version strategy.
        force_parser:     Phase 1 parser override.

    Returns:
        (parse_result, chunking_result)
        chunking_result is None when parsing fails.
    """
    parse_result = ingest_document(
        filename=filename,
        content=content,
        version_handling=version_handling,
        force_parser=force_parser,
    )

    if not parse_result.success or parse_result.document is None:
        return parse_result, None

    chunking_result = chunk_document(parse_result.document, chunking_config)
    return parse_result, chunking_result


# ─── Phase 3: Embed ───────────────────────────────────────────────────────────

def embed_chunks(
    chunks:         list,        # list[Chunk]
    doc_id:         str = "",
    doc_name:       str = "",
    pipeline=None,               # EmbeddingPipeline | None
    known_hashes:   set | None  = None,
) -> "EmbeddingResult":
    """
    Run Phase 3 embedding on a list of chunks.

    Args:
        chunks:       Output of chunk_document() (ChunkingResult.all_chunks).
        doc_id:       Source document ID.
        doc_name:     Source filename.
        pipeline:     EmbeddingPipeline instance.  When None, one is built
                      from backend.config.settings (requires OPENAI_API_KEY).
        known_hashes: Content hashes already in the vector store.

    Returns:
        EmbeddingResult with EmbeddedChunk objects ready for Pinecone.
    """
    from backend.ingestion.embedders.embedding_pipeline import build_embedding_pipeline
    emb_pipeline = pipeline or build_embedding_pipeline()
    return emb_pipeline.embed_chunks(
        chunks=chunks,
        doc_id=doc_id,
        doc_name=doc_name,
        known_hashes=known_hashes,
    )


# ─── Combined Phase 1 + 2 + 3 convenience function ───────────────────────────

def ingest_chunk_embed(
    filename:            str,
    content:             bytes,
    chunking_config=None,        # ChunkingConfig | None
    embedding_pipeline=None,     # EmbeddingPipeline | None
    version_handling:    str = "replace",
    force_parser:        str | None = None,
    known_hashes:        set | None = None,
) -> tuple["ParseResult", "ChunkingResult | None", "EmbeddingResult | None"]:
    """
    Run the complete Phase 1 → 2 → 3 ingestion pipeline in one call.

    Returns:
        (parse_result, chunking_result, embedding_result)
        chunking_result and embedding_result are None when the preceding
        phase fails.
    """
    parse_result = ingest_document(
        filename=filename,
        content=content,
        version_handling=version_handling,
        force_parser=force_parser,
    )
    if not parse_result.success or parse_result.document is None:
        return parse_result, None, None

    chunking_result = chunk_document(parse_result.document, chunking_config)

    if not chunking_result.all_chunks:
        return parse_result, chunking_result, None

    embedding_result = embed_chunks(
        chunks=chunking_result.all_chunks,
        doc_id=parse_result.document.doc_id,
        doc_name=parse_result.document.filename,
        pipeline=embedding_pipeline,
        known_hashes=known_hashes,
    )
    return parse_result, chunking_result, embedding_result

from backend.models.parsed_document import ParsedDocument  # noqa: E402

# ─── Phase 4: Store ───────────────────────────────────────────────────────────

def store_embeddings(
    embedding_result,               # EmbeddingResult
    tenant_id:   str        = "default",
    corpus:      str        = "default",
    version:     int        = 1,
    source_url:  str | None = None,
    vs_pipeline=None,               # VectorStorePipeline | None
):
    """
    Run Phase 4 — upsert EmbeddedChunks into Pinecone.

    Args:
        embedding_result: EmbeddingResult from Phase 3.
        tenant_id:        Tenant identifier (namespace component).
        corpus:           Corpus name (namespace component).
        version:          Document version integer (vector ID component).
        source_url:       Optional origin URL stored in Pinecone metadata.
        vs_pipeline:      VectorStorePipeline instance.  When None, one is
                          built from backend.config.settings.

    Returns:
        VectorStoreResult with upserted_count and timing.
    """
    from backend.vector_store.vector_store_pipeline import build_vector_store_pipeline
    pipeline = vs_pipeline or build_vector_store_pipeline()
    result = pipeline.store_embedding_result(
        result=embedding_result,
        tenant_id=tenant_id,
        corpus=corpus,
        version=version,
        source_url=source_url,
    )

    # ── Register chunks in the chunk-level corpus store ───────────────────────
    # BM25 must be built from the SAME chunks that were embedded (dense +
    # SPLADE) and upserted to Pinecone, so all three retrieval arms agree on
    # what a single "hit" is (see chunk_corpus_store.py docstring).
    try:
        from backend.ingestion.chunk_corpus_store import chunk_corpus_store
        chunk_corpus_store.add_chunks(embedding_result.embedded_chunks)
    except Exception as _corpus_exc:
        logger.warning("Chunk corpus store update failed: %s", _corpus_exc)

    # ── Post-store: rebuild BM25 index so SparseRetriever stays current ───────
    # This runs in the same thread (ingestion worker) so it's non-blocking
    # for request handlers.  Failures are logged but don't abort the pipeline.
    try:
        _rebuild_bm25_index()
    except Exception as _bm25_exc:
        logger.warning("BM25 index rebuild failed (sparse retrieval may be stale): %s", _bm25_exc)

    return result


def _rebuild_bm25_index() -> None:
    """
    Rebuild the BM25 sparse index from the chunk-level corpus store.

    Called automatically after every successful Phase 4 store so that
    SparseRetriever always has an up-to-date, chunk-aligned corpus (i.e.
    the same chunk IDs used by dense + SPLADE retrieval against Pinecone).
    """
    from backend.ingestion.chunk_corpus_store import chunk_corpus_store
    from backend.retrieval.sparse_retriever import SparseRetriever, SparseRetrieverConfig, IndexDocument

    entries = chunk_corpus_store.all()
    if not entries:
        logger.debug("BM25 rebuild skipped — chunk corpus store is empty.")
        return

    corpus = [
        IndexDocument(id=chunk_id, text=v["text"], metadata=v["metadata"])
        for chunk_id, v in entries
        if v["text"]
    ]
    if not corpus:
        logger.debug("BM25 rebuild skipped — no chunks with extractable text.")
        return

    retriever = SparseRetriever(SparseRetrieverConfig())
    retriever.build_index(corpus)
    logger.info("BM25 index rebuilt with %d chunk(s).", len(corpus))


# ─── Full Phase 1+2+3+4 pipeline ─────────────────────────────────────────────

from dataclasses import dataclass  # noqa: E402


@dataclass
class FullPipelineResult:
    """Aggregated result from all four ingestion phases."""
    filename:         str
    success:          bool
    error:            str | None           = None
    parse_result:     ParseResult | None   = None
    chunking_result:  ChunkingResult | None = None
    embedding_result: EmbeddingResult | None = None
    store_result:     "VectorStoreResult | None" = None

    def summary(self) -> dict:
        doc_id = (self.parse_result.document.doc_id
                  if self.parse_result and self.parse_result.document else None)
        return {
            "filename":      self.filename,
            "success":       self.success,
            "error":         self.error,
            "doc_id":        doc_id,
            "chunks":        self.chunking_result.total_chunks if self.chunking_result else 0,
            "embedded":      self.embedding_result.total_embedded if self.embedding_result else 0,
            "upserted":      self.store_result.upserted_count if self.store_result else 0,
            "namespace":     self.store_result.namespace if self.store_result else None,
        }


def ingest_full_pipeline(
    filename:            str,
    content:             bytes,
    chunking_config=None,
    embedding_pipeline=None,
    vs_pipeline=None,
    version_handling:    str        = "replace",
    force_parser:        str | None = None,
    tenant_id:           str        = "default",
    corpus:              str        = "default",
    version:             int        = 1,
    source_url:          str | None = None,
    known_hashes:        set | None = None,
) -> FullPipelineResult:
    """
    Run the complete Phase 1 → 2 → 3 → 4 ingestion pipeline in one call.

    Args:
        filename:         Original filename.
        content:          Raw file bytes.
        chunking_config:  Phase 2 config (auto-selected if None).
        embedding_pipeline: Phase 3 EmbeddingPipeline (built from settings if None).
        vs_pipeline:      Phase 4 VectorStorePipeline (built from settings if None).
        version_handling: Phase 1 version strategy.
        force_parser:     Phase 1 parser override.
        tenant_id:        Pinecone namespace tenant component.
        corpus:           Pinecone namespace corpus component.
        version:          Document version integer for vector IDs.
        source_url:       Optional origin URL for metadata.
        known_hashes:     Phase 3 cross-doc deduplication hashes.

    Returns:
        FullPipelineResult with per-phase results and summary().
    """
    # Phase 1
    parse_result = ingest_document(
        filename=filename, content=content,
        version_handling=version_handling, force_parser=force_parser,
    )
    if not parse_result.success or parse_result.document is None:
        return FullPipelineResult(
            filename=filename, success=False,
            error=parse_result.error, parse_result=parse_result,
        )

    # Phase 2
    chunking_result = chunk_document(parse_result.document, chunking_config)
    if not chunking_result.all_chunks:
        return FullPipelineResult(
            filename=filename, success=False,
            error="No chunks produced after Phase 2.",
            parse_result=parse_result, chunking_result=chunking_result,
        )

    # Phase 3
    embedding_result = embed_chunks(
        chunks=chunking_result.all_chunks,
        doc_id=parse_result.document.doc_id,
        doc_name=parse_result.document.filename,
        pipeline=embedding_pipeline,
        known_hashes=known_hashes,
    )
    if not embedding_result.embedded_chunks:
        return FullPipelineResult(
            filename=filename, success=False,
            error="No embedded chunks produced after Phase 3.",
            parse_result=parse_result, chunking_result=chunking_result,
            embedding_result=embedding_result,
        )

    # Phase 4
    store_result = store_embeddings(
        embedding_result=embedding_result,
        tenant_id=tenant_id, corpus=corpus,
        version=version, source_url=source_url,
        vs_pipeline=vs_pipeline,
    )

    return FullPipelineResult(
        filename=filename,
        success=store_result.upserted_count > 0,
        parse_result=parse_result,
        chunking_result=chunking_result,
        embedding_result=embedding_result,
        store_result=store_result,
    )


__all__ = [
    "ingest_document",
    "chunk_document",
    "embed_chunks",
    "store_embeddings",
    "ingest_and_chunk",
    "ingest_chunk_embed",
    "ingest_full_pipeline",
    "FullPipelineResult",
    "ParsedDocument",
    "ParseResult",
    "ChunkingConfig",
    "ChunkingResult",
    "EmbeddingResult",
]
