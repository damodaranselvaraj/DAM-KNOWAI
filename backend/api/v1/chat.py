"""
Chat endpoints  —  /api/v1/chat
================================
Every request goes through the same five-stage pipeline:

    Stage 1  Retrieval          dense | hybrid_bm25 (Dense+BM25 RRF) | hybrid_splade (native Pinecone hybrid)
    Stage 2  Metadata Filter    ACL / business-rule pass-through (extensible)
    Stage 3  Reranking          CrossEncoder listwise reranker (optional)
    Stage 4  Prompt Build       System prompt + numbered context chunks
    Stage 5  LLM Generation     GPT-4o  —  streaming (SSE) or blocking (JSON)

Session memory (conversation history) is stored via the injected
BaseMemorySaver (SQLite by default, InMemory in dev/test).  Each session
checkpoint holds the ordered list of ChatMessage dicts so the LLM sees
the full prior conversation on every turn.

Endpoints
─────────
POST  /chat/session              create session → { session_id }
POST  /chat/query                blocking JSON response
GET   /chat/stream/{session_id}  SSE stream  ?query=...&top_k=...
GET   /chat/history/{session_id} all turns for a session
DELETE/chat/session/{session_id} wipe session memory
GET   /chat/sessions             list all sessions (admin)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.dependencies import memory_saver
from backend.memory.base import BaseMemorySaver
from backend.retrieval.query_validator import (
    FilterValidationError,
    QueryValidationError,
    sanitize_query,
    validate_filters,
)
from backend.models.chat import (
    ChatMessage,
    ChatQuery,
    ChatResponse,
    ChatSession,
    Citation,
    RagContext,
    SessionCreatedResponse,
    StreamCitationsEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamTokenEvent,
)
from backend.api.v1.guardrails import get_current_config, record_audit
from backend.guardrails.pipeline import (
    AuditAccumulator,
    run_context_guardrails,
    run_post_generation_guardrails,
    run_pre_retrieval_guardrails,
)
from backend.guardrails.retrieval_safety import resolve_top_n

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


class GuardrailBlocked(Exception):
    """
    Raised internally when any guardrail layer blocks a request.

    Carries the fixed, user-facing refusal message and the audit
    accumulator so both the blocking (JSON) and streaming (SSE) endpoints
    can surface the same behaviour: return the safe message, skip the
    LLM call entirely, and still emit a complete audit record.
    """

    def __init__(self, message: str, audit: "AuditAccumulator"):
        super().__init__(message)
        self.message = message
        self.audit = audit


# ─── Lazy singletons (constructed once, reused across requests) ───────────────
# We construct these lazily so the app still starts even if Pinecone / BM25
# are not configured (useful for partial deployments / unit tests).

_hybrid_retriever  = None
_reranker          = None
_prompt_builder    = None
_llm_client        = None


def _get_hybrid_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from backend.retrieval.dense_retriever import DenseRetrieverConfig, DenseRetriever
        from backend.retrieval.sparse_retriever import (
            IndexNotFoundError,
            SparseRetrieverConfig,
            SparseRetriever,
        )
        from backend.retrieval.hybrid_retriever import HybridRetrieverConfig, HybridRetriever
        dense   = DenseRetriever(DenseRetrieverConfig())
        sparse  = SparseRetriever(SparseRetrieverConfig())
        try:
            # Load the persisted BM25 corpus (built from the same chunks
            # that were embedded/upserted — see chunk_corpus_store.py) so
            # "hybrid_bm25" mode actually has an index to score against.
            # Without this the SparseRetriever silently has no index and
            # every query fails with IndexNotFoundError, degrading to
            # dense-only.
            sparse.load_index()
        except IndexNotFoundError:
            logger.warning(
                "BM25 corpus not found — sparse retrieval will be empty "
                "until documents are ingested via the pipeline."
            )
        _hybrid_retriever = HybridRetriever(HybridRetrieverConfig(), dense, sparse)
    return _hybrid_retriever


def _get_reranker():
    """
    Return the reranker singleton.
    Prefers the instance pre-warmed at startup (stored in app.state).
    Falls back to lazy construction if somehow not yet initialised.
    """
    global _reranker
    if _reranker is None:
        try:
            from backend.retrieval.reranker import RerankerConfig, Reranker
            _reranker = Reranker(RerankerConfig())
        except Exception as exc:
            logger.warning("Reranker construction failed: %s", exc)
    return _reranker


def _set_reranker_from_app_state(reranker) -> None:
    """Called from main.py lifespan after pre-warming so requests skip lazy init."""
    global _reranker
    _reranker = reranker


def _get_prompt_builder():
    global _prompt_builder
    if _prompt_builder is None:
        from backend.llm.prompt_builder import PromptBuilder
        _prompt_builder = PromptBuilder()
    return _prompt_builder


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from backend.llm.openai_client import OpenAIClient
        _llm_client = OpenAIClient()
    return _llm_client


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_session_config(session_id: str) -> Dict[str, Any]:
    """LangGraph-style config dict used by the memory saver."""
    return {"configurable": {"thread_id": session_id}}


def _load_history(saver: BaseMemorySaver, session_id: str) -> List[Dict[str, str]]:
    """
    Load prior conversation turns from the memory saver.

    Returns a list of  ``{"role": ..., "content": ...}``  dicts suitable
    for direct injection into the OpenAI messages list.
    """
    cfg   = _make_session_config(session_id)
    state = saver.get(cfg)
    if not state:
        return []
    messages = state.get("messages", [])
    # Strip citation / metadata fields — LLM only needs role + content
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]


def _save_turn(
    saver: BaseMemorySaver,
    session_id: str,
    user_msg: ChatMessage,
    assistant_msg: ChatMessage,
) -> None:
    """Append a user + assistant turn to the memory saver checkpoint."""
    cfg   = _make_session_config(session_id)
    state = saver.get(cfg) or {"messages": []}
    state["messages"].append(user_msg.model_dump())
    state["messages"].append(assistant_msg.model_dump())
    saver.put(cfg, state, {"updated_at": _now()})


def _apply_output_safety_only(answer: str, guardrail_config):
    """Layer 6 only — used for direct (greeting/casual) responses that skip Layer 5."""
    from backend.guardrails.output_safety import apply_output_safety
    return apply_output_safety(
        answer,
        enable_pii_masking=guardrail_config.toggles.enable_pii_masking_output,
        enable_toxicity_check=guardrail_config.toggles.enable_output_toxicity_check,
    )


def _build_citations(citation_meta: List[Dict]) -> List[Citation]:
    """Convert PromptBuilder citation_meta dicts into Citation objects."""
    citations: List[Citation] = []
    for cm in citation_meta:
        meta = cm.get("metadata", {})
        citations.append(
            Citation(
                source_n    = cm["n"],
                chunk_id    = cm.get("chunk_id", ""),
                doc_id      = meta.get("doc_id", ""),
                doc_name    = cm.get("doc_name", "Unknown"),
                page_number = str(cm.get("pages") or meta.get("page", "—")),
                chunk_index = str(meta.get("chunk_index", "?")),
                score       = round(float(cm.get("score", 0.0)), 4),
                snippet     = cm.get("snippet", "")[:200],
                metadata    = meta,
            )
        )
    return citations


# ─── Core pipeline (shared by both blocking and streaming paths) ──────────────

class _PipelineResult:
    """Internal holder for pipeline outputs before LLM generation."""
    __slots__ = (
        "messages", "citation_meta", "rag_context",
        "query_id", "retrieval_ms", "rerank_ms", "sanitized_query",
        "context_chunks", "audit", "direct_response",
    )
    def __init__(self, messages, citation_meta, rag_context,
                 query_id, retrieval_ms, rerank_ms, sanitized_query,
                 context_chunks, audit, direct_response=False):
        self.messages        = messages
        self.citation_meta   = citation_meta
        self.rag_context     = rag_context
        self.query_id        = query_id
        self.retrieval_ms    = retrieval_ms
        self.rerank_ms       = rerank_ms
        self.sanitized_query = sanitized_query
        self.context_chunks  = context_chunks
        self.audit           = audit
        # True for greeting/casual turns routed straight to the LLM with
        # no retrieval — used so the caller skips Layer 5 groundedness/
        # hallucination checks (there's no context to ground against).
        self.direct_response = direct_response


def _run_pipeline(
    payload:  ChatQuery,
    saver:    BaseMemorySaver,
    query_id: str,
) -> _PipelineResult:
    """
    Run Layer 1-2 guardrails, stages 1-4 (retrieval → rerank → context
    control → prompt build). Fully synchronous — called from both the
    blocking and async-stream paths.

    Raises ``GuardrailBlocked`` when any guardrail layer blocks the
    request; the caller must catch this and return the carried message
    without calling the LLM.
    """
    guardrail_config = get_current_config()
    audit = AuditAccumulator()

    # ── Stage 0 — Query & filter validation ────────────────────────────────────
    # Raw user input must never flow unvalidated into the OpenAI embedding
    # call, the local BM25 index, or Pinecone's filter= — see
    # backend.retrieval.query_validator for the sanitisation / allow-list
    # rules enforced here.
    try:
        pre_sanitized_query = sanitize_query(payload.query)
    except QueryValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        validated_filters = validate_filters(payload.filters)
    except FilterValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── Layer 1 + 2 — Input guardrails, intent classification / routing ───────
    pre_retrieval = run_pre_retrieval_guardrails(
        pre_sanitized_query, guardrail_config, audit,
    )
    if pre_retrieval.blocked:
        raise GuardrailBlocked(pre_retrieval.user_message, audit)

    sanitized_query = pre_retrieval.sanitized_query

    # Greeting / casual — direct LLM response, no retrieval at all.
    if not pre_retrieval.route_to_rag:
        history  = _load_history(saver, payload.session_id)
        messages = history + [{"role": "user", "content": sanitized_query}]
        rag_context = RagContext(
            retrieved_count=0, after_filter_count=0, after_rerank_count=0,
            context_chunks_used=0,
        )
        return _PipelineResult(
            messages=messages, citation_meta=[], rag_context=rag_context,
            query_id=query_id, retrieval_ms=0.0, rerank_ms=0.0,
            sanitized_query=sanitized_query, context_chunks=[], audit=audit,
            direct_response=True,
        )

    # ── Stage 1 — Retrieval (dense | hybrid_bm25 | hybrid_splade) ─────────────
    # `hybrid_search=False` is kept as a backward-compatible override that
    # forces dense-only retrieval regardless of `retrieval_mode`.
    effective_mode = payload.retrieval_mode if payload.hybrid_search else "dense"

    # Layer 3 — Retrieval Safety: Top-N before rerank is sourced from the
    # guardrail config (default 20) rather than the ad-hoc `top_k * 5`
    # heuristic, when hybrid retrieval / metadata filtering is enabled.
    top_n = resolve_top_n(guardrail_config) if guardrail_config.toggles.enable_hybrid_retrieval else payload.top_k
    over_fetch = max(top_n, payload.top_k)

    t_ret = time.perf_counter()
    retrieved = []
    retrieval_failed = False
    retrieval_error: str | None = None
    try:
        retriever = _get_hybrid_retriever()
        retrieved = retriever.retrieve(
            query           = sanitized_query,
            top_k           = over_fetch,
            namespace       = payload.namespace,
            metadata_filter = validated_filters,
            query_id        = query_id,
            mode            = effective_mode,
        )
    except Exception as exc:
        logger.warning("Retrieval failed, continuing with empty context: %s", exc)
        # Distinguish "retrieval ran and found nothing" from "retrieval
        # errored out" — both otherwise present identically to the caller
        # as retrieved_count == 0.
        retrieval_failed = True
        retrieval_error = str(exc)
    retrieval_ms = (time.perf_counter() - t_ret) * 1_000

    retrieved_count     = len(retrieved)
    after_filter_count  = retrieved_count   # no separate filter stage here
    after_rerank_count  = retrieved_count

    # ── Stage 2 — Deduplication (lightweight, inline id-based pass) ───────────
    # Coarse id-based de-dup runs here; the finer-grained content/semantic
    # de-dup (Layer 4.B) runs after reranking below.
    seen: set[str] = set()
    deduped = []
    for r in retrieved:
        if r.id not in seen:
            seen.add(r.id)
            deduped.append(r)

    # ── Stage 3 — Reranking ───────────────────────────────────────────────────
    t_rerank = time.perf_counter()
    reranked = deduped
    if payload.reranker_enabled and deduped:
        try:
            reranked = _get_reranker().rerank(
                query      = sanitized_query,
                candidates = deduped,
                top_k      = payload.top_k,
                query_id   = query_id,
            )
            after_rerank_count = len(reranked)
        except Exception as exc:
            logger.warning("Reranker failed, using raw retrieval order: %s", exc)
            reranked = deduped[: payload.top_k]
            after_rerank_count = len(reranked)
    else:
        reranked = deduped[: payload.top_k]
        after_rerank_count = len(reranked)
    rerank_ms = (time.perf_counter() - t_rerank) * 1_000

    # ── Layer 4 — Context Control (relevance gate → dedup → relevance score) ──
    context_result = run_context_guardrails(
        query=sanitized_query,
        retrieved_chunks=retrieved,
        reranked_chunks=reranked,
        config=guardrail_config,
        audit=audit,
    )
    if context_result.blocked:
        raise GuardrailBlocked(context_result.user_message, audit)

    context_chunks = context_result.chunks

    # ── Stage 4 — Prompt build ────────────────────────────────────────────────
    history  = _load_history(saver, payload.session_id)
    messages, citation_meta = _get_prompt_builder().build(
        query   = sanitized_query,
        chunks  = context_chunks,
        history = history,
    )

    rag_context = RagContext(
        retrieved_count    = retrieved_count,
        after_filter_count = after_filter_count,
        after_rerank_count = after_rerank_count,
        context_chunks_used= len(citation_meta),
        retrieval_ms       = round(retrieval_ms, 1),
        rerank_ms          = round(rerank_ms, 1),
        retrieval_failed   = retrieval_failed,
        retrieval_error    = retrieval_error,
    )

    return _PipelineResult(
        messages        = messages,
        citation_meta   = citation_meta,
        rag_context     = rag_context,
        query_id        = query_id,
        retrieval_ms    = retrieval_ms,
        rerank_ms       = rerank_ms,
        sanitized_query = sanitized_query,
        context_chunks  = context_chunks,
        audit           = audit,
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/session",
    response_model=SessionCreatedResponse,
    summary="Create a new chat session",
)
async def create_session(
    saver: BaseMemorySaver = Depends(memory_saver),
):
    """
    Allocates a new session ID and initialises an empty checkpoint in the
    memory saver.  Returns the ``session_id`` the client must send with
    every subsequent query.
    """
    session_id = str(uuid.uuid4())
    created_at = _now()
    cfg = _make_session_config(session_id)
    saver.put(cfg, {"messages": [], "created_at": created_at}, {"created_at": created_at})

    backend_name = type(saver).__name__.replace("Saver", "").lower()
    logger.info(json.dumps({"event": "session_created", "session_id": session_id}))
    return SessionCreatedResponse(
        session_id=session_id,
        created_at=created_at,
        backend=backend_name,
    )


@router.post(
    "/query",
    response_model=ChatResponse,
    summary="RAG query — blocking JSON response",
)
async def query_rag(
    payload: ChatQuery,
    saver:   BaseMemorySaver = Depends(memory_saver),
):
    """
    Full RAG pipeline returning a single JSON response.

    Set ``stream: true`` in the request body to be redirected to the SSE
    endpoint instead (or call GET /chat/stream/{session_id} directly).

    Pipeline:
        Hybrid Retrieval → Reranker → Prompt Build → GPT-4o → Citations
    """
    if payload.stream:
        # Convenience: tell the client to use the SSE endpoint
        raise HTTPException(
            status_code=400,
            detail=(
                "Set stream=false for JSON responses, or use "
                "GET /api/v1/chat/stream/{session_id}?query=... for SSE streaming."
            ),
        )

    query_id = str(uuid.uuid4())
    t_total  = time.perf_counter()

    # ── Layer 1-4 (input guardrails, routing, retrieval, rerank, context) ─────
    try:
        pipe = _run_pipeline(payload, saver, query_id)
    except GuardrailBlocked as exc:
        record_audit(exc.audit.finalize())
        return ChatResponse(
            session_id  = payload.session_id,
            query_id    = query_id,
            query       = "",
            answer      = exc.message,
            citations   = [],
            timestamp   = _now(),
            model_used  = "guardrail",
            rag_context = RagContext(),
        )

    # ── Stage 5 — LLM generation (blocking) ──────────────────────────────────
    t_llm = time.perf_counter()
    try:
        result = _get_llm_client().complete(
            messages    = pipe.messages,
            temperature = payload.temperature,
            max_tokens  = payload.max_tokens,
        )
        answer    = result.text
        llm_ms    = (time.perf_counter() - t_llm) * 1_000
        usage     = result.usage
    except Exception as exc:
        logger.exception("LLM completion failed for query_id=%s", query_id)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    # ── Layer 5-6 — Generation safety + output safety ─────────────────────────
    guardrail_config = get_current_config()
    if not pipe.direct_response:
        post_gen = run_post_generation_guardrails(
            answer          = answer,
            context_chunks  = pipe.context_chunks,
            citation_meta   = pipe.citation_meta,
            config          = guardrail_config,
            audit           = pipe.audit,
        )
        if post_gen.blocked:
            record_audit(pipe.audit.finalize())
            return ChatResponse(
                session_id  = payload.session_id,
                query_id    = query_id,
                query       = pipe.sanitized_query,
                answer      = post_gen.user_message,
                citations   = [],
                timestamp   = _now(),
                model_used  = "guardrail",
                rag_context = pipe.rag_context,
            )
        answer = post_gen.safe_answer
    else:
        # Direct (greeting/casual) responses still pass through output
        # safety (Layer 6) even though generation safety (Layer 5) is
        # skipped — there's no retrieved context to ground against.
        output_result = _apply_output_safety_only(answer, guardrail_config)
        if output_result.blocked:
            record_audit(pipe.audit.finalize())
            return ChatResponse(
                session_id  = payload.session_id,
                query_id    = query_id,
                query       = pipe.sanitized_query,
                answer      = output_result.user_message,
                citations   = [],
                timestamp   = _now(),
                model_used  = "guardrail",
                rag_context = pipe.rag_context,
            )
        answer = output_result.safe_text
        pipe.audit.pii_types_detected_output.extend(output_result.pii_types_found)

    total_ms   = (time.perf_counter() - t_total) * 1_000
    citations  = _build_citations(pipe.citation_meta)

    # Finalise RagContext with LLM timings
    pipe.rag_context.llm_ms             = round(llm_ms, 1)
    pipe.rag_context.total_ms           = round(total_ms, 1)
    pipe.rag_context.prompt_tokens      = usage.prompt_tokens
    pipe.rag_context.completion_tokens  = usage.completion_tokens
    pipe.rag_context.total_tokens       = usage.total_tokens

    # ── Persist turn ──────────────────────────────────────────────────────────
    timestamp = _now()
    _save_turn(
        saver,
        payload.session_id,
        ChatMessage(role="user",      content=pipe.sanitized_query, timestamp=timestamp, query_id=query_id),
        ChatMessage(role="assistant", content=answer,               timestamp=timestamp, query_id=query_id, citations=citations),
    )

    record_audit(pipe.audit.finalize())

    logger.info(
        json.dumps({
            "event":       "rag_query_complete",
            "query_id":    query_id,
            "session_id":  payload.session_id,
            "total_ms":    round(total_ms, 1),
            "tokens_used": usage.total_tokens,
            "citations":   len(citations),
        })
    )

    return ChatResponse(
        session_id  = payload.session_id,
        query_id    = query_id,
        query       = pipe.sanitized_query,
        answer      = answer,
        citations   = citations,
        timestamp   = timestamp,
        model_used  = result.model,
        rag_context = pipe.rag_context,
    )


@router.get(
    "/stream/{session_id}",
    summary="RAG query — SSE streaming response",
)
async def stream_rag(
    session_id:       str,
    request:          Request,
    query:            str  = Query(..., min_length=1, max_length=4_000),
    top_k:            int  = Query(10,  ge=1, le=50),
    temperature:      float= Query(0.2, ge=0.0, le=2.0),
    max_tokens:       int  = Query(1_024, ge=64, le=8_191),
    hybrid_search:    bool = Query(True),
    reranker_enabled: bool = Query(True),
    retrieval_mode:   str  = Query("hybrid_bm25", pattern="^(dense|hybrid_bm25|hybrid_splade)$"),
    namespace:        Optional[str] = Query(None),
    saver:            BaseMemorySaver = Depends(memory_saver),
):
    """
    Stream the RAG answer token-by-token via Server-Sent Events.

    SSE event types (all JSON-encoded in the ``data:`` field):
    ┌──────────────┬────────────────────────────────────────────────────┐
    │ type         │ payload                                            │
    ├──────────────┼────────────────────────────────────────────────────┤
    │ token        │ { "type": "token",     "token": "<text delta>" }   │
    │ citations    │ { "type": "citations", "citations": [...] }        │
    │ done         │ { "type": "done",      "answer": "...", ... }      │
    │ error        │ { "type": "error",     "message": "..." }          │
    └──────────────┴────────────────────────────────────────────────────┘

    The client should concatenate all ``token`` events to build the answer,
    then use the ``done`` event to display citations and latency stats.
    """
    query_id = str(uuid.uuid4())

    # Build a synthetic ChatQuery for the shared pipeline
    payload = ChatQuery(
        session_id       = session_id,
        query            = query,
        top_k            = top_k,
        temperature      = temperature,
        max_tokens       = max_tokens,
        hybrid_search    = hybrid_search,
        reranker_enabled = reranker_enabled,
        retrieval_mode   = retrieval_mode,
        namespace        = namespace,
        stream           = True,
    )

    async def event_generator() -> AsyncIterator[str]:
        t_total   = time.perf_counter()
        answer_parts: list[str] = []

        # ── Stages 1–4 (sync, run in thread pool via run_in_executor) ─────────
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            pipe = await loop.run_in_executor(
                None, _run_pipeline, payload, saver, query_id
            )
        except GuardrailBlocked as exc:
            record_audit(exc.audit.finalize())
            done_evt = StreamDoneEvent(
                query_id=query_id, answer=exc.message, citations=[],
                rag_context=RagContext(),
            )
            yield f"data: {done_evt.model_dump_json()}\n\n"
            return
        except HTTPException as exc:
            # Query/filter validation failures (Stage 0) — 400-equivalent.
            err = StreamErrorEvent(message="Invalid request", detail=str(exc.detail))
            yield f"data: {err.model_dump_json()}\n\n"
            return
        except Exception as exc:
            logger.exception("Pipeline failed for query_id=%s", query_id)
            err = StreamErrorEvent(message="Retrieval pipeline error", detail=str(exc))
            yield f"data: {err.model_dump_json()}\n\n"
            return

        # ── Stage 5 — LLM streaming ───────────────────────────────────────────
        t_llm  = time.perf_counter()
        try:
            async for token_text in _get_llm_client().stream(
                messages    = pipe.messages,
                temperature = payload.temperature,
                max_tokens  = payload.max_tokens,
            ):
                if await request.is_disconnected():
                    logger.info("Client disconnected mid-stream, query_id=%s", query_id)
                    return
                answer_parts.append(token_text)
                evt = StreamTokenEvent(token=token_text)
                yield f"data: {evt.model_dump_json()}\n\n"

        except Exception as exc:
            logger.exception("LLM stream error for query_id=%s", query_id)
            err = StreamErrorEvent(message="LLM streaming error", detail=str(exc))
            yield f"data: {err.model_dump_json()}\n\n"
            return

        llm_ms    = (time.perf_counter() - t_llm)   * 1_000
        total_ms  = (time.perf_counter() - t_total) * 1_000
        answer    = "".join(answer_parts)

        # ── Layer 5-6 — Generation safety + output safety ─────────────────────
        # Tokens already streamed to the client are best-effort; the `done`
        # event (and the persisted turn) reflect the guardrail-checked
        # final answer, which is authoritative.
        guardrail_config = get_current_config()
        if not pipe.direct_response:
            post_gen = run_post_generation_guardrails(
                answer=answer, context_chunks=pipe.context_chunks,
                citation_meta=pipe.citation_meta, config=guardrail_config,
                audit=pipe.audit,
            )
            if post_gen.blocked:
                record_audit(pipe.audit.finalize())
                done_evt = StreamDoneEvent(
                    query_id=query_id, answer=post_gen.user_message,
                    citations=[], rag_context=pipe.rag_context,
                )
                yield f"data: {done_evt.model_dump_json()}\n\n"
                return
            answer = post_gen.safe_answer
        else:
            output_result = _apply_output_safety_only(answer, guardrail_config)
            if output_result.blocked:
                record_audit(pipe.audit.finalize())
                done_evt = StreamDoneEvent(
                    query_id=query_id, answer=output_result.user_message,
                    citations=[], rag_context=pipe.rag_context,
                )
                yield f"data: {done_evt.model_dump_json()}\n\n"
                return
            answer = output_result.safe_text
            pipe.audit.pii_types_detected_output.extend(output_result.pii_types_found)

        citations = _build_citations(pipe.citation_meta)

        # ── Emit citations ────────────────────────────────────────────────────
        cit_evt = StreamCitationsEvent(citations=citations)
        yield f"data: {cit_evt.model_dump_json()}\n\n"

        # ── Finalise RagContext ───────────────────────────────────────────────
        pipe.rag_context.llm_ms   = round(llm_ms, 1)
        pipe.rag_context.total_ms = round(total_ms, 1)

        # ── Emit done ─────────────────────────────────────────────────────────
        done_evt = StreamDoneEvent(
            query_id    = query_id,
            answer      = answer,
            citations   = citations,
            rag_context = pipe.rag_context,
        )
        yield f"data: {done_evt.model_dump_json()}\n\n"
        record_audit(pipe.audit.finalize())

        # ── Persist turn ──────────────────────────────────────────────────────
        timestamp = _now()
        try:
            _save_turn(
                saver, session_id,
                ChatMessage(role="user",      content=pipe.sanitized_query, timestamp=timestamp, query_id=query_id),
                ChatMessage(role="assistant", content=answer,               timestamp=timestamp, query_id=query_id, citations=citations),
            )
        except Exception as exc:
            logger.warning("Failed to persist turn for query_id=%s: %s", query_id, exc)

        logger.info(
            json.dumps({
                "event":      "rag_stream_complete",
                "query_id":   query_id,
                "session_id": session_id,
                "total_ms":   round(total_ms, 1),
                "citations":  len(citations),
            })
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",        # disable nginx buffering
            "Connection":       "keep-alive",
        },
    )


@router.get(
    "/history/{session_id}",
    response_model=ChatSession,
    summary="Retrieve full conversation history",
)
async def get_chat_history(
    session_id: str,
    saver: BaseMemorySaver = Depends(memory_saver),
):
    """Return all stored turns for *session_id*."""
    cfg   = _make_session_config(session_id)
    state = saver.get(cfg)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    raw_messages = state.get("messages", [])
    messages = []
    for m in raw_messages:
        try:
            messages.append(ChatMessage(**m))
        except Exception:
            pass   # skip malformed records

    return ChatSession(
        session_id = session_id,
        created_at = state.get("created_at", ""),
        messages   = messages,
    )


@router.delete(
    "/session/{session_id}",
    summary="Clear / delete a chat session",
)
async def clear_session(
    session_id: str,
    saver: BaseMemorySaver = Depends(memory_saver),
):
    """
    Delete all checkpoints for *session_id*.

    Returns the number of checkpoint records removed.
    """
    deleted = saver.clear_session(session_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    logger.info(json.dumps({"event": "session_cleared", "session_id": session_id}))
    return {"status": "cleared", "session_id": session_id, "records_deleted": deleted, "timestamp": _now()}


@router.get(
    "/sessions",
    summary="List all sessions (admin)",
)
async def list_sessions(
    limit:  int = Query(50, ge=1, le=500),
    offset: int = Query(0,  ge=0),
    saver:  BaseMemorySaver = Depends(memory_saver),
):
    """
    Return summary info for all stored sessions, sorted by last activity.

    Intended for admin / monitoring dashboards.
    """
    sessions = saver.list_sessions(limit=limit, offset=offset)
    return {
        "sessions": [s.model_dump() for s in sessions],
        "count":    len(sessions),
        "limit":    limit,
        "offset":   offset,
    }
