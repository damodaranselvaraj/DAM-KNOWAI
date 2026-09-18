"""
Pydantic models for chat / RAG queries and responses.

Model hierarchy
───────────────
Request side:
    ChatQuery           — user submits query + pipeline options

Response side:
    Citation            — one cited source chunk
    RagContext          — snapshot of what was retrieved / reranked
    ChatResponse        — full non-streaming response (JSON)

Streaming side:
    StreamEvent         — SSE envelope  { type, data }
    StreamTokenEvent    — token delta
    StreamDoneEvent     — final event carrying full answer + citations

Session / history:
    ChatMessage         — one turn stored in memory saver
    ChatSession         — ordered list of turns
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─── Request ──────────────────────────────────────────────────────────────────

class ChatQuery(BaseModel):
    """Payload posted to POST /api/v1/chat/query."""

    model_config = {"protected_namespaces": ()}

    session_id: str = Field(
        ...,
        description="Conversation session ID.  Create one via POST /chat/session.",
    )
    query: str = Field(..., min_length=1, max_length=4_000)

    # LLM knobs
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(1_024, ge=64, le=8_191)

    # Retrieval knobs
    top_k: int = Field(10, ge=1, le=50, description="Final results after reranking.")
    retrieval_mode: Literal["dense", "hybrid_bm25", "hybrid_splade"] = Field(
        "hybrid_bm25",
        description=(
            "'dense' = Pinecone ANN only. "
            "'hybrid_bm25' = Dense + BM25 (local index) fused via RRF (default). "
            "'hybrid_splade' = single native Pinecone hybrid query using Dense + SPLADE sparse vectors."
        ),
    )
    # Deprecated: kept for backward compatibility with older frontend builds.
    # When explicitly set to False, forces retrieval_mode="dense" regardless
    # of the retrieval_mode field above.
    hybrid_search: bool = True
    reranker_enabled: bool = True

    # Optional pre-filters forwarded to Pinecone
    filters: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

    # Citation granularity
    citation_mode: Literal["sentence", "paragraph"] = "paragraph"

    # When True the /query endpoint returns SSE instead of JSON
    stream: bool = False


# ─── Citation ─────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A single cited source returned with the LLM answer."""

    source_n: int = Field(..., description="[Source N] reference number in answer.")
    chunk_id: str
    doc_id: str = ""
    doc_name: str
    page_number: str = Field("—", description="Page number or range, e.g. '3' or 'pp. 3-5'.")
    chunk_index: str = Field("?", description="0-based position in the document chunk list.")
    score: float = Field(..., description="Rerank score (or retrieval score if no reranker).")
    snippet: str = Field("", description="First 200 chars of the chunk used.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── RAG context snapshot ─────────────────────────────────────────────────────

class RagContext(BaseModel):
    """
    Diagnostic snapshot attached to every response.

    Captures counts and latencies at each pipeline stage so the frontend
    can show a "Sources" drawer and latency breakdown.
    """

    model_config = {"protected_namespaces": ()}

    retrieved_count: int = 0       # after hybrid retrieval
    after_filter_count: int = 0    # after metadata / ACL filter
    after_rerank_count: int = 0    # after reranker
    context_chunks_used: int = 0   # chunks actually sent to the LLM

    # True when retrieval raised an exception and the pipeline fell back to
    # an empty context. Without this flag, retrieved_count == 0 is ambiguous
    # between "the query legitimately had no relevant matches" and
    # "retrieval failed outright" (Pinecone/embedding error) — both look
    # identical to the caller otherwise.
    retrieval_failed: bool = False
    retrieval_error: Optional[str] = None

    # Stage latencies in ms
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0

    # Token accounting
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ─── Non-streaming response ───────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Full JSON response returned by POST /api/v1/chat/query (stream=false)."""

    model_config = {"protected_namespaces": ()}

    session_id: str
    query_id: str
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    timestamp: str
    model_used: str
    rag_context: RagContext = Field(default_factory=RagContext)


# ─── SSE stream events ────────────────────────────────────────────────────────

class StreamTokenEvent(BaseModel):
    """Carries a single text delta from the LLM stream."""
    type: Literal["token"] = "token"
    token: str


class StreamCitationsEvent(BaseModel):
    """Sent once, just before [DONE], carrying all citations."""
    type: Literal["citations"] = "citations"
    citations: List[Citation]


class StreamDoneEvent(BaseModel):
    """
    Final SSE event.

    ``answer`` contains the fully-assembled answer text so the client can
    display it immediately if it missed any token events (e.g. reconnect).
    """
    type: Literal["done"] = "done"
    query_id: str
    answer: str
    citations: List[Citation]
    rag_context: RagContext


class StreamErrorEvent(BaseModel):
    """Sent when the pipeline raises an unrecoverable error mid-stream."""
    type: Literal["error"] = "error"
    message: str
    detail: str = ""


# ─── Session / history ────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """One stored turn in a chat session (persisted via memory saver)."""

    role: Literal["user", "assistant"]
    content: str
    citations: List[Citation] = Field(default_factory=list)
    timestamp: str
    query_id: str = ""


class ChatSession(BaseModel):
    """All turns for a session, returned by GET /chat/history/{session_id}."""

    session_id: str
    created_at: str
    messages: List[ChatMessage] = Field(default_factory=list)


# ─── Session creation response ────────────────────────────────────────────────

class SessionCreatedResponse(BaseModel):
    session_id: str
    created_at: str
    backend: str = Field("", description="Memory backend in use (sqlite | memory).")
