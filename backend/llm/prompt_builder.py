"""
Prompt builder — assembles system + user messages for the RAG chat loop.

Responsibilities
────────────────
1. Render a grounded system prompt that instructs the model to answer
   strictly from the provided context chunks and cite sources.
2. Format each reranked RetrievalResult into a numbered context block
   that carries enough provenance for accurate citations.
3. Inject conversation history (prior turns from the memory saver) so
   the model maintains multi-turn coherence.
4. Enforce a hard token budget: if the assembled context would exceed
   MAX_CONTEXT_TOKENS the oldest / lowest-scored chunks are trimmed first.

All templates live here — no prompt strings scattered across the codebase.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)

# ─── Token budget constants ───────────────────────────────────────────────────
# These are read from settings so they can be tuned via .env without code changes.
# GPT-4o context window is 128k tokens; we leave room for the completion.
MAX_CONTEXT_TOKENS: int = settings.prompt_max_context_tokens
MAX_HISTORY_TOKENS: int = settings.prompt_max_history_tokens
# Rough chars-per-token for the fast heuristic (avoids importing tiktoken)
_CHARS_PER_TOKEN: float = 3.8


def _approx_tokens(text: str) -> int:
    """Fast char-based token approximation — no tiktoken import needed."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ─── System prompt template ───────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are a precise, helpful AI assistant with access to a curated knowledge base.

INSTRUCTIONS
────────────
1. Answer the user's question using ONLY the context chunks provided below.
2. Do not use context if it is irrelevant to the question.
3. Do not force a connection between the question and retrieved documents.
4. Never invent document references.
5. For casual conversation or greetings, respond naturally without
   citing enterprise documents.
6. If the answer is not contained in the context, say:
   "I don't have enough information in the available documents to answer that."
   Do NOT fabricate facts or draw on knowledge outside the provided context.
7. Cite every factual claim inline using the format [Source N] where N is the
   chunk number shown in the context.  Place the citation immediately after
   the sentence it supports.
8. At the end of your answer, include a "Sources" section listing each cited
   chunk in the format:
       [Source N] <doc_name>, p.<page>  — <one-line summary>
9. Be concise but complete.  Use bullet points or numbered lists when helpful.
10. If the user asks a follow-up question, use the conversation history below
   to maintain context — but still ground your answer in the provided chunks.

Today's date: {today}
"""

# ─── Context block template ───────────────────────────────────────────────────

_CHUNK_TEMPLATE = """\
[Source {n}]
Document : {doc_name}
Page(s)  : {pages}
Score    : {score:.3f}
Content  :
{text}
"""

# ─── No-context fallback ──────────────────────────────────────────────────────

_NO_CONTEXT_SYSTEM = """\
You are a helpful AI assistant.  No relevant documents were retrieved for
this query, so answer from your general knowledge and clearly state that
the response is not grounded in uploaded documents.
"""


# ─── Public API ───────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles the OpenAI messages list for a single RAG turn.

    Usage::

        builder  = PromptBuilder()
        messages = builder.build(
            query        = "What is the claims process?",
            chunks       = reranked_results,      # List[RetrievalResult]
            history      = prior_messages,         # from memory saver
        )
        # messages → List[{"role": ..., "content": ...}]
    """

    def __init__(
        self,
        max_context_tokens: int = 0,
        max_history_tokens: int = 0,
    ) -> None:
        self.max_context_tokens = max_context_tokens or settings.prompt_max_context_tokens
        self.max_history_tokens = max_history_tokens or settings.prompt_max_history_tokens

    # ── Context formatting ────────────────────────────────────────────────────

    def format_chunks(self, chunks: List[RetrievalResult]) -> tuple[str, List[Dict]]:
        """
        Render reranked chunks into a numbered context string.

        Applies a token budget: chunks are included in score-descending
        order (rerank_score > score) until MAX_CONTEXT_TOKENS is reached.
        Returns:
            context_text  — the full formatted context block
            citation_meta — list of dicts used to build Citation objects
        """
        # Sort: prefer rerank_score when present, fall back to raw score
        ordered = sorted(
            chunks,
            key=lambda r: r.rerank_score if r.rerank_score is not None else r.score,
            reverse=True,
        )

        context_parts: List[str] = []
        citation_meta: List[Dict] = []
        token_budget = self.max_context_tokens

        for n, chunk in enumerate(ordered, start=1):
            doc_name = chunk.metadata.get("doc_name") or chunk.metadata.get("doc_ref_name", "Unknown Document")
            pages    = chunk.metadata.get("source_pages") or chunk.metadata.get("page", "—")
            text     = (chunk.text or "").strip()

            block = _CHUNK_TEMPLATE.format(
                n=n,
                doc_name=doc_name,
                pages=pages,
                score=chunk.rerank_score if chunk.rerank_score is not None else chunk.score,
                text=text,
            )
            block_tokens = _approx_tokens(block)
            if block_tokens > token_budget:
                # Try a truncated version
                max_chars  = int(token_budget * _CHARS_PER_TOKEN)
                trunc_text = text[: max(100, max_chars - 200)]
                block = _CHUNK_TEMPLATE.format(
                    n=n,
                    doc_name=doc_name,
                    pages=pages,
                    score=chunk.rerank_score if chunk.rerank_score is not None else chunk.score,
                    text=trunc_text + " …[truncated]",
                )
                context_parts.append(block)
                citation_meta.append({
                    "n":        n,
                    "chunk_id": chunk.id,
                    "doc_name": doc_name,
                    "pages":    pages,
                    "score":    chunk.rerank_score or chunk.score,
                    "snippet":  trunc_text[:200],
                    "metadata": chunk.metadata,
                })
                break   # budget exhausted
            else:
                token_budget -= block_tokens
                context_parts.append(block)
                citation_meta.append({
                    "n":        n,
                    "chunk_id": chunk.id,
                    "doc_name": doc_name,
                    "pages":    pages,
                    "score":    chunk.rerank_score or chunk.score,
                    "snippet":  text[:200],
                    "metadata": chunk.metadata,
                })

        context_text = "\n".join(context_parts)
        logger.debug(
            "PromptBuilder: %d chunks → %d context tokens (budget=%d)",
            len(context_parts), _approx_tokens(context_text), self.max_context_tokens,
        )
        return context_text, citation_meta

    # ── History trimming ──────────────────────────────────────────────────────

    def _trim_history(
        self, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Trim conversation history to MAX_HISTORY_TOKENS.

        Keeps the most-recent turns (pair-wise user/assistant) within budget.
        """
        budget = self.max_history_tokens
        trimmed: List[Dict[str, str]] = []
        # Walk from newest to oldest, keep until budget runs out
        for msg in reversed(history):
            tokens = _approx_tokens(msg.get("content", ""))
            if tokens > budget:
                break
            budget -= tokens
            trimmed.append(msg)
        trimmed.reverse()
        return trimmed

    # ── Main build ────────────────────────────────────────────────────────────

    def build(
        self,
        query:   str,
        chunks:  List[RetrievalResult],
        history: Optional[List[Dict[str, str]]] = None,
        today:   str = "",
    ) -> tuple[List[Dict[str, str]], List[Dict]]:
        """
        Assemble the full OpenAI messages list for one RAG turn.

        Parameters
        ----------
        query:
            The current user question.
        chunks:
            Reranked RetrievalResults from the retrieval pipeline.
        history:
            Prior conversation turns as ``[{"role": ..., "content": ...}, ...]``.
            Should NOT include the current user turn.
        today:
            ISO date string injected into the system prompt.  Defaults to
            the current UTC date when empty.

        Returns
        -------
        (messages, citation_meta)
            messages      — OpenAI-compatible messages list
            citation_meta — list of dicts for building Citation objects
        """
        from datetime import date
        if not today:
            today = date.today().isoformat()

        # ── System prompt ─────────────────────────────────────────────────────
        if chunks:
            context_text, citation_meta = self.format_chunks(chunks)
            system_content = (
                _SYSTEM_TEMPLATE.format(today=today)
                + "\nCONTEXT CHUNKS\n"
                + "─" * 60 + "\n"
                + context_text
                + "\n" + "─" * 60
            )
        else:
            context_text   = ""
            citation_meta  = []
            system_content = _NO_CONTEXT_SYSTEM

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        # ── Conversation history ──────────────────────────────────────────────
        if history:
            trimmed = self._trim_history(history)
            messages.extend(trimmed)

        # ── Current user turn ─────────────────────────────────────────────────
        messages.append({"role": "user", "content": query})

        total_tokens = sum(_approx_tokens(m["content"]) for m in messages)
        logger.debug(
            "PromptBuilder.build: %d messages, ~%d tokens, %d citations",
            len(messages), total_tokens, len(citation_meta),
        )
        return messages, citation_meta
