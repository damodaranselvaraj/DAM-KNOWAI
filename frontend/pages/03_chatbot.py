"""
Chatbot — Streaming RAG chat with citations, controls sidebar, session management.
"""
import os, sys, time
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timezone

from frontend.utils.session_state import (
    init_session_state, get_session_id, add_message, clear_chat_history,
)
from frontend.utils.api_client import get_api_client
from frontend.utils.formatters import (
    format_timestamp_short, format_number, get_score_badge,
    format_score, truncate_text,
)
from frontend.utils.icons import icon, icon_text

from frontend.components.nav_sidebar import render_sidebar

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chatbot · KNOWAI", page_icon="🧠", layout="wide"
)

_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "custom.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
client = get_api_client()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()

def _render_citation_card(cit: dict, idx: int) -> str:
    badge   = get_score_badge(cit.get("score", 0))
    score   = format_score(cit.get("score", 0))
    snippet = truncate_text(cit.get("snippet", ""), 110)
    doc_icon = icon("file-text", size=14, color="#069494")
    page_icon = icon("book-open", size=12, color="#8A9BAC")
    cal_icon  = icon("calendar", size=12, color="#8A9BAC")
    chunk_icon= icon("layers",   size=12, color="#8A9BAC")
    return (
        f"<div class='citation-card'>"
        f"<div class='citation-header'>{doc_icon} [{idx}] {cit.get('doc_name','Unknown')}</div>"
        f"<div class='citation-meta'>"
        f"<span>{page_icon} Page {cit.get('page_number','—')}</span>"
        f"<span>{cal_icon} {cit.get('published_at','—')}</span>"
        f"<span>{badge} {score}</span>"
        f"<span>{chunk_icon} Chunk #{cit.get('chunk_index','?')}</span>"
        f"</div>"
        f"<div class='citation-snippet'>{snippet}</div>"
        f"</div>"
    )

# ─── Chat rendering (native st.chat_message / st.chat_input) ──────────────────
# Avatars are inline Lucide SVGs. Streamlit converts an SVG string passed to
# `avatar=` into a base64 data-URI and renders it as
# <img alt="{name} avatar">, which is the hook the stylesheet uses to colour
# and shape each role's avatar chip.
_AVATAR_USER = icon("user", size=24, color="#069494", stroke_width=1.9)
_AVATAR_BOT  = icon("bot",  size=24, color="#FFFFFF", stroke_width=1.9)

# Trailing marker element. While it is present inside an assistant bubble the
# stylesheet appends a blinking caret to the answer, so streamed tokens never
# have to be wrapped in raw HTML.
_STREAM_FLAG = "<span class='kb-streaming'></span>"

_THINKING_HTML = (
    "<div class='kb-thinking'>Assistant is thinking"
    "<span class='thinking-dots'><span></span><span></span><span></span></span>"
    "</div>"
)

_EMPTY_STATE_HTML = (
    f"<div class='kb-chat-empty'>"
    f"<div class='kb-chat-empty-icon'>{icon('message-circle', size=48, color='#069494')}</div>"
    f"<h3>Hello! Ask me anything about your documents.</h3>"
    f"<p>I'll search through your indexed documents and provide cited answers.</p>"
    f"</div>"
)


_MIN_CITATION_CONFIDENCE = 0.5  # hide references below 50% confidence


def _citations_html(citations: list) -> str:
    """References block rendered inside the assistant bubble."""
    citations = [c for c in citations if c.get("score", 0) >= _MIN_CITATION_CONFIDENCE]
    if not citations:
        return ""
    ref_ico = icon("link", size=13, color="#8A9BAC")
    shown, hidden = citations[:2], citations[2:]
    parts = [
        "<div class='kb-refs'>",
        f"<div class='kb-refs-title'>{ref_ico} References ({len(citations)} sources)</div>",
    ]
    for i, c in enumerate(shown, 1):
        parts.append(_render_citation_card(c, i))
    if hidden:
        parts.append(
            f"<details class='kb-refs-more'>"
            f"<summary>+ {len(hidden)} more source(s)</summary>"
        )
        for i, c in enumerate(hidden, len(shown) + 1):
            parts.append(_render_citation_card(c, i))
        parts.append("</details>")
    parts.append("</div>")
    return "".join(parts)


def _render_message(msg: dict) -> None:
    """Render one persisted message as a native chat row."""
    role = msg.get("role", "user")
    ts   = format_timestamp_short(msg.get("timestamp", ""))
    meta = f"<div class='kb-msg-meta'>{ts}</div>"

    if role == "user":
        with st.chat_message("user", avatar=_AVATAR_USER):
            st.markdown(msg.get("content", ""))
            st.markdown(meta, unsafe_allow_html=True)
        return

    with st.chat_message("assistant", avatar=_AVATAR_BOT):
        st.markdown(msg.get("content", ""))
        citations = msg.get("citations") or []
        if citations:
            st.markdown(_citations_html(citations), unsafe_allow_html=True)
        st.markdown(meta, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT: chat | controls
# ═══════════════════════════════════════════════════════════════════════════════
main_col, ctrl_col = st.columns([3, 1], gap="medium")


# ───────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Controls
# ───────────────────────────────────────────────────────────────────────────────
with ctrl_col:
    # New Chat / History — sit above the Sessions section.
    top_act1, top_act2 = st.columns([1, 1], gap="small")
    with top_act1:
        if st.button("New Chat", use_container_width=True, key="new_chat_btn"):
            clear_chat_history()
            st.rerun()
    with top_act2:
        show_history = st.button("History", use_container_width=True, key="show_hist_btn")

    st.markdown(
        f"<h4>{icon_text('bar-chart-2', 'Session', size=16, color='#069494')}</h4>",
        unsafe_allow_html=True,
    )
    sid = get_session_id()
    started = format_timestamp_short(st.session_state.get("session_started_at", ""))
    n_msgs  = len(st.session_state.get("messages", []))
    tokens  = st.session_state.get("total_tokens_used", 0)
    st.markdown(
        f"<div style='font-size:0.8rem;color:#4A6070;line-height:2.2'>"
        f"<div>{icon_text('hash', f'Session: <code>{sid[:10]}…</code>', size=13, color='#069494')}</div>"
        f"<div>{icon_text('message-square', f'Messages: {n_msgs}', size=13, color='#069494')}</div>"
        f"<div>{icon_text('zap', f'Tokens: {format_number(tokens)}', size=13, color='#069494')}</div>"
        f"<div>{icon_text('clock', f'Started: {started}', size=13, color='#069494')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("Clear Session", key="clear_session_btn", use_container_width=True):
        clear_chat_history()
        st.rerun()


# ───────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Chat
# ───────────────────────────────────────────────────────────────────────────────
with main_col:

    # Header row — wrapped in its own container so the stylesheet can hold it
    # to the same 720px width as the chat box below it. Heading + subtitle are
    # now center-aligned, with no action buttons in this row (those live in
    # the right-hand controls column now).
    with st.container():
        st.markdown("<span class='kb-chat-head'></span>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='page-header kb-chat-title'>"
            f"<h1>Knowledge Assistant</h1>"
            f"<p>Ask questions across your enterprise knowledge</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Chat shell ────────────────────────────────────────────────────────────
    # One vertical block holds everything: the scrollable message log and the
    # input row. The `.kb-chat-shell` marker span is the CSS hook the
    # stylesheet uses to find this block (Streamlit 1.35 containers expose no
    # key or class of their own).
    with st.container():
        st.markdown("<span class='kb-chat-shell'></span>", unsafe_allow_html=True)

        # Scrollable log. A fixed-height container that holds chat messages
        # gets Streamlit's native stick-to-bottom behaviour, so new and
        # streaming messages scroll into view on their own.
        chat_log = st.container(height=440, border=False)
        with chat_log:
            hero_slot = st.empty()
            if not st.session_state.get("messages"):
                hero_slot.markdown(_EMPTY_STATE_HTML, unsafe_allow_html=True)
            for _msg in st.session_state.get("messages", []):
                _render_message(_msg)

        # Input row — pinned to the bottom of the shell because it is the last
        # element inside it (nesting also switches chat_input to inline mode).
        prompt = st.chat_input(
            "Ask a question about the document...",
            key="chat_query_input",
        )

    # Streamlit 1.35 ships the send button without an accessible name.
    components.html(
        """
        <script>
        (function () {
            let tries = 0;
            function label() {
                tries += 1;
                const doc = window.parent.document;
                const btn = doc.querySelector('[data-testid="stChatInputSubmitButton"]');
                if (btn) {
                    btn.setAttribute('aria-label', 'Send message');
                    btn.setAttribute('title', 'Send message (Enter)');
                    return;
                }
                if (tries < 10) { setTimeout(label, 250); }
            }
            label();
        })();
        </script>
        """,
        height=0,
    )

    # ── Send logic ────────────────────────────────────────────────────────────
    if prompt and prompt.strip():
        query = prompt.strip()
        add_message("user", query)
        hero_slot.empty()

        # Build payload
        # NOTE: the backend validates `filters` against a strict allow-list
        # of indexed Pinecone metadata fields (see
        # backend/retrieval/query_validator.py::ALLOWED_FILTER_FIELDS) and
        # rejects any unknown key outright — even if its value is None/empty.
        # So we must (a) only send filters the user actually set, and
        # (b) map UI field names to the real indexed field names
        # (date_from/date_to -> published_at range, doc_types -> mime).
        filters: dict = {}

        corpus = st.session_state.get("chat_filter_corpus")
        if corpus and corpus != "All":
            filters["corpus"] = corpus

        date_from = st.session_state.get("chat_filter_date_from")
        date_to   = st.session_state.get("chat_filter_date_to")
        if date_from or date_to:
            published_at: dict = {}
            if date_from:
                published_at["$gte"] = date_from
            if date_to:
                published_at["$lte"] = date_to
            filters["published_at"] = published_at

        doc_types = st.session_state.get("chat_filter_doc_types", [])
        if doc_types:
            filters["mime"] = {"$in": doc_types}

        payload = {
            "session_id":       get_session_id(),
            "query":            query,
            "temperature":      float(st.session_state.get("chat_temperature", 0.2)),
            "top_k":            int(st.session_state.get("chat_top_k", 10)),
            "retrieval_mode":   st.session_state.get("chat_retrieval_mode", "hybrid_bm25"),
            "hybrid_search":    bool(st.session_state.get("chat_hybrid_search", True)),
            "reranker_enabled": bool(st.session_state.get("chat_reranker", True)),
            "citation_mode":    st.session_state.get("chat_citation_mode", "paragraph"),
            "filters":          filters or None,
        }

        # ── Append the question and an answer bubble to the log, then stream
        #    into that bubble's placeholders. Writing into `chat_log` keeps
        #    everything inside the scrollable area instead of below the input.
        with chat_log:
            _render_message(st.session_state["messages"][-1])
            with st.chat_message("assistant", avatar=_AVATAR_BOT):
                answer_slot = st.empty()
                caret_slot  = st.empty()

        answer_slot.markdown(_THINKING_HTML, unsafe_allow_html=True)
        caret_slot.markdown(_STREAM_FLAG, unsafe_allow_html=True)

        # ── Streaming: consume the SSE stream and keep the final accumulated
        #    text as the answer.  If the backend later returns a richer
        #    response (with citations, token counts, etc.) from /chat/query we
        #    use that instead — but we no longer silently throw away the
        #    streamed content: it becomes the fallback answer when the POST
        #    fails or returns an empty answer.
        stream_text = ""
        try:
            for token in client.stream_response(
                get_session_id(),
                query,
                retrieval_mode=st.session_state.get("chat_retrieval_mode", "hybrid_bm25"),
            ):
                stream_text += token
                # Tokens render as markdown (no raw HTML); the blinking caret
                # comes from the `.kb-streaming` marker still in the bubble.
                answer_slot.markdown(stream_text)
        except Exception:
            # Stream failed — fall back to whatever the POST below returns.
            pass
        finally:
            caret_slot.empty()

        # ── Full response: fetch citations and metadata from /chat/query.
        #    Use the streamed text as answer when:
        #      • the POST fails entirely, OR
        #      • the POST returns an empty/missing answer field.
        citations:   list  = []
        tokens_used: int   = 0
        answer:      str   = stream_text  # default to what was streamed

        try:
            resp = client.query_rag(payload)
            post_answer = resp.get("answer", "").strip()
            # Only replace the streamed answer if the POST gave us real content
            if post_answer:
                answer = post_answer
            citations   = resp.get("citations", [])
            tokens_used = resp.get("tokens_used", 0)
        except Exception as exc:
            # POST failed — stream_text is already set as the answer above.
            # Surface the error only when we also have nothing streamed.
            if not answer:
                answer = f"⚠️ Could not reach the API: {exc}"

        add_message("assistant", answer, citations=citations, tokens_used=tokens_used)
        st.rerun()

    # ── History modal (inline expander) ──────────────────────────────────────
    if show_history:
        st.session_state.show_history_modal = not st.session_state.get("show_history_modal", False)

    if st.session_state.get("show_history_modal", False):
        with st.expander("Session History", expanded=True):
            msgs = st.session_state.get("messages", [])
            if not msgs:
                st.info("No messages in current session.")
            else:
                for i, m in enumerate(msgs):
                    role     = m.get("role", "user")
                    role_ico = icon("user", size=14, color="#069494") if role == "user" else icon("bot", size=14, color="#069494")
                    ts       = format_timestamp_short(m.get("timestamp", ""))
                    preview  = truncate_text(m.get("content", ""), 80)
                    st.markdown(
                        f"<div style='font-size:0.82rem;padding:0.4rem 0;border-bottom:1px solid #D8E4EC;color:#4A6070'>"
                        f"{role_ico} <strong style='color:#1A2B3C'>{role.capitalize()}</strong>  {ts}<br>"
                        f"<span style='color:#8A9BAC'>{preview}</span></div>",
                        unsafe_allow_html=True,
                    )
