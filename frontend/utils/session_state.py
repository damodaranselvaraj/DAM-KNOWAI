"""
Streamlit session_state initialisation and management helpers.
Import and call `init_session_state()` once at the top of every page.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
import streamlit as st
from frontend.utils.constants import DEFAULT_PIPELINE_CONFIG


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Sets all default values in st.session_state on first run."""

    # Chat
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_started_at" not in st.session_state:
        st.session_state.session_started_at = _now()
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0

    # Pipeline
    if "pipeline_config" not in st.session_state:
        import copy
        st.session_state.pipeline_config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)
    if "current_job_id" not in st.session_state:
        st.session_state.current_job_id = None
    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False
    if "uploaded_files_meta" not in st.session_state:
        st.session_state.uploaded_files_meta = []

    # UI state
    if "health_data" not in st.session_state:
        st.session_state.health_data = None
    if "last_health_check" not in st.session_state:
        st.session_state.last_health_check = None
    if "show_history_modal" not in st.session_state:
        st.session_state.show_history_modal = False

    # Chat controls
    if "chat_temperature" not in st.session_state:
        st.session_state.chat_temperature = 0.2
    if "chat_top_k" not in st.session_state:
        st.session_state.chat_top_k = 10
    if "chat_retrieval_mode" not in st.session_state:
        st.session_state.chat_retrieval_mode = "hybrid_bm25"
    if "chat_hybrid_search" not in st.session_state:
        st.session_state.chat_hybrid_search = True
    if "chat_hybrid_alpha" not in st.session_state:
        st.session_state.chat_hybrid_alpha = 0.5
    if "chat_reranker" not in st.session_state:
        st.session_state.chat_reranker = True
    if "chat_citation_mode" not in st.session_state:
        st.session_state.chat_citation_mode = "paragraph"
    if "chat_filter_corpus" not in st.session_state:
        st.session_state.chat_filter_corpus = "All"
    if "chat_filter_date_from" not in st.session_state:
        st.session_state.chat_filter_date_from = None
    if "chat_filter_date_to" not in st.session_state:
        st.session_state.chat_filter_date_to = None
    if "chat_filter_doc_types" not in st.session_state:
        st.session_state.chat_filter_doc_types = []


# ─── Session helpers ──────────────────────────────────────────────────────────

def get_session_id() -> str:
    """Return the current session ID, creating one if it doesn't exist."""
    if "session_id" not in st.session_state or not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def add_message(
    role: str,
    content: str,
    citations: list | None = None,
    tokens_used: int = 0,
) -> None:
    """Append a chat message to session history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "citations": citations or [],
        "timestamp": _now(),
    })
    if "total_tokens_used" not in st.session_state:
        st.session_state.total_tokens_used = 0
    st.session_state.total_tokens_used += tokens_used


def clear_chat_history() -> None:
    """
    Reset conversation history and create a new session ID.

    Also asks the backend to delete the old session's stored checkpoints
    (best-effort — if the backend is offline or the session was never
    persisted, this is a no-op from the user's perspective).
    """
    old_session_id = st.session_state.get("session_id")
    if old_session_id:
        try:
            from frontend.utils.api_client import get_api_client
            get_api_client().clear_session(old_session_id)
        except Exception:
            pass  # offline / already-cleared — local reset still proceeds

    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_started_at = _now()
    st.session_state.total_tokens_used = 0


def update_pipeline_config(section: str, key: str, value) -> None:
    """Update a single key within a pipeline config section."""
    if "pipeline_config" not in st.session_state:
        import copy
        st.session_state.pipeline_config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)
    st.session_state.pipeline_config[section][key] = value


def get_pipeline_config() -> dict:
    """Return the full pipeline config dict from session state."""
    if "pipeline_config" not in st.session_state:
        import copy
        st.session_state.pipeline_config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)
    return st.session_state.pipeline_config
