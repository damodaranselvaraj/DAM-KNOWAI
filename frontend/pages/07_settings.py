"""
Settings — App-level preferences: API connection, chat defaults, and
session management. Pipeline-specific configuration (parser/chunking/
embedding/vector DB) stays in the Admin Panel.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

from frontend.utils.session_state import init_session_state, clear_chat_history
from frontend.utils.api_client import get_api_client
from frontend.utils.icons import icon, icon_text
from frontend.components.nav_sidebar import render_sidebar
from frontend.config import settings

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Settings · DAM KNOWAI", page_icon="🧠", layout="wide")

_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "custom.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
client = get_api_client()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<div class='page-header'>"
    f"<h1>{icon_text('settings', 'Settings', size=34, color='#069494')}</h1>"
    f"<p>App preferences and connection settings</p>"
    f"</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("API Connection", expanded=True):
    try:
        health = client.get_health_status()
        api_status = health.get("status", "unknown")
    except Exception:
        api_status = "unhealthy"

    dot_color = {"healthy": "#0D9E6E", "degraded": "#D97706", "unhealthy": "#DC2626"}.get(api_status, "#8A9BAC")

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("API Base URL", value=settings.API_BASE_URL, disabled=True)
    with c2:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding-top:1.9rem'>"
            f"<span style='width:10px;height:10px;border-radius:50%;background:{dot_color};"
            f"box-shadow:0 0 6px {dot_color};display:inline-block'></span>"
            f"<strong>{api_status.capitalize()}</strong></div>",
            unsafe_allow_html=True,
        )
    st.number_input("Request Timeout (s)", value=settings.API_TIMEOUT, disabled=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Chat Defaults", expanded=True):
    d1, d2 = st.columns(2)
    with d1:
        default_temp = st.slider(
            "Default Temperature", 0.0, 2.0,
            value=float(st.session_state.get("chat_temperature", 0.2)),
            step=0.05, key="settings_default_temp",
        )
        st.session_state.chat_temperature = default_temp
    with d2:
        default_topk = st.slider(
            "Default Top-K", 1, 50,
            value=int(st.session_state.get("chat_top_k", 10)),
            key="settings_default_topk",
        )
        st.session_state.chat_top_k = default_topk


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Session Management", expanded=False):
    n_msgs = len(st.session_state.get("messages", []))
    st.caption(f"Current chat session has {n_msgs} message(s).")
    if st.button("Clear Chat Session", key="settings_clear_session"):
        clear_chat_history()
        st.success("Chat session cleared.")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("About", expanded=False):
    st.markdown(
        "<p><strong>DAM KNOWAI</strong> — Enterprise RAG Intelligence Platform</p>"
        "<p style='color:#8A9BAC;font-size:0.85rem'>v1.0.0 · Powered by Pinecone, OpenAI, and Docling</p>",
        unsafe_allow_html=True,
    )
