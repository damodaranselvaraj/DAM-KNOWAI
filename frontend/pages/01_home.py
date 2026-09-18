"""
Home — RAG Pipeline Hub landing page.
Displays hero nav cards, live system status bar, and metrics.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from frontend.utils.session_state import init_session_state
from frontend.utils.api_client import get_api_client
from frontend.utils.formatters import format_timestamp_short
from frontend.utils.icons import icon, icon_text
from frontend.components.nav_sidebar import render_sidebar

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DAM KNOWAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "custom.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
client = get_api_client()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()

# ─── Fetch data ────────────────────────────────────────────────────────────────
# Health of the embedding model, vector DB, LLM model, and reranker is only
# checked once per browser session (on first load of the Home page). It is
# cached in session_state so subsequent reruns (nav clicks, widget interaction)
# reuse the same result instead of re-pinging OpenAI/Pinecone every few seconds.
if "service_status" not in st.session_state:
    with st.spinner("Loading system status…"):
        st.session_state.service_status = get_api_client().get_service_status()

service_status = st.session_state.service_status


# ═══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<div class='hero-title'>"
    f"<div style='display:flex;align-items:center;justify-content:center;gap:0.75rem;margin-top:-1.5rem;margin-bottom:0.1rem'>"
    f"{icon('brain', size=44, color='#069494')}"
    f"<h1 style='margin:0!important'>DAM KNOWAI</h1>"
    f"</div>"
    f"<p class='hero-category'>Intelligent Knowledge Platform</p>"
    f"<p class='hero-tagline'>AI-Powered Document Intelligence</p>"
    f"<p class='hero-description'>Search, Summarize, and Chat With Your Documents with AI-powered RAG</p>"
    f"</div>",
    unsafe_allow_html=True,
)

# ─── Hero navigation cards ─────────────────────────────────────────────────────
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown(
        f"<div class='nav-card nav-card-admin'>"
        f"<div class='card-icon'>"
        f"{icon('settings', size=34, color='#069494')}"
        f"</div>"
        f"<h2>Admin Panel</h2>"
        f"<ul>"
        f"<li>{icon_text('zap', 'Configure Pipeline', size=14, color='#069494')} </li>"
        f"<li>{icon_text('upload', 'Upload Documents', size=14, color='#069494')}</li>"
        f"<li>{icon_text('sliders', 'Manage Settings', size=14, color='#069494')}</li>"
        f"</ul>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nav-card-spacer'></div>", unsafe_allow_html=True)
    if st.button("Open Admin Panel", key="btn_admin", use_container_width=True):
        st.switch_page("pages/02_admin.py")

with right_col:
    st.markdown(
        f"<div class='nav-card nav-card-chat'>"
        f"<div class='card-icon'>"
        f"{icon('message-circle', size=34, color='#007080')}"
        f"</div>"
        f"<h2>DAM KNOWAI Chatbot</h2>"
        f"<ul>"
        f"<li>{icon_text('send', 'Start Conversation', size=14, color='#007080')}</li>"
        f"<li>{icon_text('search', 'Ask Questions', size=14, color='#007080')}</li>"
        f"<li>{icon_text('link', 'View References', size=14, color='#007080')}</li>"
        f"</ul>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nav-card-spacer'></div>", unsafe_allow_html=True)
    if st.button("Open Chatbot", key="btn_chat", use_container_width=True):
        st.switch_page("pages/03_chatbot.py")


st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE SERVICE STATUS BOXES
# ═══════════════════════════════════════════════════════════════════════════════
def _render_service_box(icon_name: str, label: str, model_name: str, status_text: str, healthy: bool) -> str:
    """Build one teal/cyan status box: label, model name, and a color-coded live status."""
    state_class = "is-ok" if healthy else "is-error"
    dot_color   = "var(--success)" if healthy else "var(--danger)"
    return (
        f"<div class='service-box'>"
        f"<div class='service-box-label'>{icon(icon_name, size=13, color='#069494')} {label}</div>"
        f"<div class='service-box-name' title='{model_name}'>{model_name}</div>"
        f"<div class='service-box-status {state_class}'>"
        f"<span class='status-dot' style='background:{dot_color};box-shadow:0 0 5px {dot_color}'></span>"
        f"{status_text}"
        f"</div>"
        f"</div>"
    )

emb_svc   = service_status.get("embedding_model", {})
vdb_svc   = service_status.get("vector_db", {})
llm_svc   = service_status.get("llm_model", {})
rerank_svc = service_status.get("reranking_model", {})

m1, m2, m3, m4 = st.columns(4, gap="medium")

with m1:
    st.markdown(
        _render_service_box(
            "cpu", "Embedding Model",
            emb_svc.get("name", "—"),
            "Active" if emb_svc.get("healthy") else "Not Loaded",
            bool(emb_svc.get("healthy")),
        ),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        _render_service_box(
            "database", "Vector DB",
            vdb_svc.get("name", "—"),
            "Ready" if vdb_svc.get("healthy") else "Disconnected",
            bool(vdb_svc.get("healthy")),
        ),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        _render_service_box(
            "brain", "LLM Model",
            llm_svc.get("name", "—"),
            "Online" if llm_svc.get("healthy") else "Unavailable",
            bool(llm_svc.get("healthy")),
        ),
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        _render_service_box(
            "layers", "Reranking Model",
            rerank_svc.get("name", "—"),
            "Enabled" if rerank_svc.get("healthy") else "Disabled",
            bool(rerank_svc.get("healthy")),
        ),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK START GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

with st.expander("Quick Start Guide", expanded=False):
    steps = [
        ("1", "settings",    "Configure Pipeline",
         "Open the Admin Panel and set your parser, chunker, embedding model and vector DB settings."),
        ("2", "upload",      "Upload Documents",
         "Drag & drop PDFs, DOCX, CSV or TXT files in the Admin Panel's upload section."),
        ("3", "play",        "Run Pipeline",
         "Click Run Pipeline to parse, chunk, embed and store all documents in Pinecone."),
        ("4", "message-circle", "Start Chatting",
         "Open the Chatbot, ask questions, and get AI answers with cited source references."),
    ]
    for num, ic, title, desc in steps:
        st.markdown(
            f"<div class='step-card'>"
            f"<div class='step-number'>{num}</div>"
            f"<div>"
            f"<div class='step-title'>{icon_text(ic, title, size=15, color='#069494')}</div>"
            f"<div class='step-desc'>{desc}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RECENT ACTIVITY FEED
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Recent Activity", expanded=False):
    try:
        activities = client.get_recent_activity(limit=10)
    except Exception:
        activities = []

    if not activities:
        st.info("No activity yet. Upload documents and run the pipeline to see events here.")
    else:
        for event in activities:
            title     = event.get("title", "")
            detail    = event.get("detail", "")
            timestamp = event.get("timestamp", "")
            when      = format_timestamp_short(timestamp) if timestamp else "—"

            c1, c2, c3 = st.columns([0.5, 4, 1.5])
            c1.markdown(
                f"<span style='display:flex;padding-top:4px'>{icon('activity', size=18, color='#069494')}</span>",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"<div style='font-size:0.88rem;font-weight:600;color:#1A2B3C'>{title}</div>"
                f"<div style='font-size:0.78rem;color:#4A6070'>{detail}</div>",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"<div style='font-size:0.75rem;color:#8A9BAC;text-align:right;padding-top:4px'>{when}</div>",
                unsafe_allow_html=True,
            )
            st.divider()
