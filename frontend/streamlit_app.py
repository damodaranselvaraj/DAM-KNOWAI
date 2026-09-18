"""
RAG Pipeline Hub — Streamlit entry point.
Run with: streamlit run frontend/streamlit_app.py --server.port 8501
"""
import os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from frontend.utils.session_state import init_session_state
from frontend.components.nav_sidebar import render_sidebar

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DAM KNOWAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About":        "# DAM KNOWAI\nIntelligent Knowledge Platform — Powered by Pinecone, Cohere & OpenAI",
    },
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
init_session_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()

# ─── Redirect to Home dashboard ───────────────────────────────────────────────
st.switch_page("pages/01_home.py")
