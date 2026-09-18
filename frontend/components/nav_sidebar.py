"""
Shared sidebar navigation component.

Renders the KNOWAI brand block, the primary nav group (Home / Admin Panel /
Chatbot), the secondary nav group (Evaluation / Retrieval / Knowledge Base),
and the settings + version footer — identically across every page.

Usage::

    from frontend.components.nav_sidebar import render_sidebar
    render_sidebar()
"""
from __future__ import annotations
import streamlit as st
from frontend.utils.icons import icon
from frontend.utils.api_client import get_api_client


def render_sidebar() -> None:
    """Render the full sidebar: brand block, nav groups, and footer."""
    with st.sidebar:
        # ── Brand block ──────────────────────────────────────────────────────
        st.markdown(
            f"<div class='sidebar-brand'>"
            f"{icon('brain', size=40, color='#FFFFFF')}"
            f"<p class='brand-name'>KNOWAI</p>"
            f"<p class='brand-sub'>Intelligent Knowledge Platform</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Primary navigation ───────────────────────────────────────────────
        st.page_link("pages/01_home.py",    label="Home",        icon=":material/home:")
        st.page_link("pages/02_admin.py",   label="Admin Panel", icon=":material/settings:")
        st.page_link("pages/03_chatbot.py", label="Chatbot",     icon=":material/chat:")
        st.divider()

        # ── Secondary navigation ─────────────────────────────────────────────
        st.page_link("pages/04_evaluation.py",     label="Evaluation",     icon=":material/analytics:")
        st.page_link("pages/05_retrieval.py",      label="Retrieval options", icon=":material/manage_search:")
        st.page_link("pages/06_knowledge_base.py", label="Knowledge Base", icon=":material/menu_book:")
        st.divider()

        # ── Settings ──────────────────────────────────────────────────────────
        st.page_link("pages/07_settings.py", label="Settings", icon=":material/tune:")
        st.caption("v1.0.0 · Enterprise RAG")


def render_api_status() -> None:
    """Render a small live API health dot. Used only on the entry-point shim."""
    client = get_api_client()
    try:
        health = client.get_health_status()
        status = health.get("status", "unknown")
        dot_color = {
            "healthy":   "#0D9E6E",
            "degraded":  "#D97706",
            "unhealthy": "#DC2626",
        }.get(status, "#8A9BAC")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;font-size:0.82rem;color:rgba(255,255,255,0.75);padding:0 0.25rem'>"
            f"<span style='width:8px;height:8px;border-radius:50%;background:{dot_color};"
            f"box-shadow:0 0 5px {dot_color};display:inline-block;flex-shrink:0'></span>"
            f"API {status.capitalize()}</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:8px;font-size:0.82rem;color:rgba(255,255,255,0.75);padding:0 0.25rem'>"
            "<span style='width:8px;height:8px;border-radius:50%;background:#DC2626;"
            "box-shadow:0 0 5px #DC2626;display:inline-block;flex-shrink:0'></span>"
            "API Offline</div>",
            unsafe_allow_html=True,
        )
