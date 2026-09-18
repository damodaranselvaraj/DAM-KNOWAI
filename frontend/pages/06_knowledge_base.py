"""
Knowledge Base — Browse and manage all indexed documents/corpora.
Read-only overview page (document management/edit lives in Admin Panel).
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd

from frontend.utils.session_state import init_session_state
from frontend.utils.api_client import get_api_client
from frontend.utils.formatters import format_file_size, format_timestamp
from frontend.utils.icons import icon_text
from frontend.components.nav_sidebar import render_sidebar

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Knowledge Base · KNOWAI", page_icon="🧠", layout="wide")

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
    f"<h1>{icon_text('database', 'Knowledge Base', size=34, color='#069494')}</h1>"
    f"<p>Browse all corpora and indexed documents</p>"
    f"</div>",
    unsafe_allow_html=True,
)

with st.spinner("Loading documents…"):
    documents = client.list_documents()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY METRICS
# ═══════════════════════════════════════════════════════════════════════════════
total_docs   = len(documents)
total_chunks = sum(d.get("chunk_count", 0) for d in documents)
by_type: dict = {}
for d in documents:
    ext = (d.get("file_type") or "—").upper()
    by_type[ext] = by_type.get(ext, 0) + 1

m1, m2, m3 = st.columns(3, gap="medium")
with m1:
    st.metric("Total Documents", total_docs)
with m2:
    st.metric("Total Chunks", total_chunks)
with m3:
    st.metric("File Types", len(by_type))

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT TABLE
# ═══════════════════════════════════════════════════════════════════════════════
search_col, type_col = st.columns([3, 1])
with search_col:
    search = st.text_input("Search", placeholder="Search by filename…", label_visibility="collapsed", key="kb_search")
with type_col:
    type_filter = st.selectbox("Type", ["All"] + sorted(by_type.keys()), label_visibility="collapsed", key="kb_type_filter")

filtered = documents
if search:
    filtered = [d for d in filtered if search.lower() in (d.get("filename") or d.get("name", "")).lower()]
if type_filter != "All":
    filtered = [d for d in filtered if (d.get("file_type") or "").upper() == type_filter]

if filtered:
    rows = [{
        "Name":     d.get("filename") or d.get("name", "—"),
        "Type":     (d.get("file_type") or "—").upper(),
        "Size":     format_file_size(d.get("size_bytes", 0)),
        "Chunks":   d.get("chunk_count", 0),
        "Status":   (d.get("status") or "—").capitalize(),
        "Indexed":  format_timestamp(d.get("parsed_at") or d.get("uploaded_at", "")),
    } for d in filtered]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No documents found. Upload documents from the Admin Panel to populate the knowledge base.")
