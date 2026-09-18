"""
Retrieval Options — Generation, Search and Filter controls for the RAG
pipeline. These settings are shared (via st.session_state) with the Chatbot
page, which reads them when building each query payload.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

from frontend.utils.session_state import init_session_state
from frontend.utils.api_client import get_api_client
from frontend.utils.icons import icon_text
from frontend.components.nav_sidebar import render_sidebar

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Retrieval Options · DAM KNOWAI", page_icon="🧠", layout="wide")

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
    f"<h1>{icon_text('sliders', 'Retrieval Options', size=34, color='#069494')}</h1>"
    f"<p>Configure generation, search and filter settings used by the Chatbot</p>"
    f"</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h4>{icon_text('sliders', 'Generation', size=16, color='#069494')}</h4>",
    unsafe_allow_html=True,
)

temperature = st.slider(
    "Temperature",
    0.0, 2.0,
    value=float(st.session_state.get("chat_temperature", 0.2)),
    step=0.05,
    help="Higher = more creative. Lower = more deterministic.",
    key="temp_sl",
)
st.session_state.chat_temperature = temperature
st.caption("Deterministic  ←→  Creative")

top_k = st.slider(
    "Top-K Results",
    1, 50,
    value=int(st.session_state.get("chat_top_k", 10)),
    help="Number of document chunks to retrieve",
    key="topk_sl",
)
st.session_state.chat_top_k = top_k

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h4>{icon_text('search', 'Search', size=16, color='#069494')}</h4>",
    unsafe_allow_html=True,
)

retrieval_mode_opts = {
    "dense":         "Dense only",
    "hybrid_bm25":   "Hybrid (Dense + BM25)",
    "hybrid_splade": "Hybrid (Dense + SPLADE)",
}
current_retrieval_mode = st.session_state.get("chat_retrieval_mode", "hybrid_bm25")
retrieval_mode = st.radio(
    "Retrieval Mode",
    options=list(retrieval_mode_opts.keys()),
    format_func=lambda k: retrieval_mode_opts[k],
    index=list(retrieval_mode_opts.keys()).index(current_retrieval_mode)
        if current_retrieval_mode in retrieval_mode_opts else 1,
    key="retrieval_mode_radio",
    help=(
        "Dense only — pure vector similarity search.\n"
        "Hybrid (Dense + BM25) — combines dense with a local BM25 keyword "
        "index via Reciprocal Rank Fusion. Default.\n"
        "Hybrid (Dense + SPLADE) — combines dense with a learned SPLADE "
        "sparse vector in a single native Pinecone query."
    ),
)
st.session_state.chat_retrieval_mode = retrieval_mode
# Backward-compat flag consumed by the backend: only "dense" mode
# should disable hybrid retrieval.
st.session_state.chat_hybrid_search = retrieval_mode != "dense"

if retrieval_mode == "hybrid_bm25":
    alpha = st.slider(
        "Keyword ←→ Semantic",
        0.0, 1.0,
        value=float(st.session_state.get("chat_hybrid_alpha", 0.5)),
        step=0.05,
        key="alpha_sl",
    )
    st.session_state.chat_hybrid_alpha = alpha

reranker = st.toggle(
    "Reranker",
    value=st.session_state.get("chat_reranker", True),
    key="reranker_tog",
    help="Uses cross-encoder to rerank results",
)
st.session_state.chat_reranker = reranker

cit_mode = st.radio(
    "Citation Mode",
    ["paragraph", "sentence"],
    index=0 if st.session_state.get("chat_citation_mode", "paragraph") == "paragraph" else 1,
    key="cit_mode_radio",
)
st.session_state.chat_citation_mode = cit_mode

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h4>{icon_text('filter', 'Filters', size=16, color='#069494')}</h4>",
    unsafe_allow_html=True,
)

corpus_sel = st.selectbox(
    "Corpus / Collection",
    ["All", "Policy Docs", "Risk Reports", "Compliance"],
    key="corpus_sel",
)
st.session_state.chat_filter_corpus = corpus_sel

date_from = st.date_input("From:", value=None, key="date_from_di")
date_to   = st.date_input("To:",   value=None, key="date_to_di")
st.session_state.chat_filter_date_from = str(date_from) if date_from else None
st.session_state.chat_filter_date_to   = str(date_to)   if date_to   else None

doc_types = st.multiselect(
    "Document Types",
    ["PDF", "DOCX", "CSV", "HTML", "TXT"],
    default=st.session_state.get("chat_filter_doc_types", []),
    key="doc_types_ms",
)
st.session_state.chat_filter_doc_types = doc_types

if st.button("Reset Filters", key="reset_filters"):
    st.session_state.chat_filter_corpus     = "All"
    st.session_state.chat_filter_date_from  = None
    st.session_state.chat_filter_date_to    = None
    st.session_state.chat_filter_doc_types  = []
    st.rerun()
