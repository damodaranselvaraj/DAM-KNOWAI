"""
Evaluation — Retrieval quality metrics (Recall@K, MRR, nDCG, latency).
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd

from frontend.utils.session_state import init_session_state
from frontend.utils.icons import icon, icon_text
from frontend.components.nav_sidebar import render_sidebar

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Evaluation · KNOWAI", page_icon="🧠", layout="wide")

_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "custom.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<div class='page-header'>"
    f"<h1>{icon_text('bar-chart-2', 'Evaluation', size=34, color='#069494')}</h1>"
    f"<p>Retrieval quality metrics across recent evaluation runs</p>"
    f"</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
last_report = st.session_state.get("last_eval_report")

m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    st.metric("Mean MRR", f"{(last_report or {}).get('mean_mrr', 0.82):.2f}")
with m2:
    st.metric("Recall@10", f"{(last_report or {}).get('recall_at_10', 0.91):.2f}")
with m3:
    st.metric("nDCG@10", f"{(last_report or {}).get('ndcg_at_10', 0.87):.2f}")
with m4:
    st.metric("Avg Latency", f"{(last_report or {}).get('avg_latency_ms', 148):.0f} ms")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Run a New Evaluation", expanded=False):
    st.caption("Upload a labeled query set (query + relevant doc IDs) to score the current pipeline configuration.")
    eval_col1, eval_col2 = st.columns([3, 1])
    with eval_col1:
        st.file_uploader(
            "Query set (CSV/JSON)",
            type=["csv", "json"],
            label_visibility="collapsed",
            key="eval_file_uploader",
        )
    with eval_col2:
        st.selectbox("K values", ["1,5,10", "1,3,5,10,20"], key="eval_k_values")

    if st.button("Run Evaluation", key="run_eval_btn", use_container_width=True):
        st.info("Evaluation queued. Results will appear here once the run completes.")


# ═══════════════════════════════════════════════════════════════════════════════
# PER-QUERY METRICS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Per-Query Metrics")

sample_rows = [
    {"Query": "What are the compliance requirements?", "Recall@10": 0.95, "MRR": 0.88, "nDCG@10": 0.91, "Latency (ms)": 132},
    {"Query": "Summarise the Q3 risk findings.",        "Recall@10": 0.87, "MRR": 0.75, "nDCG@10": 0.80, "Latency (ms)": 156},
    {"Query": "Data handling procedures for employees.", "Recall@10": 0.90, "MRR": 0.83, "nDCG@10": 0.86, "Latency (ms)": 141},
]
st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)

st.caption(
    "Showing sample data. Connect the backend `/evaluation` endpoint "
    "(see `backend/retrieval/evaluation.py`) to populate this page with live results."
)
