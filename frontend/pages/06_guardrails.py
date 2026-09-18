"""
Guardrails — Configure the 6-layer guardrail pipeline that every chat
query passes through: Input Guardrails, Query Control, Retrieval Safety,
Context Control, Generation Safety, and Output Safety.

Every guardrail is represented as a checkbox and defaults to enabled.
Numeric thresholds/limits are configurable below each layer's checkboxes.
Settings are persisted to the backend via GET/POST /api/v1/guardrails/config
so they apply to every chat request immediately after saving.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import copy

import streamlit as st

from frontend.utils.session_state import init_session_state
from frontend.utils.api_client import get_api_client
from frontend.utils.icons import icon, icon_text
from frontend.utils.constants import DEFAULT_GUARDRAIL_CONFIG, GUARDRAIL_LAYERS
from frontend.components.nav_sidebar import render_sidebar

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Guardrails · KNOWAI", page_icon="🧠", layout="wide")

_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "custom.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
client = get_api_client()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()


# ═══════════════════════════════════════════════════════════════════════════════
# STATE — load the active config once per session, edited locally until saved
# ═══════════════════════════════════════════════════════════════════════════════
if "guardrail_config" not in st.session_state:
    with st.spinner("Loading guardrail configuration…"):
        st.session_state.guardrail_config = client.get_guardrail_config()

cfg = st.session_state.guardrail_config
toggles = cfg.setdefault("toggles", copy.deepcopy(DEFAULT_GUARDRAIL_CONFIG["toggles"]))
thresholds = cfg.setdefault("thresholds", copy.deepcopy(DEFAULT_GUARDRAIL_CONFIG["thresholds"]))


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
hdr_col, reset_col, save_col = st.columns([4, 1, 1])
with hdr_col:
    st.markdown(
        f"<div class='page-header'>"
        f"<h1>{icon_text('shield', 'Guardrails', size=34, color='#069494')}</h1>"
        f"<p>Configure the 6-layer safety pipeline applied to every chat query</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with reset_col:
    if st.button("Reset to Defaults", use_container_width=True):
        with st.spinner("Resetting…"):
            st.session_state.guardrail_config = client.reset_guardrail_config()
        st.success("Reset to defaults.")
        st.rerun()
with save_col:
    if st.button("Save", use_container_width=True, type="primary"):
        with st.spinner("Saving…"):
            st.session_state.guardrail_config = client.save_guardrail_config(cfg)
        st.success("Guardrail configuration saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER SECTIONS — checkboxes, all enabled by default
# ═══════════════════════════════════════════════════════════════════════════════
for layer in GUARDRAIL_LAYERS:
    with st.expander(layer["title"], expanded=True):
        st.markdown(
            f"<p style='color:#8A9BAC;font-size:0.85rem;margin-top:-0.5rem'>{layer['description']}</p>",
            unsafe_allow_html=True,
        )
        for key, label, help_text in layer["checks"]:
            current = toggles.get(key, True)
            toggles[key] = st.checkbox(
                label,
                value=current,
                help=help_text,
                key=f"gr_toggle_{key}",
            )

        # ── Per-layer threshold controls ────────────────────────────────────
        if layer["key"] == "layer1":
            c1, c2 = st.columns(2)
            with c1:
                thresholds["max_characters"] = st.number_input(
                    "Max Characters", min_value=1, max_value=50_000,
                    value=int(thresholds.get("max_characters", 2000)),
                    key="gr_max_characters",
                    help="Maximum character length of the raw query.",
                )
                thresholds["max_tokens"] = st.number_input(
                    "Max Tokens", min_value=1, max_value=32_000,
                    value=int(thresholds.get("max_tokens", 500)),
                    key="gr_max_tokens",
                    help="Maximum token count of the raw query.",
                )
            with c2:
                thresholds["max_query_length"] = st.number_input(
                    "Max Query Length (words)", min_value=1, max_value=10_000,
                    value=int(thresholds.get("max_query_length", 300)),
                    key="gr_max_query_length",
                    help="Maximum word count of the query.",
                )
                thresholds["max_documents"] = st.number_input(
                    "Max Documents", min_value=1, max_value=1_000,
                    value=int(thresholds.get("max_documents", 10)),
                    key="gr_max_documents",
                    help="Maximum number of documents a user can upload/reference.",
                )

        elif layer["key"] == "layer3":
            thresholds["top_n_retrieval"] = st.slider(
                "Top-N Retrieval (before rerank)", 1, 200,
                value=int(thresholds.get("top_n_retrieval", 20)),
                key="gr_top_n_retrieval",
                help="Number of documents retrieved via hybrid search before reranking.",
            )

        elif layer["key"] == "layer4":
            c1, c2 = st.columns(2)
            with c1:
                thresholds["relevance_threshold"] = st.slider(
                    "Relevance Threshold", 0.0, 1.0,
                    value=float(thresholds.get("relevance_threshold", 0.70)),
                    step=0.01,
                    key="gr_relevance_threshold",
                    help="Minimum reranker score for a document to be used.",
                )
            with c2:
                thresholds["dedup_similarity_threshold"] = st.slider(
                    "Dedup Similarity Threshold", 0.0, 1.0,
                    value=float(thresholds.get("dedup_similarity_threshold", 0.95)),
                    step=0.01,
                    key="gr_dedup_similarity_threshold",
                    help="Cosine-similarity threshold above which chunks are treated as near-duplicates.",
                )

        elif layer["key"] == "layer5":
            c1, c2 = st.columns(2)
            with c1:
                thresholds["groundedness_threshold"] = st.slider(
                    "Groundedness Threshold", 0.0, 1.0,
                    value=float(thresholds.get("groundedness_threshold", 0.75)),
                    step=0.01,
                    key="gr_groundedness_threshold",
                    help="Minimum groundedness score before a disclaimer is appended.",
                )
            with c2:
                thresholds["hallucination_threshold"] = st.slider(
                    "Hallucination Threshold", 0.0, 1.0,
                    value=float(thresholds.get("hallucination_threshold", 0.5)),
                    step=0.01,
                    key="gr_hallucination_threshold",
                    help="Fraction of unsupported entities above which the response is blocked.",
                )


st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG — recent guardrail activity (no raw PII / query / response content)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h4>{icon_text('activity', 'Recent Guardrail Activity', size=16, color='#069494')}</h4>",
    unsafe_allow_html=True,
)
st.caption("Audit records never contain raw PII values, query text, or response text.")

with st.spinner("Loading audit log…"):
    audit_data = client.get_guardrail_audit(limit=25)

records = audit_data.get("records", [])
if records:
    import pandas as pd
    rows = [{
        "Time":          (r.get("timestamp") or "")[:19].replace("T", " "),
        "Intent":        r.get("intent_classified") or "—",
        "Blocked":       "Yes" if r.get("response_blocked") else "No",
        "Block Reason":  r.get("block_reason") or "—",
        "Guardrails Triggered": ", ".join(r.get("input_guardrails_triggered", [])) or "—",
        "Docs Retrieved": r.get("documents_retrieved", 0),
        "Docs After Threshold": r.get("documents_after_threshold", 0),
        "Groundedness":  r.get("groundedness_score") if r.get("groundedness_score") is not None else "—",
        "Hallucination": "Yes" if r.get("hallucination_detected") else "No",
        "Latency (ms)":  r.get("latency_ms", 0),
    } for r in records]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No guardrail activity recorded yet. Audit records appear here once chat queries are processed.")
