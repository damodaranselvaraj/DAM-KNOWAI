"""
Admin Panel — Pipeline configuration, document upload, live progress.
"""
import os, sys, time, copy
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from frontend.utils.session_state import init_session_state, update_pipeline_config, get_pipeline_config
from frontend.utils.api_client import get_api_client
from frontend.utils.formatters import format_file_size, format_timestamp, get_file_icon, estimate_chunks
from frontend.utils.icons import icon, icon_text
from frontend.utils.constants import (
    PARSER_OPTIONS, CHUNKING_STRATEGIES, EMBEDDING_MODELS,
    VECTOR_METRICS, PHASE_NAMES, PHASE_LABELS, PHASE_ICONS,
    SUPPORTED_FILE_TYPES,
)
from frontend.components.nav_sidebar import render_sidebar

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Admin Panel · DAM KNOWAI", page_icon="🧠", layout="wide")

_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "custom.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
client = get_api_client()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()

cfg = get_pipeline_config()


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER ROW
# ═══════════════════════════════════════════════════════════════════════════════
hdr_col, save_col, run_col = st.columns([4, 1, 1])
with hdr_col:
    st.markdown(
        f"<div class='page-header'>"
        f"<h1>{icon_text('settings', 'Admin Panel', size=34, color='#069494')}</h1>"
        f"<p>Configure your RAG pipeline</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

with save_col:
    if st.button("Save Config", use_container_width=True):
        with st.spinner("Saving…"):
            resp = client.save_pipeline_config(cfg)
        st.success("Config saved")

with run_col:
    btn_label = "Running…" if st.session_state.get("pipeline_running") else "Run Pipeline"
    has_docs   = bool(st.session_state.get("uploaded_files_meta") or client.list_documents())
    if st.button(btn_label, use_container_width=True, disabled=st.session_state.get("pipeline_running", False)):
        resp = client.run_pipeline()
        st.session_state.current_job_id = resp.get("job_id")
        st.session_state.pipeline_running = True
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PROGRESS STEPPER
# ═══════════════════════════════════════════════════════════════════════════════
job_id = st.session_state.get("current_job_id")

if job_id:
    job_data = client.get_pipeline_status(job_id)
    phases   = job_data.get("phases", {})
    job_status = job_data.get("status", "running")

    if job_status in ("complete", "failed"):
        st.session_state.pipeline_running = False

    # Build stepper HTML
    phase_html = ""
    for i, pname in enumerate(PHASE_NAMES):
        p     = phases.get(pname, {})
        pstat = p.get("status", "pending")
        pct   = p.get("progress", 0)
        icon  = PHASE_ICONS.get(pstat, "⏳")
        label = PHASE_LABELS[pname]
        spin_class = " spin" if pstat == "in_progress" else ""

        phase_html += (
            f"<div class='phase-box {pstat}'>"
            f"<span class='phase-icon{spin_class}'>{icon}</span>"
            f"<div class='phase-name'>{label}</div>"
            f"<div class='phase-pct'>{pct}%</div>"
            f"</div>"
        )
        if i < len(PHASE_NAMES) - 1:
            phase_html += "<span class='phase-arrow'>▶</span>"

    st.markdown(
        f"<div class='pipeline-stepper'>{phase_html}</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("pipeline_running"):
        time.sleep(2)
        st.rerun()
    elif job_status == "complete":
        st.success("✅ Pipeline completed successfully!")
    elif job_status == "failed":
        st.error(f"❌ Pipeline failed: {job_data.get('error', 'Unknown error')}")
else:
    # Show idle stepper
    idle_html = ""
    for i, pname in enumerate(PHASE_NAMES):
        idle_html += (
            f"<div class='phase-box pending'>"
            f"<span class='phase-icon'>⏳</span>"
            f"<div class='phase-name'>{PHASE_LABELS[pname]}</div>"
            f"<div class='phase-pct'>—</div>"
            f"</div>"
        )
        if i < len(PHASE_NAMES) - 1:
            idle_html += "<span class='phase-arrow'>▶</span>"
    st.markdown(f"<div class='pipeline-stepper'>{idle_html}</div>", unsafe_allow_html=True)


st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DOCUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Phase 1: Document Parsing", expanded=False):

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown(
        f"<h4>{icon_text('upload', 'Upload Documents', size=16, color='#069494')}</h4>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drag & Drop Documents",
        accept_multiple_files=True,
        type=list(SUPPORTED_FILE_TYPES.keys()),
        label_visibility="collapsed",
    )

    # ── Small amber status indicators (OCR / language) ─────────────────────────
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin:0.6rem 0 1.25rem'>"
        f"<span style='display:inline-flex;align-items:center;gap:6px;font-size:0.78rem;"
        f"color:var(--text-secondary);background:var(--warning-bg);border:1px solid rgba(217,119,6,0.3);"
        f"border-radius:999px;padding:3px 10px'>"
        f"<span style='width:7px;height:7px;border-radius:50%;background:var(--warning);"
        f"box-shadow:0 0 4px var(--warning);display:inline-block;flex-shrink:0'></span>"
        f"OCR not enabled. Scanned PDFs may produce limited extraction results. Enabled in upcoming release."
        f"</span>"
        f"<span style='display:inline-flex;align-items:center;gap:6px;font-size:0.78rem;"
        f"color:var(--text-secondary);background:var(--warning-bg);border:1px solid rgba(217,119,6,0.3);"
        f"border-radius:999px;padding:3px 10px'>"
        f"<span style='width:7px;height:7px;border-radius:50%;background:var(--warning);"
        f"box-shadow:0 0 4px var(--warning);display:inline-block;flex-shrink:0'></span>"
        f"English only — multilingual support planned"
        f"</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if uploaded:
        rows = []
        for uf in uploaded:
            ext      = uf.name.rsplit(".", 1)[-1].lower()
            file_ico = get_file_icon(ext)
            rows.append({
                "File": f"{file_ico} {uf.name}",
                "Size": format_file_size(uf.size),
                "Type": ext.upper(),
                "Status": "Ready to upload",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.button("Upload All Files"):
            # Uploading here only validates and stages each file's raw bytes —
            # it does NOT parse them. Parsing (and chunk/embed/store) happens
            # when "Run Pipeline" is clicked above.
            prog = st.progress(0, text="Loading files…")
            new_meta = []
            errors = []
            for idx, uf in enumerate(uploaded):
                ext = uf.name.rsplit(".", 1)[-1].lower()
                content_type = f"application/{ext}"
                try:
                    resp = client.upload_document(uf.read(), uf.name, content_type)
                    new_meta.append(resp)
                except RuntimeError as exc:
                    # Backend rejected the file (wrong type, size limit, etc.)
                    errors.append(f"**{uf.name}**: {exc}")
                except Exception as exc:
                    # Network / connection failure
                    errors.append(f"**{uf.name}**: Could not reach the API — {exc}")
                prog.progress((idx + 1) / len(uploaded), text=f"Loading {uf.name}…")

            if new_meta:
                st.session_state.uploaded_files_meta = (
                    st.session_state.get("uploaded_files_meta", []) + new_meta
                )
                st.success(
                    f"✅ {len(new_meta)} file(s) loaded. Click **Run Pipeline** above to "
                    "parse, chunk, embed, and index them."
                )
            if errors:
                st.error("The following file(s) failed to load:\n\n" + "\n\n".join(errors))

    st.divider()

    # ── Parser selection ──────────────────────────────────────────────────────
    st.markdown(
        f"<h4>{icon_text('search', 'Parser Selection', size=16, color='#069494')}</h4>",
        unsafe_allow_html=True,
    )
    parser_labels = {k: f"{v['label']} {v['badge']} — {v['description']}" for k, v in PARSER_OPTIONS.items()}
    current_parser = cfg["parser"].get("parser_type", "auto")
    selected_parser = st.radio(
        "Select parser:",
        options=list(parser_labels.keys()),
        format_func=lambda k: parser_labels[k],
        index=list(parser_labels.keys()).index(current_parser),
        key="parser_type_radio",
        label_visibility="collapsed",
    )
    update_pipeline_config("parser", "parser_type", selected_parser)

    st.divider()

    # ── Version handling ──────────────────────────────────────────────────────
    st.markdown(
        f"<h4>{icon_text('git-branch', 'Document Version Handling', size=16, color='#069494')}</h4>",
        unsafe_allow_html=True,
    )
    version_opts = {
        "replace":     "Replace existing — Overwrite document with same name",
        "keep_both":   "Keep both versions — Maintain version history",
        "soft_delete": "Soft delete old — Mark old as inactive, keep new",
    }
    current_vh = cfg["parser"].get("version_handling", "replace")
    selected_vh = st.radio(
        "Version handling:",
        options=list(version_opts.keys()),
        format_func=lambda k: version_opts[k],
        index=list(version_opts.keys()).index(current_vh),
        key="version_handling_radio",
        label_visibility="collapsed",
    )
    update_pipeline_config("parser", "version_handling", selected_vh)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHUNKING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Phase 2: Chunking Strategy", expanded=False):

    strategy_labels = {k: v["label"] for k, v in CHUNKING_STRATEGIES.items()}
    current_strat   = cfg["chunking"].get("strategy", "recursive")
    selected_strat  = st.selectbox(
        "Chunking Strategy",
        options=list(strategy_labels.keys()),
        format_func=lambda k: strategy_labels[k],
        index=list(strategy_labels.keys()).index(current_strat),
        key="chunking_strategy_sel",
    )
    update_pipeline_config("chunking", "strategy", selected_strat)

    st.markdown(
        f"<div style='padding:0.6rem 1rem;background:var(--info-bg);border-left:3px solid var(--primary);"
        f"border-radius:6px;font-size:0.85rem;color:var(--text-secondary);margin:0.5rem 0'>"
        f"{CHUNKING_STRATEGIES[selected_strat]['description']}</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        chunk_size = st.slider(
            "Chunk Size (tokens)", 100, 4096,
            value=cfg["chunking"].get("chunk_size", 512), step=32,
            help="Token count per chunk",
            key="chunk_size_sl",
        )
        update_pipeline_config("chunking", "chunk_size", chunk_size)

    with col_b:
        overlap = st.slider(
            "Overlap (tokens)", 0, 500,
            value=cfg["chunking"].get("overlap", 50), step=10,
            help="Token overlap between consecutive chunks",
            key="overlap_sl",
        )
        update_pipeline_config("chunking", "overlap", overlap)

    col_c, col_d = st.columns(2)

    with col_c:
        min_chunk = st.slider(
            "Min Chunk (tokens)", 50, 500,
            value=cfg["chunking"].get("min_chunk", 100), step=10,
            key="min_chunk_sl",
        )
        update_pipeline_config("chunking", "min_chunk", min_chunk)

    with col_d:
        max_chunk = st.slider(
            "Max Chunk (tokens)", 256, 4096,
            value=cfg["chunking"].get("max_chunk", 1024), step=64,
            key="max_chunk_sl",
        )
        update_pipeline_config("chunking", "max_chunk", max_chunk)

    if selected_strat == "recursive":
        _SEP_LABELS = {
            "\n\n": "¶  Paragraph break (\\n\\n)",
            "\n":   "↵  Line break (\\n)",
            " ":    "␣  Space",
            "":     "∅  Character (no separator)",
        }
        seps = st.multiselect(
            "Separators",
            options=["\n\n", "\n", " ", ""],
            default=cfg["chunking"].get("separators", ["\n\n", "\n", " ", ""]),
            format_func=lambda s: _SEP_LABELS.get(s, repr(s)),
            help="Tried in order — the chunker splits on the first separator, "
                 "then falls back to the next one for any piece still too large.",
            key="separators_ms",
        )
        update_pipeline_config("chunking", "separators", seps)

    if selected_strat == "semantic":
        bp_pct = st.slider(
            "Breakpoint Percentile", 50, 99,
            value=cfg["chunking"].get("breakpoint_percentile", 95),
            key="bp_pct_sl",
        )
        update_pipeline_config("chunking", "breakpoint_percentile", bp_pct)

    # ── Dynamic estimate based on the most recently uploaded document ─────────
    # Files are only staged (not parsed) at upload time, so total_words isn't
    # known yet — that only becomes available once the pipeline has parsed the
    # file. Until then, fall back to the generic sample-document estimate.
    uploaded_meta = st.session_state.get("uploaded_files_meta", [])
    latest_words  = uploaded_meta[-1].get("total_words", 0) if uploaded_meta else 0
    if latest_words:
        doc_tokens   = int(latest_words * 1.3)
        doc_label    = uploaded_meta[-1].get("filename", "your uploaded document")
        source_note  = f"based on **{doc_label}** (~{doc_tokens:,} tokens)"
    else:
        doc_tokens  = 10_000
        source_note = "based on a sample 10,000-token document — run the pipeline for a real estimate"

    estimated = estimate_chunks(doc_tokens, chunk_size, overlap)
    st.info(f"📊 Estimated chunks {source_note}: **~{estimated:,} chunks**")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — EMBEDDING MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Phase 3: Embedding Model", expanded=False):

    model_labels = {
        k: f"{k}  {v['badge']}  |  {v['dims']} dims  |  {v['speed']}  |  {v['cost']}"
        for k, v in EMBEDDING_MODELS.items()
    }
    current_model  = cfg["embedding"].get("model", "text-embedding-3-small")
    selected_model = st.radio(
        "Embedding Model",
        options=list(model_labels.keys()),
        format_func=lambda k: model_labels[k],
        index=list(model_labels.keys()).index(current_model),
        key="embed_model_radio",
        label_visibility="collapsed",
    )
    update_pipeline_config("embedding", "model", selected_model)
    # Auto-update dimensions in vector_db config
    update_pipeline_config("vector_db", "dimensions", EMBEDDING_MODELS[selected_model]["dims"])

    # Comparison table
    comp_rows = [
        {
            "Model":      k,
            "Dimensions": v["dims"],
            "Speed":      v["speed"],
            "Cost":       v["cost"],
            "Use Case":   v["use_case"],
        }
        for k, v in EMBEDDING_MODELS.items()
    ]
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    e_col1, e_col2 = st.columns(2)
    with e_col1:
        batch_size = st.slider(
            "Batch Size", 1, 500,
            value=cfg["embedding"].get("batch_size", 100),
            key="batch_size_sl",
        )
        update_pipeline_config("embedding", "batch_size", batch_size)

    with e_col2:
        max_retries = st.number_input(
            "Max Retries", 1, 10,
            value=cfg["embedding"].get("max_retries", 3),
            key="max_retries_ni",
        )
        update_pipeline_config("embedding", "max_retries", int(max_retries))

        retry_on_fail = st.toggle(
            "Retry on Failure",
            value=cfg["embedding"].get("retry_on_fail", True),
            key="retry_tog",
        )
        update_pipeline_config("embedding", "retry_on_fail", retry_on_fail)

    # Cost estimate
    costs = {"text-embedding-3-small": 0.020, "text-embedding-3-large": 0.130, "text-embedding-ada-002": 0.100}
    cost_per_m = costs.get(selected_model, 0)
    st.info(f"💰 Estimated cost for 1M tokens: **${cost_per_m:.3f}**")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — VECTOR DATABASE (PINECONE)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Phase 4: Vector Database (Pinecone)", expanded=False):

    vdb = cfg["vector_db"]

    # ── Index config ──────────────────────────────────────────────────────────
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        # Fetch available indexes from backend (cached 60 s)
        @st.cache_data(ttl=60, show_spinner=False)
        def _fetch_indexes():
            return get_api_client().list_pinecone_indexes()

        idx_data    = _fetch_indexes()
        idx_list    = idx_data.get("indexes", ["rag-intelligence"])
        active_idx  = vdb.get("index_name") or idx_data.get("active", "rag-intelligence")
        # Ensure active index is in the list
        if active_idx not in idx_list:
            idx_list = [active_idx] + idx_list

        selected_index = st.selectbox(
            "Index Name",
            options=idx_list,
            index=idx_list.index(active_idx),
            help="Pinecone indexes available in your account",
            key="index_name_sel",
        )
        update_pipeline_config("vector_db", "index_name", selected_index)

        dims = vdb.get("dimensions", 1536)
        st.number_input("Dimensions (auto)", value=dims, disabled=True, key="dims_ni")

    with v_col2:
        metric_opts = {k: f"{v['label']} {v['badge']}" for k, v in VECTOR_METRICS.items()}
        current_metric = vdb.get("metric", "cosine")
        selected_metric = st.radio(
            "Distance Metric",
            options=list(metric_opts.keys()),
            format_func=lambda k: metric_opts[k],
            index=list(metric_opts.keys()).index(current_metric),
            key="metric_radio",
        )
        update_pipeline_config("vector_db", "metric", selected_metric)

    st.divider()

    # ── Search parameters ─────────────────────────────────────────────────────
    st.markdown(
        f"<h5>{icon_text('search', 'Search Parameters', size=15, color='#069494')}</h5>",
        unsafe_allow_html=True,
    )
    st.info(
        "**ℹ️ Pinecone managed index** — `efSearch` and `nProbe` are internal HNSW/IVF "
        "parameters not exposed by Pinecone. They are configurable in self-hosted vector "
        "databases such as **Milvus**. `Top-K` and `Score Threshold` apply to all backends.",
        icon="📌",
    )
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.slider(
            "efSearch  *(not applicable — Pinecone managed)*",
            10, 500,
            value=vdb.get("ef_search", 100),
            disabled=True,
            help="HNSW efSearch is managed internally by Pinecone and cannot be set via the API. Configurable in Milvus.",
            key="ef_search_sl",
        )

        st.slider(
            "nProbe  *(not applicable — Pinecone managed)*",
            1, 100,
            value=vdb.get("n_probe", 10),
            disabled=True,
            help="IVF nProbe is managed internally by Pinecone and cannot be set via the API. Configurable in Milvus.",
            key="n_probe_sl",
        )

    with s_col2:
        top_k = st.slider(
            "Top-K", 1, 50,
            value=vdb.get("top_k", 10),
            help="Number of results to retrieve",
            key="top_k_sl",
        )
        update_pipeline_config("vector_db", "top_k", top_k)

        score_threshold = st.slider(
            "Score Threshold", 0.0, 1.0,
            value=float(vdb.get("score_threshold", 0.75)),
            step=0.01,
            help="Minimum similarity score to include in results",
            key="score_thr_sl",
        )
        update_pipeline_config("vector_db", "score_threshold", score_threshold)

    st.divider()

    # ── Decay scoring ─────────────────────────────────────────────────────────
    st.markdown(
        f"<h5>{icon_text('clock', 'Decay Score (Time-based Ranking)', size=15, color='#069494')}</h5>",
        unsafe_allow_html=True,
    )
    d_col1, d_col2, d_col3 = st.columns(3)

    with d_col1:
        decay_enabled = st.toggle(
            "Enable Decay",
            value=vdb.get("decay_enabled", True),
            key="decay_tog",
        )
        update_pipeline_config("vector_db", "decay_enabled", decay_enabled)

    with d_col2:
        decay_factor = st.slider(
            "Decay Factor", 0.0, 1.0,
            value=float(vdb.get("decay_factor", 0.5)),
            step=0.05,
            disabled=not decay_enabled,
            key="decay_factor_sl",
        )
        update_pipeline_config("vector_db", "decay_factor", decay_factor)

    with d_col3:
        scale_opts = {"7": "7 days", "30": "30 days", "90": "90 days", "180": "180 days", "365": "365 days"}
        cur_scale  = str(vdb.get("decay_scale_days", 30))
        sel_scale  = st.selectbox(
            "Decay Scale",
            options=list(scale_opts.keys()),
            format_func=lambda k: scale_opts[k],
            index=list(scale_opts.keys()).index(cur_scale) if cur_scale in scale_opts else 1,
            disabled=not decay_enabled,
            key="decay_scale_sel",
        )
        update_pipeline_config("vector_db", "decay_scale_days", int(sel_scale))

    st.code(
        f"score_final = score × e^(−{decay_factor:.2f} × age_in_scale_units)",
        language=None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT MANAGEMENT TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Indexed Documents")

doc_col1, doc_col2, doc_col3 = st.columns([2, 2, 1])
with doc_col1:
    doc_search = st.text_input("Search by name", key="doc_search_ti", label_visibility="collapsed",
                                placeholder="Search by filename…")
with doc_col2:
    type_filter = st.selectbox("Filter by type", ["All", "pdf", "docx", "csv", "html", "txt", "xlsx"],
                                key="type_filter_sel", label_visibility="collapsed")
with doc_col3:
    if st.button("Refresh", key="refresh_docs"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Loading documents…"):
    documents = client.list_documents()

# Apply filters
if doc_search:
    documents = [d for d in documents if doc_search.lower() in (d.get("filename") or d.get("name", "")).lower()]
if type_filter != "All":
    documents = [d for d in documents if d.get("file_type", "") == type_filter]

if documents:
    doc_rows = []
    for d in documents:
        doc_rows.append({
            "Name":        d.get("filename") or d.get("name", "—"),
            "Type":        d.get("file_type", "—").upper(),
            "Size":        format_file_size(d.get("size_bytes", 0)),
            "Pages":       d.get("page_count") or d.get("pages", "—"),
            "Chunks":      d.get("chunk_count", 0),
            "Status":      d.get("status", "—").capitalize(),
            "Uploaded At": format_timestamp(d.get("parsed_at") or d.get("uploaded_at", "")),
            "Doc ID":      d.get("doc_id", ""),
        })

    df = pd.DataFrame(doc_rows)
    st.markdown(
        f"<p style='color:#8888AA;font-size:0.82rem'>Showing {len(documents)} document(s)</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(df.drop(columns=["Doc ID"]), use_container_width=True, hide_index=True)

    # Delete per doc
    st.markdown(
        f"<h5>{icon_text('trash-2', 'Delete Document', size=15, color='#DC2626')}</h5>",
        unsafe_allow_html=True,
    )
    del_col1, del_col2 = st.columns([3, 1])
    with del_col1:
        del_target = st.selectbox(
            "Select document to delete",
            options=[d["doc_id"] for d in documents],
            format_func=lambda did: next(
                (d.get("filename") or d.get("name", did) for d in documents if d["doc_id"] == did), did
            ),
            key="del_doc_sel",
            label_visibility="collapsed",
        )
    with del_col2:
        if st.button("Delete", key="del_doc_btn"):
            if del_target:
                try:
                    client.delete_document(del_target)
                    st.success("Document deleted successfully.")
                except RuntimeError as exc:
                    st.error(f"Delete failed: {exc}")
                except Exception as exc:
                    st.error(f"Could not reach the API: {exc}")
                st.rerun()
else:
    st.info("No documents found. Upload documents in Phase 1 above.")
