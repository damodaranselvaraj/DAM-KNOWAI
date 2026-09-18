"""
App-wide constants for the RAG Pipeline frontend.
"""

# NOTE: API_BASE_URL is NOT defined here — use `frontend.config.settings.API_BASE_URL`
# as the single source of truth (it also reads from the same .env / API_BASE_URL var).

# ─── File Upload ──────────────────────────────────────────────────────────────
SUPPORTED_FILE_TYPES: dict = {
    "pdf":  {"label": "PDF",  "icon": "📄", "max_mb": 50, "notes": "Text-based only"},
    "docx": {"label": "DOCX", "icon": "📝", "max_mb": 25, "notes": "Full support"},
    "csv":  {"label": "CSV",  "icon": "📊", "max_mb": 10, "notes": "Tabular data"},
    "txt":  {"label": "TXT",  "icon": "📃", "max_mb": 10, "notes": "Plain text"},
    "xlsx": {"label": "XLSX", "icon": "📊", "max_mb": 10, "notes": "Tabular data"},
}

# ─── Parser Options ───────────────────────────────────────────────────────────
PARSER_OPTIONS: dict = {
    "pymupdf":    {
        "label": "PyMuPDF",
        "badge": "",
        "description": "Fast, lightweight. Best for clean, text-based PDFs.",
    },
    "docling":    {
        "label": "Docling",
        "badge": "⭐ Recommended",
        "description": "Best for complex layouts, tables, and mixed content.",
    },
    "llamaindex": {
        "label": "LlamaIndex Reader",
        "badge": "",
        "description": "Universal fallback loader with broad format support.",
    },
    "auto":       {
        "label": "Auto-Select",
        "badge": "",
        "description": "PyMuPDF for PDFs; Docling → LlamaIndex fallback for others.",
    },
}

# ─── Chunking Strategies ──────────────────────────────────────────────────────
CHUNKING_STRATEGIES: dict = {
    "recursive":        {
        "label": "Recursive Chunking",
        "description": "Hierarchically splits using a list of separators. Falls back to smaller separators when needed.",
    },
    "hierarchical":     {
        "label": "Hierarchical Chunking",
        "description": "Creates parent/child chunk structure. Enables multi-granularity retrieval.",
    },
    "semantic":         {
        "label": "Semantic Chunking",
        "description": "Groups sentences by semantic similarity using embedding distance. Best for heterogeneous documents.",
    },
    "sentence":         {
        "label": "Sentence-based",
        "description": "Splits at sentence boundaries. Preserves linguistic units for better semantic coherence.",
    },
    "fixed_overlap":    {
        "label": "Fixed-size with Overlap",
        "description": "Fixed chunks with configurable token overlap between consecutive chunks. Reduces context boundary loss.",
    },
    "fixed_size":       {
        "label": "Fixed-size",
        "description": "Splits text into equal-sized chunks by token count. Simple and predictable.",
    },
}

# ─── Embedding Models ─────────────────────────────────────────────────────────
EMBEDDING_MODELS: dict = {
    "text-embedding-3-small": {
        "dims": 1536,
        "speed": "Fast",
        "cost": "$0.020 / 1M tokens",
        "use_case": "Most use cases",
        "badge": "⭐ Recommended",
    },
    "text-embedding-3-large": {
        "dims": 3072,
        "speed": "Moderate",
        "cost": "$0.130 / 1M tokens",
        "use_case": "Complex semantic tasks",
        "badge": "",
    },
    "text-embedding-ada-002": {
        "dims": 1536,
        "speed": "Fast",
        "cost": "$0.100 / 1M tokens",
        "use_case": "Legacy / backward compat",
        "badge": "",
    },
}

# ─── Vector Metrics ───────────────────────────────────────────────────────────
VECTOR_METRICS: dict = {
    "cosine":      {"label": "Cosine Similarity",       "badge": "⭐ Recommended", "description": "Best for text embeddings. Measures angle between vectors."},
    "euclidean":   {"label": "Euclidean Distance (L2)", "badge": "",               "description": "Measures absolute distance. Sensitive to magnitude."},
    "dotproduct":  {"label": "Dot Product",             "badge": "",               "description": "Fast. Requires normalised vectors for meaningful results."},
}

# ─── Pipeline Phases ──────────────────────────────────────────────────────────
PHASE_NAMES:  list[str] = ["parse", "chunk", "embed", "store"]
PHASE_LABELS: dict = {
    "parse": "Phase 1: Parse",
    "chunk": "Phase 2: Chunk",
    "embed": "Phase 3: Embed",
    "store": "Phase 4: Store",
}
PHASE_ICONS: dict = {
    "complete":    "✅",
    "in_progress": "🔄",
    "pending":     "⏳",
    "failed":      "❌",
}

# ─── Default Pipeline Config ──────────────────────────────────────────────────
DEFAULT_PIPELINE_CONFIG: dict = {
    "parser": {
        "parser_type":       "auto",
        "version_handling":  "replace",
        "ocr_enabled":       False,
    },
    "chunking": {
        "strategy":              "recursive",
        "chunk_size":            512,
        "overlap":               50,
        "min_chunk":             100,
        "max_chunk":             1024,
        "separators":            ["\n\n", "\n", " ", ""],
        "breakpoint_percentile": 95,
    },
    "embedding": {
        "model":          "text-embedding-3-small",
        "batch_size":     100,
        "retry_on_fail":  True,
        "max_retries":    3,
    },
    "vector_db": {
        "index_name":      "rag-intelligence",
        "dimensions":      1536,
        "metric":          "cosine",
        "ef_search":       100,
        "n_probe":         10,
        "top_k":           10,
        "score_threshold": 0.75,
        "decay_enabled":   True,
        "decay_factor":    0.5,
        "decay_scale_days": 30,
    },
}

# ─── Default Guardrail Config ─────────────────────────────────────────────────
# Mirrors backend.models.guardrail_config.GuardrailConfig defaults exactly —
# every guardrail enabled by default.
DEFAULT_GUARDRAIL_CONFIG: dict = {
    "toggles": {
        "enable_input_length_check":        True,
        "enable_toxicity_check":             True,
        "enable_prompt_injection_check":     True,
        "enable_jailbreak_check":            True,
        "enable_pii_masking_input":          True,
        "enable_intent_classification":      True,
        "enable_out_of_scope_check":         True,
        "enable_metadata_filtering":         True,
        "enable_hybrid_retrieval":           True,
        "enable_relevance_threshold_gate":   True,
        "enable_context_dedup":              True,
        "enable_context_relevance_scoring":  True,
        "enable_groundedness_check":         True,
        "enable_hallucination_check":        True,
        "enable_citation_validation":        True,
        "enable_pii_masking_output":         True,
        "enable_output_toxicity_check":      True,
    },
    "thresholds": {
        "relevance_threshold":          0.70,
        "max_characters":               2000,
        "max_tokens":                   500,
        "max_documents":                10,
        "max_query_length":             300,
        "top_n_retrieval":              20,
        "dedup_similarity_threshold":   0.95,
        "groundedness_threshold":       0.75,
        "hallucination_threshold":      0.5,
    },
}

# ─── Guardrail layer/checkbox metadata ─────────────────────────────────────────
# Drives the checkbox sections on the Guardrails page. Each entry maps a
# GuardrailToggles field name to its display label + help text, grouped by
# pipeline layer.
GUARDRAIL_LAYERS: list[dict] = [
    {
        "key": "layer1",
        "title": "Layer 1 — Input Guardrails",
        "description": "Checks applied to the raw user query before any processing begins.",
        "icon": "shield",
        "checks": [
            ("enable_input_length_check", "Input Length / Token Validation",
             "Rejects queries exceeding max characters, tokens, word count, or document count."),
            ("enable_toxicity_check", "Toxicity / Abuse Detection",
             "Blocks hateful, abusive, profane, or violent language in the query."),
            ("enable_prompt_injection_check", "Prompt Injection Detection",
             "Detects attempts to override system instructions or hijack the pipeline."),
            ("enable_jailbreak_check", "Jailbreak Detection",
             "Detects roleplay / fictional-framing / encoded jailbreak attempts."),
            ("enable_pii_masking_input", "PII Detection + Masking (Input)",
             "Masks PAN, Aadhaar, credit card, bank account, email, phone, address, employee/customer IDs."),
        ],
    },
    {
        "key": "layer2",
        "title": "Layer 2 — Query Control",
        "description": "Classifies and routes the query before embedding or retrieval.",
        "icon": "git-branch",
        "checks": [
            ("enable_intent_classification", "Intent Classification / Query Router",
             "Classifies the query as greeting, casual, knowledge, document, or out-of-scope."),
            ("enable_out_of_scope_check", "Out-of-Scope Detection",
             "Blocks queries unrelated to the enterprise knowledge domain with a fixed refusal."),
        ],
    },
    {
        "key": "layer3",
        "title": "Layer 3 — Retrieval Safety",
        "description": "Applied while retrieving candidate documents.",
        "icon": "search",
        "checks": [
            ("enable_metadata_filtering", "Metadata Filtering",
             "Restricts retrieval to documents within the caller's department / access level / date range."),
            ("enable_hybrid_retrieval", "Hybrid Retrieval (Top-N)",
             "Retrieves the configured Top-N documents (dense + sparse) before reranking."),
        ],
    },
    {
        "key": "layer4",
        "title": "Layer 4 — Context Control",
        "description": "Applied after reranking, before the context reaches the LLM.",
        "icon": "filter",
        "checks": [
            ("enable_relevance_threshold_gate", "Relevance Threshold Gate",
             "Drops reranked documents scoring below the relevance threshold."),
            ("enable_context_dedup", "Context Deduplication",
             "Removes exact and near-duplicate chunks from hybrid retrieval."),
            ("enable_context_relevance_scoring", "Context Relevance Scoring",
             "Final query-to-chunk relevance check; drops chunks that still fall short."),
        ],
    },
    {
        "key": "layer5",
        "title": "Layer 5 — Generation Safety",
        "description": "Applied to the LLM's answer before any output-level checks.",
        "icon": "check-circle",
        "checks": [
            ("enable_groundedness_check", "Groundedness Check",
             "Verifies the answer's claims are supported by the retrieved context."),
            ("enable_hallucination_check", "Hallucination Detection",
             "Cross-references entities/facts in the answer against the retrieved context."),
            ("enable_citation_validation", "Citation Validation",
             "Strips citations that don't correspond to an actual retrieved source."),
        ],
    },
    {
        "key": "layer6",
        "title": "Layer 6 — Output Safety",
        "description": "The final checks before the response is returned to the user.",
        "icon": "shield-check",
        "checks": [
            ("enable_pii_masking_output", "PII Detection + Masking (Output)",
             "Masks any PII the LLM may have reproduced from the retrieved context."),
            ("enable_output_toxicity_check", "Output Toxicity / Abuse Detection",
             "Blocks toxic, harmful, or inappropriate generated content."),
        ],
    },
]

# ─── Color Scheme ─────────────────────────────────────────────────────────────
COLOR_SCHEME: dict = {
    "primary":          "#6C63FF",
    "secondary":        "#48CAE4",
    "success":          "#06D6A0",
    "warning":          "#FFD166",
    "danger":           "#EF476F",
    "bg_dark":          "#0F0F1A",
    "bg_card":          "#1A1A2E",
    "bg_surface":       "#16213E",
    "text_primary":     "#E0E0FF",
    "text_secondary":   "#8888AA",
    "border":           "#2A2A4A",
}

# ─── Score thresholds ─────────────────────────────────────────────────────────
SCORE_THRESHOLDS: dict = {
    "excellent": 0.90,  # green
    "good":      0.75,  # yellow
    "fair":      0.60,  # orange
    # below fair → red
}

# ─── Example chat prompts ─────────────────────────────────────────────────────
EXAMPLE_QUESTIONS: list[str] = [
    "What are the key compliance requirements in the policy?",
    "Summarise the risk assessment findings from Q3.",
    "What data handling procedures must employees follow?",
]
