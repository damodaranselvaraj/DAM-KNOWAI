# RAG Intelligence Platform

A production-ready Retrieval-Augmented Generation (RAG) backend built with FastAPI and a Streamlit frontend. Uploads documents, parses and chunks them, embeds with OpenAI, stores vectors in Pinecone, and answers questions via a multi-turn chat interface with inline citations.

---

## Table of Contents

1. [Getting Started — Local Setup Guide](#1-getting-started--local-setup-guide)
2. [Technical Deep Dive — RAG Pipeline Architecture](#2-technical-deep-dive--rag-pipeline-architecture)

---

## 1. Getting Started — Local Setup Guide

### Prerequisites

Before you begin, make sure you have the following installed and ready:

| Requirement | Version | How to check |
|---|---|---|
| Python | 3.10 or later | `python --version` |
| pip | latest | `pip --version` |
| Git | any recent version | `git --version` |

You also need accounts and API keys for:

- **OpenAI** — [platform.openai.com](https://platform.openai.com) → API Keys
- **Pinecone** — [app.pinecone.io](https://app.pinecone.io) → API Keys
- **Cohere** (used for reranking, enabled by default) — [cohere.com](https://cohere.com). The backend starts fine without `COHERE_API_KEY`, but reranking will be disabled for chat requests and a warning is logged at startup.

> **Windows users:** Run all commands in PowerShell or Windows Terminal. The commands below work as-is on Windows.

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/rag-intelligence-platform.git
cd rag-intelligence-platform
```

---

### Step 2 — Create and activate a virtual environment

A virtual environment keeps this project's dependencies isolated from the rest of your system.

```bash
# Create it
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate it (macOS / Linux)
source .venv/bin/activate
```

**Expected output after activation:** Your terminal prompt will show `(.venv)` at the start.

> If you see a PowerShell execution-policy error, run:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then try again.

---

### Step 3 — Install dependencies

All backend and frontend dependencies are in a single file:

```bash
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, Streamlit, OpenAI SDK, Pinecone client, PyMuPDF, Docling, LlamaIndex, Cohere (reranking), sentence-transformers, rank-bm25, nltk, and everything else the platform needs.

> `pytest` is not pinned in `requirements.txt`. If you want to run the test suite (see [Running Tests](#running-tests)), install it separately: `pip install pytest`.

**Expected output:** A long list of package installs ending with `Successfully installed ...`. This can take 2–5 minutes on a fresh environment.

> If pip complains about a dependency conflict, try upgrading pip first:
> `python -m pip install --upgrade pip` then re-run the install.

---

### Step 4 — Configure environment variables

Copy the example file and fill in your keys:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Now open `.env` in any text editor and fill in the required values:

```env
# Required — OpenAI
OPENAI_API_KEY=sk-...                  # Your OpenAI secret key

# Required — Pinecone
PINECONE_API_KEY=pcsk-...             # Your Pinecone API key
PINECONE_ENVIRONMENT=us-east-1-aws    # From your Pinecone project settings

# Optional — Cohere (leave blank to skip reranking)
COHERE_API_KEY=...
```

Everything else in `.env` has sensible defaults and does not need to change for a local setup.

**Full list of variables and what they do** (grouped to match the section comments in `backend/config.py`; all fields come from the `Settings` class there):

*OpenAI*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | — | ✅ | OpenAI authentication |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | | Embedding model |
| `OPENAI_EMBEDDING_DIMS` | `1536` | | Dimensions of the embedding model output |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | | Chat completion model |
| `OPENAI_MAX_TOKENS` | `8191` | | Max tokens per completion |

*Pinecone*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `PINECONE_API_KEY` | — | ✅ | Pinecone authentication |
| `PINECONE_INDEX_NAME` | `rag-intelligence` | | Pinecone index to use |
| `PINECONE_METRIC` | `cosine` | | `cosine`, `euclidean`, or `dotproduct` |
| `PINECONE_CLOUD` | `aws` | | Serverless cloud provider |
| `PINECONE_REGION` | `us-east-1` | | Serverless region |
| `PINECONE_ENVIRONMENT` | `us-east-1-aws` | | Legacy alias kept for `.env` compatibility; not used to create serverless indexes |

*Cohere*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `COHERE_API_KEY` | `""` | ✅ for chat | Reranker authentication — the reranker is Cohere-only (see [Phase 5d](#5d--reranking)) and construction fails without this key |

*Memory / checkpoints*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `MEMORY_BACKEND` | `sqlite` | | `sqlite` for durable memory, `memory` for dev |
| `SQLITE_DB_PATH` | `.data/memory/checkpoints.db` | | Where chat history is stored |
| `SQLITE_WAL_MODE` | `True` | | Enables WAL mode for concurrent readers |
| `SQLITE_POOL_TIMEOUT` | `30` | | Connection timeout (seconds) |
| `MEMORY_MAX_CHECKPOINTS` | `0` | | Per-thread cap (SQLite backend); `0` = unlimited |
| `MEMORY_PRUNE_DAYS` | `30` | | Delete checkpoints older than N days |
| `MEMORY_KEEP_LAST` | `5` | | Always keep N most-recent checkpoints per thread |
| `MEMORY_MAX_PER_THREAD` | `100` | | Cap for `InMemorySaver` only (ignored when `MEMORY_BACKEND=sqlite`) |

*Dense retrieval*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `DENSE_TOP_K` | `50` | | Results requested per dense query |
| `DENSE_TIMEOUT_SECONDS` | `10` | | Timeout for the dense retrieval arm |
| `DENSE_MAX_RETRIES` | `3` | | Retry attempts on failure |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | | Alias used by `DenseRetrieverConfig` |
| `EMBEDDING_DIMENSION` | `1536` | | Alias used by `DenseRetrieverConfig` |

*Sparse retrieval (BM25)*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `SPARSE_TOP_K` | `50` | | Results requested per sparse query |
| `BM25_K1` | `1.5` | | BM25 term-frequency saturation parameter |
| `BM25_B` | `0.75` | | BM25 length-normalization parameter |
| `BM25_CORPUS_PATH` | `.cache/bm25/corpus.pkl` | | Local BM25 index cache path |
| `BM25_USE_STEMMING` | `False` | | Enable stemming during tokenization |
| `BM25_VOCAB_CAP` | `50000` | | Max vocabulary size |
| `STOPWORDS_LANG` | `english` | | NLTK stopwords language |

*Hybrid retrieval*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `RETRIEVAL_MODE` | `hybrid_bm25` | | `dense`, `hybrid_bm25`, or `hybrid_splade` — see [Phase 5c](#5c--hybrid-fusion-rrf) |
| `RRF_K` | `60` | | RRF smoothing constant |
| `DENSE_WEIGHT` | `0.5` | | Dense arm weight (used by weighted fusion paths) |
| `SPARSE_WEIGHT` | `0.5` | | Sparse arm weight |
| `RETRIEVAL_TIMEOUT_SECONDS` | `10.0` | | Overall retrieval timeout |

*Reranker (Cohere)*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `RERANKER_MODEL` | `rerank-v3.5` | | Cohere rerank model name |
| `RERANKER_TOP_K` | `10` | | Candidates sent into the reranker |
| `FINAL_TOP_K` | `5` | | Results kept after reranking and passed to the prompt builder |

*Prompt / LLM token budgets*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `PROMPT_MAX_CONTEXT_TOKENS` | `6000` | | Max tokens of retrieved context in the prompt |
| `PROMPT_MAX_HISTORY_TOKENS` | `2000` | | Max tokens of conversation history in the prompt |

*CORS*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `CORS_ORIGINS` | `*` | | Comma-separated allowed origins; set explicitly in production |

*App defaults*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `APP_ENV` | `development` | | Environment name |
| `MAX_FILE_SIZE_MB` | `50` | | Max upload size |
| `ALLOWED_EXTENSIONS` | `pdf,docx,csv,html,txt,xlsx` | | Comma-separated allowed upload extensions |
| `DEFAULT_CHUNK_SIZE` | `400` | | Default chunk size (tokens) |
| `DEFAULT_CHUNK_OVERLAP` | `60` | | Default chunk overlap (tokens) |
| `DEFAULT_TOP_K` | `10` | | Default retrieval `top_k` |
| `DEFAULT_SCORE_THRESHOLD` | `0.75` | | Default minimum similarity score |
| `DEFAULT_BATCH_SIZE` | `100` | | Default embedding batch size |
| `DEFAULT_MAX_RETRIES` | `3` | | Default retry count |
| `LOG_LEVEL` | `INFO` | | Application log level |
| `MAX_CHUNKING_INPUT_CHARS` | `2000000` | | Safety ceiling on characters processed per `chunk_document()` call |

*Pipeline defaults*

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `PIPELINE_DEFAULT_TENANT` | `default` | | Default tenant id used for namespacing |
| `PIPELINE_DEFAULT_CORPUS` | `default` | | Default corpus id used for namespacing |

---

### Step 5 — Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     RAG Pipeline API starting up…
INFO:     Memory saver ready: <SQLiteSaver ...>
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Verify it works — open your browser and go to:
- **API docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check:** [http://localhost:8000/api/v1/health/ping](http://localhost:8000/api/v1/health/ping)

The health endpoint should return: `{"status": "ok", "latency_ms": ...}`

> **`--reload`** restarts the server automatically when you change a source file. Remove it in production.

**Common backend errors:**

| Error | Cause | Fix |
|---|---|---|
| `ValidationError: openai_api_key field required` | `.env` file is missing or not found | Make sure `.env` is in the project root (same folder as `requirements.txt`) |
| `pinecone.exceptions.PineconeApiException: 401` | Wrong Pinecone API key | Double-check `PINECONE_API_KEY` in `.env` |
| `Address already in use` | Port 8000 is taken | Use `--port 8001` or stop the other process |
| `ModuleNotFoundError: No module named 'backend'` | Running from wrong directory | Run the command from the project root, not inside `backend/` |

---

### Step 6 — Start the frontend

Open a **second terminal**, activate the same virtual environment, then run:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Then start Streamlit
streamlit run frontend/streamlit_app.py --server.port 8501
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL:  http://localhost:8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The sidebar will show a live API status indicator — it should say connected (green dot) if the backend is running.

> If you see API connection errors in the UI, confirm the backend is running on port 8000. The frontend is pre-configured to call `http://localhost:8000/api/v1`.

---

### Step 7 — Upload a document and chat

1. Click **Admin Panel** in the sidebar.
2. Upload a PDF, DOCX, CSV, HTML, TXT, or XLSX file (max 50 MB).
3. Click **Run Pipeline** to parse, chunk, embed, and store the document.
4. Watch the four-phase stepper (Parse → Chunk → Embed → Store) complete.
5. Go to **Chatbot** in the sidebar and ask a question about your document.

**Other frontend pages** (all under `frontend/pages/`, navigable from the sidebar via `frontend/components/nav_sidebar.py`):

| Page | File | Purpose |
|---|---|---|
| Home | `01_home.py` | Landing dashboard (the app redirects here by default from `streamlit_app.py`) |
| Admin Panel | `02_admin.py` | Upload documents and run the ingestion pipeline |
| Chatbot | `03_chatbot.py` | Multi-turn RAG chat with citations |
| Evaluation | `04_evaluation.py` | Runs and displays retrieval evaluation metrics (Recall@K, MRR, NDCG@K) |
| Retrieval Options | `05_retrieval.py` | Adjust retrieval settings shared with the Chatbot page |
| Guardrails | `06_guardrails.py` | Configure the 6-layer guardrail pipeline (input/output safety, PII masking, groundedness, etc.) applied to every chat query |
| Settings | `07_settings.py` | App-level API connection and chat defaults (separate from pipeline config) |

---

### Running both services together (quick reference)

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/streamlit_app.py --server.port 8501
```

---

### Alternative: Run with Docker

The project ships a `docker/` directory with `Dockerfile.backend`, `Dockerfile.frontend`, and `docker-compose.yml` for a containerized setup.

```bash
# From the repository root — make sure .env exists (Step 4) and is filled in
docker compose -f docker/docker-compose.yml up --build
```

- **Backend** — built from `docker/Dockerfile.backend` (base image `python:3.11-slim`), exposed on port `8000`, healthchecked against `/api/v1/health/ping` every 30s. Pre-downloads NLTK corpora (`stopwords`, `punkt`, `punkt_tab`) at build time so there are no runtime network calls.
- **Frontend** — built from `docker/Dockerfile.frontend`, exposed on port `8501`, depends on the backend service and reaches it at `http://backend:8000/api/v1` (set via the `API_BASE_URL` env var, matching `frontend/config.py`'s `FrontendSettings`) instead of `localhost`.
- Both containers read secrets from the repo-root `.env` via `env_file` — do not commit a populated `.env`.
- Persistent data (`.data/`) and cache (`.cache/`) are bind-mounted from the repo root into the backend container.

> The Docker image runs Python 3.11, while the project's declared minimum (`pyproject.toml`) is 3.10 — both are compatible, just worth knowing if you're comparing environments.

Once running, the app is available at the same URLs as the manual setup: [http://localhost:8000/docs](http://localhost:8000/docs) and [http://localhost:8501](http://localhost:8501).

---

### Running Tests

Tests live under `tests/` (`tests/unit/`, `tests/integration/`, plus top-level phase tests) and are configured in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["."]`).

```bash
# Install pytest if you haven't already (not pinned in requirements.txt)
pip install pytest

# Run the full suite from the repository root
pytest

# Run a subset
pytest tests/unit
pytest tests/integration
```

---

## 2. Technical Deep Dive — RAG Pipeline Architecture

The pipeline has six sequential phases triggered on document upload (Phases 1–4) and at query time (Phases 5–6). All configurable defaults live in `backend/models/pipeline_config.py` and `backend/config.py`.

```
Document Upload
│
├── Phase 1 — Parse          (backend/ingestion/parsers/)
├── Phase 2 — Chunk          (backend/ingestion/chunkers/)
├── Phase 3 — Embed          (backend/ingestion/embedders/)
└── Phase 4 — Store          (backend/vector_store/)

Chat Query
│
├── Phase 5 — Retrieve       (backend/retrieval/)
│   ├── Dense retrieval      (Pinecone ANN search)
│   ├── Sparse retrieval     (BM25 / SPLADE)
│   ├── RRF fusion
│   └── Reranking
└── Phase 6 — Generate       (backend/llm/)
    ├── Prompt assembly      (PromptBuilder)
    └── LLM completion       (OpenAIClient → GPT-4o)
```

Memory (conversation history) is managed orthogonally across all chat turns via `backend/memory/`.

---

### Phase 1 — Parsing

**What it does:** Converts raw uploaded bytes into a structured `ParsedDocument` (plain text + page metadata). This is the point where format-specific complexity — scanned PDFs, embedded tables, multi-column layouts — is resolved.

**Why it matters:** All downstream phases operate on the text output of this phase. A bad parse corrupts everything downstream.

**Entry point:** `ParserRouter.route(request: ParseRequest) -> ParseResult` in `backend/ingestion/parsers/parser_router.py`

**Available parsers:**

| Parser | Class | Supported types | Strength |
|---|---|---|---|
| `PyMuPDFParser` | `pymupdf_parser.py` | PDF only | Fast, reliable for text-native PDFs |
| `DoclingParser` | `docling_parser.py` | PDF, DOCX, CSV, HTML, TXT, XLSX | Complex layouts, tables, mixed content |
| `LlamaIndexParser` | `llamaindex_parser.py` | All types | Last-resort fallback |

**Fallback chains (defined in `_CHAIN` in `parser_router.py`):**
```
PDF:               PyMuPDF → Docling → LlamaIndex
DOCX:              Docling → LlamaIndex
CSV / HTML / TXT / XLSX:  Docling → LlamaIndex
```

The router tries each parser in order and stops at the first success. If a parser fails, a `ParserAttempt` record is appended to the `audit_trail` and the next parser is tried. The final `ParsedDocument` carries the full audit trail so you know exactly which parser produced the output and which ones failed before it.

**Default:** `ParserConfig.parser_type = "auto"` — uses the full fallback chain. Setting `force_parser` in the request overrides this and uses only that parser.

**Why PyMuPDF is first for PDFs:** It's the fastest option and handles the majority of text-based PDFs without needing heavy ML models. Docling is kept as a fallback for cases PyMuPDF can't handle (e.g. scanned pages, complex table structures).

**`parse_status`** on the returned document is set to `FALLBACK_USED` when the primary parser in the chain fails.

---

### Phase 2 — Chunking

**What it does:** Splits the parsed document text into smaller, semantically meaningful segments (`Chunk` objects) sized for embedding and retrieval. Each chunk carries its token count, page provenance, content hash, and the strategy used.

**Why it matters:** Chunk size and strategy directly determine retrieval quality. Chunks that are too large dilute signal; chunks that are too small lose context.

**Entry point:** `ChunkerFactory.get(strategy) -> BaseChunker` then `chunker.chunk(doc, config, pages)` in `backend/ingestion/chunkers/`

**Available strategies:**

| Strategy | Class | Description |
|---|---|---|
| `fixed_size` | `FixedSizeChunker` | Pure token-count splits, no overlap |
| `fixed_overlap` | `FixedOverlapChunker` | Fixed size with configurable token overlap between adjacent chunks |
| `sentence` | `SentenceBasedChunker` | Splits on sentence boundaries; avoids cutting mid-sentence |
| `recursive` | `RecursiveChunker` | Recursively splits on `["\n\n", "\n", " ", ""]` — similar to LangChain's `RecursiveCharacterTextSplitter` |
| `semantic` | `SemanticChunker` | Groups sentences by embedding cosine similarity; uses `breakpoint_percentile` to decide where to split |
| `hierarchical` | `HierarchicalChunker` | Produces parent/child chunk pairs for multi-granularity retrieval |

**Default strategy: `fixed_overlap`** — set in `ChunkingApiConfig.strategy` in `pipeline_config.py`.

**Why `fixed_overlap` is the default:** It gives a deterministic, predictable chunk structure (no ML calls at index time), ensures every sentence appears in at least one chunk through overlap, and works well across diverse document types without tuning. Semantic chunking is more accurate but adds latency and embedding cost during ingestion.

**Key configuration fields (`ChunkingConfig`):**
```python
strategy             = "fixed_overlap"
chunk_size           = 512   # tokens per chunk (range: 100–4096)
overlap              = 50    # overlap tokens between adjacent chunks (range: 0–500)
min_chunk            = 100   # discard chunks below this token count
max_chunk            = 1024  # hard cap, truncates oversize segments
separators           = ["\n\n", "\n", " ", ""]
breakpoint_percentile = 95   # semantic chunker only
```

**Deduplication:** `ChunkDeduplicator` (also in `chunker_factory.py`) deduplicates by SHA-256 `content_hash` — both within a single document and across documents during batch ingestion.

---

### Phase 3 — Embedding

**What it does:** Converts each chunk's text into a dense numerical vector (and optionally a sparse vector) suitable for vector database storage and similarity search.

**Why it matters:** The embedding model determines the semantic space everything is retrieved from. Model choice affects accuracy, latency, and cost.

**Entry point:** `EmbeddingPipeline.embed_chunks(chunks, doc_id, doc_name) -> EmbeddingResult` in `backend/ingestion/embedders/embedding_pipeline.py`

**Pipeline flow (in order):**
1. **Pre-embedding validation** — filters empty chunks, chunks exceeding the token limit, and duplicates. Optionally warns on non-English text.
2. **In-process cache lookup** — checks `content_hash → vector` in an in-memory dict cache. Cache hits skip the OpenAI call entirely within the same process lifetime.
3. **Dense embedding** (OpenAI) — batches remaining chunks, sends async requests via `OpenAIEmbedder`, writes results back to the in-process cache.
4. **Sparse encoding** (BM25 / SPLADE) — encodes each chunk's text as a sparse vector for hybrid search.
5. **Assembly** — packages each chunk's dense + sparse vector into an `EmbeddedChunk` ready for Pinecone upsert.

**Dense embedding options (`EmbeddingConfig.model`):**

| Model | Dimensions | Cost per 1k tokens | Notes |
|---|---|---|---|
| `text-embedding-3-small` | 1536 | $0.00002 | **Default** |
| `text-embedding-3-large` | 3072 | $0.00013 | Higher accuracy, ~6.5x cost |
| `text-embedding-ada-002` | 1536 | $0.0001 | Legacy, kept for compatibility |

**Default model: `text-embedding-3-small`** — best cost/performance ratio for most document Q&A use cases. `text-embedding-3-large` gives modestly better retrieval scores but at 6.5× the price.

**Sparse embedding:**
- Primary: SPLADE via HuggingFace (`naver/splade-cocondenser-selfdistil`) — requires `transformers` + `torch`
- Fallback: BM25 (`k1=1.5`, `b=0.75`, vocab cap 50,000 terms) — pure Python, no GPU needed
- Output: `SparseVector(indices, values)` in Pinecone's native sparse format

**`OpenAIEmbedder` internals:**
- `batch_size=100`, `max_concurrency=10` concurrent async batches via `asyncio.Semaphore`
- Token rate-limit guard approximates against 1,000,000 TPM (Tier-1 limit)
- `max_retries=3` with exponential backoff on rate-limit and 5xx errors
- L2-normalization is applied at query time (in `DenseRetriever.embed_query()`)

---

### Phase 4 — Storage (Ingestion)

**What it does:** Upserts `EmbeddedChunk` objects into Pinecone. Each vector is stored with its dense vector, sparse vector (for hybrid index), and metadata payload (doc name, page numbers, doc_id, chunk strategy, etc.).

**Why it matters:** Pinecone is the retrieval backend. Index configuration — metric, dimensions, namespace — must match what the embedder produces.

**Vector DB configuration (`VectorDBConfig`):**

| Field | Default | Notes |
|---|---|---|
| `index_name` | `rag-intelligence` | Pinecone index name |
| `dimensions` | `1536` | Must match `text-embedding-3-small` output |
| `metric` | `cosine` | Similarity metric; options: `cosine`, `euclidean`, `dotproduct` |
| `ef_search` | `100` | Retained for API compatibility only — **not supported by Pinecone Serverless** and never forwarded to Pinecone |
| `n_probe` | `10` | Retained for API compatibility only — **not supported by Pinecone Serverless** and never forwarded to Pinecone |
| `top_k` | `10` | Default results returned |
| `score_threshold` | `0.75` | Minimum similarity score to include a result |
| `decay_enabled` | `True` | Apply time-decay scoring |
| `decay_factor` | `0.5` | Decay strength (0 = no decay, 1 = full decay) |
| `decay_scale_days` | `30` | Half-life of the decay function in days |

These fields live in `VectorDBConfig` (`backend/models/pipeline_config.py`). The `ef_search`/`n_probe`/decay-related fields are not exposed as top-level `Settings` fields in `backend/config.py` — only the OpenAI/Pinecone/memory/retrieval settings listed in [Step 4](#step-4--configure-environment-variables) live there.

**Why `cosine` is the default metric:** Cosine similarity measures directional alignment between vectors (ignoring magnitude), which is well-suited for text embeddings because sentence length and frequency artifacts are factored out. `text-embedding-3-small` outputs are not L2-normalized at embedding time, making cosine the correct choice.

---

### Phase 5 — Retrieval

**What it does:** Given a user query, retrieves the most relevant chunks from Pinecone. Runs dense and sparse retrieval concurrently, fuses the results, then optionally reranks them.

**Why it matters:** Retrieval precision directly determines answer quality. Hybrid retrieval consistently outperforms dense-only on keyword-heavy and exact-match queries.

**Entry point:** `HybridRetriever.retrieve(query, top_k, namespace, metadata_filter, query_id)` in `backend/retrieval/hybrid_retriever.py`

Incoming queries are first sanitized/validated by `backend/retrieval/query_validator.py` before hitting the dense/sparse arms (e.g. rejecting empty queries).

The chat pipeline over-fetches at retrieval time: `max(top_k * 5, 50)` results are requested, then trimmed after reranking. This ensures the reranker has enough candidates to work with.

#### 5a — Dense Retrieval (`DenseRetriever`)

- Embeds the query with OpenAI (`text-embedding-3-small`), L2-normalizes the vector, then calls `pinecone_index.query()`.
- Supports namespace routing and metadata filters.
- `top_k=50` per request, `timeout=10s`, `max_retries=3`.
- Returns `list[RetrievalResult]` sorted by cosine similarity descending.

#### 5b — Sparse Retrieval (`SparseRetriever`)

- Encodes the query as a sparse BM25 / SPLADE vector (same `SparseEmbedder` used during ingestion).
- Queries the Pinecone sparse index for term-overlap matches.
- Catches keyword-heavy queries that dense embeddings may miss.

#### 5c — Hybrid Fusion (RRF)

- Dense and sparse retrieval run **concurrently** via `ThreadPoolExecutor(max_workers=2)` with a 10-second timeout per arm.
- Results are merged with **Reciprocal Rank Fusion** (Cormack et al., 2009):

```
RRF(d) = 1 / (k + dense_rank(d))  +  1 / (k + sparse_rank(d))
```

- `k=60` (smoothing constant; reduces the dominance of rank-1 results).
- Documents absent from one list are penalized with `rank = len(list) + 1`.
- If one retriever times out or errors, the pipeline degrades gracefully to single-arm results without failing the request.

**Retrieval modes (set via `RETRIEVAL_MODE` env var):**

| Mode | Behavior |
|---|---|
| `hybrid_bm25` | Dense (Pinecone) + local BM25 (`rank_bm25`) index, fused via RRF — **Default** |
| `hybrid_splade` | Single native Pinecone hybrid query combining dense + SPLADE sparse values |
| `dense` | Dense only |

**Why `hybrid_bm25` is the default:** Single-arm retrievers have complementary failure modes. Dense retrieval excels at semantic similarity but struggles with exact keyword matches and rare terms. Sparse (BM25) retrieval excels at keyword matching but misses paraphrases. RRF fusion of both consistently beats either in isolation on mixed query types, with essentially no added latency since the two arms run concurrently. `hybrid_splade` is available as an alternative that pushes sparse encoding into a single native Pinecone query instead of a separate local BM25 index.

#### 5d — Reranking

- After RRF fusion, inline deduplication removes duplicate chunk IDs.
- The reranker (`Reranker` in `backend/retrieval/reranker.py`) is **Cohere-only** — there is no local cross-encoder fallback. It uses model `rerank-v3.5` by default (`RERANKER_MODEL`) and hard-requires the `cohere` package and a valid `COHERE_API_KEY`; construction raises if either is missing. `backend/main.py` pre-warms the reranker at startup and catches this error, so a missing key doesn't crash the app — it just disables reranking (logged as a warning) until a key is provided.
- Up to `RERANKER_TOP_K` (default `10`) candidates are sent into the reranker.
- The final `FINAL_TOP_K` results (default **5**, not 10) are kept after reranking and passed to the prompt builder.
- `rerank_score` is stored on each `RetrievalResult` and takes precedence over raw `score` in prompt ordering.

I use Cohere Rerank as a managed reranking service. I don't expose or control its internal pointwise/pairwise/listwise ranking mechanism.

If I need to demonstrate and implement those ranking strategies myself, I can use a locally hosted reranking model/CrossEncoder.

```
                 RAG RERANKER

                      │

          ┌───────────┴────────────┐

          │                        │

      Cohere API              Local Model

          │                        │

    rerank-v3.5             CrossEncoder

          │                        │

          │                 ranking experiments

          │                   /     |      \

          │             pointwise pairwise listwise

          │

       Production

       managed API
```

Chunking and Storing:

```
                DOCUMENT CHUNKS

                     │

          ┌──────────┼──────────┐

          │          │          │

          ▼          ▼          ▼

       Dense       SPLADE     BM25

          │          │          │

          ▼          ▼          ▼

       Pinecone   Pinecone    BM25 Index
```

Retrieval:

```
                    RETRIEVAL

                       │

          ┌────────────┴────────────┐

          │                         │

       DENSE                      SPARSE

          │                         │

   OpenAI embedding          ┌──────┴──────┐

          │                   │             │

       Pinecone             BM25         SPLADE

                              │             │

                         Local index     Sparse DB
```

Query flow:

```
                         QUERY

                           │

            ┌──────────────┼──────────────┐

            │              │              │

            ▼              ▼              ▼

         Dense           BM25          SPLADE

       Retrieval       Retrieval      Retrieval

            │              │              │

            ▼              ▼              ▼

         Pinecone       BM25 Index    Pinecone

            │              │              │

            └──────────────┼──────────────┘

                           ▼

                          RRF

                           │

                           ▼

                      Top candidates

                           │

                           ▼

                    Cross-Encoder

                       Reranker

                           │

                           ▼

                          LLM
```

BM25 and SPLADE are not the same.

BM25 = traditional lexical sparse retrieval.

SPLADE = neural sparse retrieval using a transformer to produce/expand sparse term representations.

Dense embedding = semantic vector representation.

Pinecone = infrastructure that can perform vector retrieval; it is not itself BM25 or SPLADE.

---

### Phase 6 — Generation

**What it does:** Assembles the final OpenAI messages list from the retrieved chunks and conversation history, then calls GPT-4o to generate a grounded answer with inline citations.

**Why it matters:** The prompt structure determines whether the model stays grounded in the retrieved context or hallucinates. Token budgeting ensures the context window is used efficiently.

#### Prompt Builder (`PromptBuilder`, `backend/llm/prompt_builder.py`)

**`build(query, chunks, history, today) -> (messages, citation_meta)`**

Assembly order:
1. **System message** — instructs the model to answer only from provided chunks, cite inline as `[Source N]`, and append a Sources section. Today's date is injected.
2. **Conversation history** — prior turns from the memory saver, trimmed to `MAX_HISTORY_TOKENS=2000` (newest-first eviction).
3. **User message** — the current query.

**Token budgets (approximated as `chars / 3.8`):**
- Retrieved context: hard cap of `6,000` tokens
- Conversation history: hard cap of `2,000` tokens
- Chunks are ordered by `rerank_score` (fallback to `score`) and included until the context budget is exhausted. The last chunk is truncated rather than dropped if it partially fits.

**No-context fallback:** If no chunks are retrieved, the system prompt switches to `_NO_CONTEXT_SYSTEM`, which instructs the model to answer from general knowledge and clearly state the response is not grounded in uploaded documents.

**Citation format:**
```
[Source N]
Document : <doc_name>
Page(s)  : <page numbers>
Score    : <rerank_score>
Content  :
<chunk text>
```

#### OpenAI Client (`OpenAIClient`, `backend/llm/openai_client.py`)

| Setting | Value | Notes |
|---|---|---|
| Model | `gpt-4o` | Set via `OPENAI_CHAT_MODEL` |
| Temperature | `0.2` | Low temperature = factual, less creative |
| Max tokens | `8191` | Set via `OPENAI_MAX_TOKENS` |
| Max retries | `3` | Exponential backoff, capped at 30s |
| Timeout | `60s` | Per request |

**Two execution paths:**
- `complete(messages)` — blocking, returns full `CompletionResult` (text, token usage, latency_ms)
- `stream(messages)` — async generator, yields text deltas token-by-token via OpenAI streaming API; used by the SSE endpoint `GET /chat/stream/{session_id}`

**Retry policy:** retries on `RateLimitError` (429), `APIStatusError` (≥500), `APIConnectionError`, and `APITimeoutError`. Non-retryable errors (4xx except 429) are raised immediately.

---

### Memory — Conversation Checkpoints (`backend/memory/`)

**What it does:** Persists conversation state (message history) across turns so the model can maintain multi-turn coherence without re-uploading history on every request.

**Interface:** LangGraph-compatible `BaseMemorySaver` — `put(config, checkpoint, metadata)`, `get(config)`, `list(config)`. Config format: `{"configurable": {"thread_id": "<session_id>"}}`.

**Available backends:**

| Backend | Class | Description |
|---|---|---|
| `sqlite` | `SQLiteSaver` | File-backed, durable, survives restarts — **Default** |
| `memory` | `InMemorySaver` | Ephemeral in-process dict; data lost on restart |

**Default backend: `SQLiteSaver`** — set via `MEMORY_BACKEND=sqlite` in `.env`.

**Why SQLite is the default:** Chat history must survive server restarts for a usable UX. In-memory storage is convenient for development but inappropriate for any production or long-running session use case.

**`SQLiteSaver` internals:**
- Schema: `checkpoints(thread_id, checkpoint_id, parent_id, checkpoint_json, metadata_json, created_at)`
- Index on `(thread_id, created_at DESC)` for efficient history lookups
- WAL mode enabled by default for concurrent readers + one writer
- Write operations serialized with `threading.Lock`; connection uses `check_same_thread=False`
- Default DB path: `.data/memory/checkpoints.db` (auto-created on startup)

**Retention policy (applied by `prune()`):**
- Delete checkpoints older than `MEMORY_PRUNE_DAYS=30` days
- Always keep the `MEMORY_KEEP_LAST=5` most recent per thread
- `MEMORY_MAX_CHECKPOINTS=0` means unlimited per-thread storage

---

### Retrieval Evaluation (`backend/retrieval/evaluation.py`)

The `RetrievalEvaluator` measures pipeline quality offline:

| Metric | Description |
|---|---|
| `Recall@K` | Fraction of relevant documents found in top-K results |
| `MRR` | Mean Reciprocal Rank — position of the first relevant result |
| `NDCG@K` | Normalized Discounted Cumulative Gain — supports graded relevance |

K values evaluated: `[1, 3, 5, 10]` by default. Results are written to `./eval_results/` as JSON (or CSV). Optional W&B / MLflow tracking via `EVAL_TRACKING_URI`.

---

### API Route Reference

All routes are prefixed with `/api/v1`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health/ping` | Liveness probe |
| `GET` | `/health/status` | Full system health + document metrics |
| `GET` | `/health/services` | Live status of core RAG service components |
| `POST` | `/documents/upload` | Upload + parse a document (Phase 1) |
| `GET` | `/documents/list` | List active documents |
| `GET` | `/documents/stats` | Aggregate document statistics |
| `GET` | `/documents/formats` | List supported file formats and parser routing |
| `GET` | `/documents/activity` | Recent document and pipeline activity |
| `GET` | `/documents/{doc_id}` | Get full parsed document detail |
| `DELETE` | `/documents/{doc_id}` | Soft-delete a document |
| `GET` | `/pipeline/config` | Get current pipeline config |
| `POST` | `/pipeline/config` | Save pipeline config |
| `POST` | `/pipeline/run` | Run Phase 2–4 on all active documents |
| `GET` | `/pipeline/status/{job_id}` | Poll job progress |
| `GET` | `/pipeline/phases` | Get all phase statuses (latest job) |
| `POST` | `/chat/session` | Create a new chat session |
| `GET` | `/chat/sessions` | List all sessions (admin) |
| `POST` | `/chat/query` | Blocking RAG query with citations |
| `GET` | `/chat/stream/{session_id}` | SSE streaming RAG query |
| `GET` | `/chat/history/{session_id}` | Retrieve conversation history |
| `DELETE` | `/chat/session/{session_id}` | Wipe a session's memory |
| `GET` | `/config/` | Runtime config (non-sensitive fields) |
| `GET` | `/config/indexes` | List available Pinecone indexes |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
