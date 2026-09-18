"""
Pydantic Settings — single source of truth for all configuration.
Loaded once at startup. All modules import from here.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dims: int = 1536
    openai_chat_model: str = "gpt-4o"
    openai_max_tokens: int = 8191

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "rag-intelligence"
    pinecone_metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_environment: str = "us-east-1-aws"  # legacy alias kept for .env compat

    # ── Cohere ────────────────────────────────────────────────────────────────
    cohere_api_key: str = ""

    # ── Memory / Checkpoint backend ───────────────────────────────────────────
    # "sqlite"  → durable file-backed store (DEFAULT, recommended for production)
    # "memory"  → ephemeral in-process store (dev / testing only)
    memory_backend: Literal["sqlite", "memory"] = "sqlite"

    # SQLite backend settings
    sqlite_db_path: str = ".data/memory/checkpoints.db"
    sqlite_wal_mode: bool = True
    sqlite_pool_timeout: int = 30

    # Retention policy (applied by the prune() call)
    memory_max_checkpoints: int = 0      # 0 = unlimited per-thread cap
    memory_prune_days: int = 30          # delete checkpoints older than N days
    memory_keep_last: int = 5            # always keep N most-recent per thread

    # InMemorySaver cap (ignored when backend=sqlite)
    memory_max_per_thread: int = 100     # 0 = unlimited

    # ── Dense retrieval ───────────────────────────────────────────────────────
    dense_top_k: int = 50
    dense_timeout_seconds: int = 10
    dense_max_retries: int = 3
    embedding_model: str = "text-embedding-3-small"   # alias used by DenseRetrieverConfig
    embedding_dimension: int = 1536                   # alias used by DenseRetrieverConfig

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    sparse_top_k: int = 50
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_corpus_path: str = ".cache/bm25/corpus.pkl"
    bm25_use_stemming: bool = False
    bm25_vocab_cap: int = 50_000
    stopwords_lang: str = "english"

    # ── Hybrid retrieval ─────────────────────────────────────────────────────
    # "dense"         → dense-only Pinecone ANN query
    # "hybrid_bm25"    → dense (Pinecone) + BM25 (local rank_bm25 index) fused via RRF   [default]
    # "hybrid_splade"  → single native Pinecone hybrid query using dense + SPLADE sparse_values
    retrieval_mode: Literal["dense", "hybrid_bm25", "hybrid_splade"] = "hybrid_bm25"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    retrieval_timeout_seconds: float = 10.0

    # ── Reranker (Cohere) ─────────────────────────────────────────────────────
    reranker_model: str = "rerank-v3.5"
    reranker_top_k: int = 10
    final_top_k: int = 5

    # ── Prompt / LLM token budgets ────────────────────────────────────────────
    prompt_max_context_tokens: int = 6000
    prompt_max_history_tokens: int = 2000

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # In production set to e.g. "https://myapp.example.com"
    cors_origins: str = "*"

    # ── App defaults ──────────────────────────────────────────────────────────
    app_env: str = "development"
    max_file_size_mb: int = 50
    allowed_extensions: str = "pdf,docx,csv,html,txt,xlsx"
    default_chunk_size: int = 400
    default_chunk_overlap: int = 60
    default_top_k: int = 10
    default_score_threshold: float = 0.75
    default_batch_size: int = 100
    default_max_retries: int = 3
    log_level: str = "INFO"

    # ── Chunking safety ceiling ────────────────────────────────────────────────
    # Upper bound on how many characters of ParsedDocument.full_text a single
    # chunk_document() call will process. Without this, a pathologically
    # large upload (e.g. a multi-hundred-MB text extraction) can make the
    # chunker run for an unbounded amount of time / memory. Text beyond this
    # cap is truncated with a logged warning rather than processed in full.
    max_chunking_input_chars: int = 2_000_000

    # ── Pipeline defaults ─────────────────────────────────────────────────────
    pipeline_default_tenant: str = "default"
    pipeline_default_corpus: str = "default"

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
