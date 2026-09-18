"""
Pydantic models for pipeline configuration.
All defaults are sourced from backend.config.settings so there are no
hardcoded values in this module.
"""
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Literal, Optional


class ParserConfig(BaseModel):
    parser_type: Literal["pymupdf", "docling", "llamaindex", "auto"] = "auto"
    version_handling: Literal["replace", "keep_both", "soft_delete"] = "replace"
    ocr_enabled: bool = False


class ChunkingApiConfig(BaseModel):
    """
    API/persisted-config-facing chunking settings (Admin Panel form + saved
    PipelineConfig). Distinct from backend.models.chunk.ChunkingConfig, which
    is the internal runtime config consumed by ChunkingPipeline. See
    backend.api.v1.pipeline._translate_chunking_config() for the bridge
    between the two.
    """
    strategy: Literal[
        "fixed_size", "fixed_overlap", "sentence",
        "recursive", "semantic", "hierarchical"
    ] = "fixed_overlap"
    chunk_size: int = Field(0, ge=0, le=4096)
    overlap: int = Field(0, ge=0, le=500)
    min_chunk: int = Field(100, ge=50)
    max_chunk: int = Field(1024, ge=256)
    separators: List[str] = ["\n\n", "\n", " ", ""]
    breakpoint_percentile: int = Field(95, ge=50, le=99)

    @model_validator(mode="after")
    def _apply_settings_defaults(self) -> "ChunkingApiConfig":
        from backend.config import settings as _cfg
        if not self.chunk_size:
            self.chunk_size = _cfg.default_chunk_size
        if not self.overlap:
            self.overlap = _cfg.default_chunk_overlap
        return self


class EmbeddingConfig(BaseModel):
    model: Literal[
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ] = ""
    batch_size: int = Field(0, ge=0, le=500)
    retry_on_fail: bool = True
    max_retries: int = Field(0, ge=0, le=10)
    dimensions: Dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    @model_validator(mode="after")
    def _apply_settings_defaults(self) -> "EmbeddingConfig":
        from backend.config import settings as _cfg
        if not self.model:
            self.model = _cfg.openai_embedding_model
        if not self.batch_size:
            self.batch_size = _cfg.default_batch_size
        if not self.max_retries:
            self.max_retries = _cfg.default_max_retries
        return self


class VectorDBConfig(BaseModel):
    index_name: str = ""
    dimensions: int = 0
    metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine"
    # ef_search and n_probe are NOT supported by Pinecone Serverless — they are
    # retained here for API compatibility only and are never forwarded to Pinecone.
    ef_search: int = Field(100, ge=10, le=500)
    n_probe: int = Field(10, ge=1, le=100)
    top_k: int = Field(0, ge=0, le=50)
    # NOTE: default is 0.0 here (not settings.default_score_threshold) so
    # `_apply_settings_defaults` below can tell "unset" apart from an
    # intentional 0.0 (no filtering) and substitute the real default.
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    decay_enabled: bool = True
    decay_factor: float = Field(0.5, ge=0.0, le=1.0)
    decay_scale_days: int = Field(30, ge=1, le=365)

    @model_validator(mode="after")
    def _apply_settings_defaults(self) -> "VectorDBConfig":
        from backend.config import settings as _cfg
        if not self.index_name:
            self.index_name = _cfg.pinecone_index_name
        if not self.dimensions:
            self.dimensions = _cfg.openai_embedding_dims
        if not self.top_k:
            self.top_k = _cfg.default_top_k
        if not self.score_threshold:
            self.score_threshold = _cfg.default_score_threshold
        return self


class PipelineConfig(BaseModel):
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunking: ChunkingApiConfig = Field(default_factory=ChunkingApiConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
