"""
Vector Store package — Phase 4: Pinecone Serverless upsert and retrieval.

Public surface
──────────────
VectorStorePipeline      — main orchestrator: store_embedding_result(), query()
build_vector_store_pipeline — factory reading backend.config.settings
PineconeClient           — direct SDK wrapper (upsert, query, delete, fetch)
IndexManager             — index lifecycle (create, stats, delete, namespaces)
PineconeRecord           — Pydantic upsert schema with to_dict()
resolve_namespace        — "{tenant}__{corpus}" namespace resolver
build_vector_id          — "{doc_id}:{version}:{page}:{chunk_index}" ID builder
METADATA_FIELDS          — registry of all indexed metadata fields
"""
from backend.vector_store.schema import (
    PineconeRecord,
    METADATA_FIELDS,
    INDEXED_FIELD_NAMES,
    PINECONE_METADATA_CONFIG,
    resolve_namespace,
    parse_namespace,
    build_vector_id,
    parse_vector_id,
    build_metadata,
)
from backend.vector_store.pinecone_client import (
    PineconeClient,
    QueryResult,
    QueryMatch,
    UpsertResult,
)
from backend.vector_store.index_manager import (
    IndexManager,
    IndexConfig,
    IndexStats,
    NamespaceStats,
)
from backend.vector_store.vector_store_pipeline import (
    VectorStorePipeline,
    VectorStoreResult,
    build_vector_store_pipeline,
)

__all__ = [
    # Pipeline
    "VectorStorePipeline",
    "VectorStoreResult",
    "build_vector_store_pipeline",
    # Client
    "PineconeClient",
    "QueryResult",
    "QueryMatch",
    "UpsertResult",
    # Index management
    "IndexManager",
    "IndexConfig",
    "IndexStats",
    "NamespaceStats",
    # Schema / helpers
    "PineconeRecord",
    "METADATA_FIELDS",
    "INDEXED_FIELD_NAMES",
    "PINECONE_METADATA_CONFIG",
    "resolve_namespace",
    "parse_namespace",
    "build_vector_id",
    "parse_vector_id",
    "build_metadata",
]
