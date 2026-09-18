"""
Retrieval package — dense, sparse, hybrid, reranking, evaluation, and pipeline.
"""
from backend.retrieval.dense_retriever import (
    DenseRetriever,
    DenseRetrieverConfig,
    RetrievalResult,
    EmbeddingError,
    PineconeConnectionError,
)
from backend.retrieval.sparse_retriever import (
    SparseRetriever,
    SparseRetrieverConfig,
    IndexDocument,
    IndexNotFoundError,
    EmptyQueryError,
)
from backend.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridRetrieverConfig,
)
from backend.retrieval.reranker import (
    Reranker,
    RerankerConfig,
)
from backend.retrieval.evaluation import (
    RetrievalEvaluator,
    EvaluatorConfig,
    RelevantDoc,
    QueryMetrics,
    AggregateMetrics,
    EvaluationReport,
    LatencyBreakdown,
)
__all__ = [
    # Dense
    "DenseRetriever", "DenseRetrieverConfig",
    "RetrievalResult", "EmbeddingError", "PineconeConnectionError",
    # Sparse
    "SparseRetriever", "SparseRetrieverConfig",
    "IndexDocument", "IndexNotFoundError", "EmptyQueryError",
    # Hybrid
    "HybridRetriever", "HybridRetrieverConfig",
    # Reranker
    "Reranker", "RerankerConfig",
    # Evaluation
    "RetrievalEvaluator", "EvaluatorConfig",
    "RelevantDoc", "QueryMetrics", "AggregateMetrics",
    "EvaluationReport", "LatencyBreakdown",
]
