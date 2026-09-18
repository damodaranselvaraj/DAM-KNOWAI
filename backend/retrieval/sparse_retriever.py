"""
Sparse retriever using BM25 (rank_bm25).

Responsibilities:
- Build / persist / load a BM25 index from a document corpus
- Tokenise queries (lowercase → stopword removal → optional stemming)
- Return top-K results with min-max normalised scores

Configuration (all via environment / .env):
    SPARSE_TOP_K         (default: 50)
    BM25_K1              (default: 1.5)
    BM25_B               (default: 0.75)
    BM25_CORPUS_PATH     (default: .cache/bm25/corpus.pkl)
    BM25_USE_STEMMING    (default: false)
    STOPWORDS_LANG       (default: english)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# rank_bm25 — install: pip install rank-bm25
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "rank_bm25 is required for SparseRetriever. "
        "Install it with: pip install rank-bm25"
    ) from _exc

# nltk stopwords — install: pip install nltk  +  python -m nltk.downloader stopwords
try:
    from nltk.corpus import stopwords as _nltk_stopwords  # type: ignore
    from nltk.tokenize import word_tokenize as _word_tokenize  # type: ignore
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# Optional stemming via nltk PorterStemmer
try:
    from nltk.stem import PorterStemmer as _PorterStemmer  # type: ignore
    _STEMMER_AVAILABLE = True
except ImportError:
    _STEMMER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─── Custom exceptions ────────────────────────────────────────────────────────

class IndexNotFoundError(Exception):
    """Raised when the BM25 index has not been built or loaded."""


class EmptyQueryError(Exception):
    """Raised when all query tokens are filtered out."""


# ─── Config ───────────────────────────────────────────────────────────────────

class SparseRetrieverConfig(BaseSettings):
    """Loaded from environment variables / .env — all defaults from backend.config.settings."""

    top_k: int = Field(0, alias="SPARSE_TOP_K")
    k1: float = Field(0.0, alias="BM25_K1")
    b: float = Field(0.0, alias="BM25_B")
    corpus_path: str = Field("", alias="BM25_CORPUS_PATH")
    use_stemming: bool = Field(False, alias="BM25_USE_STEMMING")
    stopwords_lang: str = Field("", alias="STOPWORDS_LANG")

    model_config = {
        "populate_by_name": True,
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "protected_namespaces": (),
    }

    def model_post_init(self, __context) -> None:
        """Fill zero/empty fields from central settings."""
        from backend.config import settings as _cfg
        if not self.top_k:
            object.__setattr__(self, "top_k", _cfg.sparse_top_k)
        if not self.k1:
            object.__setattr__(self, "k1", _cfg.bm25_k1)
        if not self.b:
            object.__setattr__(self, "b", _cfg.bm25_b)
        if not self.corpus_path:
            object.__setattr__(self, "corpus_path", _cfg.bm25_corpus_path)
        if not self.stopwords_lang:
            object.__setattr__(self, "stopwords_lang", _cfg.stopwords_lang)


# ─── Document wrapper used when building the index ───────────────────────────

class IndexDocument(BaseModel):
    """Minimal document representation used by build_index."""
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)


# ─── Sparse retriever ─────────────────────────────────────────────────────────

class SparseRetriever:
    """
    Production-grade BM25 sparse retriever.

    Usage::

        config  = SparseRetrieverConfig()
        ret     = SparseRetriever(config)

        # Build once from corpus
        ret.build_index(documents)

        # Or load persisted index
        ret.load_index()

        results = ret.retrieve("refund policy", top_k=10)
    """

    def __init__(self, config: SparseRetrieverConfig) -> None:
        self.config = config
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_metadata: List[dict] = []

        # Stopwords
        self._stopwords: set[str] = set()
        if _NLTK_AVAILABLE:
            try:
                self._stopwords = set(
                    _nltk_stopwords.words(config.stopwords_lang)
                )
            except Exception:
                logger.warning(
                    json.dumps({
                        "event": "stopwords_load_failed",
                        "lang": config.stopwords_lang,
                        "fallback": "empty set",
                    })
                )

        # Stemmer
        self._stemmer = None
        if config.use_stemming and _STEMMER_AVAILABLE:
            self._stemmer = _PorterStemmer()
        elif config.use_stemming and not _STEMMER_AVAILABLE:
            logger.warning(
                json.dumps({
                    "event": "stemmer_unavailable",
                    "message": "BM25_USE_STEMMING=true but nltk.stem not available",
                })
            )

        logger.info(
            json.dumps({
                "event": "sparse_retriever_init",
                "k1": config.k1,
                "b": config.b,
                "top_k": config.top_k,
                "use_stemming": config.use_stemming,
                "stopwords_lang": config.stopwords_lang,
                "corpus_path": config.corpus_path,
            })
        )

    # ── Text preprocessing ────────────────────────────────────────────────────

    def preprocess(self, text: str) -> List[str]:
        """
        Tokenise and normalise text for BM25 indexing / querying.

        Pipeline: lowercase → tokenise → stopword removal → optional stemming.

        Parameters
        ----------
        text : str
            Raw text to preprocess.

        Returns
        -------
        list[str]
            Filtered, (optionally stemmed) token list.
        """
        text = text.lower()

        # Tokenise
        if _NLTK_AVAILABLE:
            tokens = _word_tokenize(text)
        else:
            # Simple whitespace + punctuation split fallback
            import re
            tokens = re.findall(r"\b[a-z]+\b", text)

        # Stopword removal + keep only alphabetic tokens
        tokens = [t for t in tokens if t.isalpha() and t not in self._stopwords]

        # Optional stemming
        if self._stemmer is not None:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens

    # ── Index management ──────────────────────────────────────────────────────

    def build_index(self, documents: List[IndexDocument]) -> None:
        """
        Tokenise *documents*, build a BM25 index, and persist it to disk.

        Parameters
        ----------
        documents : list[IndexDocument]
            Corpus to index. Each document must have an ``id`` and ``text``.
        """
        if not documents:
            raise ValueError("Cannot build BM25 index from an empty document list.")

        t0 = time.perf_counter()
        tokenised_corpus: List[List[str]] = []

        self._doc_ids = []
        self._doc_texts = []
        self._doc_metadata = []

        for doc in documents:
            tokens = self.preprocess(doc.text)
            tokenised_corpus.append(tokens)
            self._doc_ids.append(doc.id)
            self._doc_texts.append(doc.text)
            self._doc_metadata.append(doc.metadata)

        self._bm25 = BM25Okapi(tokenised_corpus, k1=self.config.k1, b=self.config.b)

        build_ms = (time.perf_counter() - t0) * 1_000

        # Persist to disk
        corpus_path = Path(self.config.corpus_path)
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "doc_ids": self._doc_ids,
            "doc_texts": self._doc_texts,
            "doc_metadata": self._doc_metadata,
            "config": {
                "k1": self.config.k1,
                "b": self.config.b,
                "use_stemming": self.config.use_stemming,
                "stopwords_lang": self.config.stopwords_lang,
            },
        }
        with open(corpus_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            json.dumps({
                "event": "bm25_index_built",
                "num_docs": len(documents),
                "corpus_path": str(corpus_path),
                "build_ms": round(build_ms, 1),
            })
        )

    def load_index(self) -> None:
        """
        Load a previously serialised BM25 index from ``config.corpus_path``.

        Raises
        ------
        IndexNotFoundError
            When the corpus file does not exist.
        ValueError
            When the file is corrupt or has a schema mismatch.
        """
        corpus_path = Path(self.config.corpus_path)
        if not corpus_path.exists():
            raise IndexNotFoundError(
                f"BM25 corpus not found at '{corpus_path}'. "
                "Run build_index() first or set BM25_CORPUS_PATH correctly."
            )

        try:
            with open(corpus_path, "rb") as fh:
                payload = pickle.load(fh)  # noqa: S301 — trusted internal file
        except Exception as exc:
            raise ValueError(
                f"Failed to deserialise BM25 corpus at '{corpus_path}': {exc}"
            ) from exc

        # Validate schema
        required = {"bm25", "doc_ids", "doc_texts", "doc_metadata"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(
                f"BM25 corpus file is missing keys: {missing}. Rebuild the index."
            )

        self._bm25 = payload["bm25"]
        self._doc_ids = payload["doc_ids"]
        self._doc_texts = payload["doc_texts"]
        self._doc_metadata = payload["doc_metadata"]

        logger.info(
            json.dumps({
                "event": "bm25_index_loaded",
                "num_docs": len(self._doc_ids),
                "corpus_path": str(corpus_path),
            })
        )

    # ── Score normalisation ───────────────────────────────────────────────────

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Min-max normalise *scores* to the [0, 1] range.

        When all scores are equal (or the array is empty), returns a
        zero array of the same shape to avoid division by zero.
        """
        if scores.size == 0:
            return scores
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_ids: List[str] | None = None,
        query_id: str | None = None,
    ) -> list:
        """
        Retrieve top-K documents matching *query* via BM25.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of results to return. Falls back to ``config.top_k``.
        filter_ids:
            If provided, only documents whose IDs appear in this list are
            considered (allowlist filtering before scoring).
        query_id:
            Caller-supplied trace ID; auto-generated when omitted.

        Returns
        -------
        list[RetrievalResult]
            Sorted descending by normalised BM25 score.

        Raises
        ------
        IndexNotFoundError
            When the BM25 index has not been built/loaded.
        EmptyQueryError
            When all query tokens are removed by preprocessing.
        """
        # Lazy import to avoid circular dependency — RetrievalResult lives in
        # dense_retriever which this module does not own at module-load time.
        from backend.retrieval.dense_retriever import RetrievalResult

        if self._bm25 is None:
            raise IndexNotFoundError(
                "BM25 index is not loaded. Call build_index() or load_index() first."
            )

        query_id = query_id or str(uuid.uuid4())
        k = top_k if top_k is not None else self.config.top_k
        t0 = time.perf_counter()

        # ── 1. Preprocess ─────────────────────────────────────────────────────
        tokens = self.preprocess(query)
        if not tokens:
            raise EmptyQueryError(
                f"All tokens were filtered from query: '{query}'. "
                "Try a less restrictive stopword list or a different query."
            )

        # ── 2. Determine candidate indices ────────────────────────────────────
        if filter_ids is not None:
            filter_id_set = set(filter_ids)
            candidate_indices = [
                i for i, doc_id in enumerate(self._doc_ids) if doc_id in filter_id_set
            ]
        else:
            candidate_indices = list(range(len(self._doc_ids)))

        # ── 3. Score ──────────────────────────────────────────────────────────
        all_scores: np.ndarray = self._bm25.get_scores(tokens)  # shape: (N,)
        candidate_scores = all_scores[candidate_indices]

        # ── 4. Normalise ──────────────────────────────────────────────────────
        norm_scores = self.normalize_scores(candidate_scores)

        # ── 5. Top-K ──────────────────────────────────────────────────────────
        # argsort ascending → take last k → reverse for descending
        sorted_local = np.argsort(norm_scores)[::-1][:k]

        results: list[RetrievalResult] = []
        for rank, local_idx in enumerate(sorted_local):
            global_idx = candidate_indices[int(local_idx)]
            results.append(
                RetrievalResult(
                    id=self._doc_ids[global_idx],
                    score=float(norm_scores[local_idx]),
                    text=self._doc_texts[global_idx],
                    metadata=self._doc_metadata[global_idx],
                    source="sparse",
                    sparse_score=float(norm_scores[local_idx]),
                    sparse_rank=rank + 1,
                )
            )

        total_ms = (time.perf_counter() - t0) * 1_000

        # ── 6. Structured log ─────────────────────────────────────────────────
        logger.info(
            json.dumps({
                "event": "sparse_retrieve",
                "query_id": query_id,
                "tokens": tokens,
                "top_k": k,
                "candidate_pool": len(candidate_indices),
                "result_count": len(results),
                "latency_ms": round(total_ms, 1),
                "filter_ids_count": len(filter_ids) if filter_ids else None,
            })
        )

        return results
