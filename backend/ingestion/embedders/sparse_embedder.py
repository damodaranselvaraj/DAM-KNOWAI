"""
Sparse embedding encoder for hybrid search.

Primary:  SPLADE (via splade library) — learned sparse representations.
Fallback: BM25-style TF-IDF scoring — deterministic, zero extra deps.

Output format (Pinecone-compatible):
    {"indices": [int, …], "values": [float, …]}

The vocabulary is built lazily on the first call and shared across the
process.  For production use you should pre-build and persist the
vocabulary from your full corpus.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from backend.config import settings
from backend.models.embedded_chunk import SparseVector

logger = logging.getLogger(__name__)

_SPLADE_AVAILABLE = False
try:
    # Optional: splade or transformers-based SPLADE
    # pip install splade  (or use the HuggingFace transformers pipeline)
    from transformers import AutoTokenizer, AutoModelForMaskedLM  # type: ignore
    import torch  # type: ignore
    _SPLADE_AVAILABLE = True
except ImportError:
    logger.info(
        "SPLADE (transformers/torch) not installed — "
        "SparseEmbedder will use BM25 fallback."
    )

# Vocabulary size cap for BM25 fallback (prevents unbounded index growth) and
# BM25 hyper-parameters — sourced from backend.config.settings so tuning via
# .env actually affects this fallback path.
_VOCAB_CAP: int   = settings.bm25_vocab_cap
_BM25_K1:   float = settings.bm25_k1
_BM25_B:    float = settings.bm25_b
# Stop-words pruned from sparse indices
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "not", "this", "that",
    "it", "its", "as", "by", "from", "up", "out", "if", "so", "no",
    "we", "you", "he", "she", "they", "i", "my", "your", "our",
})


# ─── BM25 vocabulary ─────────────────────────────────────────────────────────

class _BM25Vocab:
    """
    Incremental vocabulary mapping token → integer index.
    Built from training documents (or lazily from embedded texts).
    """

    def __init__(self) -> None:
        self._word2id:  dict[str, int] = {}
        self._df:       Counter        = Counter()   # document frequencies
        self._num_docs: int            = 0
        self._avg_dl:   float          = 0.0
        self._total_dl: int            = 0

    def fit(self, documents: list[str]) -> None:
        """Build vocabulary and IDF from a corpus."""
        for doc in documents:
            tokens = _tokenize(doc)
            unique = set(tokens)
            for t in unique:
                self._df[t] += 1
            self._total_dl   += len(tokens)
            self._num_docs   += 1
            for t in tokens:
                if t not in self._word2id and len(self._word2id) < _VOCAB_CAP:
                    self._word2id[t] = len(self._word2id)
        self._avg_dl = self._total_dl / max(self._num_docs, 1)

    def encode(self, text: str) -> SparseVector:
        """Return BM25 sparse vector for `text`."""
        tokens    = _tokenize(text)
        if not tokens:
            return SparseVector()

        dl    = len(tokens)
        tf    = Counter(tokens)
        indices: list[int]   = []
        values:  list[float] = []

        for term, freq in tf.items():
            if term not in self._word2id:
                if len(self._word2id) < _VOCAB_CAP:
                    self._word2id[term] = len(self._word2id)
                else:
                    continue
            idx   = self._word2id[term]
            n_doc = self._df.get(term, 0)
            idf   = math.log(
                (self._num_docs - n_doc + 0.5) / (n_doc + 0.5) + 1
            )
            norm  = freq * (_BM25_K1 + 1) / (
                freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(self._avg_dl, 1))
            )
            score = idf * norm
            if score > 0:
                indices.append(idx)
                values.append(round(score, 6))

        return SparseVector(indices=indices, values=values)


# Module-level shared BM25 vocabulary
_bm25_vocab = _BM25Vocab()


# ─── Sparse embedder ──────────────────────────────────────────────────────────

class SparseEmbedder:
    """
    Encodes texts as sparse vectors for hybrid Pinecone search.

    Primary path:  SPLADE via HuggingFace transformers (if available).
    Fallback path: BM25 scoring with the module-level vocabulary.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-selfdistil",
        use_splade: bool = True,
        top_k_terms: int = 128,    # Max non-zero terms per sparse vector
    ) -> None:
        self._top_k    = top_k_terms
        self._use_splade = use_splade and _SPLADE_AVAILABLE

        self._tokenizer = None
        self._model     = None

        if self._use_splade:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model     = AutoModelForMaskedLM.from_pretrained(model_name)
                self._model.eval()
                logger.info("SparseEmbedder: SPLADE model loaded (%s)", model_name)
            except Exception as exc:
                logger.warning(
                    "SPLADE model load failed (%s) — using BM25 fallback.", exc
                )
                self._use_splade = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def encode(self, text: str) -> SparseVector:
        """Encode a single text to a sparse vector."""
        if self._use_splade and self._model is not None:
            return self._splade_encode(text)
        return _bm25_vocab.encode(text)

    def encode_batch(self, texts: list[str]) -> list[SparseVector]:
        """Encode multiple texts."""
        return [self.encode(t) for t in texts]

    def fit_vocabulary(self, documents: list[str]) -> None:
        """
        Pre-train the BM25 vocabulary on a corpus.
        Call this with your full document corpus for best IDF scores.
        """
        _bm25_vocab.fit(documents)
        logger.info(
            "BM25 vocabulary fitted: %d documents, %d terms",
            len(documents), len(_bm25_vocab._word2id),
        )

    @property
    def backend(self) -> str:
        return "splade" if self._use_splade else "bm25"

    # ── SPLADE path ─────────────────────────────────────────────────────────────

    def _splade_encode(self, text: str) -> SparseVector:
        try:
            import torch
            inputs  = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                logits  = self._model(**inputs).logits        # (1, seq_len, vocab)
                # SPLADE aggregation: max pooling → ReLU → log(1+x)
                scores  = torch.log1p(torch.relu(logits)).max(dim=1).values.squeeze()

            # Top-K non-zero terms
            top_k   = min(self._top_k, (scores > 0).sum().item())
            if top_k == 0:
                return SparseVector()

            top_vals, top_idx = scores.topk(int(top_k))
            return SparseVector(
                indices=[int(i) for i in top_idx.tolist()],
                values=[round(float(v), 6) for v in top_vals.tolist()],
            )
        except Exception as exc:
            logger.warning("SPLADE encode failed (%s) — using BM25.", exc)
            return _bm25_vocab.encode(text)


# ─── Tokenisation helper ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokeniser with stop-word removal."""
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [w for w in words if w not in _STOP_WORDS]
