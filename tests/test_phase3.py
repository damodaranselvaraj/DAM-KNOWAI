"""Phase 3 embedding pipeline verification tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Models ─────────────────────────────────────────────────────────────────────
from backend.models.embedded_chunk import EmbeddedChunk, EmbeddingResult, SparseVector, ChunkValidationRecord
sv = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.1])
assert sv.nnz == 3 and not sv.is_empty()
try:
    SparseVector(indices=[1, 2], values=[0.5])
    assert False, "Should have raised"
except Exception:
    pass
print("EmbeddedChunk models OK")

# ── Language detector ──────────────────────────────────────────────────────────
from backend.utils.language_detector import detect_language, is_english
assert is_english("This is a standard English sentence about compliance policy.")
assert detect_language("x") is None
print("Language detector OK")

# ── Pre-embedding validator ────────────────────────────────────────────────────
from backend.ingestion.embedders.pre_embedding_validator import validate_chunks
from backend.models.chunk import Chunk, ChunkingStrategy, ChunkType
from backend.utils.hashing import hash_text
from backend.utils.token_counter import count_tokens

def _make_chunk(text, idx=0, doc_id="doc1"):
    return Chunk(
        chunk_id=f"{doc_id}:{idx}", doc_id=doc_id, chunk_index=idx,
        chunk_type=ChunkType.TEXT, text=text,
        text_with_title=text, token_count=0,
        doc_name="test.pdf", file_type="pdf",
        content_hash=hash_text(text), strategy=ChunkingStrategy.RECURSIVE,
    )

c_empty = _make_chunk("   ", 0)
c_valid = _make_chunk("This is a valid English sentence about compliance procedures.", 1)
c_dup   = _make_chunk("This is a valid English sentence about compliance procedures.", 2)

valid, summary = validate_chunks([c_empty, c_valid, c_dup])
assert summary.skipped_empty == 1
assert summary.skipped_dupes == 1
assert len(valid) == 1 and valid[0].chunk_id == c_valid.chunk_id
print(f"Pre-embedding validator OK — {summary.skipped_empty} empty, {summary.skipped_dupes} dupes, {len(valid)} valid")

long_text = "word " * 10_000
c_long = _make_chunk(long_text, 3)
valid_t, summary_t = validate_chunks([c_long], max_tokens=100, truncate_long=True)
assert len(valid_t) == 1 and summary_t.truncated == 1
assert count_tokens(valid_t[0].text_with_title) <= 115
print(f"Token truncation OK — {count_tokens(valid_t[0].text_with_title)} tokens")

known = {hash_text("existing content")}
c_existing = _make_chunk("existing content", 4)
valid_k, sum_k = validate_chunks([c_existing], known_hashes=known)
assert len(valid_k) == 0 and sum_k.skipped_dupes == 1
print("Cross-doc dedup OK")

# ── Embedding cache ────────────────────────────────────────────────────────────
from backend.ingestion.embedders.embedding_cache import RedisEmbeddingCache, build_cache
cache = RedisEmbeddingCache(redis_url="redis://localhost:9999", ttl_seconds=60)
vec = [0.1, 0.2, 0.3]
cache.set("abc123", vec)
assert cache.get("abc123") == vec
assert cache.get("nonexistent") is None
cache.delete("abc123")
assert cache.get("abc123") is None
stats = cache.stats()
assert "hits" in stats
print(f"EmbeddingCache OK — backend={stats['backend']}")

# ── Sparse embedder ────────────────────────────────────────────────────────────
from backend.ingestion.embedders.sparse_embedder import SparseEmbedder
se = SparseEmbedder(use_splade=False)
assert se.backend == "bm25"
sv1 = se.encode("compliance policy risk assessment data handling")
assert sv1.nnz > 0 and len(sv1.indices) == len(sv1.values)
svs = se.encode_batch(["compliance policy", "risk assessment"])
assert len(svs) == 2 and all(s.nnz > 0 for s in svs)
print(f"SparseEmbedder BM25 OK — {sv1.nnz} non-zero terms")

se.fit_vocabulary([
    "compliance policy data handling procedures",
    "risk assessment quarterly compliance officer",
    "security incident response root cause analysis",
])
sv_fitted = se.encode("compliance risk policy")
assert sv_fitted.nnz > 0
print(f"BM25 vocab fitting OK — {sv_fitted.nnz} terms after fit")

# ── OpenAI embedder (no API key — zero vectors) ────────────────────────────────
from backend.ingestion.embedders.openai_embedder import OpenAIEmbedder
embedder = OpenAIEmbedder(api_key="sk-fake", model="text-embedding-3-small",
                           dimensions=1536, max_retries=0, retry_on_fail=False)
vecs = embedder.embed_texts(["hello world", "test document"])
assert len(vecs) == 2 and all(len(v) == 1536 for v in vecs)
print(f"OpenAIEmbedder OK — {len(vecs)} zero-vectors (no valid key)")

# ── EmbeddingPipeline ──────────────────────────────────────────────────────────
from backend.ingestion.embedders.embedding_pipeline import EmbeddingPipeline

chunks = [
    _make_chunk("Compliance officers must review risk assessments quarterly.", i)
    for i in range(5)
]
chunks.append(_make_chunk("Compliance officers must review risk assessments quarterly.", 99))

pipeline = EmbeddingPipeline(
    openai_api_key="sk-fake",
    model="text-embedding-3-small",
    dimensions=1536,
    batch_size=3,
    max_retries=0,
    retry_on_fail=False,
    redis_url="redis://localhost:9999",
    enable_sparse=True,
    enable_cache=True,
)
# Force BM25 (no HuggingFace download in CI / offline tests)
if pipeline._sparse_embedder:
    pipeline._sparse_embedder._use_splade = False
    pipeline._sparse_embedder._model = None

result = pipeline.embed_chunks(chunks, doc_id="doc-test", doc_name="policy.pdf")

assert result.doc_id == "doc-test"
assert result.total_input == 6
assert result.total_embedded >= 1
assert result.total_skipped >= 1
assert len(result.embedded_chunks) == result.total_embedded
assert all(len(ec.dense_vector) == 1536 for ec in result.embedded_chunks)
assert all(ec.sparse_vector is not None for ec in result.embedded_chunks)

record = result.embedded_chunks[0].to_pinecone_record()
assert "id" in record and "values" in record and "metadata" in record
assert "doc_id" in record["metadata"]
print(f"EmbeddingPipeline OK — {result.total_embedded} embedded, {result.total_skipped} skipped")
print(f"  Pinecone record keys: {list(record.keys())}")
print(f"  cache stats: {pipeline.cache_stats()}")

# ── EmbeddedChunk.from_chunk + to_pinecone_record ─────────────────────────────
ec = EmbeddedChunk.from_chunk(
    chunks[0],
    dense_vector=[0.1] * 1536,
    sparse_vector=SparseVector(indices=[1, 2], values=[0.5, 0.3]),
)
assert ec.chunk_id == chunks[0].chunk_id
assert ec.has_dense and ec.has_sparse
pr = ec.to_pinecone_record()
assert pr["sparse_values"]["indices"] == [1, 2]
print("EmbeddedChunk.from_chunk + to_pinecone_record OK")

# ── result.summary() ──────────────────────────────────────────────────────────
s = result.summary()
assert s["total_embedded"] == result.total_embedded
assert "cost_usd" in s
print(f"EmbeddingResult.summary OK — {s}")

# ── ingestion package Phase 3 API ─────────────────────────────────────────────
from backend.ingestion import embed_chunks as _embed, ingest_chunk_embed

er = _embed(chunks[:3], doc_id="doc-test", doc_name="policy.pdf", pipeline=pipeline)
assert er.total_embedded > 0
print(f"ingestion.embed_chunks OK — {er.total_embedded} embedded")

bad_pr, bad_cr, bad_er = ingest_chunk_embed("bad.exe", b"MZ", embedding_pipeline=pipeline)
assert not bad_pr.success and bad_cr is None and bad_er is None
print("ingest_chunk_embed failure path OK")

# ── Full 3-phase pipeline smoke test ──────────────────────────────────────────
# (parse will succeed for a TXT file even without real parsers)
from backend.utils.hashing import hash_bytes
from backend.models.parsed_document import ParsedDocument, ParsedPage, FileType, ParserName, ParseStatus
from backend.ingestion import chunk_document

sha = hash_bytes(b"Test policy document text about compliance risk assessment.")
doc = ParsedDocument(
    filename="policy.txt", file_type=FileType.TXT, size_bytes=60,
    sha256=sha, parser_used=ParserName.LLAMAINDEX, parse_status=ParseStatus.SUCCESS,
    pages=[ParsedPage(page_number=1,
        text="Test policy document text about compliance risk assessment. " * 10)]
)
cr = chunk_document(doc)
assert cr.total_chunks > 0
er2 = _embed(cr.all_chunks, doc_id=doc.doc_id, doc_name=doc.filename, pipeline=pipeline)
assert er2.total_embedded > 0
assert all(ec.doc_id == doc.doc_id for ec in er2.embedded_chunks)
print(f"Full 3-phase smoke test OK — {cr.total_chunks} chunks -> {er2.total_embedded} embedded")

# ── FastAPI app ────────────────────────────────────────────────────────────────
from backend.main import app
print("FastAPI app OK")

print()
print("=== ALL PHASE 3 CHECKS PASSED ===")
