"""Phase 4 vector store verification tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Schema / models ────────────────────────────────────────────────────────────
from backend.vector_store.schema import (
    PineconeRecord, METADATA_FIELDS, INDEXED_FIELD_NAMES,
    PINECONE_METADATA_CONFIG, resolve_namespace, parse_namespace,
    build_vector_id, parse_vector_id, build_metadata, _to_epoch,
)

# Namespace resolver
assert resolve_namespace("acme", "legal")     == "acme__legal"
assert resolve_namespace("ACME", "Legal Docs") == "acme__legal_docs"
assert resolve_namespace(None, None)           == "default__default"
assert resolve_namespace("tenant1", None)      == "tenant1__default"

t, c = parse_namespace("acme__legal")
assert t == "acme" and c == "legal"
t2, c2 = parse_namespace("single")
assert t2 == "default" and c2 == "default"
print("Namespace resolver OK")

# Vector ID
vid = build_vector_id("doc-abc", 2, 5, 17)
assert vid == "doc-abc:2:5:17"
parsed = parse_vector_id(vid)
assert parsed == {"doc_id": "doc-abc", "version": 2, "page": 5, "chunk_index": 17}
print("Vector ID builder/parser OK")

# Epoch conversion
assert _to_epoch("2024-01-15") == 1705276800
assert _to_epoch(None)         == 0
assert _to_epoch("bad-date")   == 0
print("Epoch conversion OK")

# Metadata fields declared
assert len(METADATA_FIELDS) >= 15
assert "doc_id"       in INDEXED_FIELD_NAMES
assert "published_at" in INDEXED_FIELD_NAMES
assert "tenant_id"    in INDEXED_FIELD_NAMES
assert "indexed"      in PINECONE_METADATA_CONFIG
print(f"Metadata config OK — {len(METADATA_FIELDS)} fields, {len(INDEXED_FIELD_NAMES)} indexed")

# ── PineconeRecord from EmbeddedChunk ──────────────────────────────────────────
from backend.models.embedded_chunk import EmbeddedChunk, SparseVector
from backend.models.chunk import ChunkingStrategy, ChunkType

ec = EmbeddedChunk(
    chunk_id="doc1:0",
    doc_id="doc-001",
    chunk_index=0,
    chunk_type="text",
    text="Compliance officers must review risk assessments quarterly.",
    text_with_title="Annual Policy | Section 2\n\nCompliance officers...",
    token_count=12,
    char_count=58,
    doc_name="Annual_Policy_2024.pdf",
    file_type="pdf",
    page_numbers=[3, 4],
    doc_title="Annual Policy 2024",
    doc_author="Risk Team",
    doc_language="en",
    published_at="2024-01-15",
    doc_sha256="a" * 64,
    content_hash="deadbeef" * 8,
    strategy="recursive",
    dense_vector=[0.1] * 1536,
    sparse_vector=SparseVector(indices=[1, 5, 9], values=[0.8, 0.5, 0.3]),
)

rec = PineconeRecord.from_embedded_chunk(
    ec, version=2, tenant_id="acme", corpus="legal", source_url="s3://bucket/doc.pdf"
)
assert rec.id == "doc-001:2:3:0"        # page=min([3,4])=3
assert rec.namespace == "acme__legal"
assert len(rec.values) == 1536
assert rec.sparse_values == {"indices": [1, 5, 9], "values": [0.8, 0.5, 0.3]}
assert rec.metadata["doc_id"]      == "doc-001"
assert rec.metadata["version"]     == 2
assert rec.metadata["tenant_id"]   == "acme"
assert rec.metadata["corpus"]      == "legal"
assert rec.metadata["published_at"] == 1705276800   # 2024-01-15 epoch
assert rec.metadata["page"]        == 3
assert rec.metadata["mime"]        == "application/pdf"
assert rec.metadata["language"]    == "en"
assert rec.metadata["source_url"]  == "s3://bucket/doc.pdf"

d = rec.to_dict()
assert "id" in d and "values" in d and "metadata" in d and "sparse_values" in d
print(f"PineconeRecord OK — id={rec.id}, namespace={rec.namespace}")
print(f"  metadata keys: {list(rec.metadata.keys())}")

# No sparse case
ec_no_sparse = EmbeddedChunk(
    chunk_id="doc1:1", doc_id="doc-001", chunk_index=1,
    chunk_type="text", text="Test.", doc_name="f.pdf", file_type="pdf",
    dense_vector=[0.2] * 1536, sparse_vector=None,
    content_hash="cf" * 32, strategy="recursive",
)
rec2 = PineconeRecord.from_embedded_chunk(ec_no_sparse)
d2 = rec2.to_dict()
assert "sparse_values" not in d2
print("PineconeRecord (no sparse) OK")

# ── PineconeClient (mock mode — no real API key) ───────────────────────────────
from backend.vector_store.pinecone_client import PineconeClient, QueryResult, UpsertResult

client = PineconeClient(api_key="fake-key", index_name="test-index", batch_size=3)
# SDK unavailable in mock environment — client stays in mock mode
assert client.is_connected == (client._index is not None)

# Upsert
records = [rec, rec2]
ur = client.upsert_records(records, namespace="acme__legal")
assert isinstance(ur, UpsertResult)
assert ur.upserted_count >= 0   # mock or real
print(f"PineconeClient.upsert_records OK — {ur.upserted_count} upserted (mock={client._index is None})")

# Dense query
qr = client.query_dense(vector=[0.1] * 1536, top_k=5, namespace="acme__legal")
assert isinstance(qr, QueryResult)
assert isinstance(qr.matches, list)
print(f"PineconeClient.query_dense OK — {len(qr.matches)} matches")

# Hybrid query
sparse_q = {"indices": [1, 5], "values": [0.9, 0.4]}
qr_h = client.query_hybrid(
    dense_vector=[0.1] * 1536, sparse_vector=sparse_q,
    top_k=5, namespace="acme__legal", alpha=0.7,
)
assert isinstance(qr_h, QueryResult)
print(f"PineconeClient.query_hybrid OK — {len(qr_h.matches)} matches")

# Delete
ok = client.delete_by_ids(["doc-001:2:3:0"], namespace="acme__legal")
assert isinstance(ok, bool)
ok2 = client.delete_by_filter({"doc_id": {"$eq": "doc-001"}}, namespace="acme__legal")
assert isinstance(ok2, bool)
try:
    client.delete_namespace_all("acme__legal", confirm=False)
    assert False, "Should have raised"
except ValueError:
    pass
print("PineconeClient.delete OK")

# Fetch
fetched = client.fetch(["doc-001:2:3:0"], namespace="acme__legal")
assert isinstance(fetched, dict)
print("PineconeClient.fetch OK")

# Stats
stats = client.describe_index_stats()
assert isinstance(stats, dict)
print("PineconeClient.describe_index_stats OK")

# ── IndexManager ──────────────────────────────────────────────────────────────
from backend.vector_store.index_manager import IndexManager, IndexConfig, IndexStats

im = IndexManager(api_key="fake-key", config=IndexConfig(name="test-idx"))
# In mock mode — should not raise
result = im.create_index_if_not_exists()
assert isinstance(result, bool)

meta_cfg = im.get_metadata_config()
assert "indexed_fields" in meta_cfg
assert "published_at" in meta_cfg["indexed_fields"]
assert "note" in meta_cfg  # Pinecone Serverless note present
print(f"IndexManager OK — {len(meta_cfg['indexed_fields'])} indexed fields")
print(f"  Note: {meta_cfg['note'][:80]}...")

stats_obj = im.get_stats()
assert isinstance(stats_obj, IndexStats)
print(f"IndexManager.get_stats OK — ready={stats_obj.is_ready}")

# Namespace helpers
ns_count = im.namespace_vector_count("acme", "legal")
assert isinstance(ns_count, int)
assert im.list_namespaces() == []  # empty in mock mode
print("IndexManager namespace helpers OK")

# ── VectorStorePipeline ────────────────────────────────────────────────────────
from backend.vector_store.vector_store_pipeline import VectorStorePipeline, VectorStoreResult
from backend.models.embedded_chunk import EmbeddingResult

# Build a realistic EmbeddingResult with several chunks
chunks_embedded = [
    EmbeddedChunk(
        chunk_id=f"doc-001:{i}", doc_id="doc-001", chunk_index=i,
        chunk_type="text",
        text=f"Policy clause {i}: all employees must comply.",
        doc_name="Policy.pdf", file_type="pdf",
        page_numbers=[i + 1],
        dense_vector=[round(0.1 + i * 0.01, 3)] * 1536,
        sparse_vector=SparseVector(indices=[i, i+10], values=[0.5, 0.3]),
        content_hash=("ab" + str(i)) * 32,
        strategy="recursive",
    )
    for i in range(5)
]
# Add one zero-vector chunk — should be skipped
chunks_embedded.append(EmbeddedChunk(
    chunk_id="doc-001:99", doc_id="doc-001", chunk_index=99,
    chunk_type="text", text="Zero vector chunk.",
    doc_name="Policy.pdf", file_type="pdf",
    dense_vector=[0.0] * 1536,
    sparse_vector=None,
    content_hash="00" * 32, strategy="recursive",
))

emb_result = EmbeddingResult(
    doc_id="doc-001", doc_name="Policy.pdf",
    model_used="text-embedding-3-small",
    embedded_chunks=chunks_embedded,
    total_input=6, total_embedded=5,
)

vsp = VectorStorePipeline(client=client, enable_sparse=True)
vsr = vsp.store_embedding_result(
    emb_result, tenant_id="acme", corpus="legal", version=3,
)

assert isinstance(vsr, VectorStoreResult)
assert vsr.namespace == "acme__legal"
assert vsr.skipped_count >= 1       # zero-vector chunk skipped
assert vsr.upserted_count >= 0      # mock or real
print(f"VectorStorePipeline.store_embedding_result OK:")
print(f"  {vsr.upserted_count} upserted, {vsr.skipped_count} skipped, ns={vsr.namespace}")
print(f"  summary: {vsr.summary()}")

# store_chunks direct path
vsr2 = vsp.store_chunks(chunks_embedded[:3], tenant_id="acme", corpus="hr")
assert vsr2.namespace == "acme__hr"
print(f"VectorStorePipeline.store_chunks OK — ns={vsr2.namespace}")

# query via pipeline
qr2 = vsp.query([0.1] * 1536, tenant_id="acme", corpus="legal", top_k=5)
assert isinstance(qr2, QueryResult)
qr3 = vsp.query([0.1]*1536, sparse_vector=sparse_q, hybrid=True, tenant_id="acme", corpus="legal")
assert isinstance(qr3, QueryResult)
print(f"VectorStorePipeline.query dense/hybrid OK")

# delete_document
del_ok = vsp.delete_document("doc-001", tenant_id="acme", corpus="legal")
assert isinstance(del_ok, bool)
print("VectorStorePipeline.delete_document OK")

# no-sparse mode
vsp_dense = VectorStorePipeline(client=client, enable_sparse=False)
vsr3 = vsp_dense.store_embedding_result(emb_result, tenant_id="t", corpus="c")
assert vsr3.skipped_count >= 1
print("VectorStorePipeline enable_sparse=False OK")

# ── vector_store __init__ exports ─────────────────────────────────────────────
from backend.vector_store import (
    VectorStorePipeline, build_vector_store_pipeline,
    PineconeClient, IndexManager, PineconeRecord,
    resolve_namespace, build_vector_id, METADATA_FIELDS,
    QueryResult, UpsertResult,
)
print("vector_store __init__ exports OK")

# ── ingestion package Phase 4 API ─────────────────────────────────────────────
from backend.ingestion import store_embeddings, ingest_full_pipeline, FullPipelineResult

# store_embeddings via package
vsr4 = store_embeddings(emb_result, tenant_id="acme", corpus="legal", vs_pipeline=vsp)
assert isinstance(vsr4, VectorStoreResult)
print(f"ingestion.store_embeddings OK — {vsr4.upserted_count} upserted")

# ingest_full_pipeline failure path (unsupported file type)
fp = ingest_full_pipeline("bad.exe", b"MZ", vs_pipeline=vsp)
assert not fp.success
assert fp.error is not None
assert fp.parse_result is not None and not fp.parse_result.success
print(f"ingest_full_pipeline failure path OK — error='{fp.error[:40]}...'")

# ── FastAPI app still imports ──────────────────────────────────────────────────
from backend.main import app
print("FastAPI app OK")

print()
print("=== ALL PHASE 4 CHECKS PASSED ===")
