"""
Synchronous HTTP client for the RAG Pipeline FastAPI backend.
Uses httpx with a configurable base URL and timeout.
All methods fall back to realistic stub data when the API is unreachable
so the UI remains functional during development without a running backend.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Iterator

import httpx

from frontend.config import settings

logger = logging.getLogger("rag_api_client")


# ─── Stub / fallback data ─────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_STUB_HEALTH = {
    "status": "healthy",
    "timestamp": _now(),
    "services": {
        "vector_db":       {"status": True, "index_name": "rag-intelligence", "doc_count": 142},
        "embedding_model": {"status": True, "model": "text-embedding-3-small"},
        "parser":          {"status": True},
    },
    "metrics": {
        "total_documents": 142,
        "total_chunks":    8_430,
        "index_status":    "ready",
        "last_updated":    _now(),
    },
}

_STUB_SERVICES = {
    "timestamp": _now(),
    "embedding_model": {"name": "text-embedding-3-small", "status": "active",    "healthy": True},
    "vector_db":       {"name": "Pinecone", "index_name": "rag-intelligence", "status": "ready",     "healthy": True},
    "llm_model":       {"name": "gpt-4o",   "status": "online",    "healthy": True},
    "reranking_model": {"name": "rerank-v3.5", "status": "enabled",  "healthy": True},
}

_STUB_DOCUMENTS = [
    {"doc_id": "doc-001", "name": "Annual_Policy_2024.pdf",         "file_type": "pdf",  "size_bytes": 2_048_000, "pages": 45, "status": "indexed", "uploaded_at": "2024-11-01T09:00:00Z", "chunk_count": 312},
    {"doc_id": "doc-002", "name": "Compliance_Guidelines_v3.docx",  "file_type": "docx", "size_bytes": 512_000,   "pages": 18, "status": "indexed", "uploaded_at": "2024-11-05T14:22:00Z", "chunk_count": 98},
    {"doc_id": "doc-003", "name": "Risk_Assessment_Q3.pdf",         "file_type": "pdf",  "size_bytes": 1_024_000, "pages": 27, "status": "indexed", "uploaded_at": "2024-11-10T11:45:00Z", "chunk_count": 187},
]

_STUB_CITATIONS = [
    {"doc_id": "doc-001", "doc_name": "Annual_Policy_2024.pdf",        "page_number": 12, "published_at": "2024-01-15", "score": 0.94, "chunk_id": "chunk-abc123", "chunk_index": 47, "snippet": "All employees must comply with data handling procedures as outlined in Section 4.2."},
    {"doc_id": "doc-002", "doc_name": "Compliance_Guidelines_v3.docx", "page_number": 5,  "published_at": "2024-03-20", "score": 0.87, "chunk_id": "chunk-def456", "chunk_index": 22, "snippet": "Risk assessments must be conducted quarterly and reviewed by the compliance officer."},
    {"doc_id": "doc-003", "doc_name": "Risk_Assessment_Q3.pdf",        "page_number": 8,  "published_at": "2024-09-30", "score": 0.79, "chunk_id": "chunk-ghi789", "chunk_index": 31, "snippet": "The identified risk categories include operational, reputational, financial, and regulatory."},
]


# ─── Client ───────────────────────────────────────────────────────────────────

class RAGApiClient:
    """Thin synchronous wrapper around the RAG Pipeline FastAPI backend."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or settings.API_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.API_TIMEOUT

    def _get(self, path: str, **kwargs) -> dict | list:
        url = f"{self.base_url}{path}"
        try:
            r = httpx.get(url, timeout=self.timeout, **kwargs)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("GET %s failed: %s", url, exc)
            raise

    def _post(self, path: str, json: dict | None = None, **kwargs) -> dict | list:
        url = f"{self.base_url}{path}"
        try:
            r = httpx.post(url, json=json, timeout=self.timeout, **kwargs)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("POST %s failed: %s", url, exc)
            raise

    def _delete(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        try:
            r = httpx.delete(url, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("DELETE %s failed: %s", url, exc)
            raise

    # ── Health ────────────────────────────────────────────────────────────────

    def get_health_status(self) -> dict:
        try:
            return self._get("/health/status")
        except Exception:
            return _STUB_HEALTH

    def get_service_status(self) -> dict:
        """Live status of embedding model, vector DB, LLM, and reranker."""
        try:
            return self._get("/health/services")
        except Exception:
            return _STUB_SERVICES

    # ── Pipeline ──────────────────────────────────────────────────────────────

    # ── Config ────────────────────────────────────────────────────────────────

    def list_pinecone_indexes(self) -> dict:
        """Returns available Pinecone indexes from the backend."""
        try:
            return self._get("/config/indexes")
        except Exception:
            return {"indexes": ["rag-intelligence"], "active": "rag-intelligence"}

    def get_pipeline_config(self) -> dict:
        try:
            return self._get("/pipeline/config")
        except Exception:
            from frontend.utils.constants import DEFAULT_PIPELINE_CONFIG
            import copy
            return copy.deepcopy(DEFAULT_PIPELINE_CONFIG)

    def save_pipeline_config(self, config: dict) -> dict:
        try:
            return self._post("/pipeline/config", json=config)
        except Exception:
            return {"status": "saved (offline)", "timestamp": _now()}

    def run_pipeline(self) -> dict:
        try:
            return self._post("/pipeline/run")
        except Exception:
            import uuid
            return {"job_id": str(uuid.uuid4()), "status": "running", "message": "Pipeline started (offline mode)"}

    def get_pipeline_status(self, job_id: str) -> dict:
        try:
            return self._get(f"/pipeline/status/{job_id}")
        except Exception:
            return {
                "job_id": job_id,
                "status": "complete",
                "phases": {
                    "parse": {"name": "Parse", "status": "complete", "progress": 100},
                    "chunk": {"name": "Chunk", "status": "complete", "progress": 100},
                    "embed": {"name": "Embed", "status": "complete", "progress": 100},
                    "store": {"name": "Store", "status": "complete", "progress": 100},
                },
                "created_at": _now(),
                "updated_at": _now(),
            }

    def get_phase_statuses(self) -> dict:
        try:
            return self._get("/pipeline/phases")
        except Exception:
            return {"phases": {}, "job_id": None}

    # ── Documents ─────────────────────────────────────────────────────────────

    def upload_document(self, file_bytes: bytes, filename: str, content_type: str) -> dict:
        """
        Upload a document to the backend.

        Raises:
            httpx.HTTPStatusError: when the backend returns a 4xx/5xx (e.g. 415 unsupported type,
                                   422 parse failure).  The caller is responsible for handling this.
            Exception:             on network-level failures (connection refused, timeout, etc.)
                                   re-raised so the UI can distinguish "backend error" from "offline".
        """
        files = {"file": (filename, file_bytes, content_type)}
        url = f"{self.base_url}/documents/upload"
        try:
            r = httpx.post(url, files=files, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            # Structured error from the backend — surface the detail message
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            logger.warning("Upload %s rejected by backend (%s): %s", filename, exc.response.status_code, detail)
            raise RuntimeError(detail) from exc
        except Exception as exc:
            logger.warning("Upload %s failed (network/offline): %s", filename, exc)
            raise

    def list_documents(self) -> list[dict]:
        try:
            data = self._get("/documents/list")
            return data.get("documents", [])
        except Exception:
            return _STUB_DOCUMENTS

    def delete_document(self, doc_id: str) -> dict:
        """
        Delete a document by ID.

        Raises:
            RuntimeError: when the backend returns a 404 (doc not found) or other error.
        """
        try:
            r = httpx.delete(f"{self.base_url}/documents/{doc_id}", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            logger.warning("Delete doc_id=%s rejected (%s): %s", doc_id, exc.response.status_code, detail)
            raise RuntimeError(detail) from exc
        except Exception as exc:
            logger.warning("Delete doc_id=%s failed (network): %s", doc_id, exc)
            raise

    def get_document_stats(self) -> dict:
        try:
            return self._get("/documents/stats")
        except Exception:
            return {"total_documents": 0, "total_chunks": 0, "total_size_bytes": 0, "by_type": {}, "last_updated": _now()}

    def get_recent_activity(self, limit: int = 20) -> list[dict]:
        """Returns the most recent activity events from the backend."""
        try:
            data = self._get("/documents/activity", params={"limit": limit})
            return data.get("activities", [])
        except Exception:
            return []

    # ── Chat ──────────────────────────────────────────────────────────────────

    def create_session(self) -> dict:
        try:
            return self._post("/chat/session")
        except Exception:
            import uuid
            return {"session_id": str(uuid.uuid4()), "created_at": _now()}

    def query_rag(self, payload: dict) -> dict:
        try:
            return self._post("/chat/query", json=payload)
        except Exception:
            import random
            answer = (
                f"Based on the indexed documents, your query about \"{payload.get('query', '')}\" "
                "relates to the compliance framework described in the policy documents. "
                "The Annual Policy 2024 outlines that all employees must follow Section 4.2 "
                "data handling procedures. Please review the cited sources for full context."
            )
            return {
                "session_id": payload.get("session_id", ""),
                "query": payload.get("query", ""),
                "answer": answer,
                "citations": _STUB_CITATIONS,
                "timestamp": _now(),
                "model_used": "gpt-4o",
                "tokens_used": random.randint(300, 900),
                "processing_time_ms": round(random.uniform(200, 600), 2),
            }

    def clear_session(self, session_id: str) -> dict:
        try:
            url = f"{self.base_url}/chat/session/{session_id}"
            r = httpx.delete(url, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"status": "cleared", "session_id": session_id}

    def stream_response(
        self,
        session_id: str,
        query: str = "",
        retrieval_mode: str = "hybrid_bm25",
    ) -> Iterator[str]:
        """
        Yields text tokens from the SSE streaming endpoint.

        The backend emits JSON-encoded SSE events of the form
        ``data: {"type": "token", "token": "..."}``, followed by a
        ``citations`` event and a final ``done`` event. Only ``token``
        events are yielded here; ``done``/``error`` terminate the stream.
        """
        import json as _json

        url = f"{self.base_url}/chat/stream/{session_id}"
        try:
            params = {"query": query, "retrieval_mode": retrieval_mode}
            with httpx.stream("GET", url, params=params, timeout=60.0) as r:
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    try:
                        evt = _json.loads(raw)
                    except ValueError:
                        # Not JSON — treat as a raw text token (backward compat)
                        if raw == "[DONE]":
                            return
                        yield raw
                        continue

                    evt_type = evt.get("type")
                    if evt_type == "token":
                        yield evt.get("token", "")
                    elif evt_type == "error":
                        raise RuntimeError(evt.get("message", "LLM streaming error"))
                    elif evt_type in ("done", "citations"):
                        # Citations/final answer are fetched separately via
                        # query_rag() after the stream completes.
                        if evt_type == "done":
                            return
        except Exception:
            # Offline fallback — yield a mock streaming answer
            import time
            mock = (
                "Based on your documents, I can provide the following analysis. "
                "The policy framework establishes clear guidelines for compliance. "
                "All employees are expected to adhere to these standards."
            )
            for word in mock.split():
                yield word + " "
                time.sleep(0.04)


# ─── Singleton ────────────────────────────────────────────────────────────────

def get_api_client() -> RAGApiClient:
    """Return a cached client instance stored in Streamlit session_state."""
    import streamlit as st
    if "_api_client" not in st.session_state:
        from frontend.config import settings
        st.session_state._api_client = RAGApiClient(
            base_url=settings.API_BASE_URL,
            timeout=settings.API_TIMEOUT,
        )
    return st.session_state._api_client
