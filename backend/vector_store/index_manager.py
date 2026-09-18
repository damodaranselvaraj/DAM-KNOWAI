"""
IndexManager — Pinecone Serverless index lifecycle and namespace management.

Responsibilities
────────────────
1. Create a serverless index if it doesn't exist (idempotent)
2. Describe index stats (total vectors, per-namespace breakdown)
3. Delete an index (with confirmation guard)
4. List and resolve namespaces (per-tenant + per-corpus strategy)
5. Configure metadata indexing (for Pinecone pods — serverless auto-indexes all)

Pinecone Serverless notes
──────────────────────────
• No efSearch / nProbe knobs — Pinecone manages HNSW parameters internally.
• Metadata is auto-indexed on Serverless; metadata_config only applies to
  pod-based indexes.
• Namespaces are logical partitions within one index, free to create/use.
• Deletion of a namespace is done by deleting all vectors in it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from backend.vector_store.schema import (
    INDEXED_FIELD_NAMES,
    PINECONE_METADATA_CONFIG,
    resolve_namespace,
    parse_namespace,
)

logger = logging.getLogger(__name__)

_PINECONE_AVAILABLE = False
try:
    from pinecone import Pinecone as _PineconeClient
    from pinecone import ServerlessSpec as _ServerlessSpec
    _PINECONE_AVAILABLE = True
except ImportError:
    logger.warning(
        "pinecone-client not installed — IndexManager will use mock mode. "
        "Install with: pip install pinecone-client"
    )


# ─── Index configuration ──────────────────────────────────────────────────────

@dataclass
class IndexConfig:
    """Runtime configuration for a Pinecone Serverless index.
    All defaults are sourced from backend.config.settings at construction time.
    """
    name:        str   = ""
    dimension:   int   = 0
    metric:      str   = ""
    cloud:       str   = ""
    region:      str   = ""

    def __post_init__(self) -> None:
        from backend.config import settings as _cfg
        if not self.name:
            self.name = _cfg.pinecone_index_name
        if not self.dimension:
            self.dimension = _cfg.openai_embedding_dims
        if not self.metric:
            self.metric = _cfg.pinecone_metric
        if not self.cloud:
            self.cloud = _cfg.pinecone_cloud
        if not self.region:
            self.region = _cfg.pinecone_region


@dataclass
class NamespaceStats:
    namespace:    str
    tenant_id:    str
    corpus:       str
    vector_count: int


@dataclass
class IndexStats:
    index_name:       str
    total_vectors:    int = 0
    dimension:        int = 0       # populated from Pinecone describe_index_stats
    namespaces:       list[NamespaceStats] = field(default_factory=list)
    fullness:         float = 0.0
    is_ready:         bool  = False
    raw:              dict  = field(default_factory=dict)


# ─── Manager ─────────────────────────────────────────────────────────────────

class IndexManager:
    """
    Manages the lifecycle of a Pinecone Serverless index.

    All operations are idempotent — calling create_index() on an existing
    index is a no-op; calling delete_index() on a missing index logs a
    warning rather than raising.

    Args:
        api_key:  Pinecone API key.
        config:   IndexConfig dataclass with index parameters.
    """

    def __init__(
        self,
        api_key: str,
        config:  IndexConfig | None = None,
    ) -> None:
        self.config  = config or IndexConfig()
        self._client = None
        self._index  = None   # cached Index handle

        if not _PINECONE_AVAILABLE:
            logger.warning("Pinecone unavailable — IndexManager in mock mode.")
            return

        try:
            self._client = _PineconeClient(api_key=api_key)
            logger.info("IndexManager: Pinecone client initialised.")
        except Exception as exc:
            logger.error("IndexManager: Pinecone init failed: %s", exc)

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def create_index_if_not_exists(self) -> bool:
        """
        Create the serverless index if it doesn't already exist.

        Returns:
            True  — index was created.
            False — index already existed (no-op) or client is unavailable.
        """
        if self._client is None:
            logger.warning("[mock] create_index_if_not_exists — no-op (mock mode).")
            return False

        try:
            existing = [idx.name for idx in self._client.list_indexes()]
        except Exception as exc:
            logger.warning(
                "create_index_if_not_exists: could not list indexes (%s) — "
                "treating as mock/offline.", exc
            )
            return False
        if self.config.name in existing:
            logger.info("Index '%s' already exists — skipping creation.", self.config.name)
            return False

        logger.info(
            "Creating Pinecone Serverless index '%s' (dim=%d, metric=%s, %s/%s)…",
            self.config.name, self.config.dimension, self.config.metric,
            self.config.cloud, self.config.region,
        )
        try:
            self._client.create_index(
                name=self.config.name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=_ServerlessSpec(
                    cloud=self.config.cloud,
                    region=self.config.region,
                ),
            )
            self._wait_for_ready()
            logger.info("Index '%s' created and ready.", self.config.name)
            return True
        except Exception as exc:
            logger.error("Index creation failed: %s", exc)
            raise

    def describe_index(self) -> dict:
        """Return the raw index description from Pinecone."""
        if self._client is None:
            return {"name": self.config.name, "status": "mock"}
        try:
            return self._client.describe_index(self.config.name)
        except Exception as exc:
            logger.error("describe_index failed: %s", exc)
            return {}

    def delete_index(self, confirm: bool = False) -> bool:
        """
        Delete the index permanently.  Requires confirm=True as a safety guard.

        Returns:
            True on success, False if index didn't exist or deletion failed.
        """
        if not confirm:
            raise ValueError(
                "delete_index() requires confirm=True. "
                "This operation is IRREVERSIBLE and deletes all vectors."
            )
        if self._client is None:
            logger.warning("[mock] delete_index — no-op (mock mode).")
            return False

        existing = [idx.name for idx in self._client.list_indexes()]
        if self.config.name not in existing:
            logger.warning("Index '%s' not found — nothing to delete.", self.config.name)
            return False

        try:
            self._client.delete_index(self.config.name)
            self._index = None
            logger.info("Index '%s' deleted.", self.config.name)
            return True
        except Exception as exc:
            logger.error("delete_index failed: %s", exc)
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> IndexStats:
        """
        Fetch index statistics including per-namespace vector counts.

        NOTE: Pinecone Serverless does not expose efSearch/nProbe.
        Those parameters apply to Milvus/Qdrant/FAISS only.
        """
        if self._client is None:
            return IndexStats(
                index_name=self.config.name,
                is_ready=False,
                raw={"status": "mock"},
            )

        try:
            idx   = self._get_index()
            stats = idx.describe_index_stats()
            raw   = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)

            ns_stats: list[NamespaceStats] = []
            for ns_name, ns_data in raw.get("namespaces", {}).items():
                tenant, corpus = parse_namespace(ns_name)
                ns_stats.append(NamespaceStats(
                    namespace=ns_name,
                    tenant_id=tenant,
                    corpus=corpus,
                    vector_count=ns_data.get("vector_count", 0),
                ))

            return IndexStats(
                index_name=self.config.name,
                total_vectors=raw.get("total_vector_count", 0),
                dimension=raw.get("dimension", self.config.dimension),
                namespaces=ns_stats,
                fullness=raw.get("index_fullness", 0.0),
                is_ready=True,
                raw=raw,
            )
        except Exception as exc:
            logger.error("get_stats failed: %s", exc)
            return IndexStats(index_name=self.config.name, is_ready=False)

    # ── Namespace management ──────────────────────────────────────────────────

    def list_namespaces(self) -> list[str]:
        """Return all active namespace strings in this index."""
        stats = self.get_stats()
        return [ns.namespace for ns in stats.namespaces]

    def namespace_vector_count(
        self,
        tenant_id: str = "default",
        corpus:    str = "default",
    ) -> int:
        """Return the vector count for a specific namespace."""
        ns    = resolve_namespace(tenant_id, corpus)
        stats = self.get_stats()
        for ns_stat in stats.namespaces:
            if ns_stat.namespace == ns:
                return ns_stat.vector_count
        return 0

    def delete_namespace(
        self,
        tenant_id: str = "default",
        corpus:    str = "default",
        confirm:   bool = False,
    ) -> int:
        """
        Delete all vectors in a namespace (effectively removes the namespace).

        Returns:
            Number of vectors deleted, or 0 in mock mode.
        """
        if not confirm:
            raise ValueError("delete_namespace() requires confirm=True.")

        ns    = resolve_namespace(tenant_id, corpus)
        count = self.namespace_vector_count(tenant_id, corpus)

        if self._client is None or count == 0:
            logger.info("Namespace '%s' empty or mock — nothing to delete.", ns)
            return 0

        try:
            idx = self._get_index()
            idx.delete(delete_all=True, namespace=ns)
            logger.info("Namespace '%s' cleared (%d vectors).", ns, count)
            return count
        except Exception as exc:
            logger.error("delete_namespace '%s' failed: %s", ns, exc)
            return 0

    # ── Metadata config (pod-based only) ─────────────────────────────────────

    def get_metadata_config(self) -> dict:
        """
        Return the metadata indexing configuration.

        On Pinecone Serverless all metadata is auto-indexed — this returns
        our declared list for documentation / Qdrant migration purposes.
        """
        return {
            "indexed_fields":  INDEXED_FIELD_NAMES,
            "pinecone_config": PINECONE_METADATA_CONFIG,
            "note": (
                "Pinecone Serverless auto-indexes all metadata fields. "
                "The explicit list is provided for documentation and "
                "cross-engine parity (Milvus/Qdrant/FAISS)."
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_index(self):
        """Return cached Index handle, creating it if needed."""
        if self._index is None and self._client is not None:
            self._index = self._client.Index(self.config.name)
        return self._index

    def _wait_for_ready(self, timeout_s: int = 120, poll_s: int = 3) -> None:
        """Poll until the index status is READY or timeout."""
        if self._client is None:
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            desc   = self._client.describe_index(self.config.name)
            status = getattr(getattr(desc, "status", None), "ready", False)
            if status:
                return
            logger.debug("Waiting for index '%s' to be ready…", self.config.name)
            time.sleep(poll_s)
        logger.warning(
            "Index '%s' did not become ready within %d s.", self.config.name, timeout_s
        )
