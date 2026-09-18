"""
Retrieval evaluation module.

Implements the standard IR metrics used to measure retrieval pipeline quality:

    • Recall@K      — fraction of relevant docs found in top-K results
    • MRR           — Mean Reciprocal Rank (position of first relevant doc)
    • NDCG@K        — Normalised Discounted Cumulative Gain (supports graded relevance)

Aggregates per-query metrics into a report with latency percentiles, and
writes the report as JSON or CSV.  A rich console summary table is printed
when the ``rich`` library is available.

Configuration (all via environment / .env):
    EVAL_K_VALUES               comma-separated K list  (default: "1,3,5,10")
    EVAL_RELEVANCE_THRESHOLD    min relevance to count as relevant  (default: 1)
    EVAL_OUTPUT_DIR             (default: ./eval_results)
    EVAL_LOG_FORMAT             "json" | "csv"  (default: json)
    EVAL_LATENCY_PERCENTILES    comma-separated  (default: "50,90,95,99")
    EVAL_TRACKING_URI           optional WandB / MLflow URI
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

class EvaluatorConfig(BaseSettings):
    """Loaded from environment variables / .env."""

    k_values_str: str = Field("1,3,5,10", alias="EVAL_K_VALUES")
    relevance_threshold: float = Field(1.0, alias="EVAL_RELEVANCE_THRESHOLD")
    output_dir: str = Field("./eval_results", alias="EVAL_OUTPUT_DIR")
    log_format: str = Field("json", alias="EVAL_LOG_FORMAT")
    latency_percentiles_str: str = Field("50,90,95,99", alias="EVAL_LATENCY_PERCENTILES")
    tracking_uri: Optional[str] = Field(None, alias="EVAL_TRACKING_URI")

    @property
    def k_values(self) -> List[int]:
        return [int(v.strip()) for v in self.k_values_str.split(",") if v.strip()]

    @property
    def latency_percentiles(self) -> List[int]:
        return [int(v.strip()) for v in self.latency_percentiles_str.split(",") if v.strip()]

    model_config = {
        "populate_by_name": True,
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "protected_namespaces": (),
    }


# ─── Data models ──────────────────────────────────────────────────────────────

class RelevantDoc(BaseModel):
    """A ground-truth relevant document with optional graded relevance."""
    id: str
    relevance: float = 1.0   # 0–3 for graded; binary: 0 or 1


class LatencyBreakdown(BaseModel):
    """Per-stage latency in milliseconds."""
    dense_ms:   float = 0.0
    sparse_ms:  float = 0.0
    fusion_ms:  float = 0.0
    filter_ms:  float = 0.0
    rerank_ms:  float = 0.0
    total_ms:   float = 0.0


class QueryMetrics(BaseModel):
    """All metrics for a single query evaluation."""
    query_id:        str
    query:           str
    recall_at_k:     Dict[int, float] = Field(default_factory=dict)
    mrr:             float = 0.0
    ndcg_at_k:       Dict[int, float] = Field(default_factory=dict)
    latency:         LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    retrieved_count: int = 0
    relevant_found:  int = 0


class AggregateMetrics(BaseModel):
    """Macro-averaged metrics across the whole evaluation set."""
    mean_recall_at_k: Dict[int, float] = Field(default_factory=dict)
    mean_mrr:         float = 0.0
    mean_ndcg_at_k:   Dict[int, float] = Field(default_factory=dict)
    std_mrr:          float = 0.0
    std_ndcg_at_k:    Dict[int, float] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Full evaluation report written to disk and (optionally) tracked."""
    timestamp:          datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    config:             dict = Field(default_factory=dict)
    num_queries:        int = 0
    aggregate_metrics:  AggregateMetrics = Field(default_factory=AggregateMetrics)
    per_query_metrics:  List[QueryMetrics] = Field(default_factory=list)
    latency_percentiles: Dict[str, float] = Field(default_factory=dict)


# ─── Evaluator ────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Compute Recall@K, MRR, and NDCG@K for retrieval pipeline output.

    Usage::

        config    = EvaluatorConfig()
        evaluator = RetrievalEvaluator(config)

        for query, relevant_docs, retrieved_ids, latency in test_set:
            evaluator.add_query_result(
                query=query,
                relevant_docs=relevant_docs,
                retrieved_ids=retrieved_ids,
                latency=latency,
            )

        report = evaluator.compute_report()
        evaluator.save_report(report)
        evaluator.print_summary(report)
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config
        self._query_metrics: List[QueryMetrics] = []

        logger.info(
            json.dumps({
                "event": "evaluator_init",
                "k_values": config.k_values,
                "relevance_threshold": config.relevance_threshold,
                "output_dir": config.output_dir,
                "log_format": config.log_format,
            })
        )

    # ── Core metrics ──────────────────────────────────────────────────────────

    def recall_at_k(
        self,
        relevant_ids: Set[str],
        retrieved_ids: List[str],
        k: int,
    ) -> float:
        """
        Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

        Returns 0.0 when ``relevant_ids`` is empty.
        """
        if not relevant_ids:
            return 0.0
        hits = len(relevant_ids & set(retrieved_ids[:k]))
        return hits / len(relevant_ids)

    def mrr(
        self,
        relevant_ids: Set[str],
        retrieved_ids: List[str],
    ) -> float:
        """
        Mean Reciprocal Rank for a single query.

        MRR = 1 / rank_of_first_relevant

        Returns 0.0 when no relevant document is found.
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(
        self,
        relevance_scores: Dict[str, float],   # doc_id → relevance grade
        retrieved_ids: List[str],
        k: int,
    ) -> float:
        """
        NDCG@K — Normalised Discounted Cumulative Gain.

        Supports both binary (0/1) and graded (0, 1, 2, 3) relevance.

        Formula:
            DCG@K  = Σ_{i=1}^{K}  (2^rel_i − 1) / log2(i + 1)
            IDCG@K = DCG of the ideal (perfectly sorted) ranking
            NDCG@K = DCG@K / IDCG@K

        Returns 0.0 when IDCG is 0 (no relevant documents exist).

        Parameters
        ----------
        relevance_scores:
            Mapping from document ID to relevance grade.  Documents not
            present are treated as having relevance 0.
        retrieved_ids:
            Ordered list of retrieved document IDs (index 0 = rank 1).
        k:
            Cut-off depth.
        """
        def _dcg(ids: List[str], grades: Dict[str, float], depth: int) -> float:
            gain = 0.0
            for i, doc_id in enumerate(ids[:depth], start=1):
                rel = grades.get(doc_id, 0.0)
                gain += (2.0 ** rel - 1.0) / math.log2(i + 1)
            return gain

        dcg = _dcg(retrieved_ids, relevance_scores, k)

        # Ideal: sort all known relevant docs descending by grade
        ideal_ids = sorted(
            relevance_scores.keys(),
            key=lambda d: relevance_scores[d],
            reverse=True,
        )
        idcg = _dcg(ideal_ids, relevance_scores, k)

        return dcg / idcg if idcg > 0 else 0.0

    # ── Per-query ingestion ───────────────────────────────────────────────────

    def add_query_result(
        self,
        query: str,
        relevant_docs: List[RelevantDoc],
        retrieved_ids: List[str],
        latency: LatencyBreakdown | None = None,
        query_id: str | None = None,
    ) -> QueryMetrics:
        """
        Compute and store all metrics for a single query.

        Parameters
        ----------
        query:
            Query string (for reporting).
        relevant_docs:
            Ground-truth relevant documents with optional relevance grades.
        retrieved_ids:
            Ordered list of retrieved document IDs from the pipeline.
        latency:
            Per-stage latency breakdown (optional).
        query_id:
            Caller-supplied trace ID; auto-generated when omitted.

        Returns
        -------
        QueryMetrics
            The computed metrics for this query.
        """
        query_id = query_id or str(uuid.uuid4())

        # Build relevance lookups
        threshold = self.config.relevance_threshold
        relevant_ids: Set[str] = {
            d.id for d in relevant_docs if d.relevance >= threshold
        }
        relevance_scores: Dict[str, float] = {
            d.id: d.relevance for d in relevant_docs
        }

        # Recall@K and NDCG@K for every configured K
        recall: Dict[int, float] = {}
        ndcg:   Dict[int, float] = {}
        for k in self.config.k_values:
            recall[k] = self.recall_at_k(relevant_ids, retrieved_ids, k)
            ndcg[k]   = self.ndcg_at_k(relevance_scores, retrieved_ids, k)

        # MRR
        mrr_score = self.mrr(relevant_ids, retrieved_ids)

        # Relevant found in retrieved list
        relevant_found = len(relevant_ids & set(retrieved_ids))

        qm = QueryMetrics(
            query_id=query_id,
            query=query,
            recall_at_k=recall,
            mrr=mrr_score,
            ndcg_at_k=ndcg,
            latency=latency or LatencyBreakdown(),
            retrieved_count=len(retrieved_ids),
            relevant_found=relevant_found,
        )
        self._query_metrics.append(qm)
        return qm

    # ── Aggregate report ──────────────────────────────────────────────────────

    def compute_report(self, pipeline_config: dict | None = None) -> EvaluationReport:
        """
        Aggregate all stored per-query metrics into an EvaluationReport.

        Parameters
        ----------
        pipeline_config:
            Snapshot of the full pipeline configuration to embed in the report.

        Returns
        -------
        EvaluationReport
        """
        if not self._query_metrics:
            logger.warning(json.dumps({"event": "eval_no_queries"}))
            return EvaluationReport(config=pipeline_config or {})

        n = len(self._query_metrics)
        k_values = self.config.k_values

        # ── Recall@K means ───────────────────────────────────────────────────
        mean_recall: Dict[int, float] = {}
        for k in k_values:
            vals = [qm.recall_at_k.get(k, 0.0) for qm in self._query_metrics]
            mean_recall[k] = float(np.mean(vals))

        # ── MRR ───────────────────────────────────────────────────────────────
        mrr_vals = np.array([qm.mrr for qm in self._query_metrics])
        mean_mrr  = float(mrr_vals.mean())
        std_mrr   = float(mrr_vals.std())

        # ── NDCG@K means + stds ───────────────────────────────────────────────
        mean_ndcg: Dict[int, float] = {}
        std_ndcg:  Dict[int, float] = {}
        for k in k_values:
            vals = np.array([qm.ndcg_at_k.get(k, 0.0) for qm in self._query_metrics])
            mean_ndcg[k] = float(vals.mean())
            std_ndcg[k]  = float(vals.std())

        # ── Latency percentiles ───────────────────────────────────────────────
        total_latencies = np.array([qm.latency.total_ms for qm in self._query_metrics])
        pct_dict: Dict[str, float] = {}
        for p in self.config.latency_percentiles:
            pct_dict[f"p{p}"] = round(float(np.percentile(total_latencies, p)), 2)

        aggregate = AggregateMetrics(
            mean_recall_at_k=mean_recall,
            mean_mrr=mean_mrr,
            mean_ndcg_at_k=mean_ndcg,
            std_mrr=std_mrr,
            std_ndcg_at_k=std_ndcg,
        )

        report = EvaluationReport(
            config=pipeline_config or {},
            num_queries=n,
            aggregate_metrics=aggregate,
            per_query_metrics=self._query_metrics,
            latency_percentiles=pct_dict,
        )

        logger.info(
            json.dumps({
                "event": "eval_report_computed",
                "num_queries": n,
                "mean_mrr": round(mean_mrr, 4),
                "mean_ndcg_at_k": {str(k): round(v, 4) for k, v in mean_ndcg.items()},
                "latency_percentiles": pct_dict,
            })
        )
        return report

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_report(self, report: EvaluationReport) -> Path:
        """
        Persist *report* to ``EVAL_OUTPUT_DIR`` in JSON or CSV format.

        Returns the path of the written file.
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = report.timestamp.strftime("%Y%m%d_%H%M%S")
        fmt = self.config.log_format.lower()

        if fmt == "csv":
            path = out_dir / f"eval_{ts}.csv"
            self._write_csv(report, path)
        else:
            path = out_dir / f"eval_{ts}.json"
            self._write_json(report, path)

        # Optional experiment tracking
        if self.config.tracking_uri:
            self._track(report)

        logger.info(json.dumps({"event": "eval_saved", "path": str(path)}))
        return path

    def _write_json(self, report: EvaluationReport, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.model_dump(mode="json"), fh, indent=2, default=str)

    def _write_csv(self, report: EvaluationReport, path: Path) -> None:
        """Write per-query metrics to CSV (one row per query)."""
        k_values = self.config.k_values
        fieldnames = (
            ["query_id", "query", "mrr", "retrieved_count", "relevant_found",
             "total_ms"]
            + [f"recall_at_{k}" for k in k_values]
            + [f"ndcg_at_{k}" for k in k_values]
        )
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for qm in report.per_query_metrics:
                row: dict = {
                    "query_id": qm.query_id,
                    "query": qm.query,
                    "mrr": round(qm.mrr, 4),
                    "retrieved_count": qm.retrieved_count,
                    "relevant_found": qm.relevant_found,
                    "total_ms": round(qm.latency.total_ms, 1),
                }
                for k in k_values:
                    row[f"recall_at_{k}"] = round(qm.recall_at_k.get(k, 0.0), 4)
                    row[f"ndcg_at_{k}"]   = round(qm.ndcg_at_k.get(k, 0.0), 4)
                writer.writerow(row)

    def _track(self, report: EvaluationReport) -> None:
        """Log aggregate metrics to WandB or MLflow if a URI is configured."""
        uri = self.config.tracking_uri or ""
        agg = report.aggregate_metrics

        if uri.startswith("wandb"):
            try:
                import wandb  # type: ignore
                wandb.init(project="rag-eval", reinit=True)
                flat: dict[str, Any] = {
                    "mrr": agg.mean_mrr,
                    **{f"recall_at_{k}": v for k, v in agg.mean_recall_at_k.items()},
                    **{f"ndcg_at_{k}": v for k, v in agg.mean_ndcg_at_k.items()},
                    **report.latency_percentiles,
                }
                wandb.log(flat)
                wandb.finish()
                logger.info(json.dumps({"event": "eval_wandb_logged"}))
            except Exception as exc:
                logger.warning(json.dumps({"event": "eval_wandb_failed", "error": str(exc)}))

        elif "mlflow" in uri or uri.startswith("http"):
            try:
                import mlflow  # type: ignore
                mlflow.set_tracking_uri(uri)
                with mlflow.start_run():
                    mlflow.log_metric("mrr", agg.mean_mrr)
                    for k, v in agg.mean_recall_at_k.items():
                        mlflow.log_metric(f"recall_at_{k}", v)
                    for k, v in agg.mean_ndcg_at_k.items():
                        mlflow.log_metric(f"ndcg_at_{k}", v)
                    for pct_name, pct_val in report.latency_percentiles.items():
                        mlflow.log_metric(f"latency_{pct_name}_ms", pct_val)
                logger.info(json.dumps({"event": "eval_mlflow_logged"}))
            except Exception as exc:
                logger.warning(json.dumps({"event": "eval_mlflow_failed", "error": str(exc)}))

    # ── Console summary ───────────────────────────────────────────────────────

    def print_summary(self, report: EvaluationReport) -> None:
        """
        Print a formatted summary table to the console.

        Uses the ``rich`` library when available; falls back to plain text.
        """
        try:
            from rich.console import Console  # type: ignore
            from rich.table import Table  # type: ignore
            self._rich_summary(report, Console())
        except ImportError:
            self._plain_summary(report)

    def _rich_summary(self, report: EvaluationReport, console: Any) -> None:
        from rich.table import Table  # type: ignore

        agg = report.aggregate_metrics
        k_values = self.config.k_values

        # ── Metrics table ─────────────────────────────────────────────────────
        table = Table(title=f"Retrieval Evaluation — {report.num_queries} queries")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")

        table.add_row("MRR (mean ± std)", f"{agg.mean_mrr:.4f} ± {agg.std_mrr:.4f}")
        for k in k_values:
            table.add_row(f"Recall@{k}", f"{agg.mean_recall_at_k.get(k, 0):.4f}")
        for k in k_values:
            table.add_row(
                f"NDCG@{k} (mean ± std)",
                f"{agg.mean_ndcg_at_k.get(k, 0):.4f} ± {agg.std_ndcg_at_k.get(k, 0):.4f}",
            )

        console.print(table)

        # ── Latency table ─────────────────────────────────────────────────────
        lat_table = Table(title="Latency Percentiles (ms)")
        lat_table.add_column("Percentile", style="bold yellow")
        lat_table.add_column("Total (ms)", justify="right")
        for pct_name, val in report.latency_percentiles.items():
            lat_table.add_row(pct_name.upper(), f"{val:.1f}")

        console.print(lat_table)

    def _plain_summary(self, report: EvaluationReport) -> None:
        agg = report.aggregate_metrics
        k_values = self.config.k_values
        sep = "─" * 50
        print(f"\n{sep}")
        print(f"  Retrieval Evaluation — {report.num_queries} queries")
        print(sep)
        print(f"  MRR          : {agg.mean_mrr:.4f} ± {agg.std_mrr:.4f}")
        for k in k_values:
            print(f"  Recall@{k:<3}  : {agg.mean_recall_at_k.get(k, 0):.4f}")
        for k in k_values:
            print(
                f"  NDCG@{k:<3}    : "
                f"{agg.mean_ndcg_at_k.get(k, 0):.4f} ± "
                f"{agg.std_ndcg_at_k.get(k, 0):.4f}"
            )
        print(sep)
        print("  Latency Percentiles (ms)")
        for pct_name, val in report.latency_percentiles.items():
            print(f"    {pct_name.upper():<5}: {val:.1f} ms")
        print(sep + "\n")
