"""
Pipeline metrics collector for tracking ingestion and reconciliation quality.
Thread-safe singleton that accumulates stats across the processing pipeline.
"""
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class DocumentMetrics:
    """Metrics for a single document's ingestion."""
    filename: str = ""
    total_pages: int = 0
    pages_after_filter: int = 0
    total_chunks_raw: int = 0
    chunks_filtered_short: int = 0
    chunks_filtered_noise: int = 0
    chunks_filtered_header_footer: int = 0
    chunks_kept: int = 0
    facts_extracted: int = 0
    facts_rejected_quality: int = 0
    facts_kept: int = 0
    avg_chunk_length: float = 0.0


@dataclass
class ClusterMetrics:
    """Metrics for clustering and reconciliation quality."""
    total_facts_clustered: int = 0
    num_clusters: int = 0
    num_singletons: int = 0
    avg_cluster_size: float = 0.0
    avg_intra_cluster_similarity: float = 0.0
    edges_before_structural_filter: int = 0
    edges_after_structural_filter: int = 0
    structural_filter_rejection_rate: float = 0.0


class PipelineMetricsCollector:
    """Thread-safe singleton collecting pipeline quality metrics."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._doc_metrics: Dict[str, DocumentMetrics] = {}
        self._cluster_metrics: ClusterMetrics = ClusterMetrics()
        self._lock_data = threading.Lock()

    def get_or_create_doc_metrics(self, filename: str) -> DocumentMetrics:
        with self._lock_data:
            if filename not in self._doc_metrics:
                self._doc_metrics[filename] = DocumentMetrics(filename=filename)
            return self._doc_metrics[filename]

    def update_cluster_metrics(self, **kwargs) -> None:
        with self._lock_data:
            for key, value in kwargs.items():
                if hasattr(self._cluster_metrics, key):
                    setattr(self._cluster_metrics, key, value)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock_data:
            doc_summaries = []
            total_chunks_raw = 0
            total_chunks_kept = 0
            total_facts_extracted = 0
            total_facts_kept = 0
            total_pages = 0
            total_pages_kept = 0

            for dm in self._doc_metrics.values():
                total_chunks_raw += dm.total_chunks_raw
                total_chunks_kept += dm.chunks_kept
                total_facts_extracted += dm.facts_extracted
                total_facts_kept += dm.facts_kept
                total_pages += dm.total_pages
                total_pages_kept += dm.pages_after_filter

                doc_summaries.append({
                    "filename": dm.filename,
                    "pages": f"{dm.pages_after_filter}/{dm.total_pages}",
                    "chunks": f"{dm.chunks_kept}/{dm.total_chunks_raw}",
                    "chunks_filtered": {
                        "short": dm.chunks_filtered_short,
                        "noise": dm.chunks_filtered_noise,
                        "header_footer": dm.chunks_filtered_header_footer
                    },
                    "facts": f"{dm.facts_kept}/{dm.facts_extracted}",
                    "facts_rejected": dm.facts_rejected_quality,
                    "avg_chunk_length": round(dm.avg_chunk_length, 1)
                })

            chunk_keep_rate = (total_chunks_kept / total_chunks_raw * 100) if total_chunks_raw > 0 else 0
            fact_yield = (total_facts_kept / total_chunks_kept) if total_chunks_kept > 0 else 0
            fact_quality_rate = (total_facts_kept / total_facts_extracted * 100) if total_facts_extracted > 0 else 0

            cm = self._cluster_metrics
            return {
                "ingestion": {
                    "documents_processed": len(self._doc_metrics),
                    "total_pages": total_pages,
                    "pages_after_filter": total_pages_kept,
                    "total_chunks_raw": total_chunks_raw,
                    "chunks_kept": total_chunks_kept,
                    "chunk_keep_rate_pct": round(chunk_keep_rate, 1),
                    "total_facts_extracted": total_facts_extracted,
                    "facts_kept": total_facts_kept,
                    "fact_quality_rate_pct": round(fact_quality_rate, 1),
                    "fact_yield_per_chunk": round(fact_yield, 2),
                    "per_document": doc_summaries
                },
                "clustering": {
                    "total_facts_clustered": cm.total_facts_clustered,
                    "num_clusters": cm.num_clusters,
                    "num_singletons": cm.num_singletons,
                    "avg_cluster_size": round(cm.avg_cluster_size, 2),
                    "avg_intra_cluster_similarity": round(cm.avg_intra_cluster_similarity, 4),
                    "edges_before_structural_filter": cm.edges_before_structural_filter,
                    "edges_after_structural_filter": cm.edges_after_structural_filter,
                    "structural_filter_rejection_rate_pct": round(cm.structural_filter_rejection_rate * 100, 1)
                }
            }

    def reset(self) -> None:
        with self._lock_data:
            self._doc_metrics.clear()
            self._cluster_metrics = ClusterMetrics()
