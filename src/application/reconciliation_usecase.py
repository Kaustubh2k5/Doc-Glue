"""
Application Usecase: ReconciliationUseCase.
Orchestrates Cluster-First fact candidate search, vector clustering, single-pass LLM cluster evaluations, and storage.
Adheres strictly to SOLID principles (Dependency Inversion).
"""
import logging
from typing import List, Set, Tuple
from src.domain.interfaces import (
    IFactRepository,
    IReconciliationRepository,
    IReconciliationEvaluator,
    IEmbeddingService
)
from src.domain.models import Fact, FactComparison
from src.infrastructure.clustering.vector_clusterer import VectorFactClusterer

logger = logging.getLogger(__name__)


class ReconciliationUseCase:
    """Orchestrates Cluster-First fact reconciliation analysis."""

    def __init__(
        self,
        fact_repository: IFactRepository,
        reconciliation_repository: IReconciliationRepository,
        evaluator: IReconciliationEvaluator,
        embedding_service: IEmbeddingService
    ):
        self.fact_repository = fact_repository
        self.reconciliation_repository = reconciliation_repository
        self.evaluator = evaluator
        self.embedding_service = embedding_service
        self.clusterer = VectorFactClusterer(embedding_service=self.embedding_service, similarity_threshold=0.85)
        self._evaluated_pairs: Set[Tuple[str, str]] = set()

    def process_new_facts(self, new_facts: List[Fact]) -> List[FactComparison]:
        """
        Executes Cluster-First reconciliation on newly ingested facts against the total knowledge pool.

        Steps:
        1. Pool new_facts with existing facts from repository.
        2. Run VectorFactClusterer to form dense semantic clusters based on normalized fact vectors.
        3. Singletons (unique facts without vector neighbors) skip LLM calls entirely.
        4. Multi-fact clusters are evaluated in single batched LLM calls.
        5. Comparisons are saved to IReconciliationRepository.
        """
        if not new_facts:
            return []

        existing_facts = self.fact_repository.get_all_facts()
        # Combine existing and new facts cleanly by fact_id
        combined_map = {f.fact_id: f for f in (existing_facts + new_facts)}
        all_facts = list(combined_map.values())

        if len(all_facts) < 2:
            return []

        # Cluster facts
        clusters, singletons = self.clusterer.group_into_clusters(all_facts)
        logger.info(f"Cluster-First Engine formed {len(clusters)} cluster(s) and {len(singletons)} singleton(s). Singletons skipped LLM evaluation.")

        saved_comparisons: List[FactComparison] = []

        for cluster in clusters:
            # Single-pass batched LLM evaluation per cluster
            cluster_comparisons = self.evaluator.evaluate_cluster(
                cluster_id=cluster.cluster_id,
                candidate_facts=cluster.candidate_facts
            )

            for comp in cluster_comparisons:
                pair_key = tuple(sorted([comp.fact_a.fact_id, comp.fact_b.fact_id]))
                if pair_key not in self._evaluated_pairs:
                    self._evaluated_pairs.add(pair_key)
                    self.reconciliation_repository.save_comparison(comp)
                    saved_comparisons.append(comp)

        return saved_comparisons

    def process_new_fact(self, new_fact: Fact, top_k: int = 5) -> List[FactComparison]:
        """Backward-compatible single fact helper calling process_new_facts."""
        return self.process_new_facts([new_fact])
