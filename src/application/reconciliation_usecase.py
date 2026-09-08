"""
Application Usecase: ReconciliationUseCase.
Orchestrates cross-document fact candidate search, deduplication, pairwise LLM reconciliation evaluation, and storage.
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

logger = logging.getLogger(__name__)


class ReconciliationUseCase:
    """Orchestrates candidate lookup and pairwise fact reconciliation analysis."""

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
        self._evaluated_pairs: Set[Tuple[str, str]] = set()

    def process_new_fact(self, new_fact: Fact, top_k: int = 5) -> List[FactComparison]:
        """
        Processes a newly ingested fact against stored candidates to evaluate cross-document relations.

        Args:
            new_fact: The newly ingested Fact object.
            top_k: Number of nearest vector candidates to evaluate against.

        Returns:
            List of generated FactComparison objects.
        """
        # Step 1: Generate vector embedding for new_fact
        fact_text = f"{new_fact.subject} {new_fact.property_name} {new_fact.value} {new_fact.temporal_context or ''} {new_fact.scope_context or ''}"
        query_vector = self.embedding_service.generate_embedding(fact_text)

        # Step 2: Retrieve top-k candidates matching vector similarity
        candidates = self.fact_repository.find_candidates(query_vector=query_vector, top_k=top_k)
        
        comparisons: List[FactComparison] = []

        # Step 3: Iterate through candidates and evaluate pairs
        for candidate in candidates:
            # Skip self-comparison (same fact_id)
            if candidate.fact_id == new_fact.fact_id:
                continue

            # Skip duplicate pairs (Fact A -> Fact B is identical to Fact B -> Fact A)
            pair_key = tuple(sorted([new_fact.fact_id, candidate.fact_id]))
            if pair_key in self._evaluated_pairs:
                continue

            # Mark pair as evaluated
            self._evaluated_pairs.add(pair_key)

            # Step 4: Run evaluate_pair
            comparison = self.evaluator.evaluate_pair(fact_a=new_fact, fact_b=candidate)
            
            # Step 5: Save comparison to repository
            self.reconciliation_repository.save_comparison(comparison)
            comparisons.append(comparison)

        logger.info(f"Reconciled fact '{new_fact.fact_id}' against {len(comparisons)} candidate(s).")
        return comparisons
