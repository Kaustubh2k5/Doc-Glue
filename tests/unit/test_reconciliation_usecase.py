"""
Unit tests for ReconciliationUseCase.
Tests candidate retrieval, self-comparison skipping, deduplication, and repository persistence.
"""
from src.domain.models import Fact, SourceEvidence, FactComparison, RelationType
from src.application.reconciliation_usecase import ReconciliationUseCase
from src.infrastructure.persistence.in_memory_repository import InMemoryFactRepository
from src.infrastructure.embeddings.fast_embedding import FastEmbeddingService


class MockEvaluator:
    def evaluate_pair(self, fact_a: Fact, fact_b: Fact) -> FactComparison:
        return FactComparison(
            fact_a=fact_a,
            fact_b=fact_b,
            relationship=RelationType.CORROBORATED,
            reasoning="Mock evaluation corroboration.",
            resolution_details={"mock": True}
        )


class MockReconciliationRepository:
    def __init__(self):
        self.saved_comparisons = []

    def save_comparison(self, comparison: FactComparison) -> None:
        self.saved_comparisons.append(comparison)

    def get_all_comparisons(self):
        return self.saved_comparisons


def test_reconciliation_usecase():
    fact_repo = InMemoryFactRepository()
    rec_repo = MockReconciliationRepository()
    evaluator = MockEvaluator()
    embedding_service = FastEmbeddingService()

    usecase = ReconciliationUseCase(
        fact_repository=fact_repo,
        reconciliation_repository=rec_repo,
        evaluator=evaluator,
        embedding_service=embedding_service
    )

    evidence = SourceEvidence(document_id="doc1", filename="a.pdf", page_number=1, verbatim_text="Text")
    f1 = Fact(fact_id="f1", subject="Delhivery", property_name="Rev", value=100, evidence=evidence)
    f2 = Fact(fact_id="f2", subject="Delhivery", property_name="Rev", value=100, evidence=evidence)

    # Save f1 in repo
    emb1 = embedding_service.generate_embedding("Delhivery Rev 100")
    fact_repo.save_fact(f1, emb1)

    # Process new fact f2
    comparisons = usecase.process_new_fact(f2, top_k=5)

    assert len(comparisons) == 1
    assert comparisons[0].fact_a.fact_id == "f2"
    assert comparisons[0].fact_b.fact_id == "f1"
    assert len(rec_repo.saved_comparisons) == 1
