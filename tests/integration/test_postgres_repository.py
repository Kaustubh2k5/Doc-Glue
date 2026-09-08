"""
Integration tests for PostgresFactRepository and PostgresReconciliationRepository.
Tests schema auto-migration, fact persistence, and pgvector HNSW similarity search.
"""
import os
import pytest
from src.domain.models import Fact, SourceEvidence, FactComparison, RelationType
from src.infrastructure.persistence.postgres_repository import (
    PostgresFactRepository,
    PostgresReconciliationRepository
)
from src.infrastructure.embeddings.openai_embedding import OpenAIEmbeddingService


@pytest.mark.integration
def test_postgres_repository_operations():
    # Only run if Postgres database is reachable
    try:
        fact_repo = PostgresFactRepository()
    except Exception as e:
        pytest.skip(f"PostgreSQL database connection failed: {e}")

    rec_repo = PostgresReconciliationRepository()

    # 1. Test save_fact
    evidence = SourceEvidence(
        document_id="doc-test-db",
        filename="integration_test.pdf",
        page_number=2,
        verbatim_text="Revenue was 81,407 Million INR"
    )

    fact = Fact(
        fact_id="fact-integration-1",
        subject="Delhivery",
        property_name="Revenue",
        value=81407.2,
        unit="Million INR",
        temporal_context="FY24",
        scope_context="Consolidated",
        evidence=evidence
    )

    embedding_service = OpenAIEmbeddingService()
    vector = embedding_service.generate_embedding("Delhivery Revenue 81407.2 FY24")

    fact_repo.save_fact(fact=fact, embedding=vector)

    # 2. Test get_all_facts
    all_facts = fact_repo.get_all_facts()
    assert len(all_facts) > 0
    saved_fact = [f for f in all_facts if f.fact_id == "fact-integration-1"][0]
    assert saved_fact.subject == "Delhivery"
    assert saved_fact.evidence.filename == "integration_test.pdf"

    # 3. Test find_candidates vector search
    candidates = fact_repo.find_candidates(query_vector=vector, top_k=3)
    assert len(candidates) > 0
    assert candidates[0].fact_id == "fact-integration-1"

    # 4. Test save_comparison
    comparison = FactComparison(
        fact_a=fact,
        fact_b=fact,
        relationship=RelationType.CORROBORATED,
        reasoning="Identical claim across test documents.",
        resolution_details={"matching_unit": "Million INR"}
    )
    rec_repo.save_comparison(comparison)

    comparisons = rec_repo.get_all_comparisons()
    assert len(comparisons) > 0
    assert comparisons[0].relationship == RelationType.CORROBORATED

    fact_repo.close()
    rec_repo.close()
