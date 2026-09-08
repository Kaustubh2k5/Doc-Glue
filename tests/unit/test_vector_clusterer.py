"""
Unit tests for VectorFactClusterer.
Tests semantic grouping of facts, cluster formation, and singleton isolation.
"""
from src.domain.models import Fact, SourceEvidence
from src.infrastructure.clustering.vector_clusterer import VectorFactClusterer
from src.infrastructure.embeddings.fast_embedding import FastEmbeddingService


def create_fact(fact_id: str, subject: str, property_name: str, value: str, scope: str) -> Fact:
    evidence = SourceEvidence(document_id="doc1", filename="a.pdf", page_number=1, verbatim_text="Text")
    return Fact(
        fact_id=fact_id,
        subject=subject,
        property_name=property_name,
        value=value,
        scope_context=scope,
        evidence=evidence
    )


def test_vector_clusterer_grouping():
    embedding_service = FastEmbeddingService()
    clusterer = VectorFactClusterer(embedding_service=embedding_service, similarity_threshold=0.85)

    f1 = create_fact("f1", "Acme Corp", "Performance Score", "0.662", "Easy Test Set")
    f2 = create_fact("f2", "Acme Corp", "Performance Score", "0.197", "Hard Test Set")
    f3 = create_fact("f3", "Other Corp", "Unrelated Metric", "999", "Global")

    facts = [f1, f2, f3]
    clusters, singletons = clusterer.group_into_clusters(facts)

    assert len(clusters) == 1
    assert len(clusters[0].candidate_facts) == 2
    assert set(f.fact_id for f in clusters[0].candidate_facts) == {"f1", "f2"}

    assert len(singletons) == 1
    assert singletons[0].fact_id == "f3"
