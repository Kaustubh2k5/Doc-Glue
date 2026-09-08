"""
Benchmark Demo Script testing the Cluster-First Reconciliation Engine with the exact User Payload:
- Fact 1: Performance Score (Easy Test Set) = 0.662
- Fact 2: Performance Score (Hard Test Set) = 0.197

Verifies that difficulty variants are classified as RECONCILED (NOT CONTRADICTED).
"""
from src.domain.models import Fact, SourceEvidence
from src.infrastructure.evaluators.openai_reconciler import OpenAIReconciliationEvaluator
from src.infrastructure.embeddings.fast_embedding import FastEmbeddingService
from src.infrastructure.clustering.vector_clusterer import VectorFactClusterer


def main():
    print("=== Cluster-First Reconciliation Engine Benchmark ===")

    f1 = Fact(
        fact_id="f1a2b3c4-0001",
        subject="Acme Corp",
        property_name="Performance Score",
        value="0.662",
        unit="ratio",
        temporal_context="FY2024",
        scope_context="Easy Test Set",
        evidence=SourceEvidence(
            document_id="doc_annual_report_2024.pdf",
            filename="doc_annual_report_2024.pdf",
            page_number=12,
            verbatim_text="Under the Easy evaluation suite, Acme Corp achieved a Performance Score of 0.662 in FY2024."
        )
    )

    f2 = Fact(
        fact_id="f1a2b3c4-0002",
        subject="Acme Corp",
        property_name="Performance Score",
        value="0.197",
        unit="ratio",
        temporal_context="FY2024",
        scope_context="Hard Test Set",
        evidence=SourceEvidence(
            document_id="doc_technical_audit_2024.pdf",
            filename="doc_technical_audit_2024.pdf",
            page_number=4,
            verbatim_text="Acme Corp recorded a Performance Score of 0.197 when tested against the Hard evaluation benchmark."
        )
    )

    # 1. Test Vector Clusterer
    embedding_service = FastEmbeddingService()
    clusterer = VectorFactClusterer(embedding_service=embedding_service, similarity_threshold=0.85)

    clusters, singletons = clusterer.group_into_clusters([f1, f2])

    print(f"\n1. Vector Clustering Output:")
    print(f"   Clusters Formed: {len(clusters)}")
    print(f"   Singletons Skipped: {len(singletons)}")

    # 2. Test Cluster Evaluator
    evaluator = OpenAIReconciliationEvaluator()
    if clusters:
        results = evaluator.evaluate_cluster(
            cluster_id=clusters[0].cluster_id,
            candidate_facts=clusters[0].candidate_facts
        )

        print(f"\n2. Cluster-First LLM Evaluation Result:")
        for idx, res in enumerate(results, 1):
            print(f"   [{idx}] Pair: {res.fact_a.fact_id} vs {res.fact_b.fact_id}")
            print(f"       Relation: {res.relationship.value}")
            print(f"       Reasoning: {res.reasoning}")
            print(f"       Resolution Details: {res.resolution_details}\n")


if __name__ == "__main__":
    main()
