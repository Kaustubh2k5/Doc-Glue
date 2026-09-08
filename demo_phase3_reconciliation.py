"""
Live Demo Script for Phase 3: Cross-Document Reconciliation Logic.
Demonstrates the 3 core cases required by the problem statement:
1. CORROBORATED: Fact corroborated across documents (different phrasing/unit conversions).
2. CONTRADICTED: Direct conflict under identical time and scope conditions.
3. RECONCILED: Values differ, but explained by context (different years, currencies, regions).
"""
from src.domain.models import Fact, SourceEvidence
from src.infrastructure.evaluators.openai_reconciler import OpenAIReconciliationEvaluator


def main():
    print("=== Phase 3 Cross-Document Reconciliation Engine Demo ===")
    evaluator = OpenAIReconciliationEvaluator()

    # Case 1: Corroborated
    f_corrob_a = Fact(
        fact_id="fact-c1",
        subject="Delhivery",
        property_name="Revenue from operations",
        value=81407.2,
        unit="Million INR",
        temporal_context="FY24",
        scope_context="Consolidated",
        evidence=SourceEvidence(
            document_id="doc-1",
            filename="delhivery-q4-fy24-earnings.pdf",
            page_number=6,
            verbatim_text="FY24 Revenue from operations stood at ₹81,407.2 Million INR"
        )
    )

    f_corrob_b = Fact(
        fact_id="fact-c2",
        subject="Delhivery",
        property_name="Total Revenue",
        value="8,140.72",
        unit="Crore INR",
        temporal_context="FY24",
        scope_context="Consolidated",
        evidence=SourceEvidence(
            document_id="doc-2",
            filename="delhivery-annual-report-fy24.pdf",
            page_number=12,
            verbatim_text="Total revenue for FY24 reached ₹8,140.72 Cr"
        )
    )

    # Case 2: Contradicted
    f_contra_a = Fact(
        fact_id="fact-k1",
        subject="Delhivery",
        property_name="Express Parcel Shipments",
        value=740,
        unit="Million",
        temporal_context="FY24",
        scope_context="Express Parcel",
        evidence=SourceEvidence(
            document_id="doc-1",
            filename="delhivery-q4-fy24-earnings.pdf",
            page_number=6,
            verbatim_text="Express parcel shipments in FY24 reached 740 Mn"
        )
    )

    f_contra_b = Fact(
        fact_id="fact-k2",
        subject="Delhivery",
        property_name="Express Parcel Shipments",
        value=520,
        unit="Million",
        temporal_context="FY24",
        scope_context="Express Parcel",
        evidence=SourceEvidence(
            document_id="doc-3",
            filename="delhivery-industry-analysis.pdf",
            page_number=4,
            verbatim_text="Express parcel shipments volume in FY24 reported as 520 Mn"
        )
    )

    # Case 3: Reconciled
    f_recon_a = Fact(
        fact_id="fact-r1",
        subject="Delhivery",
        property_name="Revenue from services",
        value=72253.0,
        unit="Million INR",
        temporal_context="FY23",
        scope_context="Consolidated",
        evidence=SourceEvidence(
            document_id="doc-1",
            filename="delhivery-q4-fy24-earnings.pdf",
            page_number=6,
            verbatim_text="FY23 revenue from services: ₹72,253.0 Million INR"
        )
    )

    f_recon_b = Fact(
        fact_id="fact-r2",
        subject="Delhivery",
        property_name="Revenue from services",
        value=81407.2,
        unit="Million INR",
        temporal_context="FY24",
        scope_context="Consolidated",
        evidence=SourceEvidence(
            document_id="doc-1",
            filename="delhivery-q4-fy24-earnings.pdf",
            page_number=6,
            verbatim_text="FY24 revenue from services: ₹81,407.2 Million INR"
        )
    )

    test_cases = [
        ("1. CORROBORATED CASE", f_corrob_a, f_corrob_b),
        ("2. CONTRADICTED CASE", f_contra_a, f_contra_b),
        ("3. RECONCILED CASE", f_recon_a, f_recon_b)
    ]

    for title, fact_a, fact_b in test_cases:
        print(f"\n--- [ {title} ] ---")
        print(f"Fact A ({fact_a.evidence.filename}): {fact_a.subject} | {fact_a.property_name} = {fact_a.value} {fact_a.unit or ''} [{fact_a.temporal_context or ''}]")
        print(f"Fact B ({fact_b.evidence.filename}): {fact_b.subject} | {fact_b.property_name} = {fact_b.value} {fact_b.unit or ''} [{fact_b.temporal_context or ''}]")
        
        result = evaluator.evaluate_pair(fact_a, fact_b)
        
        print(f" Relationship: {result.relationship.value}")
        print(f" Reasoning: {result.reasoning}")
        print(f" Details: {result.resolution_details}")


if __name__ == "__main__":
    main()
