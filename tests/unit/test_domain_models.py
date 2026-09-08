"""
Unit tests for domain models (Fact, SourceEvidence, RelationType, FactComparison).
"""
import pytest
from src.domain.models import RelationType, SourceEvidence, Fact, FactComparison


def test_source_evidence_immutability():
    evidence = SourceEvidence(
        document_id="doc-123",
        filename="test.pdf",
        page_number=1,
        verbatim_text="Revenue reached 81407 Million INR"
    )
    assert evidence.document_id == "doc-123"
    assert evidence.filename == "test.pdf"
    assert evidence.page_number == 1
    
    with pytest.raises(Exception):
        evidence.page_number = 2  # Frozen check


def test_fact_model_creation():
    evidence = SourceEvidence(
        document_id="doc-100",
        filename="report.pdf",
        page_number=3,
        verbatim_text="EBITDA margin stood at 12.5%"
    )
    fact = Fact(
        fact_id="fact-1",
        subject="Delhivery",
        property_name="EBITDA margin",
        value=12.5,
        unit="%",
        temporal_context="FY24",
        scope_context="Consolidated",
        evidence=evidence
    )
    assert fact.subject == "Delhivery"
    assert fact.value == 12.5
    assert fact.evidence.page_number == 3


def test_relation_type_enum():
    assert RelationType.CORROBORATED.value == "CORROBORATED"
    assert RelationType.CONTRADICTED.value == "CONTRADICTED"
    assert RelationType.RECONCILED.value == "RECONCILED"
