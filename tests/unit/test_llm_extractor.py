"""
Unit tests for MultiProviderFactExtractor.
Tests fact extraction formatting and mock fallback behavior.
"""
from src.infrastructure.extractors.llm_extractor import MultiProviderFactExtractor


def test_extractor_mock_fallback():
    # Instantiated without API keys uses mock extraction
    extractor = MultiProviderFactExtractor()
    facts = extractor.extract_facts(
        text_chunk="Revenue reached 81,407 Million INR in FY24.",
        document_id="doc-123",
        filename="report.pdf",
        page_number=1
    )

    assert len(facts) > 0
    fact = facts[0]
    assert fact.evidence.document_id == "doc-123"
    assert fact.evidence.page_number == 1
    assert fact.evidence.filename == "report.pdf"


def test_extractor_empty_text():
    extractor = MultiProviderFactExtractor()
    facts = extractor.extract_facts(
        text_chunk="   ",
        document_id="doc-123",
        filename="report.pdf",
        page_number=1
    )
    assert len(facts) == 0
