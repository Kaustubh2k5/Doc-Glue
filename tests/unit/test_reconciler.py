"""
Unit tests for OpenAIReconciliationEvaluator testing all 3 relation categories:
1. CORROBORATED (Identical claim / unit conversion across documents)
2. CONTRADICTED (Direct conflict under identical time/scope)
3. RECONCILED (Explained by context like time, currency, or scope)
"""
from unittest.mock import MagicMock
from src.domain.models import Fact, SourceEvidence, RelationType
from src.infrastructure.evaluators.openai_reconciler import OpenAIReconciliationEvaluator


def create_sample_fact(fact_id: str, subject: str, property_name: str, value: str, unit: str, temporal: str, scope: str, filename: str) -> Fact:
    evidence = SourceEvidence(
        document_id=f"doc-{fact_id}",
        filename=filename,
        page_number=1,
        verbatim_text=f"{subject} {property_name} is {value} {unit} for {temporal} ({scope})"
    )
    return Fact(
        fact_id=fact_id,
        subject=subject,
        property_name=property_name,
        value=value,
        unit=unit,
        temporal_context=temporal,
        scope_context=scope,
        evidence=evidence
    )


def test_evaluator_corroborated_category():
    fact_a = create_sample_fact("f1", "Delhivery", "Revenue", 81407.2, "Million INR", "FY24", "Consolidated", "doc1.pdf")
    fact_b = create_sample_fact("f2", "Delhivery", "Revenue from operations", "8,140.72", "Crore INR", "FY24", "Consolidated", "doc2.pdf")

    evaluator = OpenAIReconciliationEvaluator()
    
    # Mock LLM API response returning CORROBORATED
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = """
    {
      "relationship": "CORROBORATED",
      "reasoning": "Both documents report identical revenue for FY24 (81,407.2 Million INR equals 8,140.72 Crore INR).",
      "resolution_details": {"unit_conversion": "1 Crore = 10 Million"}
    }
    """
    mock_completion.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_completion
    evaluator.client = mock_client

    comparison = evaluator.evaluate_pair(fact_a, fact_b)
    assert comparison.relationship == RelationType.CORROBORATED
    assert "identical" in comparison.reasoning.lower()


def test_evaluator_contradicted_category():
    fact_a = create_sample_fact("f1", "Delhivery", "Revenue", 81407.2, "Million INR", "FY24", "Consolidated", "doc1.pdf")
    fact_b = create_sample_fact("f2", "Delhivery", "Revenue", 50000.0, "Million INR", "FY24", "Consolidated", "doc2.pdf")

    evaluator = OpenAIReconciliationEvaluator()

    # Mock LLM API response returning CONTRADICTED
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = """
    {
      "relationship": "CONTRADICTED",
      "reasoning": "Direct conflict: doc1 reports 81,407.2 Million INR while doc2 reports 50,000 Million INR under identical FY24 consolidated conditions.",
      "resolution_details": {"conflict_field": "value"}
    }
    """
    mock_completion.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_completion
    evaluator.client = mock_client

    comparison = evaluator.evaluate_pair(fact_a, fact_b)
    assert comparison.relationship == RelationType.CONTRADICTED
    assert "conflict" in comparison.reasoning.lower()


def test_evaluator_reconciled_category():
    fact_a = create_sample_fact("f1", "Delhivery", "Revenue", 72253.0, "Million INR", "FY23", "Consolidated", "doc1.pdf")
    fact_b = create_sample_fact("f2", "Delhivery", "Revenue", 81407.2, "Million INR", "FY24", "Consolidated", "doc2.pdf")

    evaluator = OpenAIReconciliationEvaluator()

    # Mock LLM API response returning RECONCILED
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = """
    {
      "relationship": "RECONCILED",
      "reasoning": "Values differ (72,253 vs 81,407.2) because they cover different financial periods (FY23 vs FY24).",
      "resolution_details": {"explaining_factor": "different_temporal_periods"}
    }
    """
    mock_completion.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_completion
    evaluator.client = mock_client

    comparison = evaluator.evaluate_pair(fact_a, fact_b)
    assert comparison.relationship == RelationType.RECONCILED
    assert "different financial periods" in comparison.reasoning.lower()
