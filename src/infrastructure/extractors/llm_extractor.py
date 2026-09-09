"""
Multi-provider LLM Fact Extractor supporting OpenRouter (free tier), Google Gemini API, and OpenAI.
Implements automatic provider/model fallback, structured JSON extraction, and verbatim text citation grounding.
"""
import os
import json
import uuid
import logging
from typing import List, Optional, Dict, Any
from openai import OpenAI
from src.domain.interfaces import IFactExtractor
from src.domain.models import Fact, SourceEvidence
from src.infrastructure.metrics import PipelineMetricsCollector

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """
You are an expert financial and domain data extraction system.
Your goal is to extract ALL atomic, domain-agnostic numerical or semantic facts from the provided text chunk.

Rules:
1. Extract explicit claims, financial metrics, operational KPIs, dates, policies, or organizational facts.
2. For every fact, provide:
   - subject: The entity being described (e.g., 'Delhivery', 'India GDP', 'RBI').
   - property_name: The metric or attribute name (e.g., 'Revenue from operations', 'Inflation Rate', 'Active Users').
   - value: The exact value reported (e.g., 81407.2, '6.8%', 'Resigned', 'Active').
   - unit: Unit of measurement if applicable (e.g., 'Million INR', '%', 'USD', 'tonnes'). Null if none.
   - temporal_context: Time period or date applicable (e.g., 'FY24', 'Q4 FY24', '2023-24'). Null if none.
   - scope_context: Scope, region, or segment (e.g., 'Express Parcel', 'Consolidated', 'Gross'). Null if none.
   - verbatim_text: Exact verbatim sentence or phrase from the source text supporting this claim.

3. Return ONLY a valid JSON object matching this schema:
{
  "facts": [
    {
      "subject": "string",
      "property_name": "string",
      "value": "string or number",
      "unit": "string or null",
      "temporal_context": "string or null",
      "scope_context": "string or null",
      "verbatim_text": "string"
    }
  ]
}

4. QUALITY RULES:
   - Do NOT extract page numbers, table headers, section titles, or boilerplate text as facts.
   - Do NOT extract facts where the value is missing, unclear, or purely decorative.
   - Only extract facts that carry meaningful, verifiable domain information.
   - If the chunk contains no extractable facts, return {"facts": []}.

Do not add conversational commentary or extra keys outside "facts".
"""


class MultiProviderFactExtractor(IFactExtractor):
    """
    LLM Fact Extractor with OpenRouter / Gemini / OpenAI fallback chains.
    Reads configuration from environment variables (OPENROUTER_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY).
    """

    DEFAULT_FALLBACK_MODELS = [
        "google/gemini-2.5-flash:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat:free",
        "gpt-4o-mini"
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        models: Optional[List[str]] = None
    ):
        # Resolve API keys and base URL
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            self.api_key = api_key
            self.base_url = base_url
        elif self.openrouter_key:
            self.api_key = self.openrouter_key
            self.base_url = base_url or "https://openrouter.ai/api/v1"
        elif self.openai_key:
            self.api_key = self.openai_key
            self.base_url = base_url
        elif self.gemini_key:
            self.api_key = self.gemini_key
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            # Mock / fallback mode if no key provided yet
            self.api_key = "mock-key"
            self.base_url = base_url

        self.models = models or self.DEFAULT_FALLBACK_MODELS
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key != "mock-key" else None

    def extract_facts(
        self,
        text_chunk: str,
        document_id: str,
        filename: str,
        page_number: int
    ) -> List[Fact]:
        if not text_chunk.strip():
            return []

        # Pre-extraction quality gate: skip very short chunks
        if len(text_chunk.strip()) < 50:
            return []

        if not self.client:
            # Fallback mock extraction for local testing without API keys
            return self._mock_extraction(text_chunk, document_id, filename, page_number)

        user_prompt = f"Text Chunk (Page {page_number}):\n\"\"\"\n{text_chunk}\n\"\"\""

        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                raw_json = response.choices[0].message.content
                if not raw_json:
                    continue

                parsed = self._parse_json_response(raw_json)
                facts_data = parsed.get("facts", [])
                
                facts: List[Fact] = []
                metrics_collector = PipelineMetricsCollector()
                doc_metrics = metrics_collector.get_or_create_doc_metrics(filename)
                doc_metrics.facts_extracted += len(facts_data)
                
                for item in facts_data:
                    # Post-extraction quality validation
                    if not self._validate_fact(item):
                        doc_metrics.facts_rejected_quality += 1
                        continue
                        
                    evidence = SourceEvidence(
                        document_id=document_id,
                        filename=filename,
                        page_number=page_number,
                        verbatim_text=item.get("verbatim_text") or text_chunk[:150]
                    )
                    
                    fact = Fact(
                        fact_id=f"fact-{uuid.uuid4().hex[:12]}",
                        subject=str(item.get("subject", "Unknown")),
                        property_name=str(item.get("property_name", "Property")),
                        value=item.get("value"),
                        unit=item.get("unit"),
                        temporal_context=item.get("temporal_context"),
                        scope_context=item.get("scope_context"),
                        evidence=evidence
                    )
                    facts.append(fact)
                
                doc_metrics.facts_kept += len(facts)
                return facts

            except Exception as e:
                logger.warning(f"Fact extraction failed with model {model}: {e}. Retrying with next model...")

        logger.error("All model extraction attempts failed. Returning empty list.")
        return []

    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """Parses LLM response, stripping markdown JSON backticks if present."""
        clean_text = raw_response.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        return json.loads(clean_text)

    GENERIC_SUBJECTS = {"", "unknown", "n/a", "na", "table", "header", "none", "figure", "chart", "source", "note", "notes"}

    def _validate_fact(self, item: dict) -> bool:
        """Validates an extracted fact against quality gates. Returns True if the fact is valid."""
        subject = str(item.get("subject", "")).strip().lower()
        if subject in self.GENERIC_SUBJECTS or len(subject) < 2:
            return False

        value = item.get("value")
        if value is None or str(value).strip() == "" or str(value).strip().lower() in ("n/a", "none", "null", "-"):
            return False

        prop = str(item.get("property_name", "")).strip()
        if len(prop) < 2 or prop.isdigit():
            return False

        verbatim = str(item.get("verbatim_text", "")).strip()
        if len(verbatim) < 15:
            return False

        return True

    def _mock_extraction(
        self,
        text_chunk: str,
        document_id: str,
        filename: str,
        page_number: int
    ) -> List[Fact]:
        """Mock extraction fallback for offline / mock testing."""
        return [
            Fact(
                fact_id=f"mock-{uuid.uuid4().hex[:8]}",
                subject="Sample Subject",
                property_name="Sample Property",
                value="Sample Value",
                unit="Units",
                temporal_context="FY24",
                scope_context="Global",
                evidence=SourceEvidence(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    verbatim_text=text_chunk[:100]
                )
            )
        ]
