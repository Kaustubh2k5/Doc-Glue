"""
Domain models for the Fact Knowledge Layer.
Defines core entities: RelationType, SourceEvidence, Fact, and FactComparison.
"""
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class RelationType(str, Enum):
    CORROBORATED = "CORROBORATED"
    CONTRADICTED = "CONTRADICTED"
    RECONCILED = "RECONCILED"


class SourceEvidence(BaseModel):
    """Refers to exact evidence grounding in a source document."""
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., description="Unique identifier of the source document")
    filename: str = Field(..., description="Original filename of the source document")
    page_number: int = Field(..., description="1-indexed page number where the fact was found")
    verbatim_text: str = Field(..., description="Exact verbatim text excerpt from document supporting the fact")


class Fact(BaseModel):
    """
    Represents an extracted numerical or semantic fact.
    Designed to be domain-agnostic and fully grounded.
    """
    model_config = ConfigDict(frozen=True)

    fact_id: str = Field(..., description="Unique identifier for the fact")
    subject: str = Field(..., description="The entity or subject of the claim (e.g., Delhivery, India GDP)")
    property_name: str = Field(..., description="The attribute or metric being reported (e.g., Revenue, Inflation Rate)")
    value: Any = Field(..., description="Extracted numerical or semantic value (e.g., 81407.2, 6.8%)")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., Million INR, USD, %, metric tonnes)")
    temporal_context: Optional[str] = Field(None, description="Time period or date applicable (e.g., FY24, Q4 FY24, 2023-24)")
    scope_context: Optional[str] = Field(None, description="Geographic, operational, or segment scope (e.g., Express Parcel, Gross, Consolidated)")
    evidence: SourceEvidence = Field(..., description="Grounding source evidence snippet and page reference")


class FactComparison(BaseModel):
    """Represents pairwise reconciliation analysis between two facts."""
    model_config = ConfigDict(frozen=True)

    fact_a: Fact
    fact_b: Fact
    relationship: RelationType
    reasoning: str = Field(..., description="Step-by-step reasoning explaining the relation")
    resolution_details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Detailed contextual breakdown (e.g., unit conversion, period mapping, scope differences)"
    )
