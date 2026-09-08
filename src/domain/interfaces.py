"""
Abstract Base Classes (Interfaces) for the Fact Knowledge Layer system.
Ensures Dependency Inversion Principle (DIP) across domain, application, and infrastructure layers.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from src.domain.models import Fact, SourceEvidence, FactComparison


class IDocumentParser(ABC):
    """Abstract interface for parsing PDF documents into text/markdown chunks per page."""

    @abstractmethod
    def parse(self, file_path: str, document_id: str) -> List[Tuple[int, str]]:
        """
        Parses a PDF file into list of tuples: (page_number, text_content).
        Page numbers should be 1-indexed.
        """
        pass


class IFactExtractor(ABC):
    """Abstract interface for extracting structured facts from document text."""

    @abstractmethod
    def extract_facts(
        self,
        text_chunk: str,
        document_id: str,
        filename: str,
        page_number: int
    ) -> List[Fact]:
        """Extract atomic numerical or semantic facts from a given text chunk."""
        pass


class IEmbeddingService(ABC):
    """Abstract interface for generating vector embeddings for text search."""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generates a vector embedding representation for the text string."""
        pass


class IFactRepository(ABC):
    """Abstract interface for persisting and retrieving extracted facts and vectors."""

    @abstractmethod
    def save_fact(self, fact: Fact, embedding: Optional[List[float]] = None) -> None:
        """Persists a fact and optional vector embedding."""
        pass

    @abstractmethod
    def get_all_facts(self) -> List[Fact]:
        """Retrieves all persisted facts."""
        pass

    @abstractmethod
    def find_candidates(self, query_vector: List[float], top_k: int = 5) -> List[Fact]:
        """Finds top-k candidate facts matching vector similarity."""
        pass

    @abstractmethod
    def clear_all_data(self) -> None:
        """Clears all facts and vector records from the database."""
        pass


class IReconciliationEvaluator(ABC):
    """Abstract interface for evaluating pairwise and cluster relationships between facts."""

    @abstractmethod
    def evaluate_pair(self, fact_a: Fact, fact_b: Fact) -> FactComparison:
        """Evaluates whether two facts corroborate, contradict, or can be reconciled."""
        pass

    @abstractmethod
    def evaluate_cluster(self, cluster_id: str, candidate_facts: List[Fact]) -> List[FactComparison]:
        """Evaluates a cluster of semantically related facts in a single pass."""
        pass


class IReconciliationRepository(ABC):
    """Abstract interface for persisting and querying reconciliation records."""

    @abstractmethod
    def save_comparison(self, comparison: FactComparison) -> None:
        """Saves a reconciliation comparison result."""
        pass

    @abstractmethod
    def get_all_comparisons(self) -> List[FactComparison]:
        """Retrieves all stored comparison results."""
        pass

    @abstractmethod
    def clear_all_data(self) -> None:
        """Clears all reconciliation records from the database."""
        pass
