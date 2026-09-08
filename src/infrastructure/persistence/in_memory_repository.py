"""
Lightweight in-memory repository for storing facts during Phase 1 operations.
Zero external database or C-library dependency, ultra-fast execution, low RAM footprint.
"""
import math
from typing import List, Optional, Dict
from src.domain.interfaces import IFactRepository
from src.domain.models import Fact


class InMemoryFactRepository(IFactRepository):
    """Simple, fast in-memory fact repository."""

    def __init__(self):
        self._facts: Dict[str, Fact] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._document_hashes: Dict[str, Dict[str, str]] = {}

    def find_document_by_hash(self, file_hash: str) -> Optional[Dict[str, str]]:
        return self._document_hashes.get(file_hash)

    def save_document_hash(self, doc_id: str, filename: str, file_hash: str) -> None:
        self._document_hashes[file_hash] = {"id": doc_id, "filename": filename}

    def save_fact(self, fact: Fact, embedding: Optional[List[float]] = None) -> None:
        self._facts[fact.fact_id] = fact
        if embedding:
            self._embeddings[fact.fact_id] = embedding

    def get_all_facts(self) -> List[Fact]:
        return list(self._facts.values())

    def find_candidates(self, query_vector: List[float], top_k: int = 5) -> List[Fact]:
        if not self._embeddings or not query_vector:
            return list(self._facts.values())[:top_k]

        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0:
            return list(self._facts.values())[:top_k]

        scores = []
        for fact_id, emb in self._embeddings.items():
            dot = sum(q * e for q, e in zip(query_vector, emb))
            emb_norm = math.sqrt(sum(e * e for e in emb))
            if emb_norm > 0:
                sim = dot / (q_norm * emb_norm)
            else:
                sim = 0.0
            scores.append((sim, self._facts[fact_id]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scores[:top_k]]

    def clear_all_data(self) -> None:
        self._facts.clear()
        self._embeddings.clear()
        self._document_hashes.clear()
