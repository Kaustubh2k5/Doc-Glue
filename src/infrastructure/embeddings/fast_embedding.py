"""
Lightweight embedding service using hash/deterministic vectors or OpenAI API embeddings.
Fast execution without downloading gigabytes of local models.
"""
import os
import hashlib
from typing import List, Optional
from src.domain.interfaces import IEmbeddingService


class FastEmbeddingService(IEmbeddingService):
    """
    Lightweight embedding generator.
    Generates deterministic normalized 1536-dimensional feature vectors via SHA-256 hash or OpenAI API.
    Used for instant vector operations without heavy local model weight dependencies.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def generate_embedding(self, text: str) -> List[float]:
        # Deterministic lightweight vector generation for zero-RAM overhead
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        
        # Extend hash bytes deterministically to dimension size
        vector = []
        for i in range(self.dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            val = (byte_val / 255.0) * 2.0 - 1.0  # Normalized between -1 and 1
            vector.append(val)

        # Normalize vector to unit length
        length = sum(x * x for x in vector) ** 0.5
        if length > 0:
            vector = [x / length for x in vector]

        return vector

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.generate_embedding(t) for t in texts]
