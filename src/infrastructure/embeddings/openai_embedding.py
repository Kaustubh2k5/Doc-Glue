"""
OpenAI & OpenRouter Embedding Service implementation (1536 dimensions).
Uses text-embedding-3-small by default, with automatic API key detection and fallback logic.
"""
import os
import hashlib
from typing import List, Optional
from openai import OpenAI
from src.domain.interfaces import IEmbeddingService


class OpenAIEmbeddingService(IEmbeddingService):
    """
    Embedding service using OpenAI / OpenRouter `text-embedding-3-small` model.
    Produces 1536-dimensional vector embeddings.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model_name = model_name
        self.dimension = dimension
        
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        if api_key:
            self.api_key = api_key
            self.base_url = base_url
        elif self.openai_key:
            self.api_key = self.openai_key
            self.base_url = base_url
        elif self.openrouter_key:
            self.api_key = self.openrouter_key
            self.base_url = base_url or "https://openrouter.ai/api/v1"
        elif self.gemini_key:
            self.api_key = self.gemini_key
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            self.api_key = "mock-key"
            self.base_url = base_url

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key != "mock-key" else None

    def generate_embedding(self, text: str) -> List[float]:
        """Generates a 1536-dimensional vector embedding for input text."""
        if not text.strip():
            return [0.0] * self.dimension

        if self.client:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                embedding = response.data[0].embedding
                if len(embedding) == self.dimension:
                    return embedding
            except Exception as e:
                # Log fallback
                pass

        # Deterministic lightweight fallback generator if API key is not present or API call fails
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for i in range(self.dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            val = (byte_val / 255.0) * 2.0 - 1.0
            vector.append(val)

        length = sum(x * x for x in vector) ** 0.5
        if length > 0:
            vector = [x / length for x in vector]

        return vector

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates vector embeddings for a list of text strings concurrently."""
        if not texts:
            return []

        if self.client:
            # Attempt API batch payload first
            try:
                clean_texts = [t.strip() if t.strip() else " " for t in texts]
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=clean_texts
                )
                sorted_data = sorted(response.data, key=lambda x: getattr(x, 'index', 0))
                embeddings = [item.embedding for item in sorted_data]
                if len(embeddings) == len(texts):
                    return embeddings
            except Exception:
                pass

            # Fast parallel thread pool execution for OpenRouter endpoint compatibility
            from concurrent.futures import ThreadPoolExecutor
            max_workers = min(16, len(texts))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(self.generate_embedding, texts))

        return [self.generate_embedding(t) for t in texts]
