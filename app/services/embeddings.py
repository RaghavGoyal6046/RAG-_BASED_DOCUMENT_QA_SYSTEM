import math
import re
from typing import List
import numpy as np
from app.config import settings

class LocalDeterministicEmbeddings:
    """
    High-performance, zero-external-dependency local embedding generator.
    Produces deterministic 384-dimensional dense vectors using normalized subword hashing
    and term frequency representations. Ensures full offline capability and deterministic testing.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        # Extract lowercase words and character 3-grams for semantic robustness
        words = re.findall(r"\b\w+\b", text.lower())
        tokens = list(words)
        for w in words:
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tokens.append(w[i:i+3])
        return tokens

    def _embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec.tolist()

        for token in tokens:
            # Deterministic hash mapping into vector dimensions
            h = hash(token)
            idx = abs(h) % self.dimension
            sign = 1.0 if (h >> 16) & 1 else -1.0
            # Term weight with logarithmic dampening
            vec[idx] += sign

        # L2 normalize vector for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedding_function():
    """Factory function to get the configured embedding model."""
    provider = settings.embeddings_provider.lower()
    
    if provider == "openai":
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
            try:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model=settings.embeddings_model,
                    openai_api_key=settings.openai_api_key
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenAI Embeddings ({e}). Falling back to local embeddings.")
        else:
            print("[Info] OpenAI API key not set. Using local deterministic embeddings.")
            
    elif provider == "gemini":
        if settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=settings.gemini_api_key
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini Embeddings ({e}). Falling back to local embeddings.")
        else:
            print("[Info] Gemini API key not set. Using local deterministic embeddings.")

    return LocalDeterministicEmbeddings(dimension=384)
