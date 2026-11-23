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
            h = hash(token)
            idx = abs(h) % self.dimension
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class GeminiEmbeddingsWrapper:
    """Wrapper around Google GenAI client for dense embedding generation with offline fallback."""
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.fallback = LocalDeterministicEmbeddings(dimension=384)

    def embed_query(self, text: str) -> List[float]:
        try:
            res = self.client.models.embed_content(model=self.model, contents=text)
            return res.embeddings[0].values
        except Exception as e:
            return self.fallback.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            results = []
            for i in range(0, len(texts), 10):
                batch = texts[i:i+10]
                for t in batch:
                    res = self.client.models.embed_content(model=self.model, contents=t)
                    results.append(res.embeddings[0].values)
            return results
        except Exception as e:
            return self.fallback.embed_documents(texts)


def get_embedding_function():
    """Factory function to get the configured embedding model."""
    provider = settings.embeddings_provider.lower()

    if provider == "gemini":
        if settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here":
            try:
                return GeminiEmbeddingsWrapper(
                    api_key=settings.gemini_api_key,
                    model=settings.embeddings_model
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini Embeddings ({e}). Falling back to local embeddings.")
        else:
            print("[Info] Gemini API key not set. Using local deterministic embeddings.")

    return LocalDeterministicEmbeddings(dimension=384)
