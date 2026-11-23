import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain_core.documents import Document

from app.config import settings
from app.services.embeddings import get_embedding_function

class VectorStoreManager:
    """
    Persistent in-memory vector store with disk persistence.
    Provides vector indexing, cosine similarity ranking, metadata filtering, and document management.
    """
    
    def __init__(self, persistence_dir: Optional[Path] = None):
        self.persistence_dir = persistence_dir or settings.vector_store_dir
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.persistence_dir / "vector_index.json"
        self.embeddings_file = self.persistence_dir / "embeddings.npy"
        
        self.embedding_fn = get_embedding_function()
        self.documents: List[Document] = []
        self.vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.documents_metadata: Dict[str, Dict[str, Any]] = {}
        
        self.load()

    def add_documents(self, documents: List[Document]) -> int:
        """Embeds and indexes document chunks."""
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        new_embeddings = self.embedding_fn.embed_documents(texts)
        new_vectors = np.array(new_embeddings, dtype=np.float32)

        # Normalize new vectors for cosine similarity
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_vectors = new_vectors / norms

        if self.vectors.size == 0 or len(self.documents) == 0:
            self.vectors = new_vectors
        else:
            # Handle potential dimension mismatch if provider changed
            if self.vectors.shape[1] == new_vectors.shape[1]:
                self.vectors = np.vstack([self.vectors, new_vectors])
            else:
                print("[Warning] Vector dimension mismatch. Resetting index with new embeddings.")
                self.vectors = new_vectors
                self.documents = []

        self.documents.extend(documents)

        # Update document level metadata registry
        for doc in documents:
            doc_id = doc.metadata.get("document_id")
            if doc_id and doc_id not in self.documents_metadata:
                self.documents_metadata[doc_id] = {
                    "id": doc_id,
                    "filename": doc.metadata.get("filename", "unknown"),
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "file_size_bytes": doc.metadata.get("file_size_bytes", 0),
                    "uploaded_at": doc.metadata.get("uploaded_at", ""),
                    "chunk_count": 0
                }
            if doc_id in self.documents_metadata:
                self.documents_metadata[doc_id]["chunk_count"] += 1

        self.save()
        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.0,
        document_ids: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Executes cosine similarity search over indexed chunks.
        Returns a list of (Document, score) sorted in descending order of similarity.
        """
        if not self.documents or self.vectors.size == 0:
            return []

        query_vec = np.array(self.embedding_fn.embed_query(query), dtype=np.float32)
        if query_vec.shape[0] != self.vectors.shape[1]:
            from app.services.embeddings import LocalDeterministicEmbeddings
            query_vec = np.array(LocalDeterministicEmbeddings(dimension=self.vectors.shape[1]).embed_query(query), dtype=np.float32)

        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.vectors, query_vec)

        # Filter and rank
        results = []
        for idx, score in enumerate(scores):
            score_val = float(score)
            if score_val < threshold:
                continue

            doc = self.documents[idx]
            if document_ids and doc.metadata.get("document_id") not in document_ids:
                continue

            results.append((doc, score_val))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def delete_document(self, document_id: str) -> bool:
        """Removes a document and all its chunks from the vector store."""
        if document_id not in self.documents_metadata:
            return False

        keep_indices = [
            i for i, doc in enumerate(self.documents)
            if doc.metadata.get("document_id") != document_id
        ]

        self.documents = [self.documents[i] for i in keep_indices]
        if keep_indices and self.vectors.size > 0:
            self.vectors = self.vectors[keep_indices]
        else:
            self.vectors = np.empty((0, 384), dtype=np.float32)

        del self.documents_metadata[document_id]
        self.save()
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata for all registered documents."""
        return list(self.documents_metadata.values())

    def clear(self) -> None:
        """Clears all indexed documents and vector data."""
        self.documents = []
        self.vectors = np.empty((0, 384), dtype=np.float32)
        self.documents_metadata = {}
        if self.index_file.exists():
            self.index_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        dim = self.vectors.shape[1] if self.vectors.size > 0 else 0
        return {
            "total_documents": len(self.documents_metadata),
            "total_chunks": len(self.documents),
            "vector_dimension": dim,
            "llm_provider": settings.llm_provider,
            "embeddings_provider": settings.embeddings_provider,
            "active_model": settings.llm_model,
            "status": "ready"
        }

    def save(self) -> None:
        """Persists the document catalog and vector array to disk."""
        try:
            data = {
                "metadata": self.documents_metadata,
                "documents": [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in self.documents
                ]
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if self.vectors.size > 0:
                np.save(str(self.embeddings_file), self.vectors)
        except Exception as e:
            print(f"[Error] Failed to persist vector store: {e}")

    def load(self) -> None:
        """Loads index and embeddings from disk if available."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents_metadata = data.get("metadata", {})
                self.documents = [
                    Document(page_content=item["page_content"], metadata=item["metadata"])
                    for item in data.get("documents", [])
                ]
                if self.embeddings_file.exists():
                    self.vectors = np.load(str(self.embeddings_file))
                print(f"[Info] Loaded {len(self.documents)} chunks from vector store persistence.")
            except Exception as e:
                print(f"[Warning] Could not load persisted vector store: {e}")
                self.clear()
