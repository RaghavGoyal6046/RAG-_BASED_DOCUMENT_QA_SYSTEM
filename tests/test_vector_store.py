import tempfile
import shutil
from pathlib import Path
from langchain_core.documents import Document
from app.services.vector_store import VectorStoreManager

def test_vector_store_add_search_delete():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        vstore = VectorStoreManager(persistence_dir=temp_dir)
        docs = [
            Document(
                page_content="Retrieval-Augmented Generation blends vector search with LLM reasoning.",
                metadata={"document_id": "doc-rag", "filename": "rag.txt", "chunk_index": 0}
            ),
            Document(
                page_content="Cooking pasta requires boiling water and adding salt.",
                metadata={"document_id": "doc-cooking", "filename": "pasta.txt", "chunk_index": 0}
            )
        ]

        count = vstore.add_documents(docs)
        assert count == 2
        assert len(vstore.documents) == 2

        # Search for RAG
        results = vstore.similarity_search_with_score("What is Retrieval-Augmented Generation?", k=1)
        assert len(results) == 1
        top_doc, score = results[0]
        assert top_doc.metadata["document_id"] == "doc-rag"
        assert score > 0.0

        # Delete document
        deleted = vstore.delete_document("doc-cooking")
        assert deleted is True
        assert len(vstore.documents) == 1
        assert len(vstore.list_documents()) == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
