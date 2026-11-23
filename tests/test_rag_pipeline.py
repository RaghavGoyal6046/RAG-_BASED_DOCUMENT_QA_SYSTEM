import tempfile
import shutil
from pathlib import Path
from langchain_core.documents import Document
from app.models.schemas import QueryRequest
from app.services.vector_store import VectorStoreManager
from app.services.rag_pipeline import RAGPipelineService

def test_rag_pipeline_grounded_answer():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        vstore = VectorStoreManager(persistence_dir=temp_dir)
        docs = [
            Document(
                page_content="The speed of light in vacuum is approximately 299,792 kilometers per second.",
                metadata={"document_id": "physics-1", "filename": "physics.txt", "page": 1, "chunk_index": 0}
            )
        ]
        vstore.add_documents(docs)
        pipeline = RAGPipelineService(vector_store=vstore)

        request = QueryRequest(question="What is the speed of light in vacuum?", top_k=2)
        response = pipeline.answer_query(request)

        assert response.has_grounded_context is True
        assert len(response.sources) > 0
        assert "299,792" in response.answer
        assert response.sources[0].document_name == "physics.txt"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_rag_pipeline_hallucination_guard():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        vstore = VectorStoreManager(persistence_dir=temp_dir)
        pipeline = RAGPipelineService(vector_store=vstore)

        # Empty vector store or irrelevant threshold
        request = QueryRequest(question="What is the capital of Mars?", similarity_threshold=0.9)
        response = pipeline.answer_query(request)

        assert response.has_grounded_context is False
        assert "could not find information" in response.answer.lower()
        assert len(response.sources) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
