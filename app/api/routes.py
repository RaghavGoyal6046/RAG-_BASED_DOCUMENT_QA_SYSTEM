import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status

from app.config import settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentInfo,
    DocumentUploadResponse,
    SystemStatsResponse,
    HealthResponse
)
from app.services.document_loader import DocumentLoaderService
from app.services.text_splitter import TextSplitterService
from app.services.vector_store import VectorStoreManager
from app.services.rag_pipeline import RAGPipelineService

router = APIRouter()

# Shared singleton services
vector_store = VectorStoreManager()
splitter = TextSplitterService(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
rag_pipeline = RAGPipelineService(vector_store=vector_store)

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint to verify API readiness."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@router.get("/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_system_stats():
    """Retrieves real-time statistics of indexed documents and configuration."""
    stats = vector_store.get_stats()
    return SystemStatsResponse(**stats)

@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests and indexes a document:
    1. Validates document format (.pdf, .txt, .md, .docx, .csv)
    2. Saves file to persistent storage
    3. Extracts text and metadata
    4. Partitions content into semantic overlapping chunks
    5. Computes vector embeddings and stores chunks in vector index
    """
    filename = file.filename or "unknown.txt"
    if not DocumentLoaderService.is_supported(filename):
        supported = ", ".join(DocumentLoaderService.SUPPORTED_EXTENSIONS)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{Path(filename).suffix}'. Supported formats: {supported}"
        )

    doc_id = str(uuid.uuid4())
    ext = Path(filename).suffix
    saved_filename = f"{doc_id}_{filename}"
    saved_path = settings.upload_dir / saved_filename

    try:
        # Save file to upload directory
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = saved_path.stat().st_size

        # Load raw document pages/sections
        raw_documents = DocumentLoaderService.load_document(saved_path, doc_id, filename)
        for doc in raw_documents:
            doc.metadata["file_size_bytes"] = file_size
            doc.metadata["uploaded_at"] = datetime.now(timezone.utc).isoformat()

        # Semantic chunking
        chunked_documents = splitter.split_documents(raw_documents)

        # Index chunks in vector store
        indexed_count = vector_store.add_documents(chunked_documents)

        doc_info = DocumentInfo(
            id=doc_id,
            filename=filename,
            file_type=ext.lstrip(".").lower(),
            file_size_bytes=file_size,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=indexed_count,
            status="indexed"
        )

        return DocumentUploadResponse(
            message=f"Document '{filename}' successfully processed and indexed into {indexed_count} chunks.",
            document=doc_info
        )

    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

@router.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """Lists all ingested documents and their chunk counts."""
    docs = vector_store.list_documents()
    return [DocumentInfo(**d) for d in docs]

@router.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    """Deletes a document and its chunks from the vector store and disk."""
    success = vector_store.delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )

    # Clean up file on disk if exists
    for f in settings.upload_dir.glob(f"{document_id}_*"):
        try:
            f.unlink()
        except Exception:
            pass

    return {"message": f"Document '{document_id}' successfully deleted."}

@router.post("/query", response_model=QueryResponse, tags=["RAG Query"])
async def query_documents(request: QueryRequest):
    """
    Query the document corpus using Retrieval-Augmented Generation (RAG):
    - Embeds question
    - Retrieves top-k most semantically relevant chunks
    - Filters low-confidence matches to mitigate hallucinations
    - Synthesizes grounded answer backed by source citations
    """
    if vector_store.vectors.size == 0 or len(vector_store.documents) == 0:
        return QueryResponse(
            question=request.question,
            answer="No documents have been indexed yet. Please upload one or more documents first.",
            sources=[],
            confidence_score=0.0,
            has_grounded_context=False,
            latency_ms=0.0,
            model_used="none"
        )

    response = rag_pipeline.answer_query(request)
    return response

@router.delete("/clear", tags=["System"])
async def clear_all_documents():
    """Clears all indexed documents, vector indices, and uploads."""
    vector_store.clear()
    for f in settings.upload_dir.glob("*"):
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass
    return {"message": "All documents and vector stores have been cleared."}
