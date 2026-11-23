from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="The question to query against indexed documents")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of context chunks to retrieve")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum similarity score")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional filter to specific documents")

class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_index: int
    page: Optional[int] = None
    content: str
    similarity_score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    confidence_score: float
    has_grounded_context: bool
    latency_ms: float
    model_used: str

class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    uploaded_at: str
    chunk_count: int
    status: str = "indexed"

class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentInfo

class SystemStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    vector_dimension: int
    llm_provider: str
    embeddings_provider: str
    active_model: str
    status: str

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    timestamp: str
