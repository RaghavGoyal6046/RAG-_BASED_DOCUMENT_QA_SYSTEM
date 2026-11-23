import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "RAG-Based Document Q&A System"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # LLM Settings (Groq, Gemini, or Offline)
    llm_provider: Literal["groq", "gemini", "offline"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    
    # Embedding Settings (Gemini or Local)
    embeddings_provider: Literal["gemini", "local"] = "gemini"
    embeddings_model: str = "gemini-embedding-001"
    
    # API Keys
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    
    # RAG Ingestion & Retrieval Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 4
    similarity_threshold: float = 0.25
    
    # File Paths
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    vector_store_dir: Path = Path("./data/vector_store")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
