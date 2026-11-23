import os
import csv
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader
from docx import Document as DocxDocument

class DocumentLoaderService:
    """Service to load and extract text from various document formats."""
    
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".csv"}
    
    @classmethod
    def is_supported(cls, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def load_document(cls, file_path: Path, document_id: str, original_filename: str) -> List[Document]:
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            return cls._load_pdf(file_path, document_id, original_filename)
        elif ext in {".txt", ".md", ".markdown"}:
            return cls._load_text(file_path, document_id, original_filename)
        elif ext == ".docx":
            return cls._load_docx(file_path, document_id, original_filename)
        elif ext == ".csv":
            return cls._load_csv(file_path, document_id, original_filename)
        else:
            # Fallback to plain text read
            return cls._load_text(file_path, document_id, original_filename)

    @classmethod
    def _load_pdf(cls, file_path: Path, document_id: str, original_filename: str) -> List[Document]:
        documents = []
        reader = PdfReader(str(file_path))
        
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                metadata = {
                    "document_id": document_id,
                    "filename": original_filename,
                    "page": idx + 1,
                    "file_type": "pdf",
                    "total_pages": len(reader.pages)
                }
                documents.append(Document(page_content=text, metadata=metadata))
        
        # If no extractable text found (e.g. empty pdf)
        if not documents:
            documents.append(Document(
                page_content="[Empty or scanned PDF with no extractable text]",
                metadata={"document_id": document_id, "filename": original_filename, "page": 1, "file_type": "pdf"}
            ))
        return documents

    @classmethod
    def _load_text(cls, file_path: Path, document_id: str, original_filename: str) -> List[Document]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        
        metadata = {
            "document_id": document_id,
            "filename": original_filename,
            "page": 1,
            "file_type": file_path.suffix.lstrip(".").lower()
        }
        return [Document(page_content=content.strip(), metadata=metadata)]

    @classmethod
    def _load_docx(cls, file_path: Path, document_id: str, original_filename: str) -> List[Document]:
        doc = DocxDocument(str(file_path))
        full_text = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
                
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    full_text.append(row_text)
                    
        content = "\n\n".join(full_text)
        metadata = {
            "document_id": document_id,
            "filename": original_filename,
            "page": 1,
            "file_type": "docx"
        }
        return [Document(page_content=content, metadata=metadata)]

    @classmethod
    def _load_csv(cls, file_path: Path, document_id: str, original_filename: str) -> List[Document]:
        rows = []
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    rows.append("Headers: " + ", ".join(header))
                for row in reader:
                    if row:
                        rows.append(", ".join(row))
        except UnicodeDecodeError:
            with open(file_path, mode="r", encoding="latin-1") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        rows.append(", ".join(row))
                        
        content = "\n".join(rows)
        metadata = {
            "document_id": document_id,
            "filename": original_filename,
            "page": 1,
            "file_type": "csv"
        }
        return [Document(page_content=content, metadata=metadata)]
