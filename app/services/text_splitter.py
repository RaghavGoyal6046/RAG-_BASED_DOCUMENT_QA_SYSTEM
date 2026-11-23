from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitterService:
    """Service to partition loaded documents into semantic, overlapping chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            keep_separator=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits raw documents into chunked documents and enriches their metadata."""
        raw_chunks = self.splitter.split_documents(documents)
        enriched_chunks = []
        
        for idx, chunk in enumerate(raw_chunks):
            # Shallow copy metadata and add chunk index
            meta = dict(chunk.metadata)
            meta["chunk_index"] = idx
            meta["chunk_size_chars"] = len(chunk.page_content)
            enriched_chunks.append(Document(page_content=chunk.page_content, metadata=meta))
            
        return enriched_chunks
