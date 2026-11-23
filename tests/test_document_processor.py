import tempfile
from pathlib import Path
from app.services.document_loader import DocumentLoaderService
from app.services.text_splitter import TextSplitterService

def test_document_loader_txt():
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp.write("Artificial Intelligence and Machine Learning are transforming modern technology.")
        tmp_path = Path(tmp.name)

    try:
        docs = DocumentLoaderService.load_document(tmp_path, "doc-1", "test.txt")
        assert len(docs) == 1
        assert "Artificial Intelligence" in docs[0].page_content
        assert docs[0].metadata["document_id"] == "doc-1"
        assert docs[0].metadata["filename"] == "test.txt"
    finally:
        tmp_path.unlink()

def test_document_loader_csv():
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp:
        tmp.write("Name,Role,Department\nAlice,Engineer,AI\nBob,Designer,Product")
        tmp_path = Path(tmp.name)

    try:
        docs = DocumentLoaderService.load_document(tmp_path, "doc-2", "team.csv")
        assert len(docs) == 1
        assert "Alice" in docs[0].page_content
        assert "Designer" in docs[0].page_content
    finally:
        tmp_path.unlink()

def test_text_splitter():
    splitter = TextSplitterService(chunk_size=50, chunk_overlap=10)
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp.write("Sentence one is here. Sentence two is also here. Sentence three continues the thought. Sentence four wraps it up nicely.")
        tmp_path = Path(tmp.name)

    try:
        docs = DocumentLoaderService.load_document(tmp_path, "doc-3", "long.txt")
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 1
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == idx
            assert len(chunk.page_content) > 0
    finally:
        tmp_path.unlink()
