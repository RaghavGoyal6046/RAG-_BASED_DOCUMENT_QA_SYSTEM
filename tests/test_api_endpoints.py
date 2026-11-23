import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_system_stats():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data

def test_document_lifecycle_and_query():
    # 1. Clear existing
    client.delete("/api/v1/clear")

    # 2. Upload file
    file_content = b"Python is an interpreted, high-level, general-purpose programming language created by Guido van Rossum."
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("python_history.txt", io.BytesIO(file_content), "text/plain")}
    )
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["document"]["id"]
    assert upload_data["document"]["chunk_count"] >= 1

    # 3. List documents
    list_res = client.get("/api/v1/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["id"] == doc_id for d in docs)

    # 4. Query RAG
    query_res = client.post(
        "/api/v1/query",
        json={"question": "Who created Python?"}
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["has_grounded_context"] is True
    assert "Guido van Rossum" in q_data["answer"]
    assert len(q_data["sources"]) > 0

    # 5. Delete document
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
