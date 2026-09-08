"""
Integration tests for FastAPI REST API endpoints (/health, /facts, /reconciliations, /upload).
Uses FastAPI TestClient for fast, isolated API contract validation.
"""
import os
import io
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

STARTER_PDF_PATH = "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Doc-Glue" in data["service"]


def test_get_facts_endpoint():
    response = client.get("/facts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_reconciliations_endpoint():
    response = client.get("/reconciliations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_upload_invalid_file_type():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(b"Hello world"), "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_upload_valid_pdf_endpoint():
    if not os.path.exists(STARTER_PDF_PATH):
        pytest.skip(f"Starter PDF not found at {STARTER_PDF_PATH}")

    with open(STARTER_PDF_PATH, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("delhivery_q4.pdf", f, "application/pdf")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "delhivery_q4.pdf"
    assert data["facts_extracted"] > 0
