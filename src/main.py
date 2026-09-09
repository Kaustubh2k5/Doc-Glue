"""
FastAPI Backend Application for Doc-Glue Fact Knowledge Layer.
Provides REST API endpoints for document ingestion, SHA-256 deduplication,
cluster-first reconciliation, data retrieval, and DB reset.
"""
import os
import shutil
import uuid
import hashlib
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.infrastructure.parsers.docling_parser import DoclingParser
from src.infrastructure.parsers.fast_pdf_parser import FastPDFParser
from src.infrastructure.extractors.llm_extractor import MultiProviderFactExtractor
from src.infrastructure.embeddings.openai_embedding import OpenAIEmbeddingService
from src.infrastructure.persistence.postgres_repository import PostgresFactRepository, PostgresReconciliationRepository
from src.infrastructure.persistence.in_memory_repository import InMemoryFactRepository
from src.infrastructure.evaluators.openai_reconciler import OpenAIReconciliationEvaluator
from src.infrastructure.metrics import PipelineMetricsCollector
from src.application.ingestion_usecase import IngestionUseCase
from src.application.reconciliation_usecase import ReconciliationUseCase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docglue-api")

app = FastAPI(
    title="Doc-Glue Fact Knowledge Layer API",
    description="Cluster-First REST API for PDF ingestion, grounded fact extraction, and cross-document reconciliation.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class DependencyContainer:
    def __init__(self):
        self.parser = FastPDFParser()
        logger.info("Initialized standard high-speed PDF parser: FastPDFParser (PyMuPDF)")

        self.extractor = MultiProviderFactExtractor()
        self.embedding_service = OpenAIEmbeddingService()
        self.evaluator = OpenAIReconciliationEvaluator()

        try:
            self.fact_repo = PostgresFactRepository()
            self.reconciliation_repo = PostgresReconciliationRepository()
            logger.info("Connected to PostgreSQL database with pgvector.")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed ({e}). Falling back to InMemory repositories...")
            self.fact_repo = InMemoryFactRepository()
            class InMemoryReconciliationRepo:
                def __init__(self): self._records = []
                def save_comparison(self, comp): self._records.append(comp)
                def get_all_comparisons(self): return self._records
                def clear_all_data(self): self._records.clear()
            self.reconciliation_repo = InMemoryReconciliationRepo()

        self.ingestion_usecase = IngestionUseCase(
            parser=self.parser,
            extractor=self.extractor,
            embedding_service=self.embedding_service,
            repository=self.fact_repo
        )

        self.reconciliation_usecase = ReconciliationUseCase(
            fact_repository=self.fact_repo,
            reconciliation_repository=self.reconciliation_repo,
            evaluator=self.evaluator,
            embedding_service=self.embedding_service
        )

container = DependencyContainer()


def run_cluster_reconciliation_task(extracted_facts: List[Any]):
    """Background task running Cluster-First reconciliation."""
    logger.info(f"Starting Cluster-First reconciliation task for {len(extracted_facts)} new fact(s)...")
    try:
        container.reconciliation_usecase.process_new_facts(new_facts=extracted_facts)
    except Exception as e:
        logger.error(f"Cluster reconciliation error: {e}")
    logger.info("Cluster reconciliation task completed.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Doc-Glue API",
        "active_model": getattr(container.evaluator, "active_model", "OpenRouter/Gemini")
    }


@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Uploads a PDF file.
    Computes SHA-256 file hash to check if already ingested.
    If cached, returns existing data instantly without re-processing.
    Otherwise parses text/tables, extracts atomic facts, and triggers Cluster-First reconciliation.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check for cached document in repository
    if hasattr(container.fact_repo, "find_document_by_hash"):
        existing_doc = container.fact_repo.find_document_by_hash(file_hash)
        if existing_doc:
            logger.info(f"Document '{file.filename}' already ingested (Hash: {file_hash[:8]}). Returning cached records.")
            return {
                "document_id": str(existing_doc["id"]),
                "filename": file.filename,
                "cached": True,
                "status": "cached_document_retrieved"
            }

    file_id = f"doc-{uuid.uuid4().hex[:10]}"
    saved_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    try:
        with open(saved_path, "wb") as buffer:
            buffer.write(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        extracted_facts = container.ingestion_usecase.execute(
            file_path=saved_path,
            filename=file.filename
        )
        if hasattr(container.fact_repo, "save_document_hash"):
            container.fact_repo.save_document_hash(doc_id=file_id, filename=file.filename, file_hash=file_hash)
    except Exception as e:
        logger.error(f"Ingestion failed for file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion processing failed: {e}")

    if extracted_facts:
        background_tasks.add_task(run_cluster_reconciliation_task, extracted_facts)

    return {
        "document_id": file_id,
        "filename": file.filename,
        "cached": False,
        "facts_extracted": len(extracted_facts),
        "status": "processing_cluster_reconciliations_in_background"
    }


@app.get("/facts")
def get_facts():
    """Returns all extracted facts with document and page citations."""
    facts = container.fact_repo.get_all_facts()
    return [fact.model_dump() for fact in facts]


@app.get("/reconciliations")
def get_reconciliations():
    """Returns all pairwise reconciliations with relationship status, reasoning, and evidence snippets."""
    comparisons = container.reconciliation_repo.get_all_comparisons()
    return [comp.model_dump() for comp in comparisons]


@app.get("/metrics")
def get_metrics():
    """Returns ingestion pipeline and clustering quality metrics."""
    metrics = PipelineMetricsCollector()
    return metrics.get_summary()


@app.delete("/clear")
def clear_all_data():
    """Clears all stored documents, facts, and reconciliations for clean state startup."""
    try:
        container.fact_repo.clear_all_data()
        container.reconciliation_repo.clear_all_data()
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        return {"status": "success", "message": "Database and uploads cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
