"""
FastAPI Backend Application for Doc-Glue Fact Knowledge Layer.
Provides REST API endpoints for document ingestion, fact retrieval, and pairwise reconciliations.
"""
import os
import shutil
import uuid
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.infrastructure.parsers.fast_pdf_parser import FastPDFParser
from src.infrastructure.extractors.llm_extractor import MultiProviderFactExtractor
from src.infrastructure.embeddings.openai_embedding import OpenAIEmbeddingService
from src.infrastructure.persistence.postgres_repository import PostgresFactRepository, PostgresReconciliationRepository
from src.infrastructure.persistence.in_memory_repository import InMemoryFactRepository
from src.infrastructure.evaluators.openai_reconciler import OpenAIReconciliationEvaluator
from src.application.ingestion_usecase import IngestionUseCase
from src.application.reconciliation_usecase import ReconciliationUseCase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docglue-api")

app = FastAPI(
    title="Doc-Glue Fact Knowledge Layer API",
    description="REST API for parsing PDFs, extracting grounded facts, and performing cross-document reconciliation.",
    version="1.0.0"
)

# Enable CORS for Streamlit / Web UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dependency Wireup Container
class DependencyContainer:
    def __init__(self):
        self.parser = FastPDFParser()
        self.extractor = MultiProviderFactExtractor()
        self.embedding_service = OpenAIEmbeddingService()
        self.evaluator = OpenAIReconciliationEvaluator()

        # Wire up repositories (Postgres if available, InMemory fallback)
        try:
            self.fact_repo = PostgresFactRepository()
            self.reconciliation_repo = PostgresReconciliationRepository()
            logger.info("Connected to PostgreSQL database with pgvector.")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed ({e}). Falling back to InMemory repositories...")
            self.fact_repo = InMemoryFactRepository()
            # Dual-class fallback wrapper for reconciliation repo
            class InMemoryReconciliationRepo:
                def __init__(self):
                    self._records = []
                def save_comparison(self, comp):
                    self._records.append(comp)
                def get_all_comparisons(self):
                    return self._records
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


def run_reconciliation_task(extracted_facts: List[Any]):
    """Background task function to process newly ingested facts through reconciliation engine."""
    logger.info(f"Starting background reconciliation task for {len(extracted_facts)} fact(s)...")
    for fact in extracted_facts:
        try:
            container.reconciliation_usecase.process_new_fact(new_fact=fact, top_k=5)
        except Exception as e:
            logger.error(f"Reconciliation error for fact {fact.fact_id}: {e}")
    logger.info("Background reconciliation task completed.")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Doc-Glue API"}


@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Uploads a PDF file, parses text/tables, extracts atomic facts,
    and asynchronously triggers pairwise cross-document reconciliation.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_id = f"doc-{uuid.uuid4().hex[:10]}"
    saved_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        extracted_facts = container.ingestion_usecase.execute(
            file_path=saved_path,
            filename=file.filename
        )
    except Exception as e:
        logger.error(f"Ingestion failed for file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion processing failed: {e}")

    # Trigger async reconciliation in background
    if extracted_facts:
        background_tasks.add_task(run_reconciliation_task, extracted_facts)

    return {
        "document_id": file_id,
        "filename": file.filename,
        "facts_extracted": len(extracted_facts),
        "status": "processing_reconciliations_in_background"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
