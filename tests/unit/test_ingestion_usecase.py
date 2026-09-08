"""
Unit tests for IngestionUseCase.
Tests end-to-end ingestion pipeline with dependency inversion.
"""
import os
import pytest
from src.application.ingestion_usecase import IngestionUseCase
from src.infrastructure.parsers.fast_pdf_parser import FastPDFParser
from src.infrastructure.extractors.llm_extractor import MultiProviderFactExtractor
from src.infrastructure.embeddings.fast_embedding import FastEmbeddingService
from src.infrastructure.persistence.in_memory_repository import InMemoryFactRepository

STARTER_PDF_PATH = "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"


def test_ingestion_usecase_execution():
    if not os.path.exists(STARTER_PDF_PATH):
        pytest.skip(f"Starter PDF not found at {STARTER_PDF_PATH}")

    parser = FastPDFParser()
    extractor = MultiProviderFactExtractor()  # Uses mock extractor if no keys
    embedding_service = FastEmbeddingService()
    repository = InMemoryFactRepository()

    usecase = IngestionUseCase(
        parser=parser,
        extractor=extractor,
        embedding_service=embedding_service,
        repository=repository
    )

    facts = usecase.execute(file_path=STARTER_PDF_PATH, filename="delhivery-q4.pdf")

    assert len(facts) > 0
    saved_facts = repository.get_all_facts()
    assert len(saved_facts) == len(facts)
    assert saved_facts[0].evidence.filename == "delhivery-q4.pdf"
