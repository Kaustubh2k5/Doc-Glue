"""
Application Usecase: IngestionUseCase.
Orchestrates PDF parsing, LLM fact extraction, vector embedding generation, and repository persistence.
Adheres strictly to SOLID principles (Dependency Inversion: depends ONLY on abstract interfaces).
"""
import uuid
import logging
from typing import List
from src.domain.interfaces import (
    IDocumentParser,
    IFactExtractor,
    IEmbeddingService,
    IFactRepository
)
from src.domain.models import Fact

logger = logging.getLogger(__name__)


class IngestionUseCase:
    """Orchestrates parsing, extraction, embedding, and storage of facts from documents."""

    def __init__(
        self,
        parser: IDocumentParser,
        extractor: IFactExtractor,
        embedding_service: IEmbeddingService,
        repository: IFactRepository
    ):
        self.parser = parser
        self.extractor = extractor
        self.embedding_service = embedding_service
        self.repository = repository

    def execute(self, file_path: str, filename: str) -> List[Fact]:
        """
        Executes document ingestion for a PDF file.

        Args:
            file_path: Local filesystem path to PDF.
            filename: Original document filename.

        Returns:
            List of extracted and persisted Fact entities.
        """
        document_id = f"doc-{uuid.uuid4().hex[:10]}"
        logger.info(f"Starting ingestion for document '{filename}' (ID: {document_id}) from {file_path}")

        # Step 1: Parse PDF into page chunks (1-indexed)
        pages_content = self.parser.parse(file_path=file_path, document_id=document_id)
        logger.info(f"Successfully parsed {len(pages_content)} page(s) from '{filename}'")

        extracted_facts: List[Fact] = []

        # Step 2: Extract facts per page chunk
        for page_number, page_text in pages_content:
            page_facts = self.extractor.extract_facts(
                text_chunk=page_text,
                document_id=document_id,
                filename=filename,
                page_number=page_number
            )

            # Step 3: Generate embeddings and persist each fact
            for fact in page_facts:
                fact_text_for_embedding = f"{fact.subject} {fact.property_name} {fact.value} {fact.temporal_context or ''} {fact.scope_context or ''}"
                embedding = self.embedding_service.generate_embedding(fact_text_for_embedding)
                
                self.repository.save_fact(fact=fact, embedding=embedding)
                extracted_facts.append(fact)

        logger.info(f"Completed ingestion for '{filename}': extracted and saved {len(extracted_facts)} total fact(s)")
        return extracted_facts
