"""
Demo script for Phase 2: Ingestion with PostgresFactRepository & Vector Candidate Search.
Runs full PDF parsing, LLM fact extraction, embedding generation, database storage, and vector similarity retrieval.
"""
import sys
import time
from src.infrastructure.parsers.fast_pdf_parser import FastPDFParser
from src.infrastructure.extractors.llm_extractor import MultiProviderFactExtractor
from src.infrastructure.embeddings.openai_embedding import OpenAIEmbeddingService
from src.infrastructure.persistence.postgres_repository import PostgresFactRepository, PostgresReconciliationRepository
from src.infrastructure.persistence.in_memory_repository import InMemoryFactRepository
from src.application.ingestion_usecase import IngestionUseCase

STARTER_PDF = "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"


def main():
    print("=== Phase 2 Database & Vector Search Demo ===")
    
    # 1. Initialize components
    parser = FastPDFParser()
    extractor = MultiProviderFactExtractor()
    embedding_service = OpenAIEmbeddingService()

    # Try connecting to PostgreSQL, fallback to InMemory if DB container not ready yet
    try:
        fact_repo = PostgresFactRepository()
        print(" Connected to PostgreSQL database with pgvector extension enabled.")
    except Exception as e:
        print(f" PostgreSQL container not ready ({e}). Falling back to InMemoryFactRepository...")
        fact_repo = InMemoryFactRepository()

    usecase = IngestionUseCase(
        parser=parser,
        extractor=extractor,
        embedding_service=embedding_service,
        repository=fact_repo
    )

    # 2. Ingest document
    t0 = time.time()
    facts = usecase.execute(file_path=STARTER_PDF, filename="delhivery-q4-fy24.pdf")
    elapsed = time.time() - t0

    print(f"\n Successfully ingested '{STARTER_PDF}' in {elapsed:.3f}s")
    print(f" Total Facts Saved: {len(facts)}")

    # 3. Test Vector Similarity Candidate Search
    query_text = "Delhivery revenue and express parcel shipments in FY24"
    print(f"\n Searching Vector Candidates for Query: '{query_text}'...")
    query_vector = embedding_service.generate_embedding(query_text)
    
    candidates = fact_repo.find_candidates(query_vector=query_vector, top_k=3)
    print(f" Found {len(candidates)} top matching candidate facts:\n")

    for idx, c in enumerate(candidates, 1):
        print(f"[{idx}] Fact ID: {c.fact_id}")
        print(f"    Subject: {c.subject}")
        print(f"    Property: {c.property_name}")
        print(f"    Value: {c.value} {c.unit or ''}")
        print(f"    Evidence: Page {c.evidence.page_number} in {c.evidence.filename}")
        print(f"    Verbatim: {c.evidence.verbatim_text[:100]}...\n")


if __name__ == "__main__":
    main()
