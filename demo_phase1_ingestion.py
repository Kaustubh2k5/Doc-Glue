"""
Demo Script for Phase 1 Ingestion Pipeline.
Measures parsing time, memory footprint, and fact extraction output on starter dataset PDFs.
"""
import time
import tracemalloc
from src.infrastructure.parsers.fast_pdf_parser import FastPDFParser
from src.infrastructure.extractors.llm_extractor import MultiProviderFactExtractor
from src.infrastructure.embeddings.fast_embedding import FastEmbeddingService
from src.infrastructure.persistence.in_memory_repository import InMemoryFactRepository
from src.application.ingestion_usecase import IngestionUseCase

STARTER_PDFS = [
    ("Delhivery Earnings Q4 FY24", "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"),
    ("India Economic Survey Excerpt", "starter-datasets/india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf")
]

def main():
    tracemalloc.start()
    start_time = time.time()

    parser = FastPDFParser()
    extractor = MultiProviderFactExtractor()
    embedding_service = FastEmbeddingService()
    repository = InMemoryFactRepository()

    usecase = IngestionUseCase(
        parser=parser,
        extractor=extractor,
        embedding_service=embedding_service,
        repository=repository
    )

    print("=== Phase 1 Ingestion Benchmark ===")
    total_facts = 0

    for title, path in STARTER_PDFS:
        t0 = time.time()
        facts = usecase.execute(file_path=path, filename=title)
        elapsed = time.time() - t0
        total_facts += len(facts)
        print(f"Processed '{title}' ({len(facts)} facts extracted) in {elapsed:.3f}s")

    total_elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n=== Performance Metrics ===")
    print(f"Total Processing Time: {total_elapsed:.3f} seconds")
    print(f"Peak RAM Usage: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Total Stored Facts in Repo: {len(repository.get_all_facts())}")

    print("\n=== Sample Extracted Fact ===")
    all_facts = repository.get_all_facts()
    if all_facts:
        f = all_facts[0]
        print(f"Subject: {f.subject}")
        print(f"Property: {f.property_name}")
        print(f"Value: {f.value} {f.unit or ''}")
        print(f"Context: {f.temporal_context or ''} | {f.scope_context or ''}")
        print(f"Evidence: Page {f.evidence.page_number} in '{f.evidence.filename}'")
        print(f"Verbatim Text: {f.evidence.verbatim_text[:120]}...")

if __name__ == "__main__":
    main()
