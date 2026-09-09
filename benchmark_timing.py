"""
Standalone benchmark: measures end-to-end ingestion time on a real uploaded PDF.
Phase 1 = PyMuPDF parsing only (no API calls needed)
Phase 2 = Full pipeline (parse + extract + embed + store)
"""
import time
import tracemalloc
import sys
import os
import uuid

sys.path.insert(0, '/app')

from src.infrastructure.parsers.fast_pdf_parser import FastPDFParser
from src.infrastructure.extractors.llm_extractor import MultiProviderFactExtractor
from src.infrastructure.embeddings.fast_embedding import FastEmbeddingService
from src.infrastructure.persistence.in_memory_repository import InMemoryFactRepository
from src.application.ingestion_usecase import IngestionUseCase

PDF_PATH = "/app/uploads/doc-44813c7823_delhivery_q4.pdf"

def fmt(s): return f"{s:.3f}s"

def main():
    print("=" * 62)
    print("  Doc-Glue   Ingestion Benchmark  (PyMuPDF + OpenRouter)")
    print("=" * 62)
    file_size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"  File : {os.path.basename(PDF_PATH)}")
    print(f"  Size : {file_size_kb:.0f} KB  ({file_size_kb/1024:.2f} MB)")
    print("=" * 62)

    # ─── Phase 1: Parse only ────────────────────────────────────────
    parser = FastPDFParser()
    t0 = time.perf_counter()
    pages = parser.parse(file_path=PDF_PATH, document_id="bench-parse")
    parse_time = time.perf_counter() - t0

    total_chars = sum(len(p[1]) for p in pages)
    print(f"\n  Phase 1 — PDF Parsing (PyMuPDF, zero network)")
    print(f"    Pages extracted : {len(pages)}")
    print(f"    Total chars     : {total_chars:,}")
    print(f"    Parse time      : {fmt(parse_time)}")
    print(f"    Throughput      : {file_size_kb / parse_time:,.0f} KB/s")

    # ─── Phase 2: Full pipeline ─────────────────────────────────────
    extractor = MultiProviderFactExtractor()
    embedding_service = FastEmbeddingService()
    repository = InMemoryFactRepository()
    usecase = IngestionUseCase(
        parser=parser, extractor=extractor,
        embedding_service=embedding_service, repository=repository
    )

    tracemalloc.start()
    t1 = time.perf_counter()
    facts = usecase.execute(file_path=PDF_PATH, filename="delhivery_q4.pdf")
    full_time = time.perf_counter() - t1
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    llm_time = max(0, full_time - parse_time)

    print(f"\n  Phase 2 — Full Pipeline")
    print(f"    Facts extracted : {len(facts)}")
    print(f"    Full time       : {fmt(full_time)}")
    print(f"    Peak RAM        : {peak_mem / (1024*1024):.1f} MB")

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  SUMMARY                                │")
    pct_parse = parse_time / full_time * 100 if full_time > 0 else 0
    pct_llm   = llm_time   / full_time * 100 if full_time > 0 else 0
    print(f"  │  Parse (PyMuPDF)   {fmt(parse_time):>10}  {pct_parse:>4.0f}%  │")
    print(f"  │  LLM Extraction    {fmt(llm_time):>10}  {pct_llm:>4.0f}%  │")
    print(f"  │  TOTAL             {fmt(full_time):>10}         │")
    print(f"  │  Facts/sec         {len(facts)/full_time if full_time>0 else 0:>10.1f}         │")
    print(f"  └─────────────────────────────────────────┘")

    if facts:
        f = facts[0]
        print(f"\n  Sample Fact:")
        print(f"    {f.subject} | {f.property_name} = {f.value} {f.unit or ''}")
        print(f"    Page {f.evidence.page_number}: {f.evidence.verbatim_text[:80]}...")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
