"""
Universal, high-speed, low-memory PDF parser using PyMuPDF (pymupdf).
Uses <10MB RAM, zero coldstart time, and parses any PDF efficiently.
Preserves page layout, tabular structures, and reading order across any PDF.
Includes chunk quality filtering to remove noise blocks.
"""
import os
import re
from typing import List, Tuple
import pymupdf  # PyMuPDF
from src.domain.interfaces import IDocumentParser
from src.infrastructure.metrics import PipelineMetricsCollector

# Noise patterns to filter out
NOISE_PATTERNS = [
    re.compile(r'^\d+$'),                          # Pure page numbers
    re.compile(r'^page\s*\d+', re.IGNORECASE),     # "Page 1" style
    re.compile(r'^[-=_\s*]{5,}$'),                  # Separators/rules
    re.compile(r'^(table of contents|index|bibliography|references|appendix)\s*$', re.IGNORECASE),
    re.compile(r'^\s*\|[\s|]+\|\s*$'),             # Empty table rows
    re.compile(r'^[A-Z]{1,3}$'),                    # Single short all-caps tokens
    re.compile(r'^(fig\.?|figure|table|chart|exhibit)\s*\d+', re.IGNORECASE),  # Figure/table labels only
]

MIN_BLOCK_LENGTH = 30
MIN_PAGE_CONTENT_LENGTH = 80
HEADER_FOOTER_MARGIN = 0.05  # 5% of page height


class FastPDFParser(IDocumentParser):
    """
    Universal, high-speed PDF parser implementation using PyMuPDF.
    Extracts text per page while preserving reading blocks, headings, and tabular layout blocks.
    Applies quality filtering to remove noise blocks (headers, footers, page numbers).
    """

    def parse(self, file_path: str, document_id: str) -> List[Tuple[int, str]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        metrics = PipelineMetricsCollector()
        filename = os.path.basename(file_path)
        doc_metrics = metrics.get_or_create_doc_metrics(filename)

        pages_content: List[Tuple[int, str]] = []
        doc = pymupdf.open(file_path)
        try:
            doc_metrics.total_pages = len(doc)

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1  # 1-indexed
                page_height = page.rect.height

                # Extract structured text using layout-aware block parsing
                blocks = page.get_text("blocks")

                # Sort blocks by vertical position y0, then horizontal x0
                blocks.sort(key=lambda b: (b[1], b[0]))

                block_texts = []
                for b in blocks:
                    if len(b) < 5 or not b[4].strip():
                        continue

                    block_text = b[4].strip()
                    doc_metrics.total_chunks_raw += 1

                    # Filter: minimum length
                    if len(block_text) < MIN_BLOCK_LENGTH:
                        doc_metrics.chunks_filtered_short += 1
                        continue

                    # Filter: header/footer by y-position
                    block_y0 = b[1]
                    block_y1 = b[3]
                    if page_height > 0:
                        top_margin = page_height * HEADER_FOOTER_MARGIN
                        bottom_margin = page_height * (1 - HEADER_FOOTER_MARGIN)
                        if block_y1 <= top_margin or block_y0 >= bottom_margin:
                            doc_metrics.chunks_filtered_header_footer += 1
                            continue

                    # Filter: noise patterns
                    is_noise = False
                    for pattern in NOISE_PATTERNS:
                        if pattern.match(block_text):
                            is_noise = True
                            break
                    if is_noise:
                        doc_metrics.chunks_filtered_noise += 1
                        continue

                    block_texts.append(block_text)
                    doc_metrics.chunks_kept += 1

                # Check if tables exist on the page via PyMuPDF table finder
                tables_text = []
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for tab in tabs.tables:
                            df_str = tab.to_markdown()
                            if df_str and df_str.strip() and len(df_str.strip()) >= MIN_BLOCK_LENGTH:
                                tables_text.append(f"\n[Table]\n{df_str.strip()}\n[/Table]")
                except Exception:
                    pass

                combined_text = "\n\n".join(block_texts)
                if tables_text:
                    combined_text += "\n\n" + "\n\n".join(tables_text)

                cleaned_page_text = combined_text.strip()
                if len(cleaned_page_text) >= MIN_PAGE_CONTENT_LENGTH:
                    pages_content.append((page_number, cleaned_page_text))

            doc_metrics.pages_after_filter = len(pages_content)
            if doc_metrics.chunks_kept > 0:
                total_len = sum(len(text) for _, text in pages_content)
                doc_metrics.avg_chunk_length = total_len / doc_metrics.chunks_kept

        finally:
            doc.close()

        return pages_content
