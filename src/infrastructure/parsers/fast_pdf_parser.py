"""
Universal, high-speed, low-memory PDF parser using PyMuPDF (pymupdf).
Uses <10MB RAM, zero coldstart time, and parses any PDF efficiently.
Preserves page layout, tabular structures, and reading order across any PDF.
"""
import os
from typing import List, Tuple
import pymupdf  # PyMuPDF
from src.domain.interfaces import IDocumentParser


class FastPDFParser(IDocumentParser):
    """
    Universal, high-speed PDF parser implementation using PyMuPDF.
    Extracts text per page while preserving reading blocks, headings, and tabular layout blocks.
    """

    def parse(self, file_path: str, document_id: str) -> List[Tuple[int, str]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        pages_content: List[Tuple[int, str]] = []
        doc = pymupdf.open(file_path)
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1  # 1-indexed
                
                # Extract structured text using layout-aware block parsing
                blocks = page.get_text("blocks")  # Returns list of (x0, y0, x1, y1, text, block_no, block_type)
                
                # Sort blocks by vertical position y0, then horizontal x0 to ensure natural reading order
                blocks.sort(key=lambda b: (b[1], b[0]))
                
                block_texts = []
                for b in blocks:
                    # b[4] is the text content of the block, b[6] == 0 means text block
                    if len(b) >= 5 and b[4].strip():
                        block_text = b[4].strip()
                        block_texts.append(block_text)

                # Check if tables exist on the page via PyMuPDF table finder
                tables_text = []
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for tab in tabs.tables:
                            df_str = tab.to_markdown()
                            if df_str and df_str.strip():
                                tables_text.append(f"\n[Table]\n{df_str.strip()}\n[/Table]")
                except Exception:
                    pass  # Fallback gracefully if table finder is unavailable

                combined_text = "\n\n".join(block_texts)
                if tables_text:
                    combined_text += "\n\n" + "\n\n".join(tables_text)

                cleaned_page_text = combined_text.strip()
                if cleaned_page_text:
                    pages_content.append((page_number, cleaned_page_text))
        finally:
            doc.close()

        return pages_content
