"""
Docling PDF parser implementation (Optional / Heavyweight).
Requires docling library and machine learning weights.
"""
import os
from typing import List, Tuple
from src.domain.interfaces import IDocumentParser


class DoclingParser(IDocumentParser):
    """
    Docling parser implementation for deep layout and table extraction.
    Note: Requires heavy ML weights (~1.5GB+ RAM).
    """

    def parse(self, file_path: str, document_id: str) -> List[Tuple[int, str]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError(
                "docling package is not installed. Use FastPDFParser for high-speed, lightweight parsing "
                "or install docling with `pip install docling`."
            )

        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        pages_content: List[Tuple[int, str]] = []
        for page_no, page in doc.pages.items():
            markdown_text = doc.export_to_markdown(page_no=page_no)
            if markdown_text.strip():
                pages_content.append((int(page_no), markdown_text.strip()))

        return pages_content
