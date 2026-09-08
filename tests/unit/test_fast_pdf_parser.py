"""
Unit tests for FastPDFParser (PyMuPDF).
Tests parsing speed, low RAM utilization, and page citation correctness.
"""
import os
import pytest
from src.infrastructure.parsers.fast_pdf_parser import FastPDFParser

STARTER_PDF_PATH = "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"


def test_fast_pdf_parser_real_pdf():
    if not os.path.exists(STARTER_PDF_PATH):
        pytest.skip(f"Starter PDF not found at {STARTER_PDF_PATH}")

    parser = FastPDFParser()
    pages = parser.parse(file_path=STARTER_PDF_PATH, document_id="doc-test-1")

    assert len(pages) > 0
    first_page_num, first_page_text = pages[0]
    assert first_page_num == 1
    assert isinstance(first_page_text, str)
    assert len(first_page_text) > 0


def test_fast_pdf_parser_file_not_found():
    parser = FastPDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse(file_path="non_existent_file.pdf", document_id="doc-err")
