"""
Unit tests for DoclingParser.
Tests missing file handling and fallback behavior.
"""
import pytest
from src.infrastructure.parsers.docling_parser import DoclingParser


def test_docling_parser_file_not_found():
    parser = DoclingParser()
    with pytest.raises(FileNotFoundError):
        parser.parse(file_path="non_existent_file.pdf", document_id="doc-err")
