#!/usr/bin/env python3
"""
Interactive / Batch PDF Parsing Inspection CLI.
Allows users to inspect how any PDF is parsed page-by-page, verifying block reading order, markdown tables, and text standards.
"""
import sys
import os
import argparse
from src.infrastructure.parsers.fast_pdf_parser import FastPDFParser


def inspect_pdf(pdf_path: str, max_pages: int = 5, page_num: int = None):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        sys.exit(1)

    print("=" * 80)
    print(f"📄 INSPECTING PARSED PDF: {pdf_path}")
    print("=" * 80)

    parser = FastPDFParser()
    document_id = "doc-inspect"
    
    pages = parser.parse(file_path=pdf_path, document_id=document_id)
    print(f"Total Parsed Pages with Content: {len(pages)}")
    print("-" * 80)

    if page_num is not None:
        target_pages = [p for p in pages if p[0] == page_num]
        if not target_pages:
            print(f"Page {page_num} not found or contains no text.")
            return
        pages_to_show = target_pages
    else:
        pages_to_show = pages[:max_pages]

    for p_no, text in pages_to_show:
        print(f"\n--- [ PAGE {p_no} ] ---")
        print(text)
        print("-" * 80)

    if page_num is None and len(pages) > max_pages:
        print(f"\nShowing first {max_pages} of {len(pages)} pages. Use --page <N> to view specific page.")


def main():
    parser = argparse.ArgumentParser(description="Inspect how a PDF is parsed by Doc-Glue FastPDFParser.")
    parser.add_argument("pdf_path", help="Path to the PDF file to inspect")
    parser.add_argument("--max-pages", type=int, default=3, help="Max number of pages to display (default: 3)")
    parser.add_argument("--page", type=int, default=None, help="Specific page number to inspect")

    args = parser.parse_args()
    inspect_pdf(args.pdf_path, max_pages=args.max_pages, page_num=args.page)


if __name__ == "__main__":
    main()
