"""Extract PDF pages and make identical, traceable chunks for every model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")

    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks


def extract_and_chunk(
    pdf_dir: Path,
    *,
    max_pdfs: int = 10,
    chunk_size: int = 1000,
    overlap: int = 150,
    sparse_page_chars: int = 40,
) -> tuple[list[dict], list[str]]:
    """Read at most ten PDFs and return shared chunks plus OCR warnings."""
    pdf_paths = sorted(pdf_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Add PDF files there and try again."
        )

    safe_limit = min(max_pdfs, 10)
    warnings: list[str] = []
    if len(pdf_paths) > safe_limit:
        warnings.append(
            f"Found {len(pdf_paths)} PDFs; processing only the first {safe_limit}."
        )
    pdf_paths = pdf_paths[:safe_limit]

    chunks: list[dict] = []
    for pdf_path in pdf_paths:
        try:
            document = pymupdf.open(pdf_path)
        except Exception as error:
            raise RuntimeError(f"Could not open {pdf_path.name}: {error}") from error

        sparse_pages: list[int] = []
        try:
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                if len(page_text) < sparse_page_chars:
                    sparse_pages.append(page_number)
                if not page_text:
                    continue

                for chunk_number, chunk_text in enumerate(
                    split_text(page_text, chunk_size, overlap), start=1
                ):
                    fingerprint = hashlib.sha256(
                        f"{pdf_path.name}:{page_number}:{chunk_number}:{chunk_text}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    chunks.append(
                        {
                            "chunk_id": fingerprint,
                            "text": chunk_text,
                            "filename": pdf_path.name,
                            "page": page_number,
                            "chunk_number": chunk_number,
                        }
                    )
        finally:
            document.close()

        if sparse_pages:
            page_list = ", ".join(str(page) for page in sparse_pages)
            warnings.append(
                f"{pdf_path.name}: little or no selectable text on page(s) {page_list}; "
                "these pages may be scanned and need OCR."
            )

    if not chunks:
        raise ValueError(
            "No selectable PDF text was found. The files may be scanned; run OCR first."
        )
    return chunks, warnings
