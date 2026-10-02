"""Read a PDF from bytes already loaded from internal object storage."""

from __future__ import annotations

import io

from app.indexing.errors import IndexFailure
from app.indexing.text import normalize_text

NATIVE_TEXT_MIN_CHARS = 24


def page_bounds(page_range: str | None, page_count: int) -> tuple[int, int]:
    if page_range is None or page_range == "":
        return 1, page_count
    if "-" in page_range:
        start_s, end_s = page_range.split("-", 1)
        start, end = int(start_s), int(end_s)
    else:
        start = end = int(page_range)
    if start < 1 or end > page_count or start > end:
        raise IndexFailure("page_range_invalid")
    return start, end


def read_bounded(chunks, max_bytes: int) -> bytes:
    """Stop a byte stream as soon as it passes the cap."""
    buf = bytearray()
    for chunk in chunks:
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise IndexFailure("pdf_oversized")
    return bytes(buf)


def extract_native(data: bytes, max_bytes: int) -> list[str]:
    return [text for _number, text in read_selected_pages(data, max_bytes, None, None)]


def read_selected_pages(data: bytes, max_bytes: int, max_pages: int | None, page_range: str | None) -> list[tuple[int, str]]:
    if len(data) > max_bytes:
        raise IndexFailure("pdf_oversized")
    if not data.startswith(b"%PDF-"):
        raise IndexFailure("mime_invalid")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise IndexFailure("pdf_encrypted")
        count = len(reader.pages)
        if max_pages is not None and count > max_pages:
            raise IndexFailure("page_limit")
        start, end = page_bounds(page_range, count)
        selected = []
        for number in range(start, end + 1):
            selected.append((number, normalize_text(reader.pages[number - 1].extract_text() or "")))
        return selected
    except IndexFailure:
        raise
    except Exception as exc:
        raise IndexFailure("pdf_corrupt") from exc
