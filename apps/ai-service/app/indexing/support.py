"""Verified claim-to-chunk map. Overlap is never supporting proof."""

from __future__ import annotations

import json
from pathlib import Path

_MAP_PATH = Path(__file__).with_name("support_map.json")
_ENTRIES: list[dict] = []


def _load() -> list[dict]:
    global _ENTRIES
    if _ENTRIES:
        return _ENTRIES
    body = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
    if body.get("schema") != "cas.verified-support.v1":
        raise RuntimeError("unsupported support map schema")
    _ENTRIES = list(body.get("entries") or [])
    return _ENTRIES


def classify_support(
    excerpt: str,
    statements: list[str],
    *,
    document_id: str = "",
    generation_id: str = "",
    page_start: int | None = None,
    page_end: int | None = None,
    text_sha256: str = "",
    procedure_code: str = "",
) -> str:
    del excerpt
    want = {str(item or "").strip() for item in (statements or []) if str(item or "").strip()}
    if not want or not document_id or not generation_id or not text_sha256:
        return "related"
    if page_start is None or page_end is None or page_start < 1 or page_end < page_start:
        return "related"
    digest = str(text_sha256).lower()
    code = str(procedure_code or "").strip()
    for entry in _load():
        statement = str(entry.get("statement") or "")
        if statement not in want:
            continue
        if code and str(entry.get("procedure_code") or "") != code:
            continue
        if str(entry.get("document_id") or "") != str(document_id):
            continue
        if str(entry.get("generation_id") or "") != str(generation_id):
            continue
        if int(entry.get("page_start") or 0) != int(page_start):
            continue
        if int(entry.get("page_end") or 0) != int(page_end):
            continue
        if str(entry.get("text_sha256") or "").lower() != digest:
            continue
        return "supporting"
    return "related"
