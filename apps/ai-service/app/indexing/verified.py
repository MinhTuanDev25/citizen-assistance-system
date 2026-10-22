"""Allowlisted verified-page ingest. Checksums in this module are the source of truth."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from dataclasses import dataclass

from app.indexing.errors import IndexFailure
from app.indexing.pdf import page_bounds

VERIFIED_SCHEMA = "cas.verified-pages.v1"
VERIFIED_EXTRACTION_VERSION = "verified-proof-1"
PAGE_MARKER = "----- PAGE"


@dataclass(frozen=True)
class VerifiedSpec:
    xa_id: str
    document_id: uuid.UUID
    pdf_sha256: str
    page_range: str
    locked_sha256: str
    page_sha256: dict[int, str]


# QĐ 163 / 1.001193 pages 10–21. Independent of sidecar JSON sha fields.
BTP_163_ORDINARY_BIRTH: VerifiedSpec = VerifiedSpec(
    xa_id="xa_chu_se",
    document_id=uuid.UUID("60b677df-acc0-4198-a3b6-a919c01482a1"),
    pdf_sha256="b47664493fa52445af0b33d0ba17cdb6f68cbba17742b400a664e9a56bf56e54",
    page_range="10-21",
    locked_sha256="b71ac8133ec6d67e9df1ac00ea14488f5eade64e2ccb19f57c1c0377b703fb20",
    page_sha256={
        10: "5b6d6763ffeaf19cecf805ab2652ccf2a515428afb354725d866636f1dc35e24",
        11: "545dfc644e6e19f585756ffc4cbcac242c98a085f52197c52b22705112a04c72",
        12: "c5482bc73bc07f52fa8546634b21b5751ddde4389993f51b192c753029931c16",
        13: "ddbf5a8ac382a312dd3b0f45c0932ee26cc99eac69494b471a6f7d0bad25fdbf",
        14: "852ebcd51d60f94728391b7db5f176549427d73e17467c2bdba1a80984b43ef6",
        15: "0fa70c03815287067821ffd84b902f566ce9facdde08180b01bd4d1f8dfa04fa",
        16: "0a23728f74f5ff2847670e4b092111ef66c72bfa555e88555900d73b315e0846",
        17: "79b7bb3c66b8d1d511af757d14c7b1aa39861f17ed76db2182b81b9042f0bb2e",
        18: "9e77031bd43131c3b25af928dd49b17f9cb736e0852b84d79e10850f2611875e",
        19: "42d4e57a19a2b55730a44e2590e6f45633eca1ba121c4329c92affd001f0e6b4",
        20: "3e3b5f96f2d1004c2ebb404109f1b2f91ffcc3803fd84d073825fc1d66be784e",
        21: "3d71c64086aaf48ca516a2dd25f38589319ce53c82c55c61cf16428a84a39ba9",
    },
)

APPROVED_BUNDLES: tuple[VerifiedSpec, ...] = (BTP_163_ORDINARY_BIRTH,)


def sidecar_object_key(pdf_key: str) -> str:
    if not pdf_key.endswith(".pdf"):
        raise IndexFailure("verified_mismatch")
    return pdf_key[: -len(".pdf")] + ".verified.json"


def spec_for(document_id: uuid.UUID, pdf_sha256: str, approved: tuple[VerifiedSpec, ...] | None = None) -> VerifiedSpec | None:
    digest = (pdf_sha256 or "").lower()
    for spec in approved if approved is not None else APPROVED_BUNDLES:
        if spec.document_id == document_id and spec.pdf_sha256 == digest:
            return spec
    return None


def sha256_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pdf_page_count(data: bytes, max_bytes: int) -> int:
    if len(data) > max_bytes:
        raise IndexFailure("pdf_oversized")
    if not data.startswith(b"%PDF-"):
        raise IndexFailure("mime_invalid")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise IndexFailure("pdf_encrypted")
        return len(reader.pages)
    except IndexFailure:
        raise
    except Exception as exc:
        raise IndexFailure("pdf_corrupt") from exc


def encode_sidecar(spec: VerifiedSpec, pages: dict[int, str]) -> bytes:
    """Build sidecar JSON. Page SHA fields are omitted; the allowlist is authoritative."""
    payload = {
        "schema": VERIFIED_SCHEMA,
        "document_id": str(spec.document_id),
        "pdf_sha256": spec.pdf_sha256,
        "page_range": spec.page_range,
        "locked_sha256": spec.locked_sha256,
        "pages": {str(number): pages[number] for number in sorted(pages)},
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_verified_pages(raw: bytes, spec: VerifiedSpec, job_page_range: str | None) -> list[tuple[int, str]]:
    if job_page_range != spec.page_range:
        raise IndexFailure("verified_mismatch")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexFailure("verified_mismatch") from exc
    if not isinstance(body, dict) or body.get("schema") != VERIFIED_SCHEMA:
        raise IndexFailure("verified_mismatch")
    if str(body.get("document_id")) != str(spec.document_id):
        raise IndexFailure("verified_mismatch")
    if str(body.get("pdf_sha256") or "").lower() != spec.pdf_sha256:
        raise IndexFailure("verified_mismatch")
    if body.get("page_range") != spec.page_range:
        raise IndexFailure("verified_mismatch")
    if body.get("locked_sha256") != spec.locked_sha256:
        raise IndexFailure("verified_mismatch")
    pages_obj = body.get("pages")
    if not isinstance(pages_obj, dict):
        raise IndexFailure("verified_mismatch")
    expected_keys = {str(number) for number in spec.page_sha256}
    if set(pages_obj.keys()) != expected_keys:
        raise IndexFailure("verified_mismatch")
    out: list[tuple[int, str]] = []
    for number in sorted(spec.page_sha256):
        text = _page_text(pages_obj[str(number)])
        if PAGE_MARKER in text:
            raise IndexFailure("verified_mismatch")
        if sha256_utf8(text) != spec.page_sha256[number]:
            raise IndexFailure("verified_mismatch")
        out.append((number, text))
    return out


def assert_pdf_covers_spec(data: bytes, spec: VerifiedSpec, job_page_range: str | None, max_bytes: int, max_pages: int) -> None:
    if job_page_range != spec.page_range:
        raise IndexFailure("verified_mismatch")
    count = pdf_page_count(data, max_bytes)
    start, end = page_bounds(job_page_range, count)
    if f"{start}-{end}" != spec.page_range:
        raise IndexFailure("verified_mismatch")
    if (end - start + 1) > max_pages:
        raise IndexFailure("page_limit")


def _page_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"]
    raise IndexFailure("verified_mismatch")
