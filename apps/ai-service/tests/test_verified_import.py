"""Verified-text ingest: allowlist is the checksum authority. No live G3."""

from __future__ import annotations

import hashlib
import json
import uuid

import pytest

from app.indexing.chunk import ChunkConfig, chunk_pages
from app.indexing.errors import IndexFailure
from app.indexing.ocr import FakeOCR
from app.indexing.pipeline import EXTRACTION_VERSION, PIPELINE_VERSION, run_pipeline
from app.indexing.verified import (
    BTP_163_ORDINARY_BIRTH,
    VERIFIED_EXTRACTION_VERSION,
    VerifiedSpec,
    encode_sidecar,
    load_verified_pages,
    sha256_utf8,
    sidecar_object_key,
)
from tests.test_p4b_pipeline import _ascii_pdf, _prepare

PAGE_ONE = "đăng ký khai sinh trang một hộ "
PAGE_TWO = "tịch lập Phiếu xin lỗi tiến hành niêm yết"


def _spec(
    data: bytes,
    pages: dict[int, str],
    page_range: str,
    document_id: uuid.UUID | None = None,
    xa_id: str = "xa_chu_se",
) -> VerifiedSpec:
    return VerifiedSpec(
        xa_id=xa_id,
        document_id=document_id or uuid.UUID("11111111-1111-1111-1111-111111111111"),
        pdf_sha256=hashlib.sha256(data).hexdigest(),
        page_range=page_range,
        locked_sha256="aa" * 32,
        page_sha256={number: sha256_utf8(text) for number, text in pages.items()},
    )


def _index(job, deps, bundles, max_pages: int = 50):
    return run_pipeline(
        job,
        deps,
        max_bytes=200_000,
        chunk_config=ChunkConfig(32, 40, 4),
        bucket="cas-documents",
        native_min_chars=24,
        max_pages=max_pages,
        approved_bundles=bundles,
    )


def _ready(pages: dict[int, str] | None = None):
    texts = pages or {1: PAGE_ONE, 2: PAGE_TWO}
    data = _ascii_pdf(["ascii filler page one xx", "ascii filler page two xx"])
    spec = _spec(data, texts, "1-2")
    ocr = FakeOCR("OCR KHÔNG ĐƯỢC GỌI đăng ký", vietnamese_capable=True)
    job, deps = _prepare(data, ocr=ocr, page_range="1-2")
    return data, spec, job, deps, ocr, texts


def test_t09_allowlisted_missing_sidecar_does_not_fallback_ocr():
    _, spec, job, deps, ocr, _ = _ready()
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps, (spec,))
    assert raised.value.code == "verified_required"
    assert ocr.calls == []
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_t10_sidecar_text_and_inner_sha_tamper_rejected():
    _, spec, job, deps, ocr, texts = _ready()
    tampered = dict(texts)
    tampered[1] = "nội dung giả mạo " + texts[1]
    payload = json.loads(encode_sidecar(spec, texts))
    payload["pages"]["1"] = {"text": tampered[1], "sha256": sha256_utf8(tampered[1])}
    deps.objects.put(sidecar_object_key(job.object_key), json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps, (spec,))
    assert raised.value.code == "verified_mismatch"
    assert ocr.calls == []
    assert deps.vectors.points == []


def test_verified_success_skips_ocr_and_marks_source():
    _, spec, job, deps, ocr, texts = _ready()
    deps.objects.put(sidecar_object_key(job.object_key), encode_sidecar(spec, texts))
    result = _index(job, deps, (spec,))
    assert ocr.calls == []
    assert result.extraction_version == VERIFIED_EXTRACTION_VERSION
    assert result.extraction_version != EXTRACTION_VERSION
    assert result.ocr_version == f"transcription-proof-1:{spec.locked_sha256}"
    assert result.native_pages == 0
    assert result.ocr_pages == 0
    assert result.pages_processed == 2
    assert all(item.source == "verified" for item in deps.chunks.saved)
    blob = " ".join(item.text for item in deps.chunks.saved)
    assert "đăng ký khai sinh" in blob
    assert "----- PAGE" not in blob
    assert min(item.page_start for item in deps.chunks.saved) == 1
    assert max(item.page_end for item in deps.chunks.saved) == 2


def test_wrong_identity_or_range_or_marker_rejected():
    data, spec, job, deps, ocr, texts = _ready()
    key = sidecar_object_key(job.object_key)

    other = json.loads(encode_sidecar(spec, texts))
    other["document_id"] = str(uuid.uuid4())
    deps.objects.put(key, json.dumps(other).encode("utf-8"))
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps, (spec,))
    assert raised.value.code == "verified_mismatch"

    deps.objects.put(key, encode_sidecar(spec, texts))
    job.page_range = "1-1"
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps, (spec,))
    assert raised.value.code == "verified_mismatch"

    marked = dict(texts)
    marked[2] = "----- PAGE 2 -----\n" + texts[2]
    bad_spec = _spec(data, marked, "1-2")
    job.page_range = "1-2"
    job.checksum = hashlib.sha256(data).hexdigest()
    deps.objects.put(key, encode_sidecar(bad_spec, marked))
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps, (bad_spec,))
    assert raised.value.code == "verified_mismatch"
    assert ocr.calls == []


def test_other_pdf_ignores_sidecar_and_keeps_native():
    from tests.test_p4b_pipeline import FIXTURES, _write_fixtures

    _write_fixtures()
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR("khong duoc goi", vietnamese_capable=True)
    job, deps = _prepare(data, ocr=ocr)
    fake_pages = {1: "sidecar không được dùng"}
    spec = _spec(data, fake_pages, "1-1", document_id=uuid.UUID("99999999-9999-9999-9999-999999999999"))
    deps.objects.put(sidecar_object_key(job.object_key), encode_sidecar(spec, fake_pages))
    result = run_pipeline(
        job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents",
        native_min_chars=24,
    )
    assert result.native_pages == 1
    assert result.extraction_version == EXTRACTION_VERSION
    assert deps.chunks.saved[0].text.startswith("Dang ky khai sinh")
    assert "sidecar không được dùng" not in deps.chunks.saved[0].text
    assert ocr.calls == []


def test_cross_page_chunk_keeps_both_page_numbers():
    chunks = chunk_pages([(10, "hộ ", "verified"), (11, "tịch lập", "verified")], ChunkConfig(2, 4, 0))
    spanning = [item for item in chunks if item.page_start == 10 and item.page_end == 11]
    assert spanning
    assert all(item.source == "verified" for item in chunks)
    assert all("----- PAGE" not in item.text for item in chunks)


def _range_spec(start: int, end: int) -> tuple[VerifiedSpec, dict[int, str]]:
    pages = {number: f"trang {number} hộ tịch " for number in range(start, end + 1)}
    spec = VerifiedSpec(
        xa_id="xa_chu_se",
        document_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        pdf_sha256="ab" * 32,
        page_range=f"{start}-{end}",
        locked_sha256="cd" * 32,
        page_sha256={number: sha256_utf8(text) for number, text in pages.items()},
    )
    return spec, pages


def test_m1_noncanonical_page_keys_rejected():
    spec, pages = _range_spec(10, 21)
    raw = encode_sidecar(spec, pages)
    got = load_verified_pages(raw, spec, "10-21")
    assert [number for number, _ in got] == list(range(10, 22))

    body = json.loads(raw.decode("utf-8"))
    body["pages"]["010"] = pages[10]
    with pytest.raises(IndexFailure) as raised:
        load_verified_pages(json.dumps(body, ensure_ascii=False).encode("utf-8"), spec, "10-21")
    assert raised.value.code == "verified_mismatch"

    for alias in ("+10", " 10", "10 ", "10\n", "021"):
        aliased = json.loads(raw.decode("utf-8"))
        source = 21 if "21" in alias else 10
        aliased["pages"][alias] = pages[source]
        with pytest.raises(IndexFailure) as raised:
            load_verified_pages(json.dumps(aliased, ensure_ascii=False).encode("utf-8"), spec, "10-21")
        assert raised.value.code == "verified_mismatch", alias

    padded_only = json.loads(raw.decode("utf-8"))
    padded_only["pages"]["010"] = padded_only["pages"].pop("10")
    with pytest.raises(IndexFailure) as raised:
        load_verified_pages(json.dumps(padded_only, ensure_ascii=False).encode("utf-8"), spec, "10-21")
    assert raised.value.code == "verified_mismatch"

    _, job_spec, job, deps, ocr, texts = _ready()
    extra = json.loads(encode_sidecar(job_spec, texts))
    extra["pages"]["01"] = texts[1]
    deps.objects.put(sidecar_object_key(job.object_key), json.dumps(extra, ensure_ascii=False).encode("utf-8"))
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps, (job_spec,))
    assert raised.value.code == "verified_mismatch"
    assert ocr.calls == []
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_m2_wrong_commune_does_not_fallback_or_ingest():
    _, spec, job, deps, ocr, texts = _ready()
    assert spec.xa_id == "xa_chu_se"
    job.xa_id = "xa_other"
    job.object_key = f"{job.xa_id}/documents/{job.document_id}/{job.checksum}.pdf"
    deps.objects.put(sidecar_object_key(job.object_key), encode_sidecar(spec, texts))
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps, (spec,))
    assert raised.value.code == "verified_mismatch"
    assert ocr.calls == []
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_pipeline_version_unchanged():
    assert PIPELINE_VERSION == "p4b.1"
    assert BTP_163_ORDINARY_BIRTH.xa_id == "xa_chu_se"
