"""Offline P4B pipeline. No network, no model download, no real OCR engine."""

from __future__ import annotations

import hashlib
import uuid
import zlib
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from app.indexing.chunk import ChunkConfig, chunk_pages, manifest_hash
from app.indexing.embed import fake_embed, load_manifest, normalize
from app.indexing.errors import IndexFailure
from app.indexing.ocr import FakeOCR, FakeRenderer, PaddleOCR
from app.indexing.pdf import extract_native
from app.indexing.pipeline import (
    PIPELINE_VERSION,
    FakeEmbedder,
    IndexJob,
    MemoryChunks,
    MemoryObjects,
    run_pipeline,
)
from app.indexing.qdrant import MemoryVectors, point_id
from app.indexing.text import normalize_text

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE_TEXT = "Dang ky khai sinh tai uy ban nhan dan xa Chu Se."
JRAI = "Bơngai Jrai kơnâm pơlei"


def _pdf(pages: list[str]) -> bytes:
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        writer.add_page  # keep the page object referenced
        _ = page
    # pypdf blank pages have no text. Draw with a content stream instead.
    return _pdf_with_text(pages)


def _pdf_with_text(pages: list[str]) -> bytes:
    writer = PdfWriter()
    for text in pages:
        packet = PdfWriter()
        packet.add_blank_page(width=612, height=792)
        page = packet.pages[0]
        page.merge_page  # attribute exists; unused
        _ = page
        writer.add_page(packet.pages[0])
    # Text is injected by a tiny content stream so extract_text can see ASCII.
    raw = _ascii_pdf(pages)
    return raw


def _ascii_pdf(pages: list[str]) -> bytes:
    chunks: list[bytes] = []
    page_ids: list[int] = []
    next_id = 4
    for text in pages:
        content = f"BT /F1 12 Tf 72 720 Td ({_pdf_escape(text)}) Tj ET".encode("ascii", "replace")
        content_id = next_id
        page_id = next_id + 1
        next_id += 2
        chunks.append(
            f"{content_id} 0 obj\n<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream\nendobj\n"
        )
        chunks.append(
            (
                f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n"
            ).encode()
        )
        page_ids.append(page_id)
    kids = " ".join(f"{item} 0 R" for item in page_ids)
    body = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>\nendobj\n".encode(),
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        *chunks,
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for item in body:
        offsets.append(len(out))
        out.extend(item)
    xref_at = len(out)
    count = len(offsets)
    out.extend(f"xref\n0 {count}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(f"trailer<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode())
    return bytes(out)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _image_pdf(label: str) -> bytes:
    width, height = 64, 16
    row = bytearray()
    for y in range(height):
        for x in range(width):
            dark = 8 <= x <= 56 and 4 <= y <= 12 and ((x // 4) % 2 == 0)
            row.extend(b"\x10\x10\x10" if dark else b"\xff\xff\xff")
    raw = zlib.compress(bytes(row))
    content = b"q 180 0 0 48 36 40 cm /Im Do Q"
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 80] /Resources << /XObject << /Im 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        (
            f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(raw)} >>\nstream\n"
        ).encode()
        + raw
        + b"\nendstream\nendobj\n",
        f"5 0 obj\n<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream\nendobj\n",
    ]
    out = bytearray(b"%PDF-1.4\n%" + label.encode("ascii") + b"\n")
    offsets = [0]
    for item in objects:
        offsets.append(len(out))
        out.extend(item)
    xref_at = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode())
    return bytes(out)


def _mixed_pdf() -> bytes:
    writer = PdfWriter()
    writer.append(PdfReader(__import__("io").BytesIO(_ascii_pdf([NATIVE_TEXT]))))
    writer.append(PdfReader(__import__("io").BytesIO(_image_pdf("SCAN"))))
    handle = __import__("io").BytesIO()
    writer.write(handle)
    return handle.getvalue()


def _write_fixtures() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "native.pdf").write_bytes(_ascii_pdf([NATIVE_TEXT]))
    (FIXTURES / "scanned.pdf").write_bytes(_image_pdf("SCAN"))
    (FIXTURES / "mixed.pdf").write_bytes(_mixed_pdf())
    (FIXTURES / "multipage.pdf").write_bytes(_ascii_pdf([NATIVE_TEXT, NATIVE_TEXT + " Trang hai.", NATIVE_TEXT + " Trang ba."]))
    (FIXTURES / "corrupt.pdf").write_bytes(b"%PDF-1.4\nthis is not a pdf")
    (FIXTURES / "empty.pdf").write_bytes(_ascii_pdf([""]))


def _job(**overrides) -> IndexJob:
    base = dict(
        schema_version="index.v2",
        xa_id="xa_chu_se",
        document_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        procedure_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        procedure_version_id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        job_id=uuid.UUID("44444444-4444-4444-4444-444444444444"),
        claim_token=uuid.UUID("55555555-5555-5555-5555-555555555555"),
        generation_id=uuid.UUID("66666666-6666-6666-6666-666666666666"),
        bucket="cas-documents",
        object_key="",
        checksum="",
        page_range=None,
        pipeline_version=PIPELINE_VERSION,
    )
    base.update(overrides)
    data = overrides.get("_data")
    job = IndexJob(**{key: value for key, value in base.items() if key != "_data"})
    if data is not None:
        job.checksum = hashlib.sha256(data).hexdigest()
        job.object_key = f"{job.xa_id}/documents/{job.document_id}/{job.checksum}.pdf"
    return job


def _run(data: bytes, ocr: FakeOCR | None = None, vectors: MemoryVectors | None = None, page_range: str | None = None, max_pages: int = 50):
    _write_fixtures()
    job = _job(_data=data, page_range=page_range)
    deps = type("Deps", (), {})()
    deps.objects = MemoryObjects(data)
    deps.ocr = ocr or FakeOCR("Ban scan khong co chu native.")
    deps.embedder = FakeEmbedder()
    deps.vectors = vectors or MemoryVectors()
    deps.chunks = MemoryChunks()
    deps.renderer = FakeRenderer()
    result = run_pipeline(
        job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents",
        native_min_chars=24, max_pages=max_pages,
    )
    return result, deps


def test_fixtures_cover_required_shapes(tmp_path):
    _write_fixtures()
    for name in ("native.pdf", "scanned.pdf", "mixed.pdf", "multipage.pdf", "corrupt.pdf", "empty.pdf"):
        assert (FIXTURES / name).stat().st_size > 20
    encrypted = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret")
    with encrypted.open("wb") as handle:
        writer.write(handle)
    (FIXTURES / "encrypted.pdf").write_bytes(encrypted.read_bytes())
    with pytest.raises(IndexFailure) as raised:
        extract_native((FIXTURES / "encrypted.pdf").read_bytes(), 200_000)
    assert raised.value.code == "pdf_encrypted"


def test_native_page_does_not_call_ocr_and_keeps_jrai_when_normalized():
    assert normalize_text("  " + JRAI + " \r\n") == JRAI
    ocr = FakeOCR("khong duoc goi")
    result, deps = _run((FIXTURES / "native.pdf").read_bytes(), ocr)
    assert ocr.calls == []
    assert result.native_pages == 1
    assert result.ocr_pages == 0
    assert result.chunk_count == result.vector_count
    assert deps.chunks.saved[0].text.startswith("Dang ky khai sinh")


def test_scanned_page_calls_ocr_and_mixed_skips_native_page():
    ocr = FakeOCR("Noi dung OCR du dai de qua nguong trang.")
    result, _ = _run((FIXTURES / "scanned.pdf").read_bytes(), ocr)
    assert ocr.calls == [1]
    assert result.ocr_pages == 1
    ocr.calls.clear()
    mixed, _ = _run((FIXTURES / "mixed.pdf").read_bytes(), ocr)
    assert ocr.calls == [2]
    assert mixed.native_pages == 1
    assert mixed.ocr_pages == 1


def test_page_range_and_failures():
    data = (FIXTURES / "multipage.pdf").read_bytes()
    ranged, _ = _run(data, page_range="2-2")
    assert ranged.pages_processed == 1
    with pytest.raises(IndexFailure) as bad_range:
        _run(data, page_range="9-9")
    assert bad_range.value.code == "page_range_invalid"
    job = _job(_data=b"%PDF-1.4\nbroken")
    deps = type("Deps", (), {})()
    deps.objects = MemoryObjects(b"%PDF-1.4\nbroken")
    deps.ocr = FakeOCR("x")
    deps.embedder = FakeEmbedder()
    deps.vectors = MemoryVectors()
    deps.chunks = MemoryChunks()
    with pytest.raises(IndexFailure) as corrupt:
        run_pipeline(job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents")
    assert corrupt.value.code == "pdf_corrupt"
    with pytest.raises(IndexFailure) as mismatch:
        run_pipeline(_job(checksum="a" * 64, object_key="xa_chu_se/documents/11111111-1111-1111-1111-111111111111/" + "a" * 64 + ".pdf"), type("Deps", (), {"objects": MemoryObjects(data), "ocr": FakeOCR("x"), "embedder": FakeEmbedder(), "vectors": MemoryVectors(), "chunks": MemoryChunks()})(), max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents")
    assert mismatch.value.code == "checksum_mismatch"
    missing = type("Deps", (), {})()
    missing.objects = MemoryObjects(fail="missing")
    missing.ocr = FakeOCR("x")
    missing.embedder = FakeEmbedder()
    missing.vectors = MemoryVectors()
    missing.chunks = MemoryChunks()
    with pytest.raises(IndexFailure) as absent:
        run_pipeline(_job(_data=data), missing, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents")
    assert absent.value.code == "source_not_found"


def test_chunk_order_overlap_and_hard_limit_are_deterministic():
    config = ChunkConfig(4, 6, 1)
    pages = [(1, "mot hai ba bon nam sau bay tam", "native")]
    first = chunk_pages(pages, config)
    second = chunk_pages(pages, config)
    assert [item.text_sha256 for item in first] == [item.text_sha256 for item in second]
    assert manifest_hash(first) == manifest_hash(second)
    assert all(item.token_count <= 6 for item in first)
    assert first[1].text.split()[0] == first[0].text.split()[-1]
    jrai_chunks = chunk_pages([(1, JRAI, "native")], ChunkConfig(8, 12, 0))
    assert JRAI.split()[0] in jrai_chunks[0].text


def test_vectors_are_deterministic_finite_and_qdrant_failure_does_not_save_chunks():
    one = fake_embed("xin chao")
    assert one == fake_embed("xin chao")
    assert len(one) == 384
    assert abs(sum(item * item for item in one) - 1) < 1e-6
    normalize(one)
    data = (FIXTURES / "native.pdf").read_bytes()
    vectors = MemoryVectors(fail=True)
    with pytest.raises(IndexFailure) as failed:
        _run(data, vectors=vectors)
    assert failed.value.code == "qdrant_failed"
    ok, deps = _run(data)
    assert deps.vectors.points[0]["id"] == point_id(ok.generation_id, 0)
    payload = deps.vectors.points[0]["payload"]
    assert payload["xa_id"] == "xa_chu_se"
    assert payload["generation_id"] == str(ok.generation_id)
    assert "chunk_id" in payload and "page_start" in payload


def test_missing_or_corrupt_local_model_fails_closed(tmp_path):
    with pytest.raises(IndexFailure) as missing:
        load_manifest(str(tmp_path))
    assert missing.value.code == "embedding_model_missing"
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"model_id":"other","revision":"nope","dimension":384,"files":{"model.onnx":"abc"}}', encoding="utf-8")
    with pytest.raises(IndexFailure) as corrupt:
        load_manifest(str(tmp_path))
    assert corrupt.value.code == "embedding_checksum_mismatch"
    with pytest.raises(IndexFailure) as ocr_missing:
        PaddleOCR(str(tmp_path / "absent-ocr"))
    assert ocr_missing.value.code == "ocr_model_missing"


_write_fixtures()


def test_empty_pdf_has_no_extractable_text():
    ocr = FakeOCR("   ")
    with pytest.raises(IndexFailure) as raised:
        _run((FIXTURES / "empty.pdf").read_bytes(), ocr)
    assert raised.value.code == "no_extractable_text"


def test_renderer_receives_only_the_weak_page():
    ocr = FakeOCR("Noi dung OCR du dai de qua nguong trang.")
    _, deps = _run((FIXTURES / "mixed.pdf").read_bytes(), ocr)
    assert ocr.calls == [2]
    assert ocr.images[0]["page"] == 2
    assert deps.renderer.calls == [2]


def test_passage_prefix_and_tokenizer_hard_limit():
    embedder = FakeEmbedder()
    assert embedder.embed("xin chao") == fake_embed("passage: xin chao")
    assert embedder.embed("xin chao") != fake_embed("xin chao")
    embedder.max_tokens = 8
    job = _job(_data=(FIXTURES / "native.pdf").read_bytes())
    deps = type("Deps", (), {})()
    deps.objects = MemoryObjects((FIXTURES / "native.pdf").read_bytes())
    deps.ocr = FakeOCR("khong")
    deps.embedder = embedder
    deps.vectors = MemoryVectors()
    deps.chunks = MemoryChunks()
    deps.renderer = type("R", (), {"calls": [], "render": staticmethod(lambda data, page, dpi: {"page": page, "dpi": dpi})})()
    result = run_pipeline(job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents")
    assert all(embedder.count_tokens(item.text) <= 8 for item in deps.chunks.saved)
    assert result.chunk_count == len(deps.chunks.saved) > 1


def test_page_limit_stops_before_ocr():
    ocr = FakeOCR("Noi dung OCR du dai de qua nguong trang.")
    with pytest.raises(IndexFailure) as raised:
        _run((FIXTURES / "mixed.pdf").read_bytes(), ocr, max_pages=1)
    assert raised.value.code == "page_limit"
    assert ocr.calls == []
