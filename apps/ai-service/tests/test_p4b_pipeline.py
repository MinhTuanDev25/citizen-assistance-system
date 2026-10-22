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
from app.indexing.ocr import FakeOCR, FakeRenderer, PaddleOCR, rec_dict_supports_vietnamese
from app.indexing.pdf import extract_native, native_text_usable, ocr_text_usable, page_looks_blank, recovered_text_usable
from app.indexing.pipeline import (
    EXTRACTION_VERSION,
    PIPELINE_VERSION,
    FakeEmbedder,
    IndexJob,
    MemoryChunks,
    MemoryObjects,
    _embed_all,
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


def _prepare(data: bytes, ocr: FakeOCR | None = None, vectors: MemoryVectors | None = None, page_range: str | None = None, renderer=None):
    _write_fixtures()
    job = _job(_data=data, page_range=page_range)
    deps = type("Deps", (), {})()
    deps.objects = MemoryObjects(data)
    deps.ocr = ocr or FakeOCR("Ban scan khong co chu native.")
    deps.embedder = FakeEmbedder()
    deps.vectors = vectors or MemoryVectors()
    deps.chunks = MemoryChunks()
    deps.renderer = renderer or FakeRenderer()
    return job, deps


def _index(job, deps, max_pages: int = 50):
    return run_pipeline(
        job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents",
        native_min_chars=24, max_pages=max_pages,
    )


def _run(data: bytes, ocr: FakeOCR | None = None, vectors: MemoryVectors | None = None, page_range: str | None = None, max_pages: int = 50, renderer=None):
    job, deps = _prepare(data, ocr=ocr, vectors=vectors, page_range=page_range, renderer=renderer)
    return _index(job, deps, max_pages=max_pages), deps


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
    ocr = FakeOCR(CLEAN_OCR, vietnamese_capable=True)
    ink = FakeRenderer(blank=False)
    result, _ = _run((FIXTURES / "scanned.pdf").read_bytes(), ocr, renderer=ink)
    assert ocr.calls == [1]
    assert result.ocr_pages == 1
    ocr.calls.clear()
    mixed, _ = _run((FIXTURES / "mixed.pdf").read_bytes(), ocr, renderer=FakeRenderer(blank=False))
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


def test_blank_stripe_page_does_not_invent_administrative_text():
    ocr = FakeOCR("")
    result, deps = _run((FIXTURES / "mixed.pdf").read_bytes(), ocr)
    assert result.ocr_pages == 0
    assert result.native_pages == 1
    assert ocr.calls == []
    assert deps.renderer.calls == [2]
    saved = " ".join(item.text for item in deps.chunks.saved)
    assert "GIAY KHAI SINH" not in saved
    assert "123" not in saved


def test_renderer_receives_only_the_weak_page():
    ocr = FakeOCR(CLEAN_OCR, vietnamese_capable=True)
    _, deps = _run((FIXTURES / "mixed.pdf").read_bytes(), ocr, renderer=FakeRenderer(blank=False))
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


def test_embed_all_honors_embedder_batch_size():
    calls = []

    class Emb:
        batch_size = 2

        def embed_batch(self, texts, deadline=None):
            calls.append(list(texts))
            return [[0.1] * 384 for _ in texts]

    vectors = _embed_all(type("D", (), {"embedder": Emb()})(), ["a", "b", "c"])
    assert calls == [["a", "b"], ["c"]]
    assert len(vectors) == 3


def test_page_limit_stops_before_ocr():
    ocr = FakeOCR("Noi dung OCR du dai de qua nguong trang.")
    with pytest.raises(IndexFailure) as raised:
        _run((FIXTURES / "mixed.pdf").read_bytes(), ocr, max_pages=1)
    assert raised.value.code == "page_limit"
    assert ocr.calls == []


# Native extract of the official BTP PDF (page 10). ToUnicode maps live CIDs to space.
GARBLED_NATIVE = (
    "10 PHẦN II. NỘI DUNG CỤ THỂ CỦA TỪNG THỦ TỤC HÀNH CHÍNH "
    "A. THỦ TỤC HÀNH CHÍNH CẤP XÃ 1. Thủ tục đ ng ký khai sinh "
    "Trình tự thực hiện: Nếu lựa chọn hình thức n p hồ s trực tiếp, "
    "ng ời yêu cầu ăng ký khai sinh n p hồ s ăng ký khai sinh t i "
    "Trung tâm Ph c v hành chính công có thẩm quy n; n p l phí nếu thu "
    "tr ờng hợp ph i n p l h ăng ký khai sinh; n p phí cấp b n sao "
    "Giấy khai sinh nếu có yêu cầu cấp b n sao Giấy khai sinh. "
    "Nếu lựa chọn hình thức n p hồ s trực tuyến, ng ời yêu cầu ăng ký "
    "khai sinh truy cập C ng dịch v công quố gia h Ứ đị h h ố. "
    "Ng ời yêu cầu ăng ký khai sinh trực tuyến cung cấp thông tin theo mẫu "
    "h tị h i n t t ng t ăng ký khai sinh, nh kè n ch p ho c b n sa "
    "các giấy tờ, tài li u the u ịnh; n p phí, l phí thông qua chứ năng thanh toán."
)

# Visual transcription of the same span, used only as FakeOCR output in tests.
CLEAN_OCR = (
    "10 PHẦN II. NỘI DUNG CỤ THỂ CỦA TỪNG THỦ TỤC HÀNH CHÍNH "
    "A. THỦ TỤC HÀNH CHÍNH CẤP XÃ 1. Thủ tục đăng ký khai sinh "
    "Trình tự thực hiện: Nếu lựa chọn hình thức nộp hồ sơ trực tiếp, "
    "người yêu cầu đăng ký khai sinh nộp hồ sơ đăng ký khai sinh tại "
    "Trung tâm Phục vụ hành chính công có thẩm quyền; nộp lệ phí nếu thuộc "
    "trường hợp phải nộp lệ phí đăng ký khai sinh; nộp phí cấp bản sao "
    "Giấy khai sinh nếu có yêu cầu cấp bản sao Giấy khai sinh. "
    "Nếu lựa chọn hình thức nộp hồ sơ trực tuyến, người yêu cầu đăng ký "
    "khai sinh truy cập Cổng dịch vụ công quốc gia hoặc Ứng dụng định danh quốc gia."
)


LATIN_OCR = (
    "10 PHAN II. NOI DUNG CU THE CUA TUNG THU TUC HANH CHINH "
    "A. THU TUC HANH CHINH CAP XA 1. Thu tuc dang ky khai sinh "
    "Trinh tu thuc hien: Neu lua chon hinh thuc nop ho so truc tiep, "
    "nguoi yeu cau dang ky khai sinh nop ho so dang ky khai sinh tai "
    "Trung tam Phuc vu hanh chinh cong co tham quyen; nop le phi neu thuoc "
    "truong hop phai nop le phi dang ky khai sinh; nop phi cap ban sao "
    "Giay khai sinh neu co yeu cau cap ban sao Giay khai sinh. "
    "Neu lua chon hinh thuc nop ho so truc tuyen, nguoi yeu cau dang ky "
    "khai sinh truy cap Cong dich vu cong quoc gia hoac Ung dung dinh danh quoc gia."
)


def test_native_text_usable_rejects_tounicode_space_substitution():
    assert native_text_usable(GARBLED_NATIVE) is False
    assert native_text_usable(CLEAN_OCR) is True
    assert native_text_usable(NATIVE_TEXT) is True
    assert native_text_usable("đ ng ký khai sinh") is False  # below min_chars
    assert native_text_usable("Thủ tục đ ng ký khai sinh tại xã") is False
    assert native_text_usable("A. THỦ TỤC HÀNH CHÍNH CẤP XÃ") is True
    assert native_text_usable("Bơngai Jrai kơnâm pơlei xa Chu Se Gia Lai Viet Nam.") is True


def test_valid_vietnamese_particles_are_not_cid_errors():
    assert native_text_usable("Hồ sơ ở xã cần giấy chứng sinh của trẻ.") is True
    assert native_text_usable("Ý kiến của cha mẹ là cần làm thủ tục khai sinh.") is True
    assert native_text_usable("Người yêu cầu đăng ký khai sinh ở nơi cư trú.") is True
    assert native_text_usable("a) Thành phần hồ sơ bao gồm tờ khai đăng ký khai sinh và giấy chứng sinh.") is True
    assert native_text_usable("b) Giấy chứng sinh của trẻ em vừa sinh ra.") is True
    assert native_text_usable("c) Tờ khai đăng ký khai sinh theo mẫu quy định.") is True
    assert native_text_usable("đ) Giấy chứng sinh của trẻ em trong trường hợp được yêu cầu.") is True
    assert native_text_usable("đ ng ký khai sinh") is False


def test_valid_vietnamese_native_is_kept_without_ocr(monkeypatch):
    text = "Người yêu cầu đăng ký khai sinh ở nơi cư trú."
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, text)],
    )
    ocr = FakeOCR("khong duoc goi")
    result, deps = _run((FIXTURES / "native.pdf").read_bytes(), ocr)
    assert ocr.calls == []
    assert result.native_pages == 1
    assert "ở nơi cư trú" in deps.chunks.saved[0].text


def test_recovered_text_usable_rejects_latin_stripped_copy():
    assert recovered_text_usable(CLEAN_OCR, GARBLED_NATIVE) is True
    assert recovered_text_usable(LATIN_OCR, GARBLED_NATIVE) is False
    assert recovered_text_usable(GARBLED_NATIVE, GARBLED_NATIVE) is False
    assert native_text_usable(LATIN_OCR) is True
    assert ocr_text_usable(CLEAN_OCR) is True
    assert ocr_text_usable(LATIN_OCR) is False


def test_latin_rec_dict_does_not_count_as_vietnamese(tmp_path):
    keys = tmp_path / "ppocr_keys_v1.txt"
    keys.write_text("a\nb\nâ\nê\nô\nđ\ná\nà\nã\n", encoding="utf-8")
    assert rec_dict_supports_vietnamese(str(tmp_path)) is False
    keys.write_text("ă\nơ\nư\nđăng\n", encoding="utf-8")
    assert rec_dict_supports_vietnamese(str(tmp_path)) is True


def test_good_native_is_kept_without_ocr():
    ocr = FakeOCR("khong duoc goi")
    result, deps = _run((FIXTURES / "native.pdf").read_bytes(), ocr)
    assert ocr.calls == []
    assert result.native_pages == 1
    assert result.ocr_pages == 0
    assert result.extraction_version == EXTRACTION_VERSION
    assert deps.chunks.saved[0].text.startswith("Dang ky khai sinh")


def test_fragmented_native_fails_when_ocr_charset_incomplete(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, GARBLED_NATIVE)],
    )
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR(CLEAN_OCR)
    job, deps = _prepare(data, ocr)
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == []
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_fragmented_native_falls_back_to_vietnamese_ocr(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, GARBLED_NATIVE)],
    )
    ocr = FakeOCR(CLEAN_OCR, vietnamese_capable=True)
    result, deps = _run((FIXTURES / "native.pdf").read_bytes(), ocr)
    assert ocr.calls == [1]
    assert result.ocr_pages == 1
    assert result.native_pages == 0
    assert result.extraction_version == EXTRACTION_VERSION
    saved = " ".join(item.text for item in deps.chunks.saved)
    assert "đăng ký khai sinh" in saved
    assert "đ ng ký" not in saved


def test_fragmented_native_fails_when_vietnamese_ocr_empty(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, GARBLED_NATIVE)],
    )
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR("", vietnamese_capable=True)
    job, deps = _prepare(data, ocr)
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [1]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_fragmented_native_fails_when_ocr_still_wrong(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, GARBLED_NATIVE)],
    )
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR(LATIN_OCR, vietnamese_capable=True)
    job, deps = _prepare(data, ocr)
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [1]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_required_page_failure_blocks_ready_generation(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, NATIVE_TEXT), (2, GARBLED_NATIVE)],
    )
    data = (FIXTURES / "multipage.pdf").read_bytes()
    ocr = FakeOCR(LATIN_OCR, vietnamese_capable=True)
    job, deps = _prepare(data, ocr)
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [2]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


SHORT_GARBLED = "đ ng ký khai sinh"
SHORT_HEADING = "Thủ tục đ ng ký khai sinh tại xã"


def test_short_garbled_native_fails_when_ocr_empty(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, SHORT_GARBLED)],
    )
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR("", vietnamese_capable=True)
    job, deps = _prepare(data, ocr)
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [1]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_short_garbled_native_rejects_nonempty_ocr_stub(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, SHORT_GARBLED)],
    )
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR("xyz", vietnamese_capable=True)
    job, deps = _prepare(data, ocr)
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [1]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_short_heading_with_cid_space_is_not_treated_as_usable(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, SHORT_HEADING)],
    )
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR("xyz")
    job, deps = _prepare(data, ocr)
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == []
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_verified_blank_second_page_may_be_skipped(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, NATIVE_TEXT), (2, "")],
    )
    ocr = FakeOCR("")
    result, deps = _run((FIXTURES / "multipage.pdf").read_bytes(), ocr)
    assert result.native_pages == 1
    assert result.ocr_pages == 0
    assert deps.chunks.saved
    assert all(item.page_start == 1 for item in deps.chunks.saved)


def test_ink_page_with_failed_ocr_fails_generation(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, NATIVE_TEXT), (2, "")],
    )
    data = (FIXTURES / "multipage.pdf").read_bytes()
    ocr = FakeOCR("", vietnamese_capable=True)
    job, deps = _prepare(data, ocr)

    class InkRenderer:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def render(self, _data, page_number: int, dpi: int):
            self.calls.append(page_number)
            return {"page": page_number, "dpi": dpi, "blank": False}

    deps.renderer = InkRenderer()
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [2]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_page_looks_blank_honors_explicit_flag_and_ink():
    assert page_looks_blank({"page": 1, "dpi": 150}) is True
    assert page_looks_blank({"page": 1, "blank": False}) is False
    white = [[[255, 255, 255] for _ in range(8)] for _ in range(8)]
    ink = [[[0, 0, 0] for _ in range(8)] for _ in range(8)]

    class _Grid:
        def __init__(self, rows):
            self._rows = rows
            self.shape = (len(rows), len(rows[0]), 3)

        def __getitem__(self, y):
            return self._rows[y]

    assert page_looks_blank(_Grid(white)) is True
    assert page_looks_blank(_Grid(ink)) is False
    assert page_looks_blank(object()) is False
    thin = [[[255, 255, 255] for _ in range(32)] for _ in range(32)]
    for x in range(32):
        thin[7][x] = [0, 0, 0]
    assert page_looks_blank(_Grid(thin)) is False
    sparse = [[[255, 255, 255] for _ in range(128)] for _ in range(128)]
    for x in range(128):
        sparse[64][x] = [0, 0, 0]
    assert (128 / (128 * 128)) < 0.008
    assert page_looks_blank(_Grid(sparse)) is False
    specks = [[[255, 255, 255] for _ in range(128)] for _ in range(128)]
    for x, y in ((3, 3), (40, 90), (100, 20), (110, 70), (50, 10)):
        specks[y][x] = [0, 0, 0]
    assert page_looks_blank(_Grid(specks)) is True
    large = [[[255, 255, 255] for _ in range(256)] for _ in range(256)]
    for x in range(256):
        large[1][x] = [0, 0, 0]
    assert page_looks_blank(_Grid(large)) is False


def test_ocr_text_usable_is_not_legal_certification():
    # Page-12 Tesseract errors still pass the structural filter.
    text = (
        "Sau khi lập biên bản, Ủy ban nhân dân cấp xã tiền hành niêm yết tại trụ sở "
        "Ủy ban nhân dân trong 07 ngày liên tục về việc trẻ bị bỏ rơi. "
        "Phần ghi về cha của trẻ để trồng nếu chưa xác định được cha."
    )
    assert ocr_text_usable(text) is True
    assert "tiến hành" not in text
    assert "để trống" not in text


def test_scan_latin_ocr_fails_without_vietnamese_charset(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, "")],
    )
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR(LATIN_OCR, vietnamese_capable=False)
    job, deps = _prepare(data, ocr, renderer=FakeRenderer(blank=False))
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == []
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_scan_latin_ocr_fails_even_when_charset_claims_vietnamese(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, "")],
    )
    data = (FIXTURES / "native.pdf").read_bytes()
    ocr = FakeOCR(LATIN_OCR, vietnamese_capable=True)
    job, deps = _prepare(data, ocr, renderer=FakeRenderer(blank=False))
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [1]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_scan_vietnamese_ocr_is_indexed(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, "")],
    )
    ocr = FakeOCR(CLEAN_OCR, vietnamese_capable=True)
    result, deps = _run((FIXTURES / "native.pdf").read_bytes(), ocr, renderer=FakeRenderer(blank=False))
    assert ocr.calls == [1]
    assert result.ocr_pages == 1
    assert "đăng ký khai sinh" in deps.chunks.saved[0].text


def test_sparse_ink_empty_ocr_fails_generation(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, NATIVE_TEXT), (2, "")],
    )
    data = (FIXTURES / "multipage.pdf").read_bytes()
    ocr = FakeOCR("", vietnamese_capable=True)
    job, deps = _prepare(data, ocr)

    class SparseInk:
        def render(self, _data, page_number: int, dpi: int):
            if page_number == 1:
                return {"page": page_number, "dpi": dpi, "blank": True}
            rows = [[[255, 255, 255] for _ in range(128)] for _ in range(128)]
            for x in range(128):
                rows[64][x] = [0, 0, 0]

            class _Grid:
                def __init__(self, grid):
                    self._rows = grid
                    self.shape = (len(grid), len(grid[0]), 3)

                def __getitem__(self, y):
                    return self._rows[y]

            return _Grid(rows)

    deps.renderer = SparseInk()
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [2]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_large_image_thin_stroke_empty_ocr_fails_generation(monkeypatch):
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(1, NATIVE_TEXT), (2, "")],
    )
    data = (FIXTURES / "multipage.pdf").read_bytes()
    ocr = FakeOCR("", vietnamese_capable=True)
    job, deps = _prepare(data, ocr)

    class LargeThinStroke:
        def render(self, _data, page_number: int, dpi: int):
            if page_number == 1:
                return {"page": page_number, "dpi": dpi, "blank": True}
            rows = [[[255, 255, 255] for _ in range(256)] for _ in range(256)]
            for x in range(256):
                rows[1][x] = [0, 0, 0]

            class _Grid:
                def __init__(self, grid):
                    self._rows = grid
                    self.shape = (len(grid), len(grid[0]), 3)

                def __getitem__(self, y):
                    return self._rows[y]

            return _Grid(rows)

    deps.renderer = LargeThinStroke()
    with pytest.raises(IndexFailure) as raised:
        _index(job, deps)
    assert raised.value.code == "extract_unusable"
    assert ocr.calls == [2]
    assert deps.chunks.saved == []
    assert deps.vectors.points == []


def test_list_mark_d_is_kept_and_cid_fragment_is_not():
    assert native_text_usable("đ) Giấy chứng sinh của trẻ em trong trường hợp được yêu cầu.") is True
    assert native_text_usable("Thủ tục đ ng ký khai sinh tại xã") is False
    assert native_text_usable(GARBLED_NATIVE) is False
