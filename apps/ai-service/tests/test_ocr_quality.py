"""Vietnamese OCR quality gate, per-page cache (resume) and progress for long PDFs."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from app.indexing.errors import IndexFailure
from app.indexing.ocr import OcrPage, PdfiumRenderer, TesseractOCR, parse_tesseract_tsv
from app.indexing.pipeline import CachedPage, OcrQuality, extraction_key, run_pipeline
from app.indexing.chunk import ChunkConfig
from tests.test_p4b_pipeline import CLEAN_OCR, FIXTURES, GARBLED_NATIVE, _prepare

TSV_HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def _tsv(rows: list[tuple[int, int, int, float, str]]) -> str:
    lines = [TSV_HEADER]
    for block, par, line, conf, word in rows:
        lines.append(f"5\t1\t{block}\t{par}\t{line}\t1\t0\t0\t10\t10\t{conf}\t{word}")
    return "\n".join(lines)


class ScoredOCR:
    """Fake engine that returns a fixed text and confidence per page."""

    vietnamese_capable = True
    version = "scored-fake-1"

    def __init__(self, pages: dict[int, OcrPage]) -> None:
        self.pages = pages
        self.calls: list[int] = []

    def read_page_scored(self, image, page_number: int, timeout_s: float = 30, deadline=None) -> OcrPage:
        self.calls.append(page_number)
        return self.pages[page_number]

    def read_page(self, image, page_number: int, timeout_s: float = 30, deadline=None) -> str:
        return self.read_page_scored(image, page_number).text


class MemoryCache:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, int], CachedPage] = {}

    def load(self, job, source_sha256, key):
        return {n: p for (sha, k, n), p in self.rows.items() if sha == source_sha256 and k == key}

    def store(self, job, source_sha256, key, page):
        self.rows[(source_sha256, key, page.number)] = page


class MemoryProgress:
    def __init__(self, lose_claim_at: int | None = None) -> None:
        self.events: list[dict] = []
        self.lose_claim_at = lose_claim_at

    def update(self, job, **fields):
        if self.lose_claim_at is not None and fields.get("pages_done") == self.lose_claim_at:
            raise IndexFailure("claim_rejected")
        self.events.append(fields)


GOOD = OcrPage(CLEAN_OCR, mean_confidence=95.0, low_confidence_ratio=0.01, word_count=60)
LOW = OcrPage(CLEAN_OCR, mean_confidence=70.0, low_confidence_ratio=0.30, word_count=60)


def _garbled_pages(monkeypatch, count: int) -> None:
    monkeypatch.setattr(
        "app.indexing.pipeline.read_selected_pages",
        lambda *_a, **_k: [(n, GARBLED_NATIVE) for n in range(1, count + 1)],
    )


def _run_scored(deps, job, quality: OcrQuality | None = None):
    return run_pipeline(
        job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents",
        native_min_chars=24, max_pages=50, max_ocr_pages=50, quality=quality or OcrQuality(),
    )


def _scored_deps(ocr, cache=None, progress=None):
    job, deps = _prepare((FIXTURES / "native.pdf").read_bytes(), ocr=ocr)
    deps.page_cache = cache
    deps.progress = progress
    return job, deps


def test_tsv_rebuilds_lines_drops_table_rules_and_scores_letters():
    page = parse_tesseract_tsv(_tsv([
        (1, 1, 1, 96.0, "Đăng"), (1, 1, 1, 94.0, "ký"), (1, 1, 1, 30.0, "|"), (1, 1, 1, 92.0, "lại"),
        (1, 1, 2, 90.0, "khai"), (1, 1, 2, 40.0, "sinh"), (1, 1, 2, 95.0, "18/2026/NĐ-CP"),
        (2, 1, 1, 88.0, "¬"), (2, 1, 1, 91.0, "—"),
    ]))
    assert page.text == "Đăng ký lại\nkhai sinh 18/2026/NĐ-CP\n—"
    assert page.word_count == 6
    assert page.mean_confidence == pytest.approx((96 + 94 + 92 + 90 + 40 + 95) / 6)
    assert page.low_confidence_ratio == pytest.approx(1 / 6)


def test_tsv_without_words_is_low_quality():
    page = parse_tesseract_tsv(TSV_HEADER + "\n1\t1\t0\t0\t0\t0\t0\t0\t10\t10\t-1\t")
    assert page.text == "" and page.word_count == 0
    assert page.mean_confidence == 0.0 and page.low_confidence_ratio == 1.0


def test_broken_text_layer_is_recovered_by_confident_vietnamese_ocr(monkeypatch):
    _garbled_pages(monkeypatch, 2)
    ocr = ScoredOCR({1: GOOD, 2: GOOD})
    progress = MemoryProgress()
    job, deps = _scored_deps(ocr, MemoryCache(), progress)
    result = _run_scored(deps, job)
    assert ocr.calls == [1, 2]
    assert result.ocr_pages == 2 and result.native_pages == 0
    saved = " ".join(item.text for item in deps.chunks.saved)
    assert "đăng ký khai sinh" in saved and "đ ng ký" not in saved
    assert result.ocr_version == "scored-fake-1"
    assert progress.events[0] == {"stage": "extract", "pages_total": 2, "pages_done": 0, "pages_ocr": 0, "pages_rejected": 0}
    done = [e["pages_done"] for e in progress.events if "pages_done" in e]
    assert done == sorted(done) and done[-1] == 2
    assert [e["stage"] for e in progress.events if "stage" in e] == ["extract", "chunk", "embed", "write"]


def test_low_confidence_page_fails_the_job_and_reports_pages(monkeypatch):
    _garbled_pages(monkeypatch, 3)
    progress = MemoryProgress()
    job, deps = _scored_deps(ScoredOCR({1: GOOD, 2: LOW, 3: GOOD}), MemoryCache(), progress)
    with pytest.raises(IndexFailure) as err:
        _run_scored(deps, job)
    assert err.value.code == "ocr_quality_low"
    report = [e["report"] for e in progress.events if "report" in e][-1]
    assert report["rejected_pages"] == [2]
    assert report["rejected_reasons"] == {"2": "ocr_low_confidence"}
    assert deps.chunks.saved == []


def test_allowed_rejection_excludes_page_from_citations(monkeypatch):
    _garbled_pages(monkeypatch, 2)
    low = OcrPage("TRANG LỖI chữ sai dấu không được trích dẫn " * 3, 60.0, 0.4, 30)
    job, deps = _scored_deps(ScoredOCR({1: GOOD, 2: low}), MemoryCache(), MemoryProgress())
    result = _run_scored(deps, job, OcrQuality(max_rejected_pages=1))
    assert result.ocr_pages == 1
    assert all(item.page_start == 1 and item.page_end == 1 for item in deps.chunks.saved)
    assert "TRANG LỖI" not in " ".join(item.text for item in deps.chunks.saved)


def test_unusable_ocr_text_is_rejected_even_with_high_confidence(monkeypatch):
    _garbled_pages(monkeypatch, 1)
    latin_only = OcrPage("dang ky khai sinh tai uy ban nhan dan xa", 99.0, 0.0, 9)
    progress = MemoryProgress()
    job, deps = _scored_deps(ScoredOCR({1: latin_only}), None, progress)
    with pytest.raises(IndexFailure) as err:
        _run_scored(deps, job)
    assert err.value.code == "ocr_quality_low"
    assert [e["report"] for e in progress.events if "report" in e][-1]["rejected_reasons"] == {"1": "recovery_failed"}


def test_retry_resumes_from_page_cache_without_new_ocr(monkeypatch):
    _garbled_pages(monkeypatch, 3)
    cache = MemoryCache()
    first = ScoredOCR({1: GOOD, 2: GOOD, 3: GOOD})
    job, deps = _scored_deps(first, cache, MemoryProgress(lose_claim_at=2))
    with pytest.raises(IndexFailure) as err:
        _run_scored(deps, job)
    assert err.value.code == "claim_rejected"
    assert first.calls == [1, 2]
    assert sorted(n for (_s, _k, n) in cache.rows) == [1, 2]

    second = ScoredOCR({1: GOOD, 2: GOOD, 3: GOOD})
    job, deps = _scored_deps(second, cache, MemoryProgress())
    result = _run_scored(deps, job)
    assert second.calls == [3]
    assert result.ocr_pages == 3
    assert sorted(n for (_s, _k, n) in cache.rows) == [1, 2, 3]


def test_cached_rejection_is_reused_and_threshold_change_invalidates_cache(monkeypatch):
    _garbled_pages(monkeypatch, 1)
    cache = MemoryCache()
    job, deps = _scored_deps(ScoredOCR({1: LOW}), cache, MemoryProgress())
    with pytest.raises(IndexFailure):
        _run_scored(deps, job)
    again = ScoredOCR({1: GOOD})
    job, deps = _scored_deps(again, cache, MemoryProgress())
    with pytest.raises(IndexFailure):
        _run_scored(deps, job)
    assert again.calls == [], "same engine + thresholds reuse the cached rejection"
    assert extraction_key("v", 150, 24, OcrQuality()) != extraction_key("v", 150, 24, OcrQuality(min_mean_confidence=80))
    looser = ScoredOCR({1: LOW})
    job, deps = _scored_deps(looser, cache, MemoryProgress())
    result = _run_scored(deps, job, OcrQuality(min_mean_confidence=60, max_low_confidence_ratio=0.4))
    assert looser.calls == [1] and result.ocr_pages == 1


def test_page_cache_failure_does_not_fail_the_job(monkeypatch):
    _garbled_pages(monkeypatch, 1)

    class Broken:
        def load(self, *_a):
            raise RuntimeError("db down")

        def store(self, *_a):
            raise RuntimeError("db down")

    job, deps = _scored_deps(ScoredOCR({1: GOOD}), Broken(), MemoryProgress())
    assert _run_scored(deps, job).ocr_pages == 1


def test_ocr_page_limit_still_applies(monkeypatch):
    _garbled_pages(monkeypatch, 3)
    job, deps = _scored_deps(ScoredOCR({1: GOOD, 2: GOOD, 3: GOOD}), MemoryCache(), MemoryProgress())
    with pytest.raises(IndexFailure) as err:
        run_pipeline(
            job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents",
            native_min_chars=24, max_pages=50, max_ocr_pages=2, quality=OcrQuality(),
        )
    assert err.value.code == "page_limit"


@pytest.mark.skipif(
    shutil.which("tesseract") is None or importlib.util.find_spec("pypdfium2") is None,
    reason="tesseract binary or pypdfium2 not installed (both run in the ai-service image)",
)
def test_real_tesseract_reads_vietnamese_diacritics():
    ocr = TesseractOCR()
    assert ocr.version.startswith("tesseract-") and "-vie-" in ocr.version
    image = PdfiumRenderer().render((FIXTURES / "ocr_vi_fields.pdf").read_bytes(), 1, 150)
    page = ocr.read_page_scored(image, 1)
    assert "KHAI SINH" in page.text.upper()
    assert "Nơi đăng ký" in page.text
    assert page.mean_confidence >= 80 and page.word_count >= 5


def test_large_sheet_renders_below_pixel_cap_instead_of_failing():
    pytest.importorskip("pypdfium2", reason="pypdfium2 ships in requirements-prod (ai-service image)")
    import io

    from pypdf import PdfWriter

    from app.indexing.ocr import MAX_PIXELS

    writer = PdfWriter()
    writer.add_blank_page(width=1191, height=1684)  # A2 in points
    buf = io.BytesIO()
    writer.write(buf)
    image = PdfiumRenderer().render(buf.getvalue(), 1, 300)
    assert image.shape[0] * image.shape[1] <= MAX_PIXELS
    assert image.shape[2] == 3
