"""Offline index pipeline. It never calls an external LLM or downloads models."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field

from app.indexing.chunk import Chunk, ChunkConfig, chunk_pages, manifest_hash
from app.indexing.embed import (
    E5_DIMENSION,
    E5_MAX_TOKENS,
    FAKE_MODEL_ID,
    FAKE_REVISION,
    PASSAGE_PREFIX,
    WordTokenCounter,
    fake_checksum,
    fake_embed,
    normalize,
)
from app.indexing.ocr import DEFAULT_DPI, FakeRenderer
from app.indexing.deadline import Deadline
from app.indexing.errors import IndexFailure
from app.indexing.pdf import NATIVE_TEXT_MIN_CHARS, read_selected_pages
from app.indexing.qdrant import point_id
from app.indexing.text import normalize_text, sha256_text

log = logging.getLogger("cas.index")

CLEANUP_BUDGET_S = 2.0
PIPELINE_VERSION = "p4b.1"
EXTRACTION_VERSION = "pypdf-native-nfc-1"
OBJECT_KEY = "{xa}/documents/{document}/{checksum}.pdf"


@dataclass
class IndexJob:
    schema_version: str
    xa_id: str
    document_id: uuid.UUID
    procedure_id: uuid.UUID
    procedure_version_id: uuid.UUID
    job_id: uuid.UUID
    claim_token: uuid.UUID
    generation_id: uuid.UUID
    bucket: str
    object_key: str
    checksum: str
    page_range: str | None
    pipeline_version: str


@dataclass
class PageRecord:
    number: int
    text: str
    source: str


@dataclass
class SavedChunk:
    index: int
    page_start: int
    page_end: int
    text: str
    text_sha256: str
    token_count: int
    source: str
    chunk_id: uuid.UUID


@dataclass
class PipelineResult:
    generation_id: uuid.UUID
    source_sha256: str
    content_sha256: str
    pages_processed: int
    native_pages: int
    ocr_pages: int
    chunk_count: int
    vector_count: int
    manifest_hash: str
    pipeline_version: str
    extraction_version: str
    ocr_version: str
    embedding_model_id: str
    embedding_revision: str
    embedding_checksum: str
    vector_dimension: int
    stage_ms: dict[str, int] = field(default_factory=dict)


class MemoryObjects:
    def __init__(self, data: bytes = b"", fail: str | None = None) -> None:
        self.data = data
        self.fail = fail

    def get(self, _bucket: str, _key: str, max_bytes: int | None = None, timeout_s: float | None = None, deadline=None) -> bytes:
        if self.fail == "timeout":
            raise IndexFailure("source_timeout")
        if self.fail == "missing":
            raise IndexFailure("source_not_found")
        if max_bytes is not None and len(self.data) > max_bytes:
            raise IndexFailure("pdf_oversized")
        return self.data


class MemoryChunks:
    def __init__(self) -> None:
        self.saved: list[SavedChunk] = []
        self.meta: dict | None = None

    def replace(self, job: IndexJob, chunks: list[SavedChunk], meta: dict, timeout_s: float | None = None, deadline=None) -> None:
        self.saved = list(chunks)
        self.meta = meta


def expected_object_key(job: IndexJob) -> str:
    return f"{job.xa_id}/documents/{job.document_id}/{job.checksum}.pdf"


def chunk_id_for(generation_id: uuid.UUID, index: int) -> uuid.UUID:
    return uuid.uuid5(uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"), f"{generation_id}:{index}")


def run_pipeline(
    job: IndexJob,
    deps,
    *,
    max_bytes: int,
    chunk_config: ChunkConfig,
    bucket: str,
    native_min_chars: int = NATIVE_TEXT_MIN_CHARS,
    max_pages: int = 50,
    max_ocr_pages: int = 20,
    dpi: int = DEFAULT_DPI,
    ocr_timeout_s: float = 30,
    deadline_s: float | None = None,
    deadline: Deadline | None = None,
) -> PipelineResult:
    clock = deadline if deadline is not None else (Deadline(deadline_s) if deadline_s is not None else None)
    started = time.perf_counter()
    if job.pipeline_version != PIPELINE_VERSION:
        raise IndexFailure("worker_mismatch")
    if job.bucket != bucket or job.object_key != expected_object_key(job):
        raise IndexFailure("source_not_found")
    if "://" in job.object_key or job.object_key.startswith("http"):
        raise IndexFailure("source_not_found")
    stage: dict[str, int] = {}
    mark = time.perf_counter()
    _check(clock)
    data = deps.objects.get(job.bucket, job.object_key, max_bytes, deadline=clock)
    _check(clock)
    source_sha = hashlib.sha256(data).hexdigest()
    if source_sha != job.checksum:
        raise IndexFailure("checksum_mismatch")
    pages = read_selected_pages(data, max_bytes, max_pages, job.page_range)
    _check(clock)
    stage["extract"] = _ms(mark)
    mark = time.perf_counter()
    prepared: list[PageRecord] = []
    renderer = getattr(deps, "renderer", None) or FakeRenderer()
    ocr_used = 0
    for number, native in pages:
        _check(clock)
        if len(native) >= native_min_chars:
            prepared.append(PageRecord(number, native, "native"))
            continue
        if ocr_used >= max_ocr_pages:
            raise IndexFailure("page_limit")
        ocr_used += 1
        _check(clock)
        image = renderer.render(data, number, dpi)
        _check(clock)
        ocr_budget = ocr_timeout_s if clock is None else clock.timeout_for_io(ocr_timeout_s)
        try:
            raw = deps.ocr.read_page(image, number, timeout_s=ocr_budget, deadline=clock)
        except TimeoutError as exc:
            raise IndexFailure("ocr_timeout") from exc
        text = normalize_text(raw)
        if not text:
            continue
        prepared.append(PageRecord(number, text, "ocr"))
    stage["ocr"] = _ms(mark)
    if not prepared:
        raise IndexFailure("no_extractable_text")
    mark = time.perf_counter()
    chunks = chunk_pages([(item.number, item.text, item.source) for item in prepared], chunk_config)
    counter = deps.embedder if hasattr(deps.embedder, "count_tokens") else WordTokenCounter()
    limit = getattr(deps.embedder, "max_tokens", E5_MAX_TOKENS)
    chunks = _fit_token_limit(chunks, counter, limit, clock)
    _check(clock)
    if not chunks or any(_tokens(counter, item.text, clock) > limit for item in chunks):
        raise IndexFailure("chunk_limit")
    stage["chunk"] = _ms(mark)
    content = "\n".join(item.text for item in prepared)
    content_sha = sha256_text(content)
    digest = manifest_hash(chunks)
    saved = [
        SavedChunk(
            index=item.index,
            page_start=item.page_start,
            page_end=item.page_end,
            text=item.text,
            text_sha256=item.text_sha256,
            token_count=item.token_count,
            source=item.source,
            chunk_id=chunk_id_for(job.generation_id, item.index),
        )
        for item in chunks
    ]
    mark = time.perf_counter()
    vectors = _embed_all(deps, [item.text for item in saved], clock)
    _check(clock)
    stage["embed"] = _ms(mark)
    model_id, revision, checksum, dimension = _model_meta(deps)
    mark = time.perf_counter()
    qdrant_started = False
    try:
        qdrant_started = True
        _check(clock)
        written = deps.vectors.upsert(
            job.generation_id,
            saved,
            vectors,
            {
                "xa_id": job.xa_id,
                "document_id": str(job.document_id),
                "procedure_version_id": str(job.procedure_version_id),
                "vector_dimension": dimension,
            },
            deadline=clock,
        )
        if written != len(saved):
            raise IndexFailure("qdrant_failed")
        meta = {
            "pipeline_version": PIPELINE_VERSION,
            "extraction_version": EXTRACTION_VERSION,
            "ocr_version": deps.ocr.version,
            "chunk_config_hash": chunk_config.hash(),
            "embedding_model_id": model_id,
            "embedding_revision": revision,
            "embedding_checksum": checksum,
            "vector_dimension": dimension,
            "source_sha256": source_sha,
            "content_sha256": content_sha,
            "manifest_hash": digest,
            "page_count": len(pages),
            "native_page_count": sum(1 for item in prepared if item.source == "native"),
            "ocr_page_count": sum(1 for item in prepared if item.source == "ocr"),
            "chunk_count": len(saved),
            "vector_count": written,
        }
        _check(clock)
        deps.chunks.replace(job, saved, meta, deadline=clock)
        _check(clock)
    except Exception:
        if qdrant_started and hasattr(deps.vectors, "delete_generation"):
            try:
                deps.vectors.delete_generation(job.generation_id, deadline=Deadline(CLEANUP_BUDGET_S))
            except IndexFailure:
                log.warning("qdrant_cleanup_failed code=qdrant_failed")
        raise
    stage["write"] = _ms(mark)
    log.info(
        "index_pipeline pages=%s native=%s ocr=%s chunks=%s vectors=%s ms=%s",
        meta["page_count"],
        meta["native_page_count"],
        meta["ocr_page_count"],
        meta["chunk_count"],
        meta["vector_count"],
        int((time.perf_counter() - started) * 1000),
    )
    return PipelineResult(
        generation_id=job.generation_id,
        source_sha256=source_sha,
        content_sha256=content_sha,
        pages_processed=meta["page_count"],
        native_pages=meta["native_page_count"],
        ocr_pages=meta["ocr_page_count"],
        chunk_count=len(saved),
        vector_count=written,
        manifest_hash=digest,
        pipeline_version=PIPELINE_VERSION,
        extraction_version=EXTRACTION_VERSION,
        ocr_version=deps.ocr.version,
        embedding_model_id=model_id,
        embedding_revision=revision,
        embedding_checksum=checksum,
        vector_dimension=dimension,
        stage_ms=stage,
    )


def _embed_all(deps, texts: list[str], deadline=None) -> list[list[float]]:
    _check(deadline)
    embedder = deps.embedder
    if hasattr(embedder, "embed_batch"):
        vectors = embedder.embed_batch(texts, deadline=deadline)
    else:
        vectors = [embedder.embed(PASSAGE_PREFIX + text) for text in texts]
    out = []
    for vector in vectors:
        if len(vector) != E5_DIMENSION:
            raise IndexFailure("embedding_dimension")
        out.append(normalize(vector))
    return out


def _fit_token_limit(chunks: list[Chunk], counter, limit: int, deadline=None) -> list[Chunk]:
    from app.indexing.embed import split_to_limit
    from app.indexing.text import sha256_text

    fitted: list[Chunk] = []
    splitter = counter.split_text if hasattr(counter, "split_text") else None
    for chunk in chunks:
        if _tokens(counter, chunk.text, deadline) <= limit:
            parts = [chunk.text]
        elif splitter is not None:
            parts = splitter(chunk.text, limit, deadline)
        else:
            parts = split_to_limit(counter, chunk.text, limit, deadline)
        for part in parts:
            if _tokens(counter, part, deadline) > limit:
                raise IndexFailure("chunk_limit")
            fitted.append(_piece(chunk, part))
    out: list[Chunk] = []
    for index, item in enumerate(fitted):
        out.append(
            Chunk(
                index=index,
                page_start=item.page_start,
                page_end=item.page_end,
                text=item.text,
                text_sha256=sha256_text(item.text),
                token_count=_tokens(counter, item.text, deadline),
                source=item.source,
            )
        )
    return out


def _piece(chunk: Chunk, text: str) -> Chunk:
    return Chunk(
        index=chunk.index,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=text,
        text_sha256=chunk.text_sha256,
        token_count=chunk.token_count,
        source=chunk.source,
    )


def _check(deadline) -> None:
    if deadline is not None:
        deadline.check()


def _tokens(counter, text: str, deadline) -> int:
    if deadline is None:
        return counter.count_tokens(text)
    return counter.count_tokens(text, deadline=deadline)


def _model_meta(deps) -> tuple[str, str, str, int]:
    embedder = deps.embedder
    if hasattr(embedder, "model_id"):
        return embedder.model_id, embedder.revision, embedder.checksum, embedder.dimension
    return FAKE_MODEL_ID, FAKE_REVISION, fake_checksum(), E5_DIMENSION


def _ms(mark: float) -> int:
    return int((time.perf_counter() - mark) * 1000)


class FakeEmbedder:
    model_id = FAKE_MODEL_ID
    revision = FAKE_REVISION
    checksum = fake_checksum()
    dimension = E5_DIMENSION
    max_tokens = E5_MAX_TOKENS

    def count_tokens(self, text: str, deadline=None) -> int:
        return WordTokenCounter().count_tokens(text, deadline=deadline)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], deadline=None) -> list[list[float]]:
        _check(deadline)
        return [fake_embed(PASSAGE_PREFIX + text, self.dimension) for text in texts]


def assert_point_id(generation_id: uuid.UUID, chunk_index: int, actual: str) -> None:
    if actual != point_id(generation_id, chunk_index):
        raise IndexFailure("qdrant_failed")
