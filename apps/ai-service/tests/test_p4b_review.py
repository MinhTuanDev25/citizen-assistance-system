"""Review regressions: tokenizer limit, slots, cleanup, page cap, readiness, manifest."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
import uuid
from pathlib import Path

import pytest

from app.indexing.chunk import Chunk
from app.indexing.embed import (
    E5_MODEL_SHA256,
    E5_REVISION,
    E5_TOKENIZER_SHA256,
    SubwordCounter,
    load_manifest,
)
from app.indexing.errors import IndexFailure
from app.indexing.pdf import read_bounded, read_selected_pages
from app.indexing.pipeline import (
    FakeEmbedder,
    IndexJob,
    MemoryChunks,
    MemoryObjects,
    _fit_token_limit,
    run_pipeline,
)
from app.indexing.qdrant import MemoryVectors, QdrantWriter
from app.indexing.runtime import IndexGate, probe_pipeline
from app.indexing.text import sha256_text


def test_subword_counter_splits_past_512_without_dropping_the_tail():
    real = SubwordCounter(width=1, overhead=4, limit=512, truncate=False)
    truncated = SubwordCounter(width=1, overhead=4, limit=512, truncate=True)
    text = "a" * 600
    assert real.count_tokens(text) == 604
    assert truncated.count_tokens(text) == 512
    chunk = Chunk(0, 1, 1, text, sha256_text(text), 604, "native")
    pieces = _fit_token_limit([chunk], real, 512)
    joined = "".join(item.text for item in pieces)
    assert joined == text
    assert sha256_text(joined) == sha256_text(text)
    assert all(real.count_tokens(item.text) <= 512 for item in pieces)
    assert len(pieces) > 1
    real.embed_batch([item.text for item in pieces])


def test_model_builder_is_called_once_and_gate_holds_the_slot():
    from app import main

    mode = main._settings.index_mode
    object.__setattr__(main._settings, "index_mode", "pipeline")

    async def boot():
        async with main.app.router.lifespan_context(main.app):
            assert main._gate.boot_count == 1
            first = main._gate.call({"op": "ping"}, 2)
            second = main._gate.call({"op": "ping"}, 2)
            assert first["pid"] == second["pid"]
            assert main._gate.boot_count == 1

    try:
        asyncio.run(boot())
    finally:
        object.__setattr__(main._settings, "index_mode", mode)
        main._models = None
        main._gate.shutdown()
        main._gate = IndexGate(main._settings.index_max_inflight)
    gate = IndexGate(1, grace_s=0.08, recovery_timeout_s=2)
    gate.start()
    started = "/tmp/cas-review-started"
    wrote = "/tmp/cas-review-wrote"
    for path in (started, wrote):
        try:
            os.remove(path)
        except OSError:
            pass
    pid = gate.worker_pids()[0]
    before = [item for item in threading.enumerate() if item.daemon and item.name == "index-pipeline"]
    try:
        with pytest.raises(IndexFailure) as timed:
            gate.call({"op": "hang", "stage": "parser", "seconds": 0.35, "started_path": started, "wrote_path": wrote}, 0.05)
        assert timed.value.code == "timeout"
        assert os.path.exists(started)
        assert os.path.exists(wrote) is False
        assert gate.pending() == 0
        assert _pid_alive(pid) is False
        nxt = gate.call({"op": "ping"}, 2)
        assert nxt["event"] == "pong"
        assert nxt["pid"] != pid
    finally:
        gate.shutdown()
        assert before == [item for item in threading.enumerate() if item.daemon and item.name == "index-pipeline"]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_health_answers_while_index_worker_is_busy():
    import httpx

    from app import main
    from tests.conftest import AUTH_HEADER

    mode = main._settings.index_mode
    gate = IndexGate(1, grace_s=0.1, recovery_timeout_s=2)
    gate.start()
    started = "/tmp/cas-health-started"
    release = "/tmp/cas-health-release"
    for path in (started, release):
        try:
            os.remove(path)
        except OSError:
            pass
    gate.arm_hold(started, release, 5)
    main._gate = gate
    object.__setattr__(main._settings, "index_mode", "pipeline")

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            index_task = asyncio.create_task(
                client.post(
                    "/v1/index",
                    headers=AUTH_HEADER,
                    json={
                        "schema_version": "index.v2",
                        "xa_id": "xa_chu_se",
                        "document_id": str(uuid.uuid4()),
                        "procedure_id": str(uuid.uuid4()),
                        "procedure_version_id": str(uuid.uuid4()),
                        "job_id": str(uuid.uuid4()),
                        "claim_token": str(uuid.uuid4()),
                        "generation_id": str(uuid.uuid4()),
                        "bucket": "cas-documents",
                        "object_key": "xa_chu_se/documents/x/a.pdf",
                        "checksum": "a" * 64,
                        "pipeline_version": "p4b.1",
                    },
                )
            )
            for _ in range(50):
                if os.path.exists(started):
                    break
                await asyncio.sleep(0.02)
            assert os.path.exists(started)
            health = await client.get("/health")
            open(release, "w", encoding="utf-8").close()
            indexed = await index_task
        return health, indexed

    try:
        health, indexed = asyncio.run(scenario())
    finally:
        open(release, "a", encoding="utf-8").close()
        main._gate.shutdown()
        main._gate = IndexGate(main._settings.index_max_inflight)
        object.__setattr__(main._settings, "index_mode", mode)
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert indexed.status_code == 200


def test_repeated_ocr_calls_do_not_grow_daemon_threads():
    from app.indexing.ocr import PaddleOCR

    ocr = PaddleOCR.__new__(PaddleOCR)

    class Engine:
        def ocr(self, _image, cls=False):
            time.sleep(0.01)
            return [[[ [[0, 0], [1, 0], [1, 1], [0, 1]], ("ok", 0.9) ]]]

    ocr._engine = Engine()
    before = threading.active_count()
    for _ in range(3):
        assert ocr.read_page(object(), 1, timeout_s=0.001) == "ok"
    assert threading.active_count() == before


def test_postgres_failure_after_qdrant_deletes_the_generation():
    class Boom:
        def replace(self, *_args, **_kwargs):
            raise IndexFailure("postgres_failed")

    data = b"%PDF-1.4\n"
    job = IndexJob(
        schema_version="index.v2",
        xa_id="xa_chu_se",
        document_id=uuid.uuid4(),
        procedure_id=uuid.uuid4(),
        procedure_version_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        claim_token=uuid.uuid4(),
        generation_id=uuid.uuid4(),
        bucket="cas-documents",
        object_key="x",
        checksum=hashlib.sha256(data).hexdigest(),
        page_range=None,
        pipeline_version="p4b.1",
    )
    # The key must match the pipeline rule. Use a tiny native pdf via the fixture runner instead.
    from tests.test_p4b_pipeline import FIXTURES, _job

    payload = (FIXTURES / "native.pdf").read_bytes()
    job = _job(_data=payload)
    vectors = MemoryVectors()
    deps = type("Deps", (), {})()
    deps.objects = MemoryObjects(payload)
    deps.ocr = type("O", (), {"version": "fake", "read_page": staticmethod(lambda *_a, **_k: "")})()
    deps.embedder = FakeEmbedder()
    deps.vectors = vectors
    deps.chunks = Boom()
    deps.renderer = type("R", (), {"render": staticmethod(lambda *_a, **_k: {})})()
    from app.indexing.chunk import ChunkConfig

    with pytest.raises(IndexFailure) as raised:
        run_pipeline(job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents")
    assert raised.value.code == "postgres_failed"
    assert vectors.points == []


def test_delete_requires_completed_and_zero_count():
    import httpx

    seen = {"count": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/points/delete"):
            return httpx.Response(200, json={"result": {"status": "acknowledged"}, "status": "ok"})
        if request.url.path.endswith("/points/count"):
            seen["count"] = True
            return httpx.Response(200, json={"result": {"count": 0}, "status": "ok"})
        return httpx.Response(500, json={"status": "error"})

    writer = QdrantWriter("http://qdrant", "knowledge_chunks")
    original = httpx.Client

    class _Client(original):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    httpx.Client = _Client
    try:
        with pytest.raises(IndexFailure) as raised:
            writer.delete_generation(uuid.uuid4())
    finally:
        httpx.Client = original
    assert raised.value.code == "qdrant_failed"
    assert seen["count"] is None


def test_page_cap_runs_before_extract_text(monkeypatch):
    calls = {"n": 0}

    class Page:
        def extract_text(self):
            calls["n"] += 1
            return "native text that is definitely long enough"

    class Reader:
        is_encrypted = False

        def __init__(self, _buffer):
            self.pages = [Page() for _ in range(8)]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    with pytest.raises(IndexFailure) as raised:
        read_selected_pages(b"%PDF-1.4\n", 1000, 2, None)
    assert raised.value.code == "page_limit"
    assert calls["n"] == 0


def test_bounded_stream_stops_early():
    def chunks():
        yield b"a" * 10
        yield b"b" * 10
        raise AssertionError("stream continued")

    with pytest.raises(IndexFailure) as raised:
        read_bounded(chunks(), 12)
    assert raised.value.code == "pdf_oversized"


def test_unreachable_dependency_is_not_ready():
    class Settings:
        index_mode = "pipeline"
        embedding_provider = "fake"
        ocr_provider = "fake"
        database_url = "postgres://cas:cas@127.0.0.1:1/none?sslmode=disable"
        qdrant_url = "http://127.0.0.1:1"
        qdrant_collection = "knowledge_chunks"
        object_bucket = "cas-documents"

    ok, reason = probe_pipeline(Settings(), True)
    assert ok is False
    assert reason == "postgres_unreachable"


def test_manifest_rejects_wrong_revision_quantization_and_checksum(tmp_path):
    body = {
        "model_id": "intfloat/multilingual-e5-small",
        "source_repository": "https://huggingface.co/intfloat/multilingual-e5-small",
        "revision": "6e0d5e48e6120c11e986346003f54bca39b85422",
        "source_file": "onnx/model_qint8_avx512_vnni.onnx",
        "dimension": 384,
        "quantization": "int8",
        "quantization_method": "qint8_avx512_vnni",
        "files": {"model.onnx": E5_MODEL_SHA256, "tokenizer.json": E5_TOKENIZER_SHA256},
    }
    import json

    (tmp_path / "manifest.json").write_text(json.dumps(body), encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "tokenizer.json").write_bytes(b"y")
    with pytest.raises(IndexFailure) as raised:
        load_manifest(str(tmp_path))
    assert raised.value.code == "embedding_checksum_mismatch"
    body["revision"] = E5_REVISION
    body["quantization"] = "fp32"
    (tmp_path / "manifest.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(IndexFailure) as quant:
        load_manifest(str(tmp_path))
    assert quant.value.code == "embedding_checksum_mismatch"


def test_routes_do_not_add_retrieval_or_activate():
    root = Path(__file__).resolve().parents[3]
    routes = (root / "apps/api/internal/api/http/v1/routes.go").read_text(encoding="utf-8")
    main = (root / "apps/ai-service/app/main.py").read_text(encoding="utf-8")
    assert "/retrieve" not in routes
    assert "/rag" not in routes
    assert "Activate" not in routes
    assert "/v1/retrieve" not in main
    assert "/v1/chat" not in main
