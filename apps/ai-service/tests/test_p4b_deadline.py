"""Absolute pipeline deadline, Qdrant cleanup, and ONNX recovery."""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from app.indexing.deadline import Deadline
from app.indexing.errors import IndexFailure


def _http_client(handler):
    original = httpx.Client

    class _Client(original):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    return original, _Client


def test_qdrant_cleanup_requires_completed_and_exact_zero():
    from app.indexing.qdrant import QdrantWriter

    cases = []

    def run(handler):
        writer = QdrantWriter("http://qdrant", "knowledge_chunks")
        original, client = _http_client(handler)
        httpx.Client = client
        try:
            with pytest.raises(IndexFailure) as raised:
                writer.delete_generation(uuid.uuid4())
        finally:
            httpx.Client = original
        assert raised.value.code == "qdrant_failed"

    def missing(_request):
        return httpx.Response(404, json={"status": "error"})

    def leftover(_request):
        if _request.url.path.endswith("/points/delete"):
            return httpx.Response(200, json={"result": {"status": "completed"}})
        return httpx.Response(200, json={"result": {"count": 2}})

    def bad_delete(_request):
        return httpx.Response(200, content=b"completed")

    def bad_count(_request):
        if _request.url.path.endswith("/points/delete"):
            return httpx.Response(200, json={"result": {"status": "completed"}})
        return httpx.Response(200, json={"result": {}})

    run(missing)
    run(leftover)
    run(bad_delete)
    run(bad_count)

    def clean(request: httpx.Request) -> httpx.Response:
        cases.append(request.url.path)
        if request.url.path.endswith("/points/delete"):
            return httpx.Response(200, json={"result": {"status": "completed"}})
        return httpx.Response(200, json={"result": {"count": 0}})

    writer = QdrantWriter("http://qdrant", "knowledge_chunks")
    original, client = _http_client(clean)
    httpx.Client = client
    try:
        writer.delete_generation(uuid.uuid4())
    finally:
        httpx.Client = original
    assert any(path.endswith("/points/count") for path in cases)


def test_four_qdrant_calls_share_one_deadline():
    from app.indexing.qdrant import QdrantWriter

    calls = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)

        def _ok(self):
            self._body()
            calls["n"] += 1
            time.sleep(0.09)
            payload = json.dumps({"status": "ok", "result": {"status": "completed", "count": 1, "points": []}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_PUT = _ok
        do_POST = _ok

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    writer = QdrantWriter(f"http://127.0.0.1:{server.server_address[1]}", "knowledge_chunks", timeout_s=30)
    chunk = type("C", (), {"index": 0, "chunk_id": uuid.uuid4(), "page_start": 1, "page_end": 1})()
    started = time.perf_counter()
    try:
        with pytest.raises(IndexFailure) as raised:
            writer.upsert(
                uuid.uuid4(),
                [chunk],
                [[0.1, 0.2]],
                {"vector_dimension": 2, "xa_id": "xa", "document_id": "d", "procedure_version_id": "v"},
                deadline=Deadline(0.2),
            )
        assert raised.value.code == "timeout"
        assert calls["n"] < 4
        assert time.perf_counter() - started < 0.35
    finally:
        server.shutdown()


def test_postgres_statement_timeouts_shrink(monkeypatch):
    from app.indexing.postgres import PostgresChunks

    values = []

    class Cursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=None):
            if "set_config" in sql:
                values.append(int(params[0]))

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

        def commit(self):
            return None

        def rollback(self):
            return None

    monkeypatch.setitem(__import__("sys").modules, "psycopg", type("M", (), {"connect": staticmethod(lambda *a, **k: Conn())}))
    import psycopg

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: Conn())

    class Clock:
        def __init__(self) -> None:
            self.left = 4.0

        def remaining(self) -> float:
            self.left -= 0.25
            if self.left <= 0:
                raise IndexFailure("timeout")
            return self.left

        def check(self) -> None:
            self.remaining()

    job = type(
        "J",
        (),
        {
            "generation_id": uuid.uuid4(),
            "xa_id": "xa",
            "document_id": uuid.uuid4(),
            "procedure_version_id": uuid.uuid4(),
            "job_id": uuid.uuid4(),
            "procedure_id": uuid.uuid4(),
        },
    )()
    chunk = type(
        "C",
        (),
        {
            "chunk_id": uuid.uuid4(),
            "index": 0,
            "text": "a",
            "source": "native",
            "page_start": 1,
            "page_end": 1,
            "text_sha256": "abc",
            "token_count": 1,
        },
    )()
    meta = {
        "pipeline_version": "p4b.1",
        "extraction_version": "x",
        "ocr_version": "x",
        "chunk_config_hash": "x",
        "embedding_model_id": "x",
        "embedding_revision": "x",
        "embedding_checksum": "x",
        "vector_dimension": 384,
        "source_sha256": "x",
        "content_sha256": "x",
        "manifest_hash": "x",
        "page_count": 1,
        "native_page_count": 1,
        "ocr_page_count": 0,
        "chunk_count": 1,
        "vector_count": 1,
    }
    PostgresChunks("postgresql://cas:hidden@127.0.0.1/cas").replace(job, [chunk], meta, deadline=Clock())
    assert len(values) >= 3
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_token_counts_share_one_deadline():
    from app.indexing.embed import split_to_limit

    class Counter:
        def __init__(self) -> None:
            self.calls = 0

        def count_tokens(self, text: str, deadline=None) -> int:
            self.calls += 1
            time.sleep(0.03)
            if deadline is not None:
                deadline.check()
            return len(text)

    counter = Counter()
    started = time.perf_counter()
    with pytest.raises(IndexFailure) as raised:
        split_to_limit(counter, "abcdefghij", 3, Deadline(0.1))
    assert raised.value.code == "timeout"
    assert counter.calls > 1
    assert time.perf_counter() - started < 0.25


def test_minio_slow_stream_does_not_reset_each_block(monkeypatch):
    from app.indexing import runtime

    class Response:
        reads = 0

        def read(self, _size):
            Response.reads += 1
            time.sleep(0.2)
            return b"abc"

        def close(self):
            return None

        def release_conn(self):
            return None

    class Client:
        def get_object(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(runtime, "_bounded_object_client", lambda *_a, **_k: Client())
    started = time.perf_counter()
    with pytest.raises(IndexFailure) as raised:
        runtime.fetch_object("bucket", "key", deadline=Deadline(0.35))
    assert raised.value.code == "timeout"
    assert Response.reads <= 2
    assert time.perf_counter() - started < 0.7


def test_expired_pipeline_does_not_write_after_the_deadline():
    import hashlib
    from pathlib import Path

    from app.indexing.pipeline import PIPELINE_VERSION, IndexJob, MemoryChunks, run_pipeline

    data = (Path(__file__).parent / "fixtures" / "native.pdf").read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    calls = {"upsert": 0, "replace": 0}

    class Objects:
        def get(self, *_args, **_kwargs):
            return data

    class Embedder:
        dimension = 384
        model_id = "fake"
        revision = "fake"
        checksum = "fake"
        max_tokens = 512

        def count_tokens(self, text, deadline=None):
            if deadline is not None:
                deadline.check()
            return 1

        def embed_batch(self, texts, deadline=None):
            time.sleep(0.25)
            if deadline is not None:
                deadline.check()
            return [[0.1] * 384 for _ in texts]

    class Vectors:
        def upsert(self, *_args, **_kwargs):
            calls["upsert"] += 1
            return 1

        def delete_generation(self, *_args, **_kwargs):
            return None

    class Chunks(MemoryChunks):
        def replace(self, job, chunks, meta, timeout_s=None, deadline=None):
            calls["replace"] += 1

    job = IndexJob(
        schema_version="index.v2",
        xa_id="xa",
        document_id=uuid.uuid4(),
        procedure_id=uuid.uuid4(),
        procedure_version_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        claim_token=uuid.uuid4(),
        generation_id=uuid.uuid4(),
        bucket="cas-documents",
        object_key="xa/documents/x/x.pdf",
        checksum=checksum,
        page_range=None,
        pipeline_version=PIPELINE_VERSION,
    )
    job.object_key = f"{job.xa_id}/documents/{job.document_id}/{checksum}.pdf"
    deps = type("D", (), {})()
    deps.objects = Objects()
    deps.ocr = type("O", (), {"version": "fake", "read_page": lambda *a, **k: "text"})()
    deps.embedder = Embedder()
    deps.vectors = Vectors()
    deps.chunks = Chunks()
    deps.renderer = type("R", (), {"render": staticmethod(lambda *_a, **_k: {})})()
    with pytest.raises(IndexFailure):
        run_pipeline(
            job,
            deps,
            max_bytes=200_000,
            chunk_config=__import__("app.indexing.chunk", fromlist=["ChunkConfig"]).ChunkConfig(32, 40, 4),
            bucket="cas-documents",
            deadline=Deadline(0.12),
        )
    assert calls == {"upsert": 0, "replace": 0}


def test_cleanup_after_partial_upsert_uses_a_separate_budget():
    from app.indexing.pipeline import PIPELINE_VERSION, FakeEmbedder, IndexJob, MemoryChunks, run_pipeline
    from pathlib import Path
    import hashlib

    data = (Path(__file__).parent / "fixtures" / "native.pdf").read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    seen = {}

    class Objects:
        def get(self, *_args, **_kwargs):
            return data

    class Vectors:
        def upsert(self, *_args, **_kwargs):
            raise IndexFailure("qdrant_failed")

        def delete_generation(self, *_args, deadline=None, **_kwargs):
            seen["left"] = deadline.remaining()

    job = IndexJob(
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
        checksum=checksum,
        page_range=None,
        pipeline_version=PIPELINE_VERSION,
    )
    job.object_key = f"{job.xa_id}/documents/{job.document_id}/{checksum}.pdf"
    deps = type("D", (), {})()
    deps.objects = Objects()
    deps.ocr = type("O", (), {"version": "fake", "read_page": lambda *a, **k: "unused"})()
    deps.embedder = FakeEmbedder()
    deps.vectors = Vectors()
    deps.chunks = MemoryChunks()
    deps.renderer = type("R", (), {"render": staticmethod(lambda *_a, **_k: {})})()
    from app.indexing.chunk import ChunkConfig

    with pytest.raises(IndexFailure):
        run_pipeline(job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents", deadline=Deadline(0.2))
    assert seen["left"] > 1


def test_onnx_recovers_after_a_later_crash_and_stops_a_loop():
    from app.indexing.embed import OnnxProcess

    ok = OnnxProcess("", behavior="crash")
    ok.recover_behavior = "echo"
    ok.start(3)
    try:
        assert ok.count_tokens("abcd") == 4
        ok._proc.kill()
        ok._proc.join(2)
        assert ok.count_tokens("z") == 1
        assert ok.spawn_count == 3
        assert ok.is_alive() is True
    finally:
        ok.shutdown()

    looping = OnnxProcess("", behavior="crash")
    looping.start(3)
    with pytest.raises(IndexFailure):
        looping.count_tokens("a")
    assert looping.spawn_count == 2
    with pytest.raises(IndexFailure):
        looping.count_tokens("b")
    assert looping.spawn_count == 2
    assert looping.is_alive() is False
    looping.shutdown()


def test_onnx_timeout_does_not_boot_a_fresh_timeout():
    from app.indexing.embed import OnnxProcess

    hung = OnnxProcess("", behavior="hang")
    hung.start(2)
    started = time.perf_counter()
    with pytest.raises(IndexFailure) as raised:
        hung.count_tokens("a", deadline=Deadline(0.25))
    assert raised.value.code == "timeout"
    assert hung.spawn_count == 1
    assert time.perf_counter() - started < 1
    hung.shutdown()


def test_onnx_lock_wait_expires_without_a_late_send():
    from app.indexing.embed import OnnxProcess

    proc = OnnxProcess("", behavior="echo")
    proc.max_respawns = 0
    proc.start(2)
    assert proc._lock.acquire(timeout=1)
    try:
        with pytest.raises(IndexFailure) as raised:
            proc.count_tokens("late", deadline=Deadline(0.2))
        assert raised.value.code == "timeout"
        assert proc._seq == 0
    finally:
        proc._lock.release()
        proc.shutdown()


def test_onnx_callers_do_not_overlap_the_pipe():
    from app.indexing.embed import OnnxProcess

    proc = OnnxProcess("", behavior="echo")
    proc.max_respawns = 0
    proc.start(2)
    state = {"n": 0, "max": 0}
    guard = threading.Lock()
    original = proc._exchange

    def wrapped(payload, timeout_s):
        with guard:
            state["n"] += 1
            state["max"] = max(state["max"], state["n"])
        try:
            return original(payload, timeout_s)
        finally:
            with guard:
                state["n"] -= 1

    proc._exchange = wrapped
    found = {}

    def count(text):
        found[text] = proc.count_tokens(text)

    threads = [threading.Thread(target=count, args=(text,)) for text in ("aa", "bbb", "cccc")]
    try:
        for item in threads:
            item.start()
        for item in threads:
            item.join(5)
        assert found == {"aa": 2, "bbb": 3, "cccc": 4}
        assert state["max"] == 1
    finally:
        proc.shutdown()


def test_dead_onnx_child_is_not_ready():
    from app.indexing import runtime
    from app.indexing.embed import OnnxProcess

    proc = OnnxProcess("", behavior="echo")
    proc.start(2)
    proc._proc.kill()
    proc._proc.join(2)
    settings = type("S", (), {"embedding_provider": "onnx", "ocr_provider": "fake"})()
    try:
        assert proc.is_alive() is False
        assert runtime._worker_problem(settings, (proc, None, None)) == "embedding_model_missing"
    finally:
        proc.shutdown()


def test_shutdown_kills_a_timed_out_worker_and_closes_the_gate():
    from app.indexing.embed import OnnxProcess
    from app.indexing.runtime import IndexGate

    proc = OnnxProcess("", behavior="hang")
    proc.max_respawns = 0
    proc.start(2)
    gate = IndexGate(1)

    def work():
        proc.count_tokens("a", timeout_s=30)

    future = gate.submit(work)
    time.sleep(0.15)
    started = time.perf_counter()
    pending = gate.shutdown(timeout_s=2, models=[proc])
    assert time.perf_counter() - started < 2
    assert pending == 0
    assert future.done()
    with pytest.raises(IndexFailure) as raised:
        gate.submit(lambda: "no")
    assert raised.value.code == "pipeline_busy"
