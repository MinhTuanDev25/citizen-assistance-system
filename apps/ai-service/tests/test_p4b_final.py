"""Final P4B regressions: CPU flags, worker recovery, data-path deadlines, packaging."""

from __future__ import annotations

import hashlib
import os
import socket
import stat
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path

import pytest

from app.indexing.embed import OnnxProcess, detect_cpu_flags, require_supported_variant
from app.indexing.errors import IndexFailure
from app.indexing.ocr_supervisor import OcrSupervisor

ROOT = Path(__file__).resolve().parents[3]
LINUX_OK = "flags : fpu avx512f avx512dq avx512bw avx512vl avx512_vnni\n"
LINUX_ALIAS = "flags : avx512f avx512bw avx512dq avx512vl avx512vnni\n"
LINUX_NO_VNNI = "flags : avx512f avx512dq avx512bw avx512vl sse2\n"


def test_linux_cpuinfo_avx512_vnni_is_detected_by_the_parser():
    parsed = detect_cpu_flags(cpuinfo_text="processor : 0\n" + LINUX_OK)
    assert "avx512vnni" in parsed
    assert "avx512_vnni" not in parsed
    assert require_supported_variant(parsed) == "qint8_avx512_vnni"
    alias = detect_cpu_flags(cpuinfo_text=LINUX_ALIAS)
    assert require_supported_variant(alias) == "qint8_avx512_vnni"
    missing = detect_cpu_flags(cpuinfo_text=LINUX_NO_VNNI)
    with pytest.raises(IndexFailure) as raised:
        require_supported_variant(missing)
    assert raised.value.code == "embedding_cpu_unsupported"


def test_cpu_override_is_explicit_and_does_not_replace_the_parser(monkeypatch):
    monkeypatch.setenv("EMBEDDING_CPU_FLAGS", "avx512vnni")
    parsed = detect_cpu_flags(cpuinfo_text=LINUX_NO_VNNI)
    assert "avx512vnni" not in parsed
    override = detect_cpu_flags()
    assert "avx512vnni" in override
    with pytest.raises(IndexFailure):
        require_supported_variant(override)


def test_onnx_pipe_serializes_count_and_embed():
    proc = OnnxProcess("", behavior="echo")
    proc.max_respawns = 0
    proc.start(3)
    try:
        texts = [f"item-{index}-" + ("x" * index) for index in range(6)]
        found = {}

        def count(text):
            found[text] = proc.count_tokens(text)

        threads = [threading.Thread(target=count, args=(text,)) for text in texts]
        for item in threads:
            item.start()
        for item in threads:
            item.join(5)
        assert found == {text: len(text) for text in texts}

        box = {}

        def embed():
            box["vectors"] = proc.embed_batch(["ab", "cdef"])

        def other():
            box["n"] = proc.count_tokens("hello")

        pair = [threading.Thread(target=embed), threading.Thread(target=other)]
        for item in pair:
            item.start()
        for item in pair:
            item.join(5)
        assert box["n"] == len("hello")
        assert len(box["vectors"]) == 2
        assert all(len(vector) == 384 for vector in box["vectors"])
        assert proc.spawn_count == 1
    finally:
        proc.shutdown()


def test_onnx_timeout_crash_eof_and_broken_pipe_are_reaped():
    hung = OnnxProcess("", behavior="hang")
    hung.max_respawns = 0
    hung.start(3)
    started = time.perf_counter()
    with pytest.raises(IndexFailure) as timed:
        hung._call({"op": "count", "text": "a"}, 0.3)
    assert timed.value.code == "timeout"
    assert time.perf_counter() - started < 3
    assert hung.status == "timed_out"
    assert hung.reaped >= 1
    assert hung.is_alive() is False
    hung.shutdown()

    for behavior in ("crash", "eof", "broken", "malformed"):
        proc = OnnxProcess("", behavior=behavior)
        proc.max_respawns = 0
        proc.start(3)
        with pytest.raises(IndexFailure):
            proc.count_tokens("ab")
        assert proc.status in ("failed", "timed_out")
        assert proc.reaped >= 1
        assert proc.is_alive() is False
        proc.shutdown()
        assert proc.spawn_count == 1


def test_onnx_respawn_succeeds_once_and_stops_when_recovery_fails():
    ok = OnnxProcess("", behavior="crash")
    ok.recover_behavior = "echo"
    ok.start(3)
    try:
        assert ok.count_tokens("abcd") == 4
        assert ok.status == "ready"
        assert ok.is_alive() is True
        assert ok.spawn_count == 2
        assert ok.count_tokens("z") == 1
        assert ok.spawn_count == 2
    finally:
        ok.shutdown()

    bad = OnnxProcess("", behavior="crash")
    bad.recover_behavior = "init_crash"
    bad.start(3)
    with pytest.raises(IndexFailure):
        bad.count_tokens("ab")
    assert bad.status == "failed"
    assert bad.spawn_count == 2
    assert bad.is_alive() is False
    bad.shutdown()


def test_ocr_replacement_timeout_crash_and_shutdown_reap_workers():
    hung = OcrSupervisor(1, {"kind": "fake", "behavior": "hang", "recover": "init_hang", "replace_timeout_s": 0.3})
    hung.start(2)
    with pytest.raises(IndexFailure):
        hung.read_page(None, 1, timeout_s=0.2)
    hung.wait_settled(2)
    assert hung.status == "failed"
    assert hung.alive_workers() == 0
    assert hung.spawn_count == 2
    hung.shutdown()

    crashed = OcrSupervisor(1, {"kind": "fake", "behavior": "crash", "recover": "ok", "replace_timeout_s": 2})
    crashed.start(2)
    try:
        with pytest.raises(IndexFailure) as raised:
            crashed.read_page(None, 1, timeout_s=2)
        assert raised.value.code == "ocr_failed"
        assert crashed.read_page(None, 1, timeout_s=2) == "ok"
        assert crashed.alive_workers() == 1
        assert crashed.spawn_count == 2
    finally:
        crashed.shutdown()

    active = OcrSupervisor(1, {"kind": "fake", "behavior": "hang", "replace_timeout_s": 2})
    active.start(2)
    error = {}

    def call():
        try:
            active.read_page(None, 1, timeout_s=3)
        except IndexFailure as exc:
            error["code"] = exc.code

    thread = threading.Thread(target=call)
    thread.start()
    time.sleep(0.2)
    assert active.active_count() == 1
    active.shutdown()
    thread.join(3)
    assert thread.is_alive() is False
    assert active.alive_workers() == 0
    assert active.status == "stopped"

    limited = OcrSupervisor(1, {"kind": "fake", "behavior": "hang", "recover": "hang", "replace_timeout_s": 2})
    limited.start(2)
    try:
        for _ in range(3):
            with pytest.raises(IndexFailure):
                limited.read_page(None, 1, timeout_s=0.2)
            limited.wait_settled(2)
        assert limited.spawn_count == 4
        assert limited.alive_workers() == 1
    finally:
        limited.shutdown()
        assert limited.alive_workers() == 0


def test_one_ocr_engine_serves_one_call():
    sup = OcrSupervisor(1, {"kind": "fake", "behavior": "slow", "delay_s": 0.4})
    sup.start(2)
    try:
        codes = []

        def call():
            try:
                sup.read_page(None, 1, timeout_s=2)
                codes.append("ok")
            except IndexFailure as exc:
                codes.append(exc.code)

        threads = [threading.Thread(target=call) for _ in range(2)]
        for item in threads:
            item.start()
        for item in threads:
            item.join(3)
        assert sorted(codes) == ["ok", "pipeline_busy"]
        assert sup.active_count() == 0
        assert sup.alive_workers() == 1
    finally:
        sup.shutdown()


def test_ready_follows_worker_health(monkeypatch):
    from app import main
    from app.indexing import runtime

    class Embed:
        def __init__(self) -> None:
            self.status = "failed"
            self.variant = "qint8_avx512_vnni"
            self.cpu_compatible = True
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

    class Settings:
        index_mode = "pipeline"
        embedding_provider = "onnx"
        ocr_provider = "fake"
        database_url = "postgres://cas:cas@127.0.0.1/cas"
        qdrant_url = "http://qdrant:6333"
        qdrant_collection = "knowledge_chunks"
        object_bucket = "cas-documents"
        service_token = "test-service-token"
        provider = "mock"
        model = "mock"

        def provider_ready(self):
            return True, None

        def status_dict(self):
            return {
                "provider": "mock",
                "model": "mock",
                "ready": True,
                "reason": None,
                "rate_per_minute": 1,
                "max_inflight": 1,
                "index_mode": self.index_mode,
                "embedding_provider": self.embedding_provider,
                "ocr_provider": self.ocr_provider,
                "models_loaded": False,
                "postgres_configured": True,
                "qdrant_configured": True,
                "object_storage_configured": True,
            }

    embed = Embed()
    settings = Settings()
    monkeypatch.setattr(runtime, "_postgres", lambda _dsn: True)
    monkeypatch.setattr(runtime, "_qdrant", lambda *_args: None)
    monkeypatch.setattr(runtime, "_minio", lambda _settings: True)
    runtime._HEALTH["ready"] = None
    runtime._HEALTH["monotonic"] = 0
    failed = runtime.current_health(settings, (embed, None, None), force=True)
    assert failed["ready"] is False
    assert failed["reason"] == "embedding_model_missing"
    assert failed["embedding_status"] == "failed"

    embed.status = "ready"
    embed._alive = True
    recovered = runtime.current_health(settings, (embed, None, None), force=True)
    assert recovered["ready"] is True
    assert recovered["cpu_compatible"] is True
    assert recovered["embedding_variant"] == "qint8_avx512_vnni"

    embed.status = "failed"
    embed._alive = False
    stale = runtime.current_health(settings, (embed, None, None))
    assert stale["ready"] is False

    pool = OcrSupervisor(1, {"kind": "fake", "behavior": "ok"})
    pool.start(2)
    ocr_settings = Settings()
    ocr_settings.embedding_provider = "fake"
    ocr_settings.ocr_provider = "paddle"
    try:
        up = runtime.current_health(ocr_settings, (None, pool, None), force=True)
        assert up["ready"] is True
        assert up["ocr_workers_alive"] == 1
        pool.shutdown()
        down = runtime.current_health(ocr_settings, (None, pool, None), force=True)
        assert down["ready"] is False
        assert down["ocr_status"] == "stopped"
    finally:
        pool.shutdown()

    monkeypatch.setattr(main, "_models", (embed, None, None))
    monkeypatch.setattr(main, "_settings", settings)
    ready = main.ready()
    status = main.status()
    assert ready.status_code == 503
    assert status["ready"] is False
    assert status["embedding_status"] == "failed"
    assert "database_url" not in status
    assert "postgres://" not in str(status["reason"])


def test_minio_data_path_times_out_on_connect_and_read():
    from app.indexing import runtime

    runtime._DATA_MINIO.clear()
    started = time.perf_counter()
    with pytest.raises(IndexFailure) as refused:
        runtime.fetch_object("cas-documents", "missing.pdf", timeout_s=0.4)
    assert refused.value.code in ("source_timeout", "source_not_found")
    assert "secret" not in str(refused.value)
    assert time.perf_counter() - started < 3

    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    release = threading.Event()

    def accept_and_stall():
        conn, _addr = held.accept()
        release.wait(3)
        conn.close()

    threading.Thread(target=accept_and_stall, daemon=True).start()
    monkey_endpoint = f"127.0.0.1:{port}"
    old = runtime.os.environ.get("OBJECT_STORAGE_ENDPOINT")
    runtime.os.environ["OBJECT_STORAGE_ENDPOINT"] = monkey_endpoint
    runtime.os.environ.setdefault("OBJECT_STORAGE_ACCESS_KEY", "minioadmin")
    runtime.os.environ.setdefault("OBJECT_STORAGE_SECRET_KEY", "minioadmin")
    runtime._DATA_MINIO.clear()
    started = time.perf_counter()
    try:
        with pytest.raises(IndexFailure) as stalled:
            runtime.fetch_object("cas-documents", "doc.pdf", timeout_s=0.4)
        assert stalled.value.code == "source_timeout"
        assert time.perf_counter() - started < 3
    finally:
        release.set()
        held.close()
        if old is None:
            runtime.os.environ.pop("OBJECT_STORAGE_ENDPOINT", None)
        else:
            runtime.os.environ["OBJECT_STORAGE_ENDPOINT"] = old
        runtime._DATA_MINIO.clear()


def test_postgres_timeouts_do_not_leak_the_dsn(monkeypatch):
    from app.indexing.postgres import PostgresChunks

    seen = {}

    class Broken(Exception):
        pass

    def connect(dsn, **kwargs):
        seen["dsn"] = dsn
        seen["connect_timeout"] = kwargs.get("connect_timeout")
        raise Broken("password=super-secret host=db.internal")

    monkeypatch.setattr("psycopg.connect", connect, raising=False)
    import psycopg

    monkeypatch.setattr(psycopg, "connect", connect)
    store = PostgresChunks("postgresql://cas:super-secret@127.0.0.1/cas")
    with pytest.raises(IndexFailure) as raised:
        store.replace(None, [], {}, timeout_s=5)
    assert raised.value.code == "postgres_failed"
    assert "super-secret" not in str(raised.value)
    assert seen["connect_timeout"] == 2


def test_postgres_sets_statement_timeout(monkeypatch):
    from app.indexing.postgres import PostgresChunks

    statements = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=None):
            statements.append((sql, params))
            if "set_config" in sql:
                return
            raise TimeoutError("canceling statement due to statement timeout")

        rowcount = 1

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

        def commit(self):
            statements.append(("commit", None))

        def rollback(self):
            statements.append(("rollback", None))

    monkeypatch.setitem(__import__("sys").modules, "psycopg", type("M", (), {"connect": staticmethod(lambda *a, **k: Conn())}))
    import psycopg

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: Conn())
    job = type(
        "J",
        (),
        {
            "generation_id": uuid.uuid4(),
            "xa_id": "xa",
            "document_id": uuid.uuid4(),
            "procedure_version_id": uuid.uuid4(),
            "job_id": uuid.uuid4(),
        },
    )()
    meta = {key: "x" for key in (
        "pipeline_version", "extraction_version", "ocr_version", "chunk_config_hash",
        "embedding_model_id", "embedding_revision", "embedding_checksum", "source_sha256",
        "content_sha256", "manifest_hash",
    )}
    meta.update({
        "vector_dimension": 384,
        "page_count": 1,
        "native_page_count": 1,
        "ocr_page_count": 0,
        "chunk_count": 0,
        "vector_count": 0,
    })
    store = PostgresChunks("postgresql://cas:hidden@127.0.0.1/cas")
    with pytest.raises(IndexFailure) as raised:
        store.replace(job, [], meta, timeout_s=5)
    assert raised.value.code == "postgres_failed"
    assert statements[0][0].startswith("SELECT set_config")
    assert int(statements[0][1][0]) <= 5000
    assert ("rollback", None) in statements
    assert "hidden" not in str(raised.value)


def test_qdrant_timeout_and_pipeline_deadline_shrinks(monkeypatch):
    import httpx

    from app.indexing.pipeline import PIPELINE_VERSION, FakeEmbedder, IndexJob, MemoryChunks, run_pipeline
    from app.indexing.qdrant import QdrantWriter

    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    release = threading.Event()

    def stall():
        conn, _addr = held.accept()
        release.wait(3)
        conn.close()

    threading.Thread(target=stall, daemon=True).start()
    writer = QdrantWriter(f"http://127.0.0.1:{port}", "knowledge_chunks", timeout_s=0.4)
    started = time.perf_counter()
    with pytest.raises(IndexFailure) as raised:
        writer.delete_generation(uuid.uuid4())
    assert raised.value.code == "qdrant_failed"
    assert time.perf_counter() - started < 3
    release.set()
    held.close()
    assert httpx.TimeoutException

    data = (Path(__file__).parent / "fixtures" / "native.pdf").read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
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

    class Objects:
        def __init__(self) -> None:
            self.left = None

        def get(self, _bucket, _key, _max_bytes=None, timeout_s=None, deadline=None):
            time.sleep(0.15)
            self.left = deadline.remaining()
            return data

    class Vectors:
        def __init__(self) -> None:
            self.left = None

        def upsert(self, *_args, deadline=None, **_kwargs):
            self.left = deadline.remaining()
            return 1

        def delete_generation(self, *_args, deadline=None, **_kwargs):
            return None

    class Chunks(MemoryChunks):
        def __init__(self) -> None:
            super().__init__()
            self.left = None

        def replace(self, job, chunks, meta, timeout_s=None, deadline=None):
            self.left = deadline.remaining()
            super().replace(job, chunks, meta)

    objects = Objects()
    chunks = Chunks()
    vectors = Vectors()
    deps = type("Deps", (), {})()
    deps.objects = objects
    deps.ocr = type("O", (), {"version": "fake", "read_page": lambda *a, **k: "unused"})()
    deps.embedder = FakeEmbedder()
    deps.vectors = vectors
    deps.chunks = chunks
    deps.renderer = type("R", (), {"render": staticmethod(lambda *_a, **_k: {})})()
    from app.indexing.chunk import ChunkConfig

    run_pipeline(job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents", deadline_s=5)
    assert objects.left < 5
    assert vectors.left < objects.left
    assert chunks.left < vectors.left


def test_index_gate_releases_the_slot_and_reports_shutdown():
    from app.indexing.runtime import IndexGate

    gate = IndexGate(1, grace_s=0.08, recovery_timeout_s=2)
    gate.start()
    try:
        with pytest.raises(IndexFailure) as timed:
            gate.call({"op": "hang", "stage": "parser", "seconds": 0.4}, 0.05)
        assert timed.value.code == "timeout"
        assert gate.pending() == 0
        assert gate.call({"op": "ping"}, 2)["event"] == "pong"
        started = "/tmp/cas-index-gate-started"
        try:
            os.remove(started)
        except OSError:
            pass

        def block():
            try:
                gate.call({"op": "hang", "stage": "parser", "seconds": 5, "started_path": started}, 5)
            except IndexFailure:
                return

        worker = threading.Thread(target=block)
        worker.start()
        end = time.monotonic() + 1
        while time.monotonic() < end and not os.path.exists(started):
            time.sleep(0.01)
        assert os.path.exists(started)
        pending = gate.shutdown(timeout_s=1)
        worker.join(1)
        assert pending == 0
        assert worker.is_alive() is False
        assert gate.pending() == 0
    finally:
        gate.shutdown()


def test_check_zip_rejects_unsafe_archives(tmp_path):
    script = ROOT / "scripts" / "check-zip.sh"

    def run(path: Path) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(script), str(path)], capture_output=True, text=True)

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    bad = run(corrupt)
    assert bad.returncode != 0
    assert "corrupt" in bad.stderr

    weight = tmp_path / "weight.zip"
    with zipfile.ZipFile(weight, "w") as archive:
        archive.writestr("model.onnx", b"weight")
    bad = run(weight)
    assert bad.returncode != 0
    assert "model.onnx" in bad.stderr

    travel = tmp_path / "travel.zip"
    with zipfile.ZipFile(travel, "w") as archive:
        info = zipfile.ZipInfo("../escape.txt")
        archive.writestr(info, b"x")
    bad = run(travel)
    assert bad.returncode != 0
    assert "traversal" in bad.stderr

    link = tmp_path / "link.zip"
    with zipfile.ZipFile(link, "w") as archive:
        info = zipfile.ZipInfo("linked.txt")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"target")
    bad = run(link)
    assert bad.returncode != 0
    assert "symlink" in bad.stderr

    duplicate = tmp_path / "dup.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("same.txt", b"a")
        archive.writestr("same.txt", b"b")
    bad = run(duplicate)
    assert bad.returncode != 0
    assert "duplicate" in bad.stderr


def test_bundle_entry_point_writes_checksums_and_sidecar(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from p4b_bundle import build_bundle

    package = tmp_path / "src"
    (package / "apps" / "ai-service" / "app" / "models").mkdir(parents=True)
    (package / "docs" / "models").mkdir(parents=True)
    (package / "apps" / "ai-service" / "app" / "models" / "extract.py").write_text("x = 1\n", encoding="utf-8")
    (package / "docs" / "models" / "embedding.manifest.example.json").write_text("{}\n", encoding="utf-8")
    (package / "skip.onnx").write_bytes(b"weight")
    dest = tmp_path / "bundle.zip"
    side = tmp_path / "bundle.zip.sha256"
    digest = build_bundle(package, dest, side)
    assert digest == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert side.read_text(encoding="utf-8").startswith(digest)
    with zipfile.ZipFile(dest) as archive:
        names = set(archive.namelist())
    assert "apps/ai-service/app/models/extract.py" in names
    assert "docs/models/embedding.manifest.example.json" in names
    assert "skip.onnx" not in names
    assert "SHA256SUMS.txt" in names
    monkeypatch.chdir(tmp_path)
