"""Bundle 6: killable jobs, Qdrant validation, OCR recovery, RunPod S3."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from app.indexing.errors import IndexFailure
from app.indexing.runtime import IndexGate


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_file(path: str, timeout: float = 1.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if os.path.exists(path):
            return True
        time.sleep(0.01)
    return False


def _gate():
    gate = IndexGate(1, grace_s=0.08, recovery_timeout_s=2)
    gate.start()
    return gate


def test_cancel_does_not_run_a_blocking_closer_on_the_caller():
    from app.indexing.deadline import Deadline

    def block():
        time.sleep(0.25)

    clock = Deadline(5)
    clock.bind(block)
    started = time.perf_counter()
    clock.cancel()
    assert time.perf_counter() - started < 0.05
    clock.cancel()


def test_parser_hang_is_killed_and_the_next_job_runs(tmp_path):
    gate = _gate()
    started = str(tmp_path / "started")
    wrote = str(tmp_path / "wrote")
    try:
        pid = gate.worker_pids()[0]
        begin = time.perf_counter()
        with pytest.raises(IndexFailure) as raised:
            gate.call({"op": "hang", "stage": "parser", "seconds": 0.35, "started_path": started, "wrote_path": wrote}, 0.05)
        assert raised.value.code == "timeout"
        assert time.perf_counter() - begin < 0.5
        assert os.path.exists(started)
        assert os.path.exists(wrote) is False
        assert gate.pending() == 0
        assert _pid_alive(pid) is False
        assert not any(item.daemon and item.name == "index-pipeline" for item in threading.enumerate())
        assert gate.call({"op": "ping"}, 2)["event"] == "pong"
    finally:
        gate.shutdown()


@pytest.mark.parametrize("stage", ["renderer", "ocr", "embedding"])
def test_stage_hang_releases_the_gate(stage):
    gate = _gate()
    try:
        with pytest.raises(IndexFailure) as raised:
            gate.call({"op": "hang", "stage": stage, "seconds": 0.35}, 0.05)
        assert raised.value.code == "timeout"
        assert gate.pending() == 0
        assert gate.call({"op": "ping"}, 2)["pid"]
    finally:
        gate.shutdown()


def test_ipc_receive_hang_is_killed(tmp_path):
    gate = _gate()
    started = str(tmp_path / "started")
    try:
        pid = gate.worker_pids()[0]
        with pytest.raises(IndexFailure):
            gate.call({"op": "ipc_block", "started_path": started}, 0.05)
        assert os.path.exists(started)
        assert gate.pending() == 0
        assert _pid_alive(pid) is False
    finally:
        gate.shutdown()


@pytest.mark.parametrize("stage", ["postgres", "qdrant", "minio"])
def test_network_hang_is_killed(stage, tmp_path):
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    stop = threading.Event()

    def accept():
        try:
            conn, _ = held.accept()
        except OSError:
            return
        stop.wait(2)
        conn.close()

    threading.Thread(target=accept).start()
    gate = _gate()
    started = str(tmp_path / "started")
    try:
        with pytest.raises(IndexFailure):
            gate.call(
                {
                    "op": "socket_block",
                    "stage": stage,
                    "host": "127.0.0.1",
                    "port": held.getsockname()[1],
                    "started_path": started,
                },
                0.05,
            )
        assert os.path.exists(started)
        assert gate.pending() == 0
    finally:
        stop.set()
        held.close()
        gate.shutdown()


@pytest.mark.parametrize("stage", ["parser", "renderer", "ocr", "embedding", "postgres", "qdrant", "minio"])
def test_shutdown_during_stage_reaps_the_job(stage, tmp_path):
    gate = _gate()
    started = str(tmp_path / "started")
    op = "socket_block" if stage in ("postgres", "qdrant", "minio") else "hang"
    held = None
    stop = threading.Event()
    message = {"op": op, "stage": stage, "seconds": 5, "started_path": started}
    if op == "socket_block":
        held = socket.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)

        def accept():
            try:
                conn, _ = held.accept()
            except OSError:
                return
            stop.wait(2)
            try:
                conn.close()
            except OSError:
                return

        threading.Thread(target=accept).start()
        message.update({"host": "127.0.0.1", "port": held.getsockname()[1]})

    def run():
        try:
            gate.call(message, 5)
        except IndexFailure:
            return

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert _wait_file(started)
        begin = time.perf_counter()
        pending = gate.shutdown(timeout_s=1)
        thread.join(1)
        assert time.perf_counter() - begin < 1
        assert pending == 0
        assert thread.is_alive() is False
        assert gate.pending() == 0
    finally:
        stop.set()
        if held is not None:
            held.close()
        gate.shutdown()


def test_client_disconnect_terminates_the_job(tmp_path):
    import httpx

    from app import main
    from tests.conftest import AUTH_HEADER

    mode = main._settings.index_mode
    timeout = main._settings.pipeline_timeout_s
    gate = IndexGate(1, grace_s=0.08, recovery_timeout_s=2)
    gate.start()
    started = str(tmp_path / "started")
    release = str(tmp_path / "release")
    gate.arm_hold(started, release, 5)
    main._gate = gate
    object.__setattr__(main._settings, "index_mode", "pipeline")
    object.__setattr__(main._settings, "pipeline_timeout_s", 2)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            task = asyncio.create_task(
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
            for _ in range(100):
                if os.path.exists(started):
                    break
                await asyncio.sleep(0.01)
            assert os.path.exists(started)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        return gate.pending()

    try:
        pending = asyncio.run(scenario())
    finally:
        open(release, "a", encoding="utf-8").close()
        gate.shutdown()
        main._gate = IndexGate(main._settings.index_max_inflight)
        object.__setattr__(main._settings, "index_mode", mode)
        object.__setattr__(main._settings, "pipeline_timeout_s", timeout)
    assert pending == 0


def test_ready_is_503_while_the_worker_recovers():
    from app import main
    from app.indexing.runtime import current_health

    mode = main._settings.index_mode
    gate = _gate()
    try:
        gate._status = "recovering"
        object.__setattr__(main._settings, "index_mode", "pipeline")
        main._gate = gate
        health = current_health(main._settings, None, force=True, gate=gate)
        ready = main.ready()
        assert health["ready"] is False
        assert health["reason"] == "worker_recovering"
        assert ready.status_code == 503
    finally:
        object.__setattr__(main._settings, "index_mode", mode)
        gate.shutdown()
        main._gate = IndexGate(main._settings.index_max_inflight)


def test_qdrant_rejects_malformed_bodies():
    from app.indexing.qdrant import QdrantWriter

    bodies = [
        b"not-json",
        b"[1,2]",
        b'{"status":"error","result":{"status":"completed"}}',
        b'{"result":{"status":"completed"}}',
        b'{"status":"ok","result":[]}',
        b'{"status":"ok","result":{"count":false}}',
        b'{"status":"ok","result":{"count":1.5}}',
        b'{"status":"ok","result":{"count":"0"}}',
    ]

    def run(content: bytes):
        def handler(_request):
            return httpx.Response(200, content=content)

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

    for body in bodies:
        run(body)


def test_cleanup_malformed_response_keeps_the_original_error():
    from app.indexing.chunk import ChunkConfig
    from app.indexing.pipeline import FakeEmbedder, MemoryObjects, run_pipeline
    from tests.test_p4b_pipeline import FIXTURES, _job

    class Vectors:
        def upsert(self, *_args, **_kwargs):
            return 1

        def delete_generation(self, *_args, **_kwargs):
            raise json.JSONDecodeError("bad", "doc", 0)

    payload = (FIXTURES / "native.pdf").read_bytes()
    job = _job(_data=payload)
    deps = type("Deps", (), {})()
    deps.objects = MemoryObjects(payload)
    deps.ocr = type("O", (), {"version": "fake", "read_page": staticmethod(lambda *_a, **_k: "")})()
    deps.embedder = FakeEmbedder()
    deps.vectors = Vectors()
    deps.chunks = type("C", (), {"replace": staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(IndexFailure("postgres_failed")))})()
    deps.renderer = type("R", (), {"render": staticmethod(lambda *_a, **_k: {})})()
    with pytest.raises(IndexFailure) as raised:
        run_pipeline(job, deps, max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents")
    assert raised.value.code == "postgres_failed"


def test_ocr_timeout_recovers_on_its_own_budget():
    from app.indexing.ocr_supervisor import OcrSupervisor

    sup = OcrSupervisor(1, {"kind": "fake", "behavior": "hang", "recover": "ok", "replace_timeout_s": 2})
    sup.start(2)
    try:
        with pytest.raises(IndexFailure) as raised:
            sup.read_page(None, 1, timeout_s=0.15)
        assert raised.value.code == "ocr_timeout"
        assert sup.status in ("recovering", "ready")
        sup.wait_settled(2)
        assert sup.status == "ready"
        assert sup.spawn_count == 2
        assert sup.accepting() is True
        assert sup.read_page(None, 2, timeout_s=2) == "ok"
    finally:
        sup.shutdown()

    exhausted = OcrSupervisor(1, {"kind": "fake", "behavior": "hang", "recover": "init_hang", "replace_timeout_s": 0.2})
    exhausted.start(2)
    try:
        with pytest.raises(IndexFailure):
            exhausted.read_page(None, 1, timeout_s=0.1)
        assert exhausted.status == "recovering"
        exhausted.wait_settled(2)
        assert exhausted.status == "failed"
        assert exhausted.accepting() is False
        assert exhausted.spawn_count == 2
    finally:
        exhausted.shutdown()
        assert exhausted.alive_workers() == 0
        assert exhausted._recovery_thread is None or exhausted._recovery_thread.is_alive() is False


def test_runpod_storage_contract(caplog, monkeypatch):
    from app.config import ConfigError
    from app.indexing.object_storage import ensure_bucket, normalize_endpoint, open_client, storage_config

    assert normalize_endpoint("https://s3api-eu-ro-1.runpod.io", False) == ("s3api-eu-ro-1.runpod.io:443", True)
    assert normalize_endpoint("minio:9000", False) == ("minio:9000", False)
    assert normalize_endpoint("http://127.0.0.1:9000", True) == ("127.0.0.1:9000", False)
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "https://s3api-eu-ro-1.runpod.io")
    monkeypatch.delenv("OBJECT_STORAGE_REGION", raising=False)
    with pytest.raises(ConfigError) as missing:
        storage_config()
    assert "OBJECT_STORAGE_REGION" in str(missing.value)

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def _ok(self):
            seen.append((self.command, self.path))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_HEAD = _ok
        do_GET = _ok
        do_PUT = _ok

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", f"127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("OBJECT_STORAGE_REGION", "eu-ro-1")
    monkeypatch.setenv("OBJECT_STORAGE_USE_SSL", "false")
    monkeypatch.setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "false")
    monkeypatch.setenv("OBJECT_STORAGE_ACCESS_KEY", "placeholder-access")
    monkeypatch.setenv("OBJECT_STORAGE_SECRET_KEY", "placeholder-secret")
    try:
        with caplog.at_level(logging.DEBUG):
            cfg = storage_config()
            client = open_client(config=cfg)
            assert client._base_url.region == "eu-ro-1"
            assert ensure_bucket(client, "network-volume", False) is True
        joined = json.dumps(seen)
        assert "PUT" not in joined
        assert "location" not in joined
        assert "placeholder-secret" not in caplog.text
        assert "placeholder-secret" not in joined
    finally:
        server.shutdown()
        server.server_close()
