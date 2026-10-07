"""Bundle 7: process-tree kill, real ASGI disconnect, scoped cancel, live OCR, RunPod TLS."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
import uuid

import pytest

from app.indexing.errors import IndexFailure
from app.indexing.procgroup import process_alive
from app.indexing.supervisor import IndexGate


def _alive(pid: int) -> bool:
    return process_alive(pid)


def _wait_file(path: str, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if os.path.exists(path):
            return True
        time.sleep(0.02)
    return False


def _pids(path: str) -> list[int]:
    return [int(line) for line in open(path, encoding="utf-8") if line.strip()]


def _expire(_signum, _frame):
    raise TimeoutError("bundle7 test exceeded 12s")


@pytest.fixture(autouse=True)
def _bound_test_runtime():
    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, 12)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _gate(limit: int = 1, kind: str = "stage") -> IndexGate:
    gate = IndexGate(limit, grace_s=0.15, recovery_timeout_s=3, kind=kind)
    gate.start(4)
    return gate


def _index_body() -> dict:
    return {
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
    }


def test_timeout_kills_the_pipeline_process_tree(tmp_path):
    gate = _gate(kind="pipeline")
    pids_path = str(tmp_path / "pids")
    wrote = str(tmp_path / "wrote")
    try:
        begin = time.perf_counter()
        with pytest.raises(IndexFailure) as raised:
            gate.call(
                {"op": "nested_hang", "seconds": 30, "pids_path": pids_path, "wrote_path": wrote},
                4,
            )
        assert raised.value.code == "timeout"
        assert time.perf_counter() - begin < 5
        pids = _pids(pids_path)
        assert len(pids) >= 4
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
            time.sleep(0.02)
        assert [pid for pid in pids if _alive(pid)] == []
        assert gate.pending() == 0
        assert os.path.exists(wrote) is False
        assert gate.call({"op": "ping"}, 3)["event"] == "pong"
        assert not any(item.daemon and item.name == "index-pipeline" for item in threading.enumerate())
    finally:
        gate.shutdown()


def test_shutdown_kills_the_pipeline_process_tree(tmp_path):
    gate = _gate(kind="pipeline")
    pids_path = str(tmp_path / "pids")
    handle = None
    try:
        handle = gate.submit({"op": "nested_hang", "seconds": 30, "pids_path": pids_path}, 8)

        def _wait():
            try:
                gate.wait(handle, 8)
            except IndexFailure:
                return

        thread = threading.Thread(target=_wait)
        thread.start()
        assert _wait_file(pids_path)
        pids = _pids(pids_path)
        begin = time.perf_counter()
        assert gate.shutdown(timeout_s=2) == 0
        thread.join(2)
        assert time.perf_counter() - begin < 2
        assert thread.is_alive() is False
        assert [pid for pid in pids if _alive(pid)] == []
        assert gate.pending() == 0
    finally:
        if gate.status != "stopped":
            gate.shutdown()


def test_cancel_is_scoped_to_one_request(tmp_path):
    gate = _gate(2)
    started_a = str(tmp_path / "a")
    started_b = str(tmp_path / "b")
    try:
        handle_a = gate.submit({"op": "hang", "seconds": 5, "started_path": started_a}, 5)
        handle_b = gate.submit({"op": "hang", "seconds": 0.3, "started_path": started_b}, 5)
        assert _wait_file(started_a) and _wait_file(started_b)
        pid_a = handle_a.identity[0]
        pid_b = handle_b.identity[0]
        assert gate.pending() == 2
        gate.cancel(handle_a)
        assert _alive(pid_a) is False
        assert _alive(pid_b) is True
        reply = gate.wait(handle_b, 2)
        assert reply["event"] == "result"
        assert gate.pending() == 0
        replacement = [pid for pid in gate.worker_pids() if pid != pid_b]
        assert replacement
        gate.cancel(handle_a)
        assert _alive(replacement[0]) is True
        gate._groups.kill(handle_a.identity, 0.1)
        assert _alive(replacement[0]) is True
    finally:
        gate.shutdown()


def test_two_cancels_do_not_cross_kill(tmp_path):
    gate = _gate(2)
    try:
        handle_a = gate.submit({"op": "hang", "seconds": 5, "started_path": str(tmp_path / "a")}, 5)
        handle_b = gate.submit({"op": "hang", "seconds": 5, "started_path": str(tmp_path / "b")}, 5)
        assert _wait_file(str(tmp_path / "a")) and _wait_file(str(tmp_path / "b"))
        pid_a = handle_a.identity[0]
        pid_b = handle_b.identity[0]
        gate.cancel(handle_a)
        gate.cancel(handle_b)
        assert _alive(pid_a) is False
        assert _alive(pid_b) is False
        assert os.getpid()
        gate.cancel(handle_a)
        gate.cancel(handle_b)
        assert gate.pending() == 0
    finally:
        gate.shutdown()


def test_timeout_and_shutdown_together(tmp_path):
    gate = _gate()
    started = str(tmp_path / "started")
    try:
        handle = gate.submit({"op": "hang", "seconds": 5, "started_path": started}, 5)

        def _wait():
            try:
                gate.wait(handle, 0.05)
            except IndexFailure:
                return

        thread = threading.Thread(target=_wait)
        thread.start()
        assert _wait_file(started)
        gate.shutdown(timeout_s=2)
        thread.join(2)
        assert thread.is_alive() is False
        assert gate.pending() == 0
        assert [pid for pid in gate.worker_pids() if _alive(pid)] == []
    finally:
        if gate.status != "stopped":
            gate.shutdown()


def test_http_disconnect_cancels_only_that_call(tmp_path):
    import json as jsonlib

    from app import main

    token = os.environ["AI_SERVICE_TOKEN"]

    mode = main._settings.index_mode
    previous = main._gate
    gate = _gate(2)
    started = str(tmp_path / "started")
    release = str(tmp_path / "release")
    other = str(tmp_path / "other")
    gate.arm_hold(started, release, 5)
    main._gate = gate
    object.__setattr__(main._settings, "index_mode", "pipeline")
    raw = jsonlib.dumps(_index_body()).encode()

    async def scenario():
        disconnect = asyncio.Event()
        sent = {"body": False}
        messages = []

        async def receive():
            if not sent["body"]:
                sent["body"] = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/index",
            "raw_path": b"/v1/index",
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "client": ("127.0.0.1", 9),
            "server": ("test", 80),
            "root_path": "",
            "state": {},
        }
        other_handle = gate.submit({"op": "hang", "seconds": 0.4, "started_path": other}, 5)
        task = asyncio.create_task(main.app(scope, receive, send))
        for _ in range(200):
            if os.path.exists(started) and os.path.exists(other) and gate.pending() == 2:
                break
            await asyncio.sleep(0.02)
        assert os.path.exists(started)
        assert task.done() is False
        victim = other_handle.identity[0]
        kept = [pid for pid in gate.worker_pids() if pid != victim]
        disconnect.set()
        await asyncio.wait_for(task, 3)
        reply = gate.wait(other_handle, 2)
        return kept, victim, reply, messages

    try:
        kept, victim, reply, sent_messages = asyncio.run(scenario())
    finally:
        open(release, "a", encoding="utf-8").close()
        gate.shutdown()
        main._gate = previous
        object.__setattr__(main._settings, "index_mode", mode)
    assert reply["event"] == "result"
    assert [pid for pid in kept if _alive(pid)] == []
    assert _alive(victim) is True or reply["pid"] == victim
    assert gate.pending() == 0
    assert sent_messages
    body = b"".join(message.get("body") or b"" for message in sent_messages)
    assert json.loads(body)["error_code"] == "timeout"


def test_live_ocr_failure_is_not_ready():
    from app import main

    gate = _gate()
    try:
        gate._slots[0].state = gate._ctx.Value("i", 3)
        gate.report = {"ocr_status": "ready", "embedding_status": "ready", "ocr_workers_expected": 1, "ocr_workers_alive": 1}
        provider = main._settings.ocr_provider
        mode = main._settings.index_mode
        object.__setattr__(main._settings, "ocr_provider", "paddle")
        object.__setattr__(main._settings, "index_mode", "pipeline")
        try:
            assert gate.health_problem(main._settings) == "ocr_model_missing"
            main._gate = gate
            ready = asyncio.run(_ready())
            assert ready.status_code == 503
            gate._slots[0].state.value = 2
            assert gate.health_problem(main._settings) == "worker_recovering"
            gate._slots[0].state.value = 1
            assert gate.health_problem(main._settings) is None
            gate._slots[0].state = None
            assert gate.health_problem(main._settings) == "ocr_model_missing"
        finally:
            object.__setattr__(main._settings, "ocr_provider", provider)
            object.__setattr__(main._settings, "index_mode", mode)
    finally:
        gate.shutdown()
        main._gate = IndexGate(main._settings.index_max_inflight)


async def _ready():
    import httpx

    from app import main

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/ready")


def test_unexpected_pipeline_death_is_not_ready():
    from app import main

    gate = _gate()
    provider = main._settings.ocr_provider
    try:
        gate._slots[0].state = gate._ctx.Value("i", 1)
        object.__setattr__(main._settings, "ocr_provider", "paddle")
        os.kill(gate.worker_pids()[0], signal.SIGKILL)
        gate._slots[0].proc.join(1)
        assert gate.health_problem(main._settings) == "worker_failed"
    finally:
        object.__setattr__(main._settings, "ocr_provider", provider)
        gate.shutdown()


def test_one_ocr_worker_per_pipeline_process():
    import inspect

    from app.indexing.ocr_supervisor import OcrSupervisor
    from app.indexing.runtime import build_models, ocr_pool_size

    class Settings:
        index_max_inflight = 4

    assert ocr_pool_size(Settings()) == 1
    assert "ocr_pool_size(settings)" in inspect.getsource(build_models)
    supers = []
    try:
        for _ in range(Settings.index_max_inflight):
            sup = OcrSupervisor(ocr_pool_size(Settings()), {"kind": "fake", "behavior": "ok"})
            sup.start(2)
            supers.append(sup)
        pids = [proc.pid for sup in supers for proc, _conn in sup._idle]
        assert len(pids) == 4
        assert len(set(pids)) == 4
        assert all(_alive(pid) for pid in pids)
    finally:
        for sup in supers:
            sup.shutdown()
        assert all(sup.alive_workers() == 0 for sup in supers)


def test_shutdown_during_ocr_spawn_reaps_the_worker():
    from app.indexing.ocr_supervisor import OcrSupervisor

    sup = OcrSupervisor(1, {"kind": "fake", "behavior": "ok", "start_delay_s": 0.4})
    box = {}

    def _start():
        try:
            sup.start(2)
            box["error"] = None
        except IndexFailure as exc:
            box["error"] = exc.code

    thread = threading.Thread(target=_start)
    thread.start()
    time.sleep(0.05)
    sup.shutdown()
    thread.join(2)
    assert thread.is_alive() is False
    assert sup.alive_workers() == 0
    assert sup._recovery_thread is None or sup._recovery_thread.is_alive() is False


def test_runpod_tls_contract():
    from app.config import ConfigError
    from app.indexing.object_storage import normalize_endpoint, storage_config

    assert normalize_endpoint("s3api-eu-ro-1.runpod.io", True) == ("s3api-eu-ro-1.runpod.io:443", True)
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_USE_SSL"):
        normalize_endpoint("s3api-eu-ro-1.runpod.io", False)
    assert normalize_endpoint("https://s3api-eu-ro-1.runpod.io", False) == ("s3api-eu-ro-1.runpod.io:443", True)
    with pytest.raises(ConfigError, match="TLS"):
        normalize_endpoint("http://s3api-eu-ro-1.runpod.io", True)
    assert normalize_endpoint("minio:9000", False) == ("minio:9000", False)
    assert normalize_endpoint("s3api-eu-ro-1.runpod.io:443", True) == ("s3api-eu-ro-1.runpod.io:443", True)
    with pytest.raises(ConfigError, match="443"):
        normalize_endpoint("https://s3api-eu-ro-1.runpod.io:9000", True)
    with pytest.raises(ConfigError, match="443"):
        normalize_endpoint("s3api-eu-ro-1.runpod.io:9000", True)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "s3api-eu-ro-1.runpod.io")
    monkeypatch.setenv("OBJECT_STORAGE_REGION", "eu-ro-1")
    monkeypatch.setenv("OBJECT_STORAGE_USE_SSL", "false")
    monkeypatch.setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "false")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_USE_SSL"):
        storage_config()
    monkeypatch.setenv("OBJECT_STORAGE_USE_SSL", "true")
    monkeypatch.setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "true")
    with pytest.raises(ConfigError, match="AUTO_CREATE_BUCKET"):
        storage_config()
    monkeypatch.delenv("OBJECT_STORAGE_REGION", raising=False)
    monkeypatch.setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "false")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_REGION"):
        storage_config()
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "minio:9000")
    monkeypatch.setenv("OBJECT_STORAGE_USE_SSL", "false")
    monkeypatch.setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "true")
    local = storage_config()
    assert local["endpoint"] == "minio:9000"
    assert local["secure"] is False
    assert local["auto_create"] is True
    monkeypatch.undo()
