"""Hard timeout, readiness, CPU gate, and bundle contents."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from app.indexing.embed import detect_cpu_flags, select_embedding_variant
from app.indexing.errors import IndexFailure
from app.indexing.ocr_supervisor import OcrSupervisor


def test_auto_selects_int8_with_vnni_and_fp32_otherwise():
    vnni = detect_cpu_flags(cpuinfo_text="flags : avx512f avx512dq avx512bw avx512vl avx512_vnni\n")
    assert select_embedding_variant(vnni, requested="auto") == "qint8_avx512_vnni"
    arm = detect_cpu_flags(cpuinfo_text="processor : 0\nflags : fp asimd aes\n")
    assert select_embedding_variant(arm, requested="auto") == "fp32"
    x86 = detect_cpu_flags(cpuinfo_text="flags : sse2 avx avx2\n")
    assert select_embedding_variant(x86, requested="auto") == "fp32"


def test_cpu_gate_refuses_vnni_without_the_flag():
    missing = detect_cpu_flags(cpuinfo_text="processor : 0\nflags : sse2 avx avx2\n")
    with pytest.raises(IndexFailure) as raised:
        select_embedding_variant(missing, requested="int8-vnni")
    assert raised.value.code == "embedding_cpu_unsupported"
    linux = detect_cpu_flags(cpuinfo_text="flags : fpu avx512f avx512dq avx512bw avx512vl avx512_vnni\n")
    assert "avx512vnni" in linux
    assert select_embedding_variant(linux, requested="int8-vnni") == "qint8_avx512_vnni"


def test_ocr_hang_is_killed_and_the_next_call_works():
    sup = OcrSupervisor(1, {"kind": "fake", "behavior": "hang", "recover": "ok", "replace_timeout_s": 2})
    sup.start(2)
    try:
        started = time.perf_counter()
        with pytest.raises(IndexFailure) as raised:
            sup.read_page(None, 1, timeout_s=0.2)
        assert raised.value.code == "ocr_timeout"
        assert time.perf_counter() - started < 1
        sup.wait_settled(2)
        assert sup.status == "ready"
        assert sup.worker_count() == 1
        assert sup.read_page(None, 1, timeout_s=2) == "ok"
        assert sup.worker_count() == 1
    finally:
        sup.shutdown()


def test_ocr_init_hang_and_crash_do_not_become_ready():
    hung = OcrSupervisor(1, {"kind": "fake", "behavior": "init_hang"})
    started = time.perf_counter()
    with pytest.raises(IndexFailure):
        hung.start(0.3)
    assert time.perf_counter() - started < 3
    assert hung.status == "timed_out"
    assert hung.worker_count() == 0
    hung.shutdown()
    crashed = OcrSupervisor(1, {"kind": "fake", "behavior": "init_crash"})
    with pytest.raises(IndexFailure):
        crashed.start(2)
    assert crashed.status == "failed"
    crashed.shutdown()


def test_embedding_init_hang_is_killed():
    from app.indexing.embed import OnnxProcess

    proc = OnnxProcess("", behavior="init_hang")
    started = time.perf_counter()
    with pytest.raises(IndexFailure):
        proc.start(0.3)
    assert time.perf_counter() - started < 3
    assert proc.status == "timed_out"
    proc.shutdown()


def test_two_ocr_slots_use_two_processes():
    sup = OcrSupervisor(2, {"kind": "fake", "behavior": "slow", "delay_s": 0.2})
    sup.start(2)
    try:
        results = []

        def call():
            results.append(sup.read_page(None, 1, timeout_s=2))

        threads = [threading.Thread(target=call) for _ in range(2)]
        for item in threads:
            item.start()
        for item in threads:
            item.join(3)
        assert results == ["ok", "ok"]
        assert sup.worker_count() == 2
    finally:
        sup.shutdown()


def test_minio_probe_times_out(monkeypatch):
    from app.indexing import runtime

    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    stop = threading.Event()

    def stall():
        conn, _ = held.accept()
        stop.wait(3)
        conn.close()

    threading.Thread(target=stall, daemon=True).start()
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", f"127.0.0.1:{port}")
    monkeypatch.setenv("OBJECT_STORAGE_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("OBJECT_STORAGE_SECRET_KEY", "minioadmin")
    runtime._MINIO_CLIENT = None

    class Settings:
        object_bucket = "cas-documents"

    started = time.perf_counter()
    assert runtime._minio(Settings()) is False
    assert time.perf_counter() - started < 4
    stop.set()
    held.close()


def test_status_and_ready_share_probe_result(monkeypatch):
    from app import main
    from app.indexing import runtime

    runtime._HEALTH["ready"] = None
    runtime._HEALTH["monotonic"] = 0
    monkeypatch.setattr(runtime, "probe_pipeline", lambda *_args, **_kwargs: (False, "postgres_unreachable"))
    object.__setattr__(main._settings, "index_mode", "pipeline")
    try:
        health = runtime.current_health(main._settings, False, force=True)
        status = main.status()
        ready = main.ready()
    finally:
        object.__setattr__(main._settings, "index_mode", "mock")
        runtime._HEALTH["ready"] = None
        runtime._HEALTH["monotonic"] = 0
    assert health["ready"] is False
    assert status["ready"] is False
    assert status["reason"] == "postgres_unreachable"
    assert ready.status_code == 503
    assert ready.body


def test_manifest_examples_parse():
    root = Path(__file__).resolve().parents[3] / "docs" / "models"
    embedding = json.loads((root / "embedding.manifest.example.json").read_text(encoding="utf-8"))
    ocr = json.loads((root / "ocr.manifest.example.json").read_text(encoding="utf-8"))
    assert embedding["revision"] == "6a0d452a575215f80b8f66276dd4ee5d504942c6"
    assert embedding["quantization"] == "int8"
    assert embedding["files"]["model.onnx"].startswith("dd476dd0")
    assert "files" in ocr
