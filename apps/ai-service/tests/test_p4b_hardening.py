"""Hard timeout, readiness, CPU gate, and bundle contents."""

from __future__ import annotations

import json
import socket
import threading
import time
import zipfile
from pathlib import Path

import pytest

from app.indexing.embed import detect_cpu_flags, require_supported_variant
from app.indexing.errors import IndexFailure
from app.indexing.ocr_supervisor import OcrSupervisor


def test_cpu_gate_refuses_vnni_without_the_flag():
    missing = detect_cpu_flags(cpuinfo_text="processor : 0\nflags : sse2 avx avx2\n")
    with pytest.raises(IndexFailure) as raised:
        require_supported_variant(missing)
    assert raised.value.code == "embedding_cpu_unsupported"
    linux = detect_cpu_flags(cpuinfo_text="flags : fpu avx512f avx512dq avx512bw avx512vl avx512_vnni\n")
    assert "avx512vnni" in linux
    assert require_supported_variant(linux) == "qint8_avx512_vnni"


def test_ocr_hang_is_killed_and_the_next_call_works():
    sup = OcrSupervisor(1, {"kind": "fake", "behavior": "hang", "recover": "ok", "replace_timeout_s": 2})
    sup.start(2)
    try:
        started = time.perf_counter()
        with pytest.raises(IndexFailure) as raised:
            sup.read_page(None, 1, timeout_s=0.2)
        assert raised.value.code == "ocr_timeout"
        assert time.perf_counter() - started < 3
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


def test_bundle_filter_keeps_model_packages(tmp_path):
    import sys

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts"))
    from p4b_bundle import iter_files, should_skip_dir

    assert should_skip_dir("models") is False
    (tmp_path / "apps" / "ai-service" / "app" / "models").mkdir(parents=True)
    (tmp_path / "docs" / "models").mkdir(parents=True)
    (tmp_path / "weights").mkdir()
    (tmp_path / "apps" / "ai-service" / "app" / "models" / "extract.py").write_text("x", encoding="utf-8")
    (tmp_path / "docs" / "models" / "embedding.manifest.example.json").write_text("{}", encoding="utf-8")
    (tmp_path / "weights" / "model.onnx").write_bytes(b"weight")
    names = set(iter_files(tmp_path))
    assert "apps/ai-service/app/models/extract.py" in names
    assert "docs/models/embedding.manifest.example.json" in names
    assert "weights/model.onnx" not in names


def test_check_zip_fails_when_a_model_module_is_missing(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "check-zip.sh"
    required = [
        "phase_1_3_final_report.md",
        "phase_1_3_test_output.txt",
        "phase_p2_llm_extract_report.md",
        "phase_p2_test_output.txt",
        "phase_p3_1_document_intake_report.md",
        "phase_p3_1_test_output.txt",
        "phase_p4a_document_indexing_report.md",
        "phase_p4a_test_output.txt",
        "deploy/migrations/000001_init_schema.up.sql",
        "deploy/migrations/000001_init_schema.down.sql",
        "deploy/migrations/000002_seed_master.up.sql",
        "deploy/migrations/000002_seed_master.down.sql",
        "deploy/migrations/000003_seed_procedures.up.sql",
        "deploy/migrations/000003_seed_procedures.down.sql",
        "deploy/migrations/000004_seed_auth_users.up.sql",
        "deploy/migrations/000004_seed_auth_users.down.sql",
        "deploy/migrations/000005_phase13_chat.up.sql",
        "deploy/migrations/000005_phase13_chat.down.sql",
        "deploy/migrations/000006_idempotency_envelope.up.sql",
        "deploy/migrations/000006_idempotency_envelope.down.sql",
        "deploy/migrations/000007_clear_demo_passwords.up.sql",
        "deploy/migrations/000007_clear_demo_passwords.down.sql",
        "deploy/migrations/000008_extract_claim.up.sql",
        "deploy/migrations/000008_extract_claim.down.sql",
        "deploy/migrations/000009_document_intake.up.sql",
        "deploy/migrations/000009_document_intake.down.sql",
        "deploy/migrations/000010_document_indexing.up.sql",
        "deploy/migrations/000010_document_indexing.down.sql",
        "deploy/migrations/000011_link_index_status.up.sql",
        "deploy/migrations/000011_link_index_status.down.sql",
        "deploy/migrations/000012_p4a_hardening.up.sql",
        "deploy/migrations/000012_p4a_hardening.down.sql",
        "deploy/migrations/000013_legacy_idempotency_replay.up.sql",
        "deploy/migrations/000013_legacy_idempotency_replay.down.sql",
        "deploy/migrations/000014_p4b_index_generations.up.sql",
        "deploy/migrations/000014_p4b_index_generations.down.sql",
        "deploy/migrations/000015_p4b_unlink_reindex.up.sql",
        "deploy/migrations/000015_p4b_unlink_reindex.down.sql",
        "docs/p4b-runbook.md",
        "phase_p4b_document_content_report.md",
        "phase_p4b_test_output.txt",
        "SHA256SUMS.txt",
        ".github/workflows/ci.yml",
        ".gitignore",
        "deploy/.env.example",
        "packages/contracts/README.md",
        "apps/api/go.mod",
        "apps/web/package-lock.json",
        "apps/api/docs/swagger.json",
        "apps/api/docs/swagger.yaml",
        "apps/api/docs/docs.go",
        "apps/ai-service/app/main.py",
        "apps/ai-service/app/config.py",
        "apps/ai-service/app/models/__init__.py",
        "apps/ai-service/app/models/extract.py",
        "apps/ai-service/app/models/index.py",
        "docs/models/embedding.manifest.example.json",
        "docs/models/ocr.manifest.example.json",
        "deploy/migrations/000016_p4b_cleanup_claim.up.sql",
        "deploy/migrations/000016_p4b_cleanup_claim.down.sql",
        "scripts/p4b_bundle.py",
    ]
    archive = tmp_path / "bundle.zip"

    def build(drop: str | None = None) -> None:
        import hashlib

        payload = {}
        for name in required:
            if name == drop or name == "SHA256SUMS.txt":
                continue
            payload[name] = b"" if name.endswith(".py") else b"x"
        lines = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in sorted(payload.items())]
        payload["SHA256SUMS.txt"] = ("\n".join(lines) + "\n").encode()
        with zipfile.ZipFile(archive, "w") as zf:
            for name, data in payload.items():
                zf.writestr(name, data)

    import subprocess

    build()
    ok = subprocess.run(["bash", str(script), str(archive)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    build(drop="apps/ai-service/app/models/extract.py")
    bad = subprocess.run(["bash", str(script), str(archive)], capture_output=True, text=True)
    assert bad.returncode != 0
    assert "app/models/extract.py" in bad.stderr


def test_manifest_examples_parse():
    root = Path(__file__).resolve().parents[3] / "docs" / "models"
    embedding = json.loads((root / "embedding.manifest.example.json").read_text(encoding="utf-8"))
    ocr = json.loads((root / "ocr.manifest.example.json").read_text(encoding="utf-8"))
    assert embedding["revision"] == "6a0d452a575215f80b8f66276dd4ee5d504942c6"
    assert embedding["quantization"] == "int8"
    assert embedding["files"]["model.onnx"].startswith("dd476dd0")
    assert "files" in ocr
