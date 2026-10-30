"""EMBEDDING_VARIANT selection, fail-closed model checks, and real ONNX inference.

Selection tests feed CPU flag text to the parser (simulated CPUs). The inference
tests run the real graphs on the CPU of the test machine and are skipped unless
the model directory is given:
  E5_FP32_MODEL_DIR  fp32 model (fetch_models.sh --variant fp32 <dir>)
  E5_INT8_MODEL_DIR  int8 model (fetch_models.sh --variant int8-vnni <dir>), VNNI CPU only
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import pytest

from app.indexing.embed import (
    E5_DIMENSION,
    E5_FP32_SHA256,
    E5_MODEL_SHA256,
    OnnxEmbedder,
    REQUIRED_CPU_FLAGS,
    detect_cpu_flags,
    load_manifest,
    requested_variant,
    select_embedding_variant,
)
from app.indexing.errors import IndexFailure

VNNI = detect_cpu_flags(cpuinfo_text="flags : avx512f avx512dq avx512bw avx512vl avx512_vnni\n")
EPYC_ZEN3 = detect_cpu_flags(cpuinfo_text="flags : fpu sse2 avx avx2 fma bmi2 sha_ni vaes vpclmulqdq\n")
ARM = detect_cpu_flags(cpuinfo_text="processor : 0\nFeatures : fp asimd aes sha2\n")

FP32_DIR = os.environ.get("E5_FP32_MODEL_DIR", "")
INT8_DIR = os.environ.get("E5_INT8_MODEL_DIR", "")


def _has_vnni() -> bool:
    seen = detect_cpu_flags()
    return all(name in seen for name in REQUIRED_CPU_FLAGS)


def test_auto_uses_fp32_on_x86_without_vnni():
    assert select_embedding_variant(EPYC_ZEN3, requested="auto") == "fp32"
    assert select_embedding_variant(VNNI, requested="auto") == "qint8_avx512_vnni"
    assert select_embedding_variant(ARM, requested="auto") == "fp32"


def test_forced_fp32_ignores_cpu_flags():
    for flags in (VNNI, EPYC_ZEN3, ARM):
        assert select_embedding_variant(flags, requested="fp32") == "fp32"


def test_forced_int8_on_cpu_without_vnni_fails_clearly():
    assert select_embedding_variant(VNNI, requested="int8-vnni") == "qint8_avx512_vnni"
    for flags in (EPYC_ZEN3, ARM):
        with pytest.raises(IndexFailure) as raised:
            select_embedding_variant(flags, requested="int8-vnni")
        assert raised.value.code == "embedding_cpu_unsupported"


def test_variant_comes_from_env_and_unknown_value_fails(monkeypatch):
    monkeypatch.delenv("EMBEDDING_VARIANT", raising=False)
    assert requested_variant() == "auto"
    monkeypatch.setenv("EMBEDDING_VARIANT", " FP32 ")
    assert select_embedding_variant(VNNI) == "fp32"
    monkeypatch.setenv("EMBEDDING_VARIANT", "int8")
    with pytest.raises(IndexFailure) as raised:
        select_embedding_variant(VNNI)
    assert raised.value.code == "embedding_variant_invalid"


def _fake_model_dir(tmp_path: Path, method: str, model_sha: str) -> Path:
    d = tmp_path / method
    d.mkdir()
    (d / "model.onnx").write_bytes(b"not the pinned model")
    (d / "tokenizer.json").write_bytes(b"{}")
    manifest = {
        "model_id": "intfloat/multilingual-e5-small",
        "source_repository": "https://huggingface.co/intfloat/multilingual-e5-small",
        "revision": "6a0d452a575215f80b8f66276dd4ee5d504942c6",
        "source_file": "onnx/model.onnx" if method == "fp32" else "onnx/model_qint8_avx512_vnni.onnx",
        "dimension": 384,
        "quantization": "fp32" if method == "fp32" else "int8",
        "quantization_method": method,
        "files": {
            "model.onnx": model_sha,
            "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_fp32_model_with_wrong_bytes_fails_closed(tmp_path):
    d = _fake_model_dir(tmp_path, "fp32", E5_FP32_SHA256)
    with pytest.raises(IndexFailure) as raised:
        load_manifest(str(d), "fp32")
    assert raised.value.code == "embedding_checksum_mismatch"


def test_int8_manifest_is_refused_when_fp32_is_selected(tmp_path):
    d = _fake_model_dir(tmp_path, "qint8_avx512_vnni", E5_MODEL_SHA256)
    with pytest.raises(IndexFailure) as raised:
        load_manifest(str(d), "fp32")
    assert raised.value.code == "embedding_checksum_mismatch"


def test_missing_model_fails_closed(tmp_path):
    with pytest.raises(IndexFailure) as raised:
        load_manifest(str(tmp_path), "fp32")
    assert raised.value.code == "embedding_model_missing"
    d = _fake_model_dir(tmp_path, "fp32", E5_FP32_SHA256)
    (d / "model.onnx").unlink()
    with pytest.raises(IndexFailure) as raised:
        load_manifest(str(d), "fp32")
    assert raised.value.code == "embedding_model_missing"


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _check_real_embedder(embedder: OnnxEmbedder) -> list[float]:
    assert embedder.dimension == E5_DIMENSION
    query = embedder.embed_query("Làm giấy khai sinh cho con cần những giấy tờ gì?")
    passages = embedder.embed_batch(
        [
            "Thủ tục đăng ký khai sinh: người đi đăng ký khai sinh nộp Tờ khai theo mẫu và Giấy chứng sinh.",
            "Thủ tục đăng ký kết hôn: hai bên nam, nữ nộp Tờ khai đăng ký kết hôn theo mẫu.",
        ]
    )
    for vec in [query, *passages]:
        assert len(vec) == E5_DIMENSION
        assert all(math.isfinite(x) for x in vec)
        assert abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-5
    # query: and passage: prefixes give different vectors for the same text.
    same = "đăng ký khai sinh"
    assert _cos(embedder.embed_query(same), embedder.embed_batch([same])[0]) < 0.9999
    assert _cos(query, passages[0]) > _cos(query, passages[1])
    return query


@pytest.mark.skipif(not FP32_DIR, reason="E5_FP32_MODEL_DIR not set")
def test_real_fp32_inference_on_this_cpu(monkeypatch):
    monkeypatch.setenv("EMBEDDING_VARIANT", "fp32")
    embedder = OnnxEmbedder(FP32_DIR)
    assert embedder.variant == "fp32"
    assert embedder.checksum == E5_FP32_SHA256
    _check_real_embedder(embedder)


@pytest.mark.skipif(not FP32_DIR, reason="E5_FP32_MODEL_DIR not set")
def test_real_auto_on_this_cpu(monkeypatch):
    monkeypatch.setenv("EMBEDDING_VARIANT", "auto")
    if _has_vnni():
        if not INT8_DIR:
            pytest.skip("VNNI CPU: auto needs E5_INT8_MODEL_DIR")
        embedder = OnnxEmbedder(INT8_DIR)
        assert embedder.variant == "qint8_avx512_vnni"
    else:
        embedder = OnnxEmbedder(FP32_DIR)
        assert embedder.variant == "fp32"
    _check_real_embedder(embedder)


@pytest.mark.skipif(not FP32_DIR, reason="E5_FP32_MODEL_DIR not set")
def test_real_int8_model_refused_for_fp32_and_vice_versa(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_VARIANT", "int8-vnni")
    if _has_vnni():
        with pytest.raises(IndexFailure) as raised:
            OnnxEmbedder(FP32_DIR)
        assert raised.value.code == "embedding_checksum_mismatch"
    else:
        with pytest.raises(IndexFailure) as raised:
            OnnxEmbedder(FP32_DIR)
        assert raised.value.code == "embedding_cpu_unsupported"
    # A corrupted copy of the real fp32 model fails closed before ONNX load.
    broken = tmp_path / "broken"
    shutil.copytree(FP32_DIR, broken)
    with open(broken / "model.onnx", "r+b") as fh:
        fh.seek(1024)
        fh.write(b"\x00\x01\x02\x03")
    monkeypatch.setenv("EMBEDDING_VARIANT", "fp32")
    with pytest.raises(IndexFailure) as raised:
        OnnxEmbedder(str(broken))
    assert raised.value.code == "embedding_checksum_mismatch"


@pytest.mark.skipif(not (FP32_DIR and INT8_DIR), reason="needs both model directories")
def test_real_int8_and_fp32_agree_on_vnni_cpu(monkeypatch):
    if not _has_vnni():
        pytest.skip("int8 graph needs AVX-512 VNNI")
    monkeypatch.setenv("EMBEDDING_VARIANT", "int8-vnni")
    q8 = _check_real_embedder(OnnxEmbedder(INT8_DIR))
    monkeypatch.setenv("EMBEDDING_VARIANT", "fp32")
    q32 = _check_real_embedder(OnnxEmbedder(FP32_DIR))
    assert _cos(q8, q32) > 0.95
