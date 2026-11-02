"""File object backend (OBJECT_STORAGE_BACKEND=file): key contract shared with the
Go file store, symlink and traversal refusal, size caps, readiness, config."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.config import ConfigError
from app.indexing.errors import IndexFailure
from app.indexing import file_objects as fo

CONTRACT = Path(__file__).resolve().parents[3] / "deploy" / "contracts" / "object_store_keys.json"


@pytest.fixture()
def store(tmp_path: Path):
    root = tmp_path.resolve() / "objects"
    (root / "cas-documents").mkdir(parents=True)
    return root


def _put(root: Path, key: str, data: bytes) -> Path:
    path = root / "cas-documents" / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_key_contract_matches_the_go_store():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for key in contract["valid"]:
        assert fo.valid_key(key), key
    for key in contract["invalid"]:
        assert not fo.valid_key(key), key
    for bucket in contract["valid_buckets"]:
        assert fo.valid_component(bucket), bucket
    for bucket in contract["invalid_buckets"]:
        assert not (bucket and fo.valid_component(bucket)), bucket
    assert not fo.valid_key("a" * 129)
    assert fo.valid_key("a" * 128)
    assert not fo.valid_key("a/" * 300 + "a")


def test_read_round_trip_and_missing(store: Path):
    key = "xa_chu_se/documents/d1/abc.pdf"
    _put(store, key, b"%PDF-1.4 body")
    assert fo.read_object(str(store), "cas-documents", key, 1024) == b"%PDF-1.4 body"
    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(store), "cas-documents", "xa_chu_se/documents/d1/missing.pdf", 1024)
    assert exc.value.code == "source_not_found"
    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(store), "cas-documents", "nodir/x.pdf", 1024)
    assert exc.value.code == "source_not_found"


def test_size_cap(store: Path):
    _put(store, "xa/big.pdf", b"x" * 2048)
    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(store), "cas-documents", "xa/big.pdf", 1024)
    assert exc.value.code == "pdf_oversized"


@pytest.mark.parametrize("key", ["../x.pdf", "xa/../../etc/passwd", "/etc/passwd", "xa/.tmp-1", ""])
def test_traversal_keys_never_reach_the_filesystem(store: Path, key: str):
    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(store), "cas-documents", key, 1024)
    assert exc.value.code == "source_unsafe_path"


def test_symlinks_are_refused(store: Path, tmp_path: Path):
    outside = tmp_path / "research"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"research data")
    bucket = store / "cas-documents"
    (bucket / "xa").symlink_to(outside)
    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(store), "cas-documents", "xa/secret.txt", 1024)
    assert exc.value.code == "source_unsafe_path"
    (bucket / "real").mkdir()
    (bucket / "real" / "link.pdf").symlink_to(outside / "secret.txt")
    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(store), "cas-documents", "real/link.pdf", 1024)
    assert exc.value.code == "source_unsafe_path"
    root2 = tmp_path / "root2"
    root2.mkdir()
    (root2 / "cas-documents").symlink_to(bucket)
    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(root2), "cas-documents", "real/x.pdf", 1024)
    assert exc.value.code == "source_unsafe_path"
    assert not fo.bucket_ready(str(root2), "cas-documents")
    link_root = tmp_path / "root-link"
    link_root.symlink_to(store)
    with pytest.raises(IndexFailure):
        fo.read_object(str(link_root), "cas-documents", "real/x.pdf", 1024)


def test_non_regular_file_is_refused(store: Path):
    (store / "cas-documents" / "xa" / "dir.pdf").mkdir(parents=True)
    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(store), "cas-documents", "xa/dir.pdf", 1024)
    assert exc.value.code in ("source_unsafe_path", "source_timeout")


def test_deadline_is_checked(store: Path):
    _put(store, "xa/doc.pdf", b"x" * 10)

    class Expired:
        def check(self):
            raise IndexFailure("timeout")

    with pytest.raises(IndexFailure) as exc:
        fo.read_object(str(store), "cas-documents", "xa/doc.pdf", 1024, Expired())
    assert exc.value.code == "timeout"


def test_bucket_ready(store: Path, tmp_path: Path):
    assert fo.bucket_ready(str(store), "cas-documents")
    assert not fo.bucket_ready(str(store), "missing")
    assert not fo.bucket_ready(str(store), "../objects")


def test_backend_config(monkeypatch):
    monkeypatch.delenv("OBJECT_STORAGE_BACKEND", raising=False)
    assert fo.backend() == "s3"
    monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "FILE")
    assert fo.backend() == "file"
    monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "nfs")
    with pytest.raises(ConfigError):
        fo.backend()
    for root in ("", "relative", "/a/../b", "/a/b/"):
        monkeypatch.setenv("OBJECT_STORAGE_FILE_ROOT", root)
        with pytest.raises(ConfigError):
            fo.file_root()
    monkeypatch.setenv("OBJECT_STORAGE_FILE_ROOT", "/workspace/cas/objects")
    assert fo.file_root() == "/workspace/cas/objects"


def test_runtime_fetch_uses_the_file_backend(store: Path, monkeypatch):
    from app.indexing.runtime import fetch_object

    monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("OBJECT_STORAGE_FILE_ROOT", str(store))
    _put(store, "xa/documents/d/abc.pdf", b"%PDF-1.4 x")
    assert fetch_object("cas-documents", "xa/documents/d/abc.pdf", 1024, timeout_s=5) == b"%PDF-1.4 x"
    with pytest.raises(IndexFailure) as exc:
        fetch_object("cas-documents", "xa/documents/d/none.pdf", 1024)
    assert exc.value.code == "source_not_found"


def test_pipeline_config_accepts_file_backend_without_s3(monkeypatch):
    from app.config import _require_pipeline_storage

    for name in ("OBJECT_STORAGE_ENDPOINT", "OBJECT_STORAGE_ACCESS_KEY", "OBJECT_STORAGE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("OBJECT_STORAGE_FILE_ROOT", "/workspace/cas/objects")
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "cas-documents")
    _require_pipeline_storage()
    monkeypatch.delenv("OBJECT_STORAGE_BUCKET")
    with pytest.raises(ConfigError):
        _require_pipeline_storage()
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "cas-documents")
    monkeypatch.setenv("OBJECT_STORAGE_FILE_ROOT", "relative")
    with pytest.raises(ConfigError):
        _require_pipeline_storage()


def test_fifo_at_an_object_path_is_refused_without_blocking(store: Path):
    d = store / "cas-documents" / "xa"
    d.mkdir(parents=True)
    os.mkfifo(d / "pipe.pdf")
    with pytest.raises(IndexFailure) as exc:   # would hang forever on a blocking open
        fo.read_object(str(store), "cas-documents", "xa/pipe.pdf", 1024)
    assert exc.value.code == "source_unsafe_path"
