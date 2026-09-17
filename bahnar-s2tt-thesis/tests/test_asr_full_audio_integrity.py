"""
Audio integrity: what makes a hydrated WAV acceptable, and what must not.

The PCM hash covers samples only, so header fields have to be checked separately;
and an all-excluded shard is a legitimate outcome, not a corrupt index.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.asr_full_data import (
    AUDIO_PCM_PIPELINE_VERSION,
    SIDECAR_REQUIRED_FIELDS,
    local_audio_ok,
    read_pcm16_payload,
    shard_index_paths,
    verify_shard_sidecar_index,
    write_shard_sidecar_index,
)
from src.asr_full_pcm import sha256_hex
from src.data_utils import safe_cache_filename

SHARD = "default/train/0000.parquet"


def _pcm(n=1600, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.integers(-2000, 2000, size=n)).astype("<i2")


def _write_wav(path: Path, pcm, sample_rate=16000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pcm if channels == 1 else np.stack([pcm, pcm], axis=1)
    sf.write(str(path), data, sample_rate, subtype="PCM_16", format="WAV")


def _record(uid="u0", *, n_samples=1600, sample_rate=16000, sha_pcm=None):
    return {
        "record_uid": uid,
        "record_id": f"id-{uid}",
        "shard_row_index": 0,
        "sha256_source": sha256_hex(("src" + uid).encode()),
        "sha256_pcm": sha_pcm or sha256_hex(uid.encode()),
        "n_samples": n_samples,
        "sample_rate": sample_rate,
        "channels": 1,
        "audio_pcm_pipeline_version": AUDIO_PCM_PIPELINE_VERSION,
        "dataset_revision": "dsrev",
        "parquet_revision": "a" * 40,
        "local_cache_relpath": safe_cache_filename(uid),
    }


class TestWavHeaderIsPartOfTheContract:
    def test_correct_wav_passes(self, tmp_path: Path):
        pcm = _pcm()
        path = tmp_path / "a.wav"
        _write_wav(path, pcm)
        got = read_pcm16_payload(path)
        ok, reason = local_audio_ok(
            path, expected_sha256_pcm=got["sha256_pcm"],
            expected_n_samples=len(pcm), expected_sample_rate=16000,
        )
        assert ok, reason

    def test_right_pcm_wrong_header_sample_rate_is_rejected(self, tmp_path: Path):
        """Identical samples at 8kHz would train at the wrong speed."""
        pcm = _pcm()
        good, bad = tmp_path / "good.wav", tmp_path / "bad.wav"
        _write_wav(good, pcm, sample_rate=16000)
        _write_wav(bad, pcm, sample_rate=8000)
        expected = read_pcm16_payload(good)["sha256_pcm"]
        # The PCM payload is byte-identical, so the hash alone cannot catch this.
        assert read_pcm16_payload(bad)["sha256_pcm"] == expected
        ok, reason = local_audio_ok(
            bad, expected_sha256_pcm=expected,
            expected_n_samples=len(pcm), expected_sample_rate=16000,
        )
        assert ok is False and reason.startswith("sample_rate_mismatch")

    def test_stereo_wav_is_rejected(self, tmp_path: Path):
        pcm = _pcm()
        path = tmp_path / "stereo.wav"
        _write_wav(path, pcm, channels=2)
        ok, reason = local_audio_ok(
            path, expected_sha256_pcm="whatever", expected_sample_rate=16000,
        )
        assert ok is False and "not_mono" in reason

    def test_truncated_pcm_is_rejected_by_sample_count(self, tmp_path: Path):
        pcm = _pcm()
        full, short = tmp_path / "full.wav", tmp_path / "short.wav"
        _write_wav(full, pcm)
        _write_wav(short, pcm[:-10])
        ok, reason = local_audio_ok(
            short, expected_sha256_pcm=read_pcm16_payload(full)["sha256_pcm"],
            expected_n_samples=len(pcm), expected_sample_rate=16000,
        )
        assert ok is False and reason == "sha256_pcm_mismatch"

    def test_missing_file_is_reported_as_missing(self, tmp_path: Path):
        ok, reason = local_audio_ok(tmp_path / "nope.wav", expected_sha256_pcm="x")
        assert ok is False and reason == "missing"

    def test_non_wav_bytes_are_reported_as_unreadable(self, tmp_path: Path):
        path = tmp_path / "junk.wav"
        path.write_bytes(b"not a wav at all")
        ok, reason = local_audio_ok(path, expected_sha256_pcm="x")
        assert ok is False and reason.startswith("unreadable:")


class TestZeroRecordSidecar:
    def test_shard_with_no_eligible_records_is_valid(self, tmp_path: Path):
        """
        Every UID excluded is a normal outcome. Reading n_records with `or -1`
        turned a legitimate 0 into "missing" and failed such a shard forever.
        """
        write_shard_sidecar_index(
            tmp_path, SHARD, records=[],
            dataset_revision="dsrev", parquet_revision="a" * 40,
        )
        out = verify_shard_sidecar_index(tmp_path, SHARD)
        assert out == {}

    def test_zero_records_meta_records_the_count_explicitly(self, tmp_path: Path):
        write_shard_sidecar_index(
            tmp_path, SHARD, records=[],
            dataset_revision="dsrev", parquet_revision="a" * 40,
        )
        _index_path, meta_path = shard_index_paths(tmp_path, SHARD)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["n_records"] == 0

    def test_meta_without_n_records_is_refused(self, tmp_path: Path):
        write_shard_sidecar_index(
            tmp_path, SHARD, records=[_record()],
            dataset_revision="dsrev", parquet_revision="a" * 40,
        )
        _index_path, meta_path = shard_index_paths(tmp_path, SHARD)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.pop("n_records")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(RuntimeError, match="lacks n_records"):
            verify_shard_sidecar_index(tmp_path, SHARD)

    def test_count_mismatch_is_still_caught(self, tmp_path: Path):
        write_shard_sidecar_index(
            tmp_path, SHARD, records=[_record("u0"), _record("u1")],
            dataset_revision="dsrev", parquet_revision="a" * 40,
        )
        _index_path, meta_path = shard_index_paths(tmp_path, SHARD)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["n_records"] = 5
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(RuntimeError, match="count mismatch"):
            verify_shard_sidecar_index(tmp_path, SHARD)


class TestSidecarProvenanceFields:
    def test_every_required_provenance_field_is_enforced(self, tmp_path: Path):
        # shard_key / revisions / pipeline version are filled in by the writer
        # from its own arguments, so only the per-record fields can be missing.
        writer_supplied = {
            "shard_key", "parquet_revision", "dataset_revision",
            "audio_pcm_pipeline_version",
        }
        for field in set(SIDECAR_REQUIRED_FIELDS) - writer_supplied:
            record = _record()
            record.pop(field, None)
            with pytest.raises(RuntimeError):
                write_shard_sidecar_index(
                    tmp_path / field, SHARD, records=[record],
                    dataset_revision="dsrev", parquet_revision="a" * 40,
                )

    def test_required_fields_cover_audio_and_provenance(self):
        assert {
            "record_uid", "sha256_source", "sha256_pcm", "n_samples",
            "sample_rate", "audio_pcm_pipeline_version",
        } <= set(SIDECAR_REQUIRED_FIELDS)

    def test_pcm_hash_is_over_raw_little_endian_samples(self, tmp_path: Path):
        pcm = _pcm()
        path = tmp_path / "a.wav"
        _write_wav(path, pcm)
        assert read_pcm16_payload(path)["sha256_pcm"] == sha256_hex(pcm.tobytes())
