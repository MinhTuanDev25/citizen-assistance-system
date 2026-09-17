"""
Deterministic PCM16 pipeline, HF parquet reference parsing and disk reclamation.

Audio is synthesised locally (PCM_16 / PCM_24 / stereo / 8kHz) so the byte-exact
contract can be tested without touching Hugging Face.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.asr_full_pcm import (
    AUDIO_PCM_PIPELINE_VERSION,
    AudioQaError,
    HfParquetRef,
    assert_disk_headroom,
    build_wav_bytes,
    canonical_pcm16_from_array,
    canonical_pcm16_from_bytes,
    cleanup_shard_download,
    delete_hf_cache_blob,
    disk_free_bytes,
    float_to_pcm16_bytes,
    normalize_parquet_ref,
    parse_hf_parquet_url,
    read_pcm16_payload,
    safe_shard_key,
    write_wav_from_pcm16,
)

REPO = "cuong06/Bahnar_Vietnamese"


def _tone(seconds=0.5, sr=16000, freq=220.0, channels=1):
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    mono = 0.4 * np.sin(2 * np.pi * freq * t)
    if channels == 1:
        return mono
    return np.stack([mono, mono * 0.5], axis=1)


def _encode(array, sr, *, fmt="FLAC", subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, array, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


class TestParquetUrlNormalization:
    def test_resolve_url_splits_filename_and_revision(self):
        """A full URL passed as hf_hub_download(filename=...) 404s; it must be split."""
        ref = parse_hf_parquet_url(
            "https://huggingface.co/datasets/cuong06/Bahnar_Vietnamese/resolve/"
            "refs%2Fconvert%2Fparquet/default/train/0033.parquet"
        )
        assert ref == HfParquetRef(
            repo_id=REPO,
            filename="default/train/0033.parquet",
            revision="refs/convert/parquet",
            repo_type="dataset",
        )

    def test_unescaped_refs_path_also_parses(self):
        ref = parse_hf_parquet_url(
            f"https://huggingface.co/datasets/{REPO}/resolve/refs/convert/parquet/"
            "default/validation/0000.parquet"
        )
        assert ref.filename == "default/validation/0000.parquet"
        assert ref.revision == "refs/convert/parquet"

    def test_plain_commit_revision_url(self):
        ref = parse_hf_parquet_url(
            f"https://huggingface.co/datasets/{REPO}/resolve/{'a' * 40}/data/train-0000.parquet"
        )
        assert ref.revision == "a" * 40
        assert ref.filename == "data/train-0000.parquet"

    @pytest.mark.parametrize("bad", [
        "",
        "not-a-url",
        "https://example.com/datasets/a/b/resolve/main/x.parquet",
        f"https://huggingface.co/datasets/{REPO}/blob/main",
    ])
    def test_invalid_urls_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_hf_parquet_url(bad)

    def test_normalize_pins_immutable_snapshot(self):
        ref = normalize_parquet_ref(
            f"https://huggingface.co/datasets/{REPO}/resolve/refs%2Fconvert%2Fparquet/d/train/0.parquet",
            expected_repo_id=REPO,
            parquet_revision="b" * 40,
        )
        assert ref.revision == "b" * 40  # alias replaced by the pinned sha
        assert ref.filename == "d/train/0.parquet"

    def test_relative_filename_passes_through(self):
        ref = normalize_parquet_ref(
            "default/train/0007.parquet", expected_repo_id=REPO, parquet_revision="c" * 40
        )
        assert ref.filename == "default/train/0007.parquet"

    def test_mutable_alias_revision_is_refused(self):
        with pytest.raises(ValueError, match="immutable commit sha"):
            normalize_parquet_ref(
                "d/train/0.parquet",
                expected_repo_id=REPO,
                parquet_revision="refs/convert/parquet",
            )

    def test_repo_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="expected"):
            normalize_parquet_ref(
                "https://huggingface.co/datasets/someone/else/resolve/main/x.parquet",
                expected_repo_id=REPO,
                parquet_revision="d" * 40,
            )

    def test_shard_keys_are_filesystem_safe_and_unique(self):
        a = safe_shard_key("default/train/0000.parquet")
        b = safe_shard_key("default/validation/0000.parquet")
        assert a != b
        assert "/" not in a and "/" not in b


class TestCanonicalPcm16:
    def test_pcm16_roundtrip_is_byte_exact(self):
        raw = _encode(_tone(), 16000, subtype="PCM_16")
        out = canonical_pcm16_from_bytes(raw, target_sr=16000)
        assert out["sample_rate"] == 16000
        assert out["n_samples"] == 8000
        assert out["resampled"] is False
        assert out["audio_pcm_pipeline_version"] == AUDIO_PCM_PIPELINE_VERSION
        assert len(out["sha256_pcm"]) == 64 and len(out["sha256_source"]) == 64
        # Re-running the pipeline on the same bytes yields the same hash.
        assert canonical_pcm16_from_bytes(raw, target_sr=16000)["sha256_pcm"] == out["sha256_pcm"]

    def test_pcm24_is_quantised_deterministically(self):
        """PCM_24 sources exist in this dataset; they must land on a stable int16 grid."""
        array = _tone()
        out24 = canonical_pcm16_from_bytes(_encode(array, 16000, subtype="PCM_24"), target_sr=16000)
        again = canonical_pcm16_from_bytes(_encode(array, 16000, subtype="PCM_24"), target_sr=16000)
        assert out24["source_subtype"] == "PCM_24"
        assert out24["sha256_pcm"] == again["sha256_pcm"]
        assert out24["n_samples"] == 8000

    def test_stereo_is_mean_mixed_to_mono(self):
        stereo = _tone(channels=2)
        raw = _encode(stereo, 16000, subtype="PCM_16")
        out = canonical_pcm16_from_bytes(raw, target_sr=16000)
        assert out["source_channels"] == 2
        assert out["n_samples"] == stereo.shape[0]
        # The contract is "mean-mix the decoded channels", so compare against a
        # decode of the same bytes rather than the pre-encode floats (encoding
        # quantises each channel first, which shifts the mix by up to 1 LSB).
        decoded, _sr = sf.read(io.BytesIO(raw), dtype="float64", always_2d=True)
        assert out["pcm"] == float_to_pcm16_bytes(decoded.mean(axis=1))
        assert canonical_pcm16_from_bytes(raw, target_sr=16000)["pcm"] == out["pcm"]

    def test_non_16k_is_resampled_and_flagged(self):
        raw = _encode(_tone(seconds=0.5, sr=8000, channels=1), 8000, subtype="PCM_16")
        out = canonical_pcm16_from_bytes(raw, target_sr=16000)
        assert out["resampled"] is True
        assert out["source_sample_rate"] == 8000
        assert out["sample_rate"] == 16000
        assert out["n_samples"] == pytest.approx(8000, rel=0.02)

    @pytest.mark.parametrize("fill", [np.nan, np.inf, -np.inf])
    def test_non_finite_audio_is_rejected_not_quantised(self, fill):
        """NaN/inf must not quantise into plausible silence or full-scale constants."""
        raw = _encode(np.full(16000, fill), 16000, fmt="WAV", subtype="FLOAT")
        with pytest.raises(AudioQaError) as excinfo:
            canonical_pcm16_from_bytes(raw, target_sr=16000)
        assert excinfo.value.reason == "non_finite_values"

    def test_empty_audio_is_rejected(self):
        raw = _encode(np.zeros(0, dtype=np.float32), 16000, fmt="WAV", subtype="FLOAT")
        with pytest.raises(AudioQaError) as excinfo:
            canonical_pcm16_from_bytes(raw, target_sr=16000)
        assert excinfo.value.reason == "empty"

    def test_digital_silence_stays_eligible(self):
        """NB02 treats near-silence as a warning, not a hard failure."""
        out = canonical_pcm16_from_bytes(
            _encode(np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16"),
            target_sr=16000,
        )
        assert out["n_samples"] == 16000

    def test_clipping_and_rounding_are_pinned(self):
        array = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        pcm = float_to_pcm16_bytes(array)
        values = np.frombuffer(pcm, dtype="<i2")
        assert values.tolist() == [-32767, -32767, -16384, 0, 16384, 32767, 32767]

    def test_empty_payload_rejected(self):
        with pytest.raises(ValueError):
            canonical_pcm16_from_bytes(b"", target_sr=16000)

    def test_wav_writer_is_deterministic_and_reads_back(self, tmp_path: Path):
        out = canonical_pcm16_from_array(_tone(), 16000, target_sr=16000)
        p1 = write_wav_from_pcm16(tmp_path / "a.wav", out["pcm"], 16000)
        p2 = write_wav_from_pcm16(tmp_path / "b.wav", out["pcm"], 16000)
        assert p1.read_bytes() == p2.read_bytes()
        assert len(build_wav_bytes(out["pcm"], 16000)) == 44 + len(out["pcm"])
        back = read_pcm16_payload(p1)
        assert back["sha256_pcm"] == out["sha256_pcm"]
        assert back["sample_rate"] == 16000 and back["channels"] == 1

    def test_truncated_wav_changes_pcm_hash(self, tmp_path: Path):
        out = canonical_pcm16_from_array(_tone(), 16000, target_sr=16000)
        path = write_wav_from_pcm16(tmp_path / "a.wav", out["pcm"], 16000)
        write_wav_from_pcm16(path, out["pcm"][: len(out["pcm"]) // 2], 16000)
        assert read_pcm16_payload(path)["sha256_pcm"] != out["sha256_pcm"]


class TestDiskGuards:
    def test_headroom_failure_is_fail_closed(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="Insufficient disk headroom"):
            assert_disk_headroom(tmp_path, need_bytes=10 ** 18, label="shard X")

    def test_headroom_passes_with_small_request(self, tmp_path: Path):
        info = assert_disk_headroom(tmp_path, need_bytes=1024, safety_bytes=0)
        assert info["free_bytes"] > 0

    def test_free_bytes_walks_up_to_existing_parent(self, tmp_path: Path):
        assert disk_free_bytes(tmp_path / "does" / "not" / "exist") > 0

    def test_symlinked_hf_blob_is_actually_deleted(self, tmp_path: Path):
        """Unlinking only the snapshot symlink would free zero bytes."""
        blobs = tmp_path / "blobs"
        snaps = tmp_path / "snapshots" / "rev"
        blobs.mkdir()
        snaps.mkdir(parents=True)
        blob = blobs / "deadbeef"
        blob.write_bytes(b"x" * 4096)
        link = snaps / "train.parquet"
        os.symlink(blob, link)
        freed = delete_hf_cache_blob(link)
        assert freed == 4096
        assert not blob.exists() and not link.exists()

    def test_cleanup_handles_plain_file_and_dirs(self, tmp_path: Path):
        plain = tmp_path / "shard.parquet"
        plain.write_bytes(b"y" * 100)
        extra = tmp_path / "tmpdir"
        extra.mkdir()
        (extra / "junk").write_bytes(b"z" * 50)
        freed = cleanup_shard_download(plain, extra_dirs=[extra])
        assert freed == 150
        assert not plain.exists() and not extra.exists()

    def test_cleanup_missing_path_is_not_an_error(self, tmp_path: Path):
        assert cleanup_shard_download(tmp_path / "nope.parquet") == 0
