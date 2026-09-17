"""Unit tests for ASR utilities (Notebook 03)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.asr_utils import (
    AUDIO_EXCLUSIONS_COLUMNS,
    CTC_FEASIBILITY_EXCLUSIONS_COLUMNS,
    EMPTY_NORMALIZED_REFERENCE,
    OOV_SUMMARY_COLUMNS,
    VALIDATION_PREDICTIONS_COLUMNS,
    CTCDataCollatorWithPadding,
    analyze_oov,
    assert_full_mode_supported,
    build_eligible_pool_and_sample,
    check_cache_pool_sufficiency,
    check_ctc_feasibility,
    checkpoint_belongs_to_run,
    classify_cache_status,
    compute_batch_metrics,
    compute_cer,
    compute_records_and_hours,
    compute_wer,
    count_min_ctc_target_length,
    create_artifact_manifest,
    derive_pilot_status,
    detect_frozen_leakage,
    empty_dataframe,
    estimate_encoder_output_frames,
    filter_empty_normalized_references,
    generate_run_id,
    get_device,
    get_environment_info,
    is_forbidden_test_path,
    preprocess_features_for_training,
    preprocess_waveform_for_training,
    require_best_checkpoint,
    resolve_positive_duration,
    resolve_training_duration_seconds,
    sample_pilot_data,
    sha256_file,
    validate_collator_batch,
    validate_prediction_lengths,
    validate_waveform_for_training,
    verify_artifacts_run_id,
    verify_notebook03_prerequisites,
    verify_pilot_sample,
)


# ---------------------------------------------------------------------------
# CTC Data Collator Tests
# ---------------------------------------------------------------------------

class TestCTCDataCollator:
    """Tests for CTCDataCollatorWithPadding."""

    @pytest.fixture
    def mock_processor(self):
        """Create mock Wav2Vec2Processor."""
        processor = MagicMock()

        def mock_fe_pad(features, **kwargs):
            batch_size = len(features)
            max_len = max(len(f["input_values"]) for f in features)
            padded = np.zeros((batch_size, max_len), dtype=np.float32)
            mask = np.zeros((batch_size, max_len), dtype=np.int64)
            for i, f in enumerate(features):
                arr = np.asarray(f["input_values"])
                padded[i, :len(arr)] = arr
                mask[i, :len(arr)] = 1
            return {
                "input_values": torch.tensor(padded),
                "attention_mask": torch.tensor(mask),
            }

        processor.feature_extractor.pad = mock_fe_pad

        def mock_tok_pad(features, **kwargs):
            if not features:
                return {"input_ids": torch.tensor([], dtype=torch.long)}
            batch_size = len(features)
            max_len = max(len(f["input_ids"]) for f in features)
            padded = torch.zeros((batch_size, max_len), dtype=torch.long)
            for i, f in enumerate(features):
                ids = f["input_ids"]
                padded[i, :len(ids)] = torch.tensor(ids)
            return {"input_ids": padded}

        processor.tokenizer.pad = mock_tok_pad
        processor.tokenizer.pad_token_id = 0
        return processor

    def test_collate_single_sample(self, mock_processor):
        collator = CTCDataCollatorWithPadding(processor=mock_processor)
        features = [{
            "input_values": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "labels": [1, 2, 3],
        }]
        batch = collator(features)
        assert "input_values" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_values"].shape == (1, 3)
        assert batch["labels"].shape == (1, 3)

    def test_collate_multiple_samples_different_lengths(self, mock_processor):
        collator = CTCDataCollatorWithPadding(processor=mock_processor)
        features = [
            {"input_values": np.array([0.1, 0.2], dtype=np.float32), "labels": [1, 2]},
            {"input_values": np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32), "labels": [3, 4, 5]},
        ]
        batch = collator(features)
        assert batch["input_values"].shape == (2, 4)
        assert batch["labels"].shape == (2, 3)

    def test_labels_padding_replaced_with_minus_100(self, mock_processor):
        collator = CTCDataCollatorWithPadding(processor=mock_processor)
        features = [
            {"input_values": np.array([0.1], dtype=np.float32), "labels": [1, 2, 3]},
            {"input_values": np.array([0.2], dtype=np.float32), "labels": [4]},
        ]
        batch = collator(features)
        labels = batch["labels"]
        assert labels[0, 0] == 1
        assert labels[0, 1] == 2
        assert labels[0, 2] == 3
        assert labels[1, 0] == 4
        assert labels[1, 1] == -100
        assert labels[1, 2] == -100

    def test_empty_batch_raises_error(self, mock_processor):
        collator = CTCDataCollatorWithPadding(processor=mock_processor)
        with pytest.raises(ValueError, match="Cannot collate empty batch"):
            collator([])

    def test_missing_input_values_raises_error(self, mock_processor):
        collator = CTCDataCollatorWithPadding(processor=mock_processor)
        with pytest.raises(ValueError, match="missing 'input_values'"):
            collator([{"labels": [1, 2]}])

    def test_empty_waveform_raises_error(self, mock_processor):
        collator = CTCDataCollatorWithPadding(processor=mock_processor)
        with pytest.raises(ValueError, match="Empty waveform"):
            collator([{"input_values": np.array([], dtype=np.float32), "labels": [1]}])

    def test_nan_in_waveform_raises_error(self, mock_processor):
        collator = CTCDataCollatorWithPadding(processor=mock_processor)
        wav = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            collator([{"input_values": wav, "labels": [1]}])

    def test_inf_in_waveform_raises_error(self, mock_processor):
        collator = CTCDataCollatorWithPadding(processor=mock_processor)
        wav = np.array([0.1, np.inf, 0.3], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            collator([{"input_values": wav, "labels": [1]}])


class TestValidateCollatorBatch:
    """Tests for validate_collator_batch."""

    def test_valid_batch_with_labels(self):
        batch = {
            "input_values": torch.randn(2, 100),
            "attention_mask": torch.ones(2, 100),
            "labels": torch.tensor([[1, 2, -100], [3, 4, 5]]),
        }
        result = validate_collator_batch(batch)
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["has_labels"] is True
        assert result["input_values_shape"] == [2, 100]
        assert result["labels_shape"] == [2, 3]

    def test_missing_input_values(self):
        batch = {"attention_mask": torch.ones(2, 100)}
        result = validate_collator_batch(batch)
        assert result["valid"] is False
        assert "Missing input_values" in result["errors"]

    def test_nan_in_input_values(self):
        batch = {"input_values": torch.tensor([[1.0, float("nan")]])}
        result = validate_collator_batch(batch)
        assert result["valid"] is False
        assert any("NaN/Inf" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Waveform Preprocessing Tests
# ---------------------------------------------------------------------------

class TestPreprocessWaveform:
    """Tests for waveform preprocessing."""

    def test_valid_float32_mono(self):
        wav = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = preprocess_waveform_for_training(wav, 16000, 16000)
        assert result is not None
        assert result.dtype == np.float32
        assert len(result) == 3

    def test_int16_converted_to_float(self):
        wav = np.array([1000, 2000, 3000], dtype=np.int16)
        result = preprocess_waveform_for_training(wav, 16000, 16000)
        assert result is not None
        assert result.dtype == np.float32
        assert np.abs(result).max() <= 1.0

    def test_stereo_converted_to_mono(self):
        wav = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        result = preprocess_waveform_for_training(wav, 16000, 16000)
        assert result is not None
        assert result.ndim == 1

    def test_resampling(self):
        wav = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000).astype(np.float32)
        result = preprocess_waveform_for_training(wav, 8000, 16000)
        assert result is not None
        assert abs(len(result) - 16000) < 100

    def test_invalid_sampling_rate_returns_none(self):
        wav = np.array([0.1, 0.2], dtype=np.float32)
        assert preprocess_waveform_for_training(wav, None, 16000) is None
        assert preprocess_waveform_for_training(wav, 0, 16000) is None
        assert preprocess_waveform_for_training(wav, -1, 16000) is None

    def test_nan_waveform_returns_none(self):
        wav = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        assert preprocess_waveform_for_training(wav, 16000, 16000) is None

    def test_empty_waveform_returns_none(self):
        wav = np.array([], dtype=np.float32)
        assert preprocess_waveform_for_training(wav, 16000, 16000) is None


class TestValidateWaveform:
    """Tests for validate_waveform_for_training."""

    def test_valid_waveform(self):
        wav = np.zeros(16000, dtype=np.float32)
        wav[100:200] = 0.1
        result = validate_waveform_for_training(wav)
        assert result["ok"] is True
        assert result["reason"] == "ok"
        assert abs(result["duration_sec"] - 1.0) < 0.01

    def test_null_waveform(self):
        result = validate_waveform_for_training(None)
        assert result["ok"] is False
        assert result["reason"] == "null_waveform"

    def test_empty_waveform(self):
        result = validate_waveform_for_training(np.array([], dtype=np.float32))
        assert result["ok"] is False
        assert result["reason"] == "empty"

    def test_too_short(self):
        wav = np.zeros(100, dtype=np.float32)
        result = validate_waveform_for_training(wav, min_duration=0.1)
        assert result["ok"] is False
        assert result["reason"] == "too_short"

    def test_too_long(self):
        wav = np.zeros(16000 * 60, dtype=np.float32)
        wav[0] = 0.1
        result = validate_waveform_for_training(wav, max_duration=30.0)
        assert result["ok"] is False
        assert result["reason"] == "too_long"

    def test_non_finite(self):
        wav = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        result = validate_waveform_for_training(wav)
        assert result["ok"] is False
        assert result["reason"] == "non_finite"


# ---------------------------------------------------------------------------
# Pilot Sampling Tests
# ---------------------------------------------------------------------------

class TestPilotSampling:
    """Tests for pilot data sampling."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "record_uid": [f"uid_{i}" for i in range(100)],
            "group_id": [f"group_{i // 10}" for i in range(100)],
            "text": [f"text_{i}" for i in range(100)],
        })

    def test_deterministic_sampling(self, sample_df):
        sample1 = sample_pilot_data(sample_df, 20, seed=42)
        sample2 = sample_pilot_data(sample_df, 20, seed=42)
        assert list(sample1["record_uid"]) == list(sample2["record_uid"])

    def test_different_seeds_different_samples(self, sample_df):
        sample1 = sample_pilot_data(sample_df, 20, seed=42)
        sample2 = sample_pilot_data(sample_df, 20, seed=123)
        assert list(sample1["record_uid"]) != list(sample2["record_uid"])

    def test_sample_size(self, sample_df):
        sample = sample_pilot_data(sample_df, 20, seed=42)
        assert len(sample) == 20

    def test_sample_covers_multiple_groups(self, sample_df):
        sample = sample_pilot_data(sample_df, 30, seed=42)
        n_groups = sample["group_id"].nunique()
        assert n_groups >= 3

    def test_sample_larger_than_df_returns_full(self, sample_df):
        sample = sample_pilot_data(sample_df, 200, seed=42)
        assert len(sample) == 100

    def test_empty_df(self):
        empty_df = pd.DataFrame(columns=["record_uid", "group_id"])
        sample = sample_pilot_data(empty_df, 10, seed=42)
        assert len(sample) == 0

    def test_preserves_original_order(self, sample_df):
        sample = sample_pilot_data(sample_df, 30, seed=42)
        indices = sample.index.tolist()
        assert indices == sorted(indices)


class TestVerifyPilotSample:
    """Tests for verify_pilot_sample."""

    def test_valid_sample(self):
        full_df = pd.DataFrame({
            "record_uid": [f"uid_{i}" for i in range(100)],
            "group_id": [f"group_{i // 10}" for i in range(100)],
        })
        sample_df = full_df.iloc[:20].copy()
        result = verify_pilot_sample(sample_df, full_df, 20)
        assert result["passed"] is True
        assert result["sample_size"] == 20

    def test_size_mismatch(self):
        full_df = pd.DataFrame({
            "record_uid": [f"uid_{i}" for i in range(100)],
            "group_id": [f"group_{i // 10}" for i in range(100)],
        })
        sample_df = full_df.iloc[:15].copy()
        result = verify_pilot_sample(sample_df, full_df, 20)
        assert result["passed"] is False
        assert any("Size mismatch" in e for e in result["errors"])

    def test_uid_not_in_full(self):
        full_df = pd.DataFrame({
            "record_uid": [f"uid_{i}" for i in range(100)],
            "group_id": [f"group_{i // 10}" for i in range(100)],
        })
        sample_df = pd.DataFrame({
            "record_uid": ["uid_0", "uid_999"],
            "group_id": ["group_0", "group_99"],
        })
        result = verify_pilot_sample(sample_df, full_df, 2)
        assert result["passed"] is False
        assert any("not in full data" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Contract Verification Tests
# ---------------------------------------------------------------------------

class TestVerifyPrerequisites:
    """Tests for verify_notebook03_prerequisites."""

    def test_valid_contract_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            contract = {
                "dataset_revision": "abc123",
                "train": {"clean_count": 1000},
                "validation": {"clean_count": 100},
            }
            contract_path = tmpdir / "contract.json"
            contract_path.write_text(json.dumps(contract))
            tok_dir = tmpdir / "tokenizer"
            tok_dir.mkdir()
            (tok_dir / "provenance.json").write_text("{}")
            (tok_dir / "vocab.json").write_text('{"[PAD]": 0, "[UNK]": 1}')
            result = verify_notebook03_prerequisites(
                contract_path=contract_path,
                tokenizer_dir=tok_dir,
                expected_dataset_revision="abc123",
                expected_train_clean_count=1000,
                expected_validation_clean_count=100,
            )
            assert result["passed"] is True
            assert result["vocab_size"] == 2

    def test_missing_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            result = verify_notebook03_prerequisites(
                contract_path=tmpdir / "missing.json",
                tokenizer_dir=tmpdir / "tokenizer",
                expected_dataset_revision="abc123",
                expected_train_clean_count=1000,
                expected_validation_clean_count=100,
            )
            assert result["passed"] is False
            assert any("not found" in e for e in result["errors"])

    def test_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            contract = {
                "dataset_revision": "abc123",
                "train": {"clean_count": 1000},
                "validation": {"clean_count": 100},
            }
            contract_path = tmpdir / "contract.json"
            contract_path.write_text(json.dumps(contract))
            tok_dir = tmpdir / "tokenizer"
            tok_dir.mkdir()
            (tok_dir / "provenance.json").write_text("{}")
            (tok_dir / "vocab.json").write_text('{"[PAD]": 0}')
            result = verify_notebook03_prerequisites(
                contract_path=contract_path,
                tokenizer_dir=tok_dir,
                expected_dataset_revision="different",
                expected_train_clean_count=1000,
                expected_validation_clean_count=100,
            )
            assert result["passed"] is False
            assert result["revision_match"] is False

    def test_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            contract = {
                "dataset_revision": "abc123",
                "train": {"clean_count": 999},
                "validation": {"clean_count": 100},
            }
            contract_path = tmpdir / "contract.json"
            contract_path.write_text(json.dumps(contract))
            tok_dir = tmpdir / "tokenizer"
            tok_dir.mkdir()
            (tok_dir / "provenance.json").write_text("{}")
            (tok_dir / "vocab.json").write_text('{"[PAD]": 0}')
            result = verify_notebook03_prerequisites(
                contract_path=contract_path,
                tokenizer_dir=tok_dir,
                expected_dataset_revision="abc123",
                expected_train_clean_count=1000,
                expected_validation_clean_count=100,
            )
            assert result["passed"] is False
            assert result["train_count_match"] is False


class TestForbiddenTestPath:
    """Tests for is_forbidden_test_path."""

    def test_forbidden_paths(self):
        assert is_forbidden_test_path("data/rq1_test.csv") is True
        assert is_forbidden_test_path("data/RQ1_TEST.csv") is True
        assert is_forbidden_test_path("/path/to/frozen_test/data.csv") is True
        assert is_forbidden_test_path("test.csv") is True

    def test_allowed_paths(self):
        assert is_forbidden_test_path("data/rq1_train.csv") is False
        assert is_forbidden_test_path("data/rq1_validation.csv") is False
        assert is_forbidden_test_path("test_utils.py") is False


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for metric computation."""

    def test_cer_identical(self):
        assert compute_cer("hello", "hello") == 0.0

    def test_cer_completely_different(self):
        cer = compute_cer("abc", "xyz")
        assert cer == 1.0

    def test_cer_empty_reference(self):
        assert compute_cer("", "hello") == 1.0
        assert compute_cer("", "") == 0.0

    def test_wer_identical(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_wer_completely_different(self):
        wer = compute_wer("hello world", "foo bar")
        assert wer == 1.0

    def test_batch_metrics(self):
        refs = ["hello world", "foo bar"]
        hyps = ["hello world", "foo baz"]
        metrics = compute_batch_metrics(refs, hyps)
        assert "cer" in metrics
        assert "wer" in metrics
        assert metrics["cer"] >= 0
        assert metrics["wer"] >= 0

    def test_batch_metrics_empty(self):
        metrics = compute_batch_metrics([], [])
        assert metrics["cer"] == 0.0
        assert metrics["wer"] == 0.0


# ---------------------------------------------------------------------------
# Environment and Device Tests
# ---------------------------------------------------------------------------

class TestEnvironmentInfo:
    """Tests for get_environment_info."""

    def test_returns_dict(self):
        info = get_environment_info()
        assert isinstance(info, dict)
        assert "python_version" in info
        assert "platform" in info
        assert "torch_version" in info


class TestGetDevice:
    """Tests for get_device."""

    def test_cpu_when_requested(self):
        device = get_device(preferred="cpu")
        assert device.type == "cpu"

    def test_auto_returns_device(self):
        device = get_device(preferred="auto", run_mode="pilot")
        assert device.type in ["cpu", "cuda", "mps"]


# ---------------------------------------------------------------------------
# Artifact Utilities Tests
# ---------------------------------------------------------------------------

class TestSha256File:
    """Tests for sha256_file."""

    def test_sha256_consistent(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            path = Path(f.name)
        try:
            hash1 = sha256_file(path)
            hash2 = sha256_file(path)
            assert hash1 == hash2
            assert len(hash1) == 64
        finally:
            path.unlink()


class TestArtifactManifest:
    """Tests for create_artifact_manifest."""

    def test_creates_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "file1.txt").write_text("content1")
            (tmpdir / "file2.json").write_text('{"key": "value"}')
            manifest = create_artifact_manifest(tmpdir)
            assert len(manifest) == 2
            assert all("path" in m for m in manifest)
            assert all("sha256" in m for m in manifest)
            assert all("size_bytes" in m for m in manifest)


class TestGenerateRunId:
    """Tests for generate_run_id."""

    def test_generates_uuid(self):
        run_id = generate_run_id()
        assert len(run_id) == 36
        assert run_id.count("-") == 4

    def test_unique_ids(self):
        ids = [generate_run_id() for _ in range(100)]
        assert len(set(ids)) == 100


# ---------------------------------------------------------------------------
# Pilot Sampling — edge cases (group-aware, deterministic)
# ---------------------------------------------------------------------------

class TestPilotSamplingEdgeCases:
    """Edge cases for sample_pilot_data."""

    def test_more_groups_than_samples(self):
        df = pd.DataFrame({
            "record_uid": [f"u{i}" for i in range(10)],
            "group_id": [f"g{i}" for i in range(10)],
        })
        sample = sample_pilot_data(df, 3, seed=1)
        assert len(sample) == 3
        assert sample["record_uid"].nunique() == 3
        assert sample["group_id"].nunique() == 3

    def test_skewed_group_sizes(self):
        df = pd.DataFrame({
            "record_uid": [f"u{i}" for i in range(6)],
            "group_id": ["a", "b", "c", "c", "c", "c"],
        })
        sample = sample_pilot_data(df, 3, seed=7)
        assert len(sample) == 3
        assert sample["record_uid"].nunique() == 3
        assert sample["group_id"].nunique() >= 2

    def test_n_samples_zero(self):
        df = pd.DataFrame({"record_uid": ["a", "b"], "group_id": ["g", "g"]})
        sample = sample_pilot_data(df, 0, seed=1)
        assert len(sample) == 0
        assert list(sample.columns) == list(df.columns)

    def test_n_samples_larger_than_dataset(self):
        df = pd.DataFrame({"record_uid": ["a", "b"], "group_id": ["g", "g"]})
        sample = sample_pilot_data(df, 999, seed=1)
        assert len(sample) == 2

    def test_single_group(self):
        df = pd.DataFrame({
            "record_uid": [f"u{i}" for i in range(20)],
            "group_id": ["only"] * 20,
        })
        sample = sample_pilot_data(df, 5, seed=3)
        assert len(sample) == 5
        assert sample["group_id"].nunique() == 1
        assert sample["record_uid"].nunique() == 5

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["record_uid", "group_id"])
        sample = sample_pilot_data(df, 5, seed=3)
        assert len(sample) == 0

    def test_duplicate_uids_collapsed(self):
        df = pd.DataFrame({
            "record_uid": ["a", "a", "b", "c"],
            "group_id": ["g", "g", "g", "g"],
        })
        sample = sample_pilot_data(df, 3, seed=2)
        assert len(sample) == 3
        assert sample["record_uid"].nunique() == 3

    def test_metadata_preserved(self):
        df = pd.DataFrame({
            "record_uid": [f"u{i}" for i in range(10)],
            "group_id": [f"g{i % 3}" for i in range(10)],
            "duration_seconds": [float(i) for i in range(10)],
            "text_bahnar": [f"t{i}" for i in range(10)],
        })
        sample = sample_pilot_data(df, 4, seed=5)
        assert set(sample.columns) == set(df.columns)
        assert sample["duration_seconds"].notna().all()

    def test_never_exceeds_request(self):
        df = pd.DataFrame({
            "record_uid": [f"u{i}" for i in range(100)],
            "group_id": [f"g{i % 5}" for i in range(100)],
        })
        for n in [1, 7, 33, 50]:
            sample = sample_pilot_data(df, n, seed=11)
            assert len(sample) == n

    def test_stable_output_order(self):
        df = pd.DataFrame({
            "record_uid": [f"u{i}" for i in range(50)],
            "group_id": [f"g{i % 4}" for i in range(50)],
        })
        sample = sample_pilot_data(df, 10, seed=9)
        idx = sample.index.tolist()
        assert idx == sorted(idx)


# ---------------------------------------------------------------------------
# Feature extractor preprocessing (single canonical path)
# ---------------------------------------------------------------------------

class TestPreprocessFeatures:
    """Tests for preprocess_features_for_training."""

    @pytest.fixture
    def mock_processor(self):
        processor = MagicMock()
        processor.feature_extractor.sampling_rate = 16000
        calls = {}

        def _call(wav, sampling_rate=None, return_tensors=None, **kw):
            calls["sampling_rate"] = sampling_rate
            calls["return_tensors"] = return_tensors
            arr = np.asarray(wav, dtype=np.float32)
            mean = arr.mean()
            std = arr.std() + 1e-7
            norm = (arr - mean) / std
            return {"input_values": norm[None, :]}

        processor.side_effect = _call
        processor._calls = calls
        return processor

    def test_called_with_16000(self, mock_processor):
        wav = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        out = preprocess_features_for_training(mock_processor, wav, 16000)
        assert mock_processor._calls["sampling_rate"] == 16000
        assert out.dtype == np.float32
        assert out.ndim == 1

    def test_output_finite(self, mock_processor):
        wav = np.random.RandomState(0).randn(2000).astype(np.float32)
        out = preprocess_features_for_training(mock_processor, wav, 16000)
        assert np.isfinite(out).all()

    def test_resamples_before_extractor(self, mock_processor):
        wav = np.sin(2 * np.pi * 220 * np.arange(8000) / 8000).astype(np.float32)
        out = preprocess_features_for_training(mock_processor, wav, 8000)
        assert abs(len(out) - 16000) < 200

    def test_train_predict_same_contract(self, mock_processor):
        wav = np.random.RandomState(1).randn(1600).astype(np.float32)
        a = preprocess_features_for_training(mock_processor, wav, 16000)
        b = preprocess_features_for_training(mock_processor, wav, 16000)
        assert np.allclose(a, b)

    def test_bad_sampling_rate_extractor_raises(self):
        processor = MagicMock()
        processor.feature_extractor.sampling_rate = 8000
        wav = np.array([0.1, 0.2], dtype=np.float32)
        with pytest.raises(ValueError, match="sampling_rate"):
            preprocess_features_for_training(processor, wav, 16000)

    def test_collator_pads_extractor_features(self):
        processor = MagicMock()

        def fe_pad(features, **kwargs):
            max_len = max(len(f["input_values"]) for f in features)
            bs = len(features)
            padded = np.zeros((bs, max_len), dtype=np.float32)
            mask = np.zeros((bs, max_len), dtype=np.int64)
            for i, f in enumerate(features):
                arr = np.asarray(f["input_values"], dtype=np.float32)
                padded[i, : len(arr)] = arr
                mask[i, : len(arr)] = 1
            return {"input_values": torch.tensor(padded), "attention_mask": torch.tensor(mask)}

        def tok_pad(features, **kwargs):
            max_len = max(len(f["input_ids"]) for f in features)
            bs = len(features)
            out = torch.zeros((bs, max_len), dtype=torch.long)
            for i, f in enumerate(features):
                out[i, : len(f["input_ids"])] = torch.tensor(f["input_ids"])
            return {"input_ids": out}

        processor.feature_extractor.pad = fe_pad
        processor.feature_extractor.sampling_rate = 16000
        processor.tokenizer.pad = tok_pad
        processor.tokenizer.pad_token_id = 0

        collator = CTCDataCollatorWithPadding(processor=processor)
        feats = [
            {"input_values": np.array([0.1, 0.2, 0.3], dtype=np.float32), "labels": [3, 4, 5]},
            {"input_values": np.array([0.4, 0.5], dtype=np.float32), "labels": [6]},
        ]
        batch = collator(feats)
        assert batch["input_values"].shape == (2, 3)
        assert batch["labels"].shape == (2, 3)
        assert batch["labels"][1, 1].item() == -100
        assert batch["labels"][1, 2].item() == -100


# ---------------------------------------------------------------------------
# Strict cache classification
# ---------------------------------------------------------------------------

class TestClassifyCacheStatus:
    """Tests for classify_cache_status using real WAV + sidecar."""

    def _make_cache(self, tmpdir, record_uid, revision, sr=16000, dur=1.0):
        from src.data_utils import (
            write_pcm16_wav,
            validate_cache_wav,
            build_cache_provenance,
            write_cache_sidecar,
        )
        path = Path(tmpdir) / "clip.wav"
        n = int(sr * dur)
        wav = (0.05 * np.sin(2 * np.pi * 200 * np.arange(n) / sr)).astype(np.float32)
        write_pcm16_wav(path, wav, sr)
        qa = validate_cache_wav(path, target_sr=16000, min_duration=0.1, max_duration=30.0)
        prov = build_cache_provenance(
            record_uid=record_uid,
            dataset_revision=revision,
            target_sr=16000,
            source_qa={"ok": True, "hard_ok": True, "reason": "ok"},
            cache_sha256=qa["cache_sha256"],
        )
        write_cache_sidecar(path, prov)
        return path, qa["cache_sha256"]

    def test_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._make_cache(tmp, "rec1", "rev1")
            res = classify_cache_status(
                path, record_uid="rec1", dataset_revision="rev1",
                min_duration=0.1, max_duration=30.0,
            )
            assert res["ok"] is True
            assert res["reason"] == "ok"

    def test_missing(self):
        res = classify_cache_status(
            "/nonexistent/none.wav", record_uid="x", dataset_revision="r",
            min_duration=0.1, max_duration=30.0,
        )
        assert res["ok"] is False
        assert res["reason"] == "cache_missing"

    def test_sidecar_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._make_cache(tmp, "rec1", "rev1")
            path.with_suffix(".json").unlink()
            res = classify_cache_status(
                path, record_uid="rec1", dataset_revision="rev1",
                min_duration=0.1, max_duration=30.0,
            )
            assert res["reason"] == "cache_sidecar_missing"

    def test_provenance_mismatch_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._make_cache(tmp, "rec1", "rev1")
            res = classify_cache_status(
                path, record_uid="rec1", dataset_revision="DIFFERENT",
                min_duration=0.1, max_duration=30.0,
            )
            assert res["reason"] == "cache_provenance_mismatch"

    def test_provenance_mismatch_processing_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._make_cache(tmp, "rec1", "rev1")
            res = classify_cache_status(
                path, record_uid="rec1", dataset_revision="rev1",
                min_duration=0.1, max_duration=30.0,
                expected_processing_version="some_other_version",
            )
            assert res["reason"] == "cache_provenance_mismatch"

    def test_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._make_cache(tmp, "rec1", "rev1")
            side = path.with_suffix(".json")
            prov = json.loads(side.read_text())
            prov["cache_sha256"] = "0" * 64
            side.write_text(json.dumps(prov))
            res = classify_cache_status(
                path, record_uid="rec1", dataset_revision="rev1",
                min_duration=0.1, max_duration=30.0,
            )
            assert res["reason"] == "cache_sha_mismatch"

    def test_duration_too_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._make_cache(tmp, "rec1", "rev1", dur=0.2)
            res = classify_cache_status(
                path, record_uid="rec1", dataset_revision="rev1",
                min_duration=0.5, max_duration=30.0,
            )
            assert res["reason"] == "duration_too_short"


# ---------------------------------------------------------------------------
# CTC feasibility
# ---------------------------------------------------------------------------

class TestCTCFeasibility:
    def test_min_ctc_length_no_repeats(self):
        assert count_min_ctc_target_length([1, 2, 3]) == 3

    def test_min_ctc_length_with_repeats(self):
        assert count_min_ctc_target_length([1, 1, 2, 2, 2]) == 5 + 3

    def test_estimate_frames_monotonic(self):
        assert estimate_encoder_output_frames(16000) > estimate_encoder_output_frames(8000)
        assert estimate_encoder_output_frames(0) == 0

    def test_feasible_long_audio(self):
        res = check_ctc_feasibility([1, 2, 3], 16000)
        assert res["feasible"] is True
        assert res["reason"] == "ok"

    def test_infeasible_short_audio(self):
        res = check_ctc_feasibility(list(range(1, 30)), 400)
        assert res["feasible"] is False
        assert res["reason"] == "insufficient_frames_for_ctc_target"

    def test_empty_target(self):
        res = check_ctc_feasibility([], 16000)
        assert res["feasible"] is False
        assert res["reason"] == "empty_target"


# ---------------------------------------------------------------------------
# OOV analysis
# ---------------------------------------------------------------------------

class TestOOV:
    def test_detects_oov_and_scope(self):
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4}
        texts = ["a b", "a z", "z z"]
        df = analyze_oov(texts, vocab, scope="active_validation")
        assert set(df.columns) == set(OOV_SUMMARY_COLUMNS)
        assert (df["scope"] == "active_validation").all()
        zrow = df[df["char"] == "z"].iloc[0]
        assert zrow["count"] == 3
        assert zrow["n_records"] == 2
        assert abs(zrow["record_ratio"] - (2 / 3)) < 1e-9

    def test_no_oov_returns_empty_with_header(self):
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4}
        df = analyze_oov(["a b", "b a"], vocab, scope="full_clean_validation")
        assert len(df) == 0
        assert list(df.columns) == OOV_SUMMARY_COLUMNS

    def test_does_not_mutate_vocab(self):
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3}
        before = dict(vocab)
        analyze_oov(["a x y"], vocab, scope="active_validation")
        assert vocab == before

    def test_uppercase_and_punctuation_not_oov_after_normalization(self):
        """Regression: OOV is counted on normalize_bahnar_ctc_v1 output, not raw text."""
        from src.data_utils import normalize_bahnar_ctc_v1

        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4, "c": 5}
        # Raw has uppercase + punctuation that would look like OOV without normalization.
        raw_texts = ["Abc!", "A, B.", "C??"]
        # Without normalization these chars would be flagged:
        raw_oov_chars = set()
        for t in raw_texts:
            for ch in t:
                if not ch.isspace() and ch not in vocab:
                    raw_oov_chars.add(ch)
        assert raw_oov_chars & set("ABC!,.?")  # sanity: raw looks OOVy

        df = analyze_oov(raw_texts, vocab, scope="full_clean_validation",
                         normalize_fn=normalize_bahnar_ctc_v1)
        reported = set(df["char"].tolist()) if len(df) else set()
        assert "A" not in reported
        assert "B" not in reported
        assert "C" not in reported
        assert "!" not in reported
        assert "," not in reported
        assert "." not in reported
        assert "?" not in reported
        # After casefold + punct→space, only a/b/c remain → no OOV
        assert len(df) == 0

    def test_true_oov_survives_normalization(self):
        from src.data_utils import normalize_bahnar_ctc_v1

        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3}
        df = analyze_oov(["A z!", "z"], vocab, scope="active_validation",
                         normalize_fn=normalize_bahnar_ctc_v1)
        assert set(df["char"]) == {"z"}


class TestResolveTrainingDuration:
    def test_prefers_positive_finite_train_runtime(self):
        out = resolve_training_duration_seconds(12.5, 99.0, global_step=5)
        assert out["training_duration_seconds"] == 12.5
        assert out["perf_counter_seconds"] == 99.0
        assert out["train_runtime_seconds"] == 12.5
        assert out["duration_source"] == "train_runtime"
        assert abs(out["step_time_seconds"] - (12.5 / 5)) < 1e-9
        assert "wall_clock_seconds" not in out

    def test_falls_back_to_perf_counter_when_runtime_invalid(self):
        for bad in (None, 0, -1, float("nan"), float("inf"), "x"):
            out = resolve_training_duration_seconds(bad, 40.0, global_step=8)
            assert out["training_duration_seconds"] == 40.0
            assert out["perf_counter_seconds"] == 40.0
            assert out["train_runtime_seconds"] is None
            assert out["duration_source"] == "perf_counter_fallback"
            assert abs(out["step_time_seconds"] - 5.0) < 1e-9

    def test_step_time_uses_official_duration(self):
        out = resolve_training_duration_seconds(20.0, 100.0, global_step=4)
        assert out["step_time_seconds"] == 5.0  # 20/4, not 100/4

    def test_both_missing_returns_zero(self):
        out = resolve_training_duration_seconds(None, None, global_step=3)
        assert out["training_duration_seconds"] == 0.0
        assert out["train_runtime_seconds"] is None
        assert out["perf_counter_seconds"] is None
        assert out["step_time_seconds"] == 0.0
        assert out["duration_source"] == "unavailable"

    def test_required_reporting_keys_present(self):
        out = resolve_training_duration_seconds(10.0, 11.0, global_step=2)
        for key in (
            "training_duration_seconds",
            "train_runtime_seconds",
            "perf_counter_seconds",
            "duration_source",
            "step_time_seconds",
        ):
            assert key in out



# ---------------------------------------------------------------------------
# Strict batch metrics
# ---------------------------------------------------------------------------

class TestBatchMetricsStrict:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            compute_batch_metrics(["a", "b"], ["a"])

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="Empty reference"):
            compute_batch_metrics(["hello", ""], ["hello", "x"])

    def test_empty_reference_allowed(self):
        res = compute_batch_metrics(["hello", ""], ["hello", "x"], allow_empty_reference=True)
        assert res["n"] == 2

    def test_known_cer(self):
        res = compute_batch_metrics(["abcd"], ["abxd"])
        assert abs(res["cer"] - 0.25) < 1e-9

    def test_known_wer(self):
        res = compute_batch_metrics(["one two three four"], ["one two three x"])
        assert abs(res["wer"] - 0.25) < 1e-9

    def test_identical_zero(self):
        res = compute_batch_metrics(["hblock bahnar", "abc"], ["hblock bahnar", "abc"])
        assert res["cer"] == 0.0
        assert res["wer"] == 0.0

    def test_empty_prediction(self):
        res = compute_batch_metrics(["abc"], [""])
        assert res["cer"] == 1.0

    def test_unicode_bahnar(self):
        ref = "ĭ ơ ư ằ"
        res = compute_batch_metrics([ref], [ref])
        assert res["cer"] == 0.0

    def test_per_sample_lengths(self):
        res = compute_batch_metrics(["abc", "defg"], ["abc", "xefg"])
        assert len(res["per_sample_cer"]) == 2
        assert res["per_sample_cer"][0] == 0.0


# ---------------------------------------------------------------------------
# Fixed schemas
# ---------------------------------------------------------------------------

class TestFixedSchemas:
    def test_empty_dataframe_headers(self):
        for cols in [
            AUDIO_EXCLUSIONS_COLUMNS,
            CTC_FEASIBILITY_EXCLUSIONS_COLUMNS,
            VALIDATION_PREDICTIONS_COLUMNS,
            OOV_SUMMARY_COLUMNS,
        ]:
            df = empty_dataframe(cols)
            assert list(df.columns) == cols
            assert len(df) == 0

    def test_audio_exclusions_schema_fields(self):
        assert "exception_type" in AUDIO_EXCLUSIONS_COLUMNS
        assert "source_duration_seconds" in AUDIO_EXCLUSIONS_COLUMNS
        assert "processed_duration_seconds" in AUDIO_EXCLUSIONS_COLUMNS


# ---------------------------------------------------------------------------
# Prediction length validation (no silent truncation)
# ---------------------------------------------------------------------------

class TestValidatePredictionLengths:
    def test_all_equal_passes(self):
        validate_prediction_lengths(50, 50, 50, 50)  # no raise

    def test_missing_prediction_raises(self):
        with pytest.raises(ValueError):
            validate_prediction_lengths(49, 50, 50, 50)

    def test_extra_prediction_raises(self):
        with pytest.raises(ValueError):
            validate_prediction_lengths(51, 50, 50, 50)

    def test_label_metadata_mismatch_raises(self):
        with pytest.raises(ValueError):
            validate_prediction_lengths(50, 50, 49, 50)

    def test_dataset_mismatch_raises(self):
        with pytest.raises(ValueError):
            validate_prediction_lengths(50, 50, 50, 48)


# ---------------------------------------------------------------------------
# Empty normalized reference exclusion
# ---------------------------------------------------------------------------

class TestFilterEmptyNormalizedReferences:
    def _df(self):
        return pd.DataFrame([
            {"record_uid": "u1", "record_id": 1, "group_id": "g1",
             "text_bahnar": "bơngай", "duration_seconds": 2.0, "processed_duration_seconds": 2.0},
            {"record_uid": "u2", "record_id": 2, "group_id": "g2",
             "text_bahnar": "   ", "duration_seconds": 1.0, "processed_duration_seconds": 1.0},
            {"record_uid": "u3", "record_id": 3, "group_id": "g3",
             "text_bahnar": "", "duration_seconds": 1.0, "processed_duration_seconds": 1.0},
        ])

    def test_empty_reference_excluded(self):
        kept, excluded = filter_empty_normalized_references(
            self._df(), text_col="text_bahnar", normalize_fn=lambda s: (s or "").strip(),
            run_id="R", split="train")
        assert set(kept["record_uid"]) == {"u1"}
        assert len(excluded) == 2
        assert all(r["reason"] == EMPTY_NORMALIZED_REFERENCE for r in excluded)
        # excluded rows match the audio-exclusions schema
        assert all(set(AUDIO_EXCLUSIONS_COLUMNS).issubset(r.keys()) for r in excluded)

    def test_no_empty_keeps_all(self):
        df = self._df().iloc[[0]]
        kept, excluded = filter_empty_normalized_references(
            df, text_col="text_bahnar", normalize_fn=lambda s: (s or "").strip())
        assert len(kept) == 1 and len(excluded) == 0


# ---------------------------------------------------------------------------
# Records + hours recomputation
# ---------------------------------------------------------------------------

class TestComputeRecordsAndHours:
    def test_uses_processed_duration(self):
        df = pd.DataFrame({"processed_duration_seconds": [3600.0, 1800.0]})
        out = compute_records_and_hours(df)
        assert out["records"] == 2
        assert out["hours"] == pytest.approx(1.5)

    def test_empty_df(self):
        out = compute_records_and_hours(pd.DataFrame())
        assert out == {"records": 0, "hours": 0.0}

    def test_falls_back_to_duration_seconds(self):
        df = pd.DataFrame({"processed_duration_seconds": [None, None],
                           "duration_seconds": [3600.0, 3600.0]})
        out = compute_records_and_hours(df)
        assert out["records"] == 2
        assert out["hours"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

class TestCheckpointHelpers:
    def test_checkpoint_inside_run_dir(self, tmp_path):
        run = tmp_path / "checkpoints" / "notebook03" / "RUNID"
        cp = run / "checkpoint-20"
        cp.mkdir(parents=True)
        assert checkpoint_belongs_to_run(str(cp), str(run)) is True

    def test_checkpoint_outside_run_dir_rejected(self, tmp_path):
        run = tmp_path / "checkpoints" / "notebook03" / "RUNID"
        run.mkdir(parents=True)
        other = tmp_path / "checkpoints" / "notebook03" / "OTHER" / "checkpoint-20"
        other.mkdir(parents=True)
        assert checkpoint_belongs_to_run(str(other), str(run)) is False

    def test_checkpoint_none_rejected(self, tmp_path):
        assert checkpoint_belongs_to_run(None, str(tmp_path)) is False

    def test_require_best_checkpoint_missing_raises(self):
        with pytest.raises(RuntimeError):
            require_best_checkpoint(None)
        with pytest.raises(RuntimeError):
            require_best_checkpoint("")

    def test_require_best_checkpoint_returns_path(self):
        assert require_best_checkpoint("/x/checkpoint-20") == "/x/checkpoint-20"


# ---------------------------------------------------------------------------
# Status derivation + full-mode lock
# ---------------------------------------------------------------------------

class TestStatusAndFullModeLock:
    def test_success_pilot_requires_pilot_passed(self):
        assert derive_pilot_status("pilot", True) == "SUCCESS_PILOT"
        assert derive_pilot_status("pilot", False) == "FAILED"

    def test_status_never_success_pilot_when_not_passed(self):
        # The core invariant: not passed => never SUCCESS_PILOT.
        assert derive_pilot_status("pilot", False) != "SUCCESS_PILOT"

    def test_full_mode_never_success_here(self):
        assert derive_pilot_status("full", True) == "FAILED"

    def test_full_mode_locked_raises(self):
        with pytest.raises(RuntimeError, match="Full training pipeline is not implemented"):
            assert_full_mode_supported("full", False)

    def test_full_mode_allowed_when_implemented(self):
        assert_full_mode_supported("full", True)  # no raise

    def test_pilot_mode_never_locked(self):
        assert_full_mode_supported("pilot", False)  # no raise


# ---------------------------------------------------------------------------
# Cache pool sufficiency (fail clearly, no download)
# ---------------------------------------------------------------------------

class TestCachePoolSufficiency:
    def test_sufficient(self):
        out = check_cache_pool_sufficiency(661, 200, "train")
        assert out["ok"] is True and out["deficit"] == 0

    def test_insufficient_reports_deficit(self):
        out = check_cache_pool_sufficiency(42, 50, "validation")
        assert out["ok"] is False
        assert out["deficit"] == 8
        assert out["pool_size"] == 42 and out["target"] == 50
        assert out["split"] == "validation"
        assert "42" in out["message"] and "Parquet" in out["message"]


# ---------------------------------------------------------------------------
# Frozen-test leakage detection
# ---------------------------------------------------------------------------

class TestDetectFrozenLeakage:
    def test_clean_is_empty(self):
        hits = detect_frozen_leakage(
            opened_paths=["/data/manifests/rq1_train.csv"],
            loaded_splits=["train", "validation"],
            source_splits=["train", "train"],
            split_values=["train", "validation"])
        assert hits == []

    def test_frozen_source_split_detected(self):
        hits = detect_frozen_leakage(
            opened_paths=[], loaded_splits=[],
            source_splits=["train", "test"], split_values=["train"])
        assert any("source_split:test" == h for h in hits)

    def test_frozen_split_value_detected(self):
        hits = detect_frozen_leakage(
            opened_paths=[], loaded_splits=[],
            source_splits=["train"], split_values=["rq1_test"])
        assert any("split_value:rq1_test" == h for h in hits)

    def test_frozen_split_loaded_detected(self):
        hits = detect_frozen_leakage(
            opened_paths=[], loaded_splits=["frozen_test"],
            source_splits=["train"], split_values=["train"])
        assert any("split:frozen_test" == h for h in hits)


# ---------------------------------------------------------------------------
# Cross-file RUN_ID validation
# ---------------------------------------------------------------------------

class TestVerifyArtifactsRunId:
    def test_all_consistent(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({"run_id": "RID", "x": 1}))
        pd.DataFrame({"run_id": ["RID", "RID"], "v": [1, 2]}).to_csv(d / "b.csv", index=False)
        assert verify_artifacts_run_id([d], "RID") == []

    def test_json_mismatch_detected(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({"run_id": "OLD"}))
        issues = verify_artifacts_run_id([d], "RID")
        assert any("a.json" in i for i in issues)

    def test_csv_mismatch_detected(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        pd.DataFrame({"run_id": ["RID", "OLD"]}).to_csv(d / "b.csv", index=False)
        issues = verify_artifacts_run_id([d], "RID")
        assert any("b.csv" in i for i in issues)

    def test_empty_csv_with_header_ignored(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        empty_dataframe(VALIDATION_PREDICTIONS_COLUMNS).to_csv(d / "preds.csv", index=False)
        assert verify_artifacts_run_id([d], "RID") == []


# ---------------------------------------------------------------------------
# build_eligible_pool_and_sample tests (fix for group-aware sampling bug)
# ---------------------------------------------------------------------------
class TestResolvePositiveDuration:
    def test_uses_primary_when_valid(self):
        row = {"processed_duration_seconds": 1.5, "duration_seconds": 9.0}
        assert resolve_positive_duration(row) == 1.5

    def test_nan_primary_falls_back(self):
        row = {"processed_duration_seconds": float("nan"), "duration_seconds": 2.0}
        assert resolve_positive_duration(row) == 2.0

    def test_zero_and_negative_rejected(self):
        row = {"processed_duration_seconds": 0.0, "duration_seconds": -1.0}
        assert resolve_positive_duration(row) is None

    def test_require_raises(self):
        row = {"processed_duration_seconds": float("nan"), "duration_seconds": 0.0}
        with pytest.raises(ValueError, match="no numeric finite duration"):
            resolve_positive_duration(row, require=True, uid="u1", split="train")


class TestBuildEligiblePoolAndSample:
    """
    Tests for build_eligible_pool_and_sample which fixes the bug where
    sample_pilot_data(pool, len(pool), ...) returned pool in original order
    instead of doing true group-aware selection.
    """

    @pytest.fixture
    def mock_vocab(self):
        """Simple vocab for testing CTC feasibility."""
        return {chr(i): i - 97 for i in range(ord('a'), ord('z') + 1)}

    @pytest.fixture
    def mock_normalize(self):
        """Normalize: strip and lowercase."""
        return lambda text: text.strip().lower() if text else ""

    @pytest.fixture
    def mock_encode(self):
        """Encode: list of char indices."""
        def _encode(text, vocab):
            return [vocab.get(c, 0) for c in text.lower() if c in vocab]
        return _encode

    @pytest.fixture
    def mock_ctc_check(self):
        """CTC check: feasible if text length <= duration * some factor."""
        def _check(token_ids, num_samples):
            # Simple: feasible if samples >= 320 * len(token_ids)
            min_frames = len(token_ids) * 2  # simplified
            estimated = num_samples // 320
            if len(token_ids) == 0:
                return {"feasible": False, "reason": "empty_target",
                        "target_token_length": 0, "min_required_frames": 0,
                        "estimated_output_frames": estimated}
            feasible = estimated >= min_frames
            return {
                "feasible": feasible,
                "reason": "" if feasible else "insufficient_frames_for_ctc_target",
                "target_token_length": len(token_ids),
                "min_required_frames": min_frames,
                "estimated_output_frames": estimated,
            }
        return _check

    def _make_pool(self, groups_with_records):
        """
        Make a pool DataFrame.
        groups_with_records: list of (group_id, n_records)
        Each record gets uid=g{group}_r{idx}, text_bahnar="text{idx}", duration=1.0
        """
        rows = []
        for gid, n in groups_with_records:
            for i in range(n):
                rows.append({
                    "record_uid": f"g{gid}_r{i}",
                    "record_id": f"rec_g{gid}_r{i}",
                    "group_id": f"group_{gid}",
                    "text_bahnar": f"abcde",  # 5 chars, feasible with duration 1.0
                    "duration_seconds": 1.0,
                    "processed_duration_seconds": 1.0,
                    "source_split": "train",
                })
        return pd.DataFrame(rows)

    def test_group_aware_selection_not_first_n(self, mock_vocab, mock_normalize,
                                                mock_encode, mock_ctc_check):
        """
        Case 1: Pool with groups in contiguous blocks.
        g1: 10 records, g2: 10, g3: 10, g4: 10, g5: 10, g6: 10 = 60 total
        With target_n=6, result must have 6 records from 6 different groups,
        NOT the first 6 records (which would all be from g1).
        """
        pool = self._make_pool([(1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10)])
        assert len(pool) == 60

        result = build_eligible_pool_and_sample(
            pool_df=pool,
            target_n=6,
            split="train",
            seed=42,
            vocab=mock_vocab,
            normalize_fn=mock_normalize,
            ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode,
            run_id="test_run",
        )

        active = result["active_df"]
        assert len(active) == 6, "Should select exactly 6 records"

        # Key assertion: should have 6 different groups, not just group_1
        unique_groups = active["group_id"].nunique()
        assert unique_groups == 6, f"Expected 6 groups, got {unique_groups}"

        # Should NOT be the first 6 records (which are all g1)
        first_six_uids = set(pool.head(6)["record_uid"])
        selected_uids = set(active["record_uid"])
        assert selected_uids != first_six_uids, \
            "Selection should be group-aware, not first N records"

    def test_same_seed_deterministic(self, mock_vocab, mock_normalize,
                                      mock_encode, mock_ctc_check):
        """Case 2: Same seed produces identical results."""
        pool = self._make_pool([(1, 5), (2, 5), (3, 5)])

        r1 = build_eligible_pool_and_sample(
            pool, 6, "train", seed=123, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r1")

        r2 = build_eligible_pool_and_sample(
            pool, 6, "train", seed=123, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r2")

        assert list(r1["active_df"]["record_uid"]) == list(r2["active_df"]["record_uid"])

    def test_different_seed_changes_selection(self, mock_vocab, mock_normalize,
                                               mock_encode, mock_ctc_check):
        """Case 3: Different seeds must change the selected UID set."""
        pool = self._make_pool([(1, 10), (2, 10), (3, 10)])

        r1 = build_eligible_pool_and_sample(
            pool, 6, "train", seed=100, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        r2 = build_eligible_pool_and_sample(
            pool, 6, "train", seed=999, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        uids1 = set(r1["active_df"]["record_uid"])
        uids2 = set(r2["active_df"]["record_uid"])
        assert len(uids1) == len(uids2) == 6
        assert uids1 != uids2

    def test_encode_uses_normalized_transcript(self, mock_vocab, mock_ctc_check):
        """encode_fn must receive the normalized transcript, not the raw text."""
        pool = self._make_pool([(1, 3), (2, 3)])
        pool["text_bahnar"] = "  ABCDE  "
        seen = []

        def _normalize(text):
            return text.strip().lower() if text else ""

        def _encode(text, vocab):
            seen.append(text)
            return [vocab.get(c, 0) for c in text if c in vocab]

        result = build_eligible_pool_and_sample(
            pool, 4, "train", seed=42, vocab=mock_vocab,
            normalize_fn=_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=_encode, run_id="r")

        assert len(result["active_df"]) == 4
        assert seen, "encode_fn should have been called"
        assert all(t == "abcde" for t in seen)

    def test_nan_duration_falls_back(self, mock_vocab, mock_normalize,
                                     mock_encode, mock_ctc_check):
        """NaN primary duration must fall back (NaN is truthy; ``a or b`` is unsafe)."""
        pool = self._make_pool([(1, 4), (2, 4)])
        pool.loc[:, "processed_duration_seconds"] = float("nan")
        pool.loc[:, "duration_seconds"] = 1.0

        result = build_eligible_pool_and_sample(
            pool, 4, "train", seed=42, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        assert len(result["active_df"]) == 4
        assert (result["active_df"]["processed_duration_seconds"] == 1.0).all()

    def test_invalid_duration_raises(self, mock_vocab, mock_normalize,
                                     mock_encode, mock_ctc_check):
        """Non-positive / non-finite durations with no fallback must raise."""
        pool = self._make_pool([(1, 3), (2, 3)])
        pool.loc[:, "processed_duration_seconds"] = 0.0
        pool.loc[:, "duration_seconds"] = float("nan")

        with pytest.raises(ValueError, match="no numeric finite duration"):
            build_eligible_pool_and_sample(
                pool, 2, "train", seed=42, vocab=mock_vocab,
                normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
                encode_fn=mock_encode, run_id="r")

    def test_group_coverage_assert_applies(self, mock_vocab, mock_normalize,
                                           mock_encode, mock_ctc_check):
        """Helper asserts active_group_count == min(target_n, eligible_group_count)."""
        pool = self._make_pool([(1, 5), (2, 5), (3, 5), (4, 5)])
        result = build_eligible_pool_and_sample(
            pool, 4, "validation", seed=7, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")
        assert result["active_group_count"] == min(4, result["eligible_group_count"])
        assert result["active_group_count"] == result["expected_active_groups"]

    def test_no_duplicate_uids(self, mock_vocab, mock_normalize, mock_encode, mock_ctc_check):
        """Case 4: No duplicate record_uids in selection."""
        pool = self._make_pool([(1, 20), (2, 20)])
        result = build_eligible_pool_and_sample(
            pool, 10, "train", seed=42, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        active = result["active_df"]
        uids = list(active["record_uid"])
        assert len(uids) == len(set(uids)), "No duplicate UIDs allowed"

    def test_empty_ref_excluded_before_sampling(self, mock_vocab, mock_normalize,
                                                 mock_encode, mock_ctc_check):
        """Case 5: Empty normalized references are excluded before sampling."""
        pool = self._make_pool([(1, 5), (2, 5)])
        # Make some records have empty text
        pool.loc[pool["record_uid"].isin(["g1_r0", "g1_r1"]), "text_bahnar"] = "   "

        result = build_eligible_pool_and_sample(
            pool, 5, "train", seed=42, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        # 2 empty refs excluded
        assert len(result["empty_ref_exclusions"]) == 2
        # Active should not include excluded UIDs
        active_uids = set(result["active_df"]["record_uid"])
        assert "g1_r0" not in active_uids
        assert "g1_r1" not in active_uids

    def test_ctc_invalid_excluded_before_sampling(self, mock_vocab, mock_normalize,
                                                   mock_encode, mock_ctc_check):
        """Case 6: CTC-invalid records are excluded before sampling."""
        pool = self._make_pool([(1, 10), (2, 10)])
        # Make some records have very long text (CTC infeasible)
        pool.loc[pool["record_uid"] == "g1_r0", "text_bahnar"] = "abcdefghijklmnopqrstuvwxyz" * 10
        pool.loc[pool["record_uid"] == "g2_r0", "text_bahnar"] = "abcdefghijklmnopqrstuvwxyz" * 10

        result = build_eligible_pool_and_sample(
            pool, 10, "train", seed=42, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        assert len(result["ctc_exclusions"]) == 2
        active_uids = set(result["active_df"]["record_uid"])
        assert "g1_r0" not in active_uids
        assert "g2_r0" not in active_uids

    def test_select_correct_count_after_exclusions(self, mock_vocab, mock_normalize,
                                                    mock_encode, mock_ctc_check):
        """Case 7: After exclusions, still select exactly target_n if eligible pool is sufficient."""
        pool = self._make_pool([(1, 20), (2, 20)])
        # Exclude 5 from group 1
        for i in range(5):
            pool.loc[pool["record_uid"] == f"g1_r{i}", "text_bahnar"] = "   "

        result = build_eligible_pool_and_sample(
            pool, 20, "train", seed=42, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        assert len(result["empty_ref_exclusions"]) == 5
        assert len(result["active_df"]) == 20  # Still get 20 from remaining 35

    def test_insufficient_eligible_raises_error(self, mock_vocab, mock_normalize,
                                                 mock_encode, mock_ctc_check):
        """Case 8: If eligible pool < target_n, raise clear error."""
        pool = self._make_pool([(1, 5), (2, 5)])  # 10 total
        # Exclude 8, leaving only 2 eligible
        for i in range(4):
            pool.loc[pool["record_uid"] == f"g1_r{i}", "text_bahnar"] = "   "
        for i in range(4):
            pool.loc[pool["record_uid"] == f"g2_r{i}", "text_bahnar"] = "   "

        with pytest.raises(RuntimeError) as exc:
            build_eligible_pool_and_sample(
                pool, 5, "train", seed=42, vocab=mock_vocab,
                normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
                encode_fn=mock_encode, run_id="r")

        err = str(exc.value)
        assert "train" in err.lower()
        assert "eligible_pool=2" in err
        assert "target=5" in err

    def test_group_metadata_computed_correctly(self, mock_vocab, mock_normalize,
                                                mock_encode, mock_ctc_check):
        """Case 9: Group coverage metadata is computed correctly."""
        pool = self._make_pool([(1, 10), (2, 10), (3, 10), (4, 10)])  # 4 groups, 40 records

        result = build_eligible_pool_and_sample(
            pool, 8, "train", seed=42, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        assert result["eligible_pool_size"] == 40
        assert result["eligible_group_count"] == 4
        # With 8 records from 4 groups, round-robin should give 2 per group
        assert result["active_group_count"] == 4  # min(8, 4) = 4 groups
        assert result["expected_active_groups"] == 4
        assert result["group_coverage_ratio"] == 1.0  # 4/4

    def test_all_active_from_eligible_pool(self, mock_vocab, mock_normalize,
                                           mock_encode, mock_ctc_check):
        """All active records must come from the eligible pool."""
        pool = self._make_pool([(1, 10), (2, 10)])
        # Make g1_r0 empty (excluded)
        pool.loc[pool["record_uid"] == "g1_r0", "text_bahnar"] = ""

        result = build_eligible_pool_and_sample(
            pool, 10, "train", seed=42, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        # g1_r0 excluded → 19 eligible
        assert result["eligible_pool_size"] == 19
        active_uids = set(result["active_df"]["record_uid"])
        assert "g1_r0" not in active_uids

    def test_group_round_robin_distribution(self, mock_vocab, mock_normalize,
                                             mock_encode, mock_ctc_check):
        """
        With 6 groups and target=6, should get exactly 1 record from each group
        (round-robin behavior of sample_pilot_data).
        """
        pool = self._make_pool([(1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10)])

        result = build_eligible_pool_and_sample(
            pool, 6, "train", seed=42, vocab=mock_vocab,
            normalize_fn=mock_normalize, ctc_check_fn=mock_ctc_check,
            encode_fn=mock_encode, run_id="r")

        active = result["active_df"]
        # Each group should have exactly 1 record
        group_counts = active["group_id"].value_counts()
        assert all(c == 1 for c in group_counts), \
            f"Expected 1 record per group, got: {group_counts.to_dict()}"
