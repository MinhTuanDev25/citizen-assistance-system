"""
Evaluate gates: metric prefix normalisation, finiteness and the frozen-split guard.

``trainer.predict()`` prefixes metrics with ``test_``, which is a Transformers
naming convention and not evidence that the frozen RQ1 test split was touched.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.asr_full_data import (
    assert_eligible_frames_no_frozen_splits,
    is_frozen_split_label,
    normalize_split_label,
)
from src.asr_full_train import (
    CANONICAL_METRIC_KEYS,
    canonicalize_trainer_metrics,
    derive_full_evaluate_status,
    metrics_are_finite,
)


class TestMetricNormalisation:
    def test_predict_prefix_is_folded_to_canonical_names(self):
        out = canonicalize_trainer_metrics(
            {"test_loss": 1.5, "test_cer": 0.3, "test_wer": 0.6, "test_runtime": 12.0}
        )
        assert out == {"loss": 1.5, "cer": 0.3, "wer": 0.6}

    def test_eval_prefix_is_folded_too(self):
        out = canonicalize_trainer_metrics(
            {"eval_loss": 1.0, "eval_cer": 0.2, "eval_wer": 0.4}
        )
        assert set(out) == set(CANONICAL_METRIC_KEYS)

    def test_already_canonical_metrics_pass_through(self):
        assert canonicalize_trainer_metrics({"loss": 1, "cer": 2, "wer": 3})["wer"] == 3.0

    def test_loss_is_required_not_just_cer_and_wer(self):
        """A run can produce finite CER/WER with a NaN loss; loss is a gate too."""
        with pytest.raises(RuntimeError, match=r"\['loss'\]"):
            canonicalize_trainer_metrics({"test_cer": 0.3, "test_wer": 0.6})

    def test_missing_keys_are_reported_with_what_was_available(self):
        with pytest.raises(RuntimeError, match="test_runtime"):
            canonicalize_trainer_metrics({"test_runtime": 1.0})

    def test_non_numeric_metric_is_treated_as_missing(self):
        with pytest.raises(RuntimeError, match="cer"):
            canonicalize_trainer_metrics(
                {"test_loss": 1.0, "test_cer": "n/a", "test_wer": 0.5}
            )

    def test_finiteness_is_checked_on_canonical_keys(self):
        good = canonicalize_trainer_metrics({"test_loss": 1.0, "test_cer": 0.1, "test_wer": 0.2})
        assert metrics_are_finite(good, keys=CANONICAL_METRIC_KEYS) is True

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_metric_fails_the_gate(self, bad):
        metrics = canonicalize_trainer_metrics(
            {"test_loss": bad, "test_cer": 0.1, "test_wer": 0.2}
        )
        assert math.isfinite(metrics["cer"])
        assert metrics_are_finite(metrics, keys=CANONICAL_METRIC_KEYS) is False

    def test_default_eval_keys_would_miss_predict_output(self):
        """Why the explicit keys matter: the default names are absent here."""
        predict_metrics = {"test_loss": 1.0, "test_cer": 0.1, "test_wer": 0.2}
        assert metrics_are_finite(predict_metrics) is False


class TestFrozenSplitGuard:
    @pytest.mark.parametrize("label", [
        "test", "Test", " TEST ", "rq1_test", "rq1-test", "RQ1 Test",
        "frozen_test", "frozen-test", "frozen  test",
    ])
    def test_every_spelling_of_a_frozen_split_is_caught(self, label):
        assert is_frozen_split_label(label) is True

    @pytest.mark.parametrize("label", ["train", "validation", "dev", "", None, "testimony"])
    def test_trainable_splits_are_not_flagged(self, label):
        assert is_frozen_split_label(label) is False

    def test_hyphen_and_underscore_normalise_to_one_token(self):
        assert normalize_split_label("RQ1-Test") == normalize_split_label("rq1_test")
        assert normalize_split_label(" frozen  test ") == "frozen_test"

    def test_guard_runs_on_the_frame_actually_used(self):
        df = pd.DataFrame({"record_uid": ["a"], "source_split": ["RQ1-Test"]})
        with pytest.raises(RuntimeError, match="Forbidden split"):
            assert_eligible_frames_no_frozen_splits(df, label="full_train_eligible")

    def test_clean_frame_passes(self):
        df = pd.DataFrame({"record_uid": ["a"], "source_split": ["train"], "split": ["train"]})
        assert_eligible_frames_no_frozen_splits(df) is None


class TestEvaluateStatusGate:
    def _verdict(self, **overrides):
        kwargs = dict(
            full_train_success=True,
            best_checkpoint_from_drive_valid=True,
            metrics_finite=True,
            frozen_test_accessed=False,
            contract_matches=True,
        )
        kwargs.update(overrides)
        return derive_full_evaluate_status(**kwargs)

    def test_all_conditions_met_is_success(self):
        assert self._verdict()["status"] == "SUCCESS_FULL_EVALUATE"

    @pytest.mark.parametrize("field,value", [
        ("full_train_success", False),
        ("best_checkpoint_from_drive_valid", False),
        ("metrics_finite", False),
        ("frozen_test_accessed", True),
        ("contract_matches", False),
    ])
    def test_any_single_failure_blocks_success(self, field, value):
        verdict = self._verdict(**{field: value})
        assert verdict["status"] != "SUCCESS_FULL_EVALUATE"
        assert verdict["failed_checks"]
