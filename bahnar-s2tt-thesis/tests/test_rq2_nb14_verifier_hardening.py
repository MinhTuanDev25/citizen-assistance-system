"""Per-seed bootstrap CI checks and aggregate chrF++ delta checks."""
from __future__ import annotations

import json

import pytest

from src.rq2_final_contract import (
    ARM_QUALITY,
    ARM_RANDOM,
    SEED_POLICY_MULTI,
    aggregate_paired_seed_metrics,
)
from src.rq2_final_evaluate import (
    EvaluationError,
    _assert_recomputed_bootstrap,
    _assert_seed_summary_matches,
    comparison_key,
    paired_cluster_bootstrap_arms,
)
from tests.test_rq2_nb14_orchestration import _frames


def _bootstrap():
    frames = _frames(42, d0_fp="aa" * 32, random_fp="bb" * 32, quality_fp="cc" * 32)
    recorded = paired_cluster_bootstrap_arms(frames, n_samples=8, seed=3, confidence=0.95)
    recomputed = paired_cluster_bootstrap_arms(frames, n_samples=8, seed=3, confidence=0.95)
    return recorded, recomputed


def _mutate(recorded, field, value):
    bad = json.loads(json.dumps(recorded))
    bad["comparisons"][comparison_key(ARM_QUALITY, ARM_RANDOM)][field] = value
    return bad


def test_per_seed_bootstrap_accepts_a_recomputed_artifact():
    recorded, recomputed = _bootstrap()
    _assert_recomputed_bootstrap(recorded, recomputed, label="seed 42")
    frames = _frames(42, d0_fp="aa" * 32, random_fp="bb" * 32, quality_fp="cc" * 32)
    zero = paired_cluster_bootstrap_arms(frames, n_samples=4, seed=0, confidence=0.95)
    _assert_recomputed_bootstrap(zero, dict(zero), label="seed 0")


@pytest.mark.parametrize(
    "field",
    [
        "sacrebleu_ci_lower",
        "sacrebleu_ci_upper",
        "chrfpp_ci_lower",
        "chrfpp_ci_upper",
        "observed_sacrebleu_delta",
        "observed_chrfpp_delta",
    ],
)
def test_per_seed_bootstrap_rejects_tampered_bounds_and_deltas(field):
    recorded, recomputed = _bootstrap()
    bad = _mutate(recorded, field, -999.0)
    with pytest.raises(EvaluationError, match=field):
        _assert_recomputed_bootstrap(bad, recomputed, label="seed 42")


def test_per_seed_bootstrap_rejects_empty_or_removed_quality_random_comparison():
    recorded, recomputed = _bootstrap()
    empty = json.loads(json.dumps(recorded))
    empty["comparisons"] = {}
    with pytest.raises(EvaluationError, match="missing"):
        _assert_recomputed_bootstrap(empty, recomputed, label="seed 42")
    removed = json.loads(json.dumps(recorded))
    del removed["comparisons"][comparison_key(ARM_QUALITY, ARM_RANDOM)]
    with pytest.raises(EvaluationError, match="Quality-Random"):
        _assert_recomputed_bootstrap(removed, recomputed, label="seed 42")


def test_aggregate_chrf_delta_list_fails_when_mean_is_left_unchanged():
    seeds = [7, 11, 13]
    rows = []
    for index, seed in enumerate(seeds):
        rows.append({
            "active_seed": seed,
            ARM_RANDOM: {"sacrebleu": 10.0 + index, "chrfpp": 20.0 + index},
            ARM_QUALITY: {"sacrebleu": 12.0 + index, "chrfpp": 23.0 + index},
        })
    summary = aggregate_paired_seed_metrics(
        rows,
        seed_policy={"seed_policy": SEED_POLICY_MULTI, "seed_policy_seeds": seeds},
    )
    _assert_seed_summary_matches(summary, summary, label="aggregate")
    tampered = json.loads(json.dumps(summary))
    tampered["treatment"]["per_seed_delta_chrfpp"][1] = 99.0
    assert tampered["treatment"]["chrfpp"]["mean"] == summary["treatment"]["chrfpp"]["mean"]
    with pytest.raises(EvaluationError, match="per_seed_delta_chrfpp"):
        _assert_seed_summary_matches(tampered, summary, label="contract seed summary")
