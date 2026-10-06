"""NB14 active-seed, arm×seed runs, and multi-seed evaluation provenance."""
from __future__ import annotations

import json

import pytest

from src.rq2_final_contract import (
    ARM_QUALITY,
    ARM_RANDOM,
    SEED_POLICY_MULTI,
    SEED_POLICY_SINGLE,
    TrainingContractError,
    aggregate_paired_seed_metrics,
    build_arm_training_contract,
    index_seed_runs,
    materialize_seed_bound_contracts,
    normalize_gold_pseudo_mix_policy,
    normalize_seed_policy,
    summarize_seed_runs,
)
from src.rq2_final_evaluate import (
    EvaluationError,
    GTestFirewallError,
    assert_declared_seed_checkpoints_ready,
    plan_paired_seed_evaluations,
    verify_published_seed_provenance,
)
from src.rq2_final_train import (
    ArmIsolationError,
    build_gold_pseudo_sampler,
    discover_latest_valid_checkpoint,
)
from tests.rq2_nb14_fixtures import write_complete_checkpoint
from tests.test_rq2_final_contract import _base_fields


def _fingerprint(seed: int) -> str:
    return f"{seed:02d}" * 32


def _best_map(seeds):
    best = {}
    for arm in (ARM_RANDOM, ARM_QUALITY):
        best[arm] = {
            seed: {
                "arm": arm,
                "active_seed": seed,
                "checkpoint_fingerprint": _fingerprint(seed if arm == ARM_RANDOM else seed + 1),
                "g_test_used": False,
                "validation_metric": 1.0,
            }
            for seed in seeds
        }
    return best


def _metrics(seeds):
    rows = []
    for index, seed in enumerate(seeds):
        rows.append({
            "active_seed": seed,
            ARM_RANDOM: {"sacrebleu": 10.0 + index, "chrfpp": 20.0 + index},
            ARM_QUALITY: {"sacrebleu": 11.0 + index, "chrfpp": 22.0 + index},
        })
    return rows


def test_single_active_seed_is_the_only_declared_seed(tmp_path):
    built = materialize_seed_bound_contracts(_base_fields(tmp_path, ARM_RANDOM))
    assert len(built) == 1
    assert built[0]["active_seed"] == 42
    assert built[0]["seed_policy_record"] == {"mode": "single", "seeds": [42]}


def test_multi_contracts_bind_each_active_seed_and_reject_missing(tmp_path):
    fields = _base_fields(tmp_path, ARM_RANDOM)
    fields["seed_policy"] = SEED_POLICY_MULTI
    fields["seed_policy_seeds"] = [7, 11, 13]
    fields.pop("seed", None)
    fields.pop("active_seed", None)
    fields.pop("dataloader_seed", None)
    with pytest.raises(TrainingContractError, match="active_seed"):
        build_arm_training_contract(fields)
    built = materialize_seed_bound_contracts(fields)
    assert [contract["active_seed"] for contract in built] == [7, 11, 13]
    alien = dict(fields)
    alien["active_seed"] = 99
    with pytest.raises(TrainingContractError, match="not in the declared"):
        build_arm_training_contract(alien)


def test_sampler_uses_contract_active_seed_not_a_generic_seed(tmp_path):
    import pandas as pd

    fields = _base_fields(tmp_path, ARM_RANDOM)
    fields["seed_policy"] = SEED_POLICY_MULTI
    fields["seed_policy_seeds"] = [7, 11]
    fields.pop("active_seed", None)
    fields["active_seed"] = 11
    contract = build_arm_training_contract(fields)
    frame = pd.DataFrame({
        "kind": ["supervised_g_train", "pseudo_nb13"],
    })
    sampler = build_gold_pseudo_sampler(frame, contract)
    assert sampler.seed == 11


def test_multi_run_index_keeps_every_seed(tmp_path):
    seeds = [7, 11, 13]
    random_fields = _base_fields(tmp_path / "r", ARM_RANDOM)
    quality_fields = _base_fields(tmp_path / "q", ARM_QUALITY)
    runs = {}
    for fields in (random_fields, quality_fields):
        fields["seed_policy"] = SEED_POLICY_MULTI
        fields["seed_policy_seeds"] = seeds
        fields.pop("active_seed", None)
        built = materialize_seed_bound_contracts(fields)
        runs[built[0]["arm"]] = index_seed_runs([{"contract": contract} for contract in built])
    assert set(runs[ARM_RANDOM]) == set(seeds)
    assert set(runs[ARM_QUALITY]) == set(seeds)
    identities = {
        (arm, seed, run["contract"]["arm_training_contract_sha256"])
        for arm, by_seed in runs.items()
        for seed, run in by_seed.items()
    }
    assert len(identities) == 6


def test_missing_fingerprint_and_cross_seed_resume_fail(tmp_path):
    from src.rq2_final_evaluate import _best_record_ready

    assert _best_record_ready({"g_test_used": False, "checkpoint_fingerprint": ""}) is False
    assert _best_record_ready({"g_test_used": False, "checkpoint_fingerprint": "ab" * 32}) is True
    layout = {"checkpoints": tmp_path / "checkpoints"}
    write_complete_checkpoint(
        layout["checkpoints"] / "seed-7",
        arm=ARM_RANDOM,
        contract_hash="11" * 32,
        data_hash="aa" * 32,
        step=3,
        fingerprint_extra={"training_seed": 7, "nb13_generation_id": "nb13", "manifest_sha256": "aa" * 32},
    )
    with pytest.raises(TrainingContractError, match="active_seed"):
        discover_latest_valid_checkpoint(
            layout,
            arm=ARM_RANDOM,
            expected_contract_hash="11" * 32,
            expected_data_hash="aa" * 32,
            expected_active_seed=11,
        )
    with pytest.raises(ArmIsolationError):
        discover_latest_valid_checkpoint(
            layout,
            arm=ARM_QUALITY,
            expected_contract_hash="11" * 32,
            expected_data_hash="aa" * 32,
            expected_active_seed=7,
        )


def test_partial_multi_seed_keeps_g_test_locked_and_complete_set_pairs():
    policy = {"seed_policy": SEED_POLICY_MULTI, "seed_policy_seeds": [7, 11, 13]}
    partial = _best_map([7, 11])
    with pytest.raises(GTestFirewallError, match="missing"):
        assert_declared_seed_checkpoints_ready(partial, seed_policy=policy)
    pairs = plan_paired_seed_evaluations(_best_map([7, 11, 13]), seed_policy=policy)
    assert [pair["active_seed"] for pair in pairs] == [7, 11, 13]
    assert pairs[0]["pairing"] == f"{ARM_RANDOM}:7|{ARM_QUALITY}:7"
    assert pairs[0][ARM_RANDOM]["checkpoint_fingerprint"] != pairs[1][ARM_RANDOM]["checkpoint_fingerprint"]


def test_aggregate_mean_std_and_delta_and_rejects_empty():
    policy = {"seed_policy": SEED_POLICY_MULTI, "seed_policy_seeds": [7, 11, 13]}
    with pytest.raises(TrainingContractError, match="empty"):
        summarize_seed_runs([], seed_policy=policy)
    summary = aggregate_paired_seed_metrics(_metrics([7, 11, 13]), seed_policy=policy)
    assert summary["n_valid_runs"] == 3
    assert summary[ARM_RANDOM]["sacrebleu"]["mean"] == pytest.approx(11.0)
    assert summary[ARM_QUALITY]["sacrebleu"]["mean"] == pytest.approx(12.0)
    assert summary["treatment"]["per_seed_delta_sacrebleu"] == [1.0, 1.0, 1.0]
    assert summary["treatment"]["sacrebleu"]["mean"] == pytest.approx(1.0)
    assert summary["treatment"]["sacrebleu"]["std"] == pytest.approx(0.0)
    single = aggregate_paired_seed_metrics(
        _metrics([42]),
        seed_policy={"seed_policy": SEED_POLICY_SINGLE, "seed_policy_seeds": [42]},
    )
    assert single["n_valid_runs"] == 1
    assert single[ARM_RANDOM]["sacrebleu"]["std"] is None
    assert single["mean_std_reported"] is False


def test_published_seed_provenance_rejects_inconsistent_aggregate():
    seeds = [7, 11, 13]
    rows = _metrics(seeds)
    policy = {"seed_policy": SEED_POLICY_MULTI, "seed_policy_seeds": seeds}
    summary = aggregate_paired_seed_metrics(rows, seed_policy=policy)
    best = _best_map(seeds)
    contract = {
        "seed_policy_record": {"mode": "multi", "seeds": seeds},
        "gold_pseudo_mix": {"policy": "slotted", "gold_slots": 1, "pseudo_slots": 1},
        "seed_run_summary": summary,
    }
    verify_published_seed_provenance(contract, best)
    drifted = json.loads(json.dumps(summary))
    drifted[ARM_RANDOM]["sacrebleu"]["mean"] = 0.0
    contract["seed_run_summary"] = drifted
    with pytest.raises(EvaluationError, match="does not recompute"):
        verify_published_seed_provenance(contract, best)
    contract["seed_run_summary"] = summary
    contract["seed_policy_record"] = {"mode": "multi", "seeds": [7, 11]}
    with pytest.raises(EvaluationError):
        verify_published_seed_provenance(contract, best)


FROZEN_TRAINING_SEEDS = [13, 17, 23]


def _frozen_treatment_fields(tmp_path, arm):
    fields = _base_fields(tmp_path, arm)
    fields["gold_pseudo_mix_policy"] = "configured_gold_pseudo_slot_ratio"
    fields["gold_slots"] = 5
    fields["pseudo_slots"] = 1
    fields["seed_policy"] = "multi_seed"
    fields["seed_policy_seeds"] = list(FROZEN_TRAINING_SEEDS)
    fields.pop("active_seed", None)
    fields.pop("seed", None)
    fields.pop("dataloader_seed", None)
    return fields


def test_predeclared_mix_and_seed_policies_normalize():
    mix = normalize_gold_pseudo_mix_policy({
        "gold_pseudo_mix_policy": "configured_gold_pseudo_slot_ratio",
        "gold_slots": 5,
        "pseudo_slots": 1,
    })
    assert mix["configured"] is True
    assert mix["configured_ratio"] == "5:1"
    assert mix["gold_slots"] == 5
    assert mix["pseudo_slots"] == 1
    seeds = normalize_seed_policy({
        "seed_policy": "multi_seed",
        "seed_policy_seeds": [13, 17, 23],
    })
    assert seeds["configured"] is True
    assert seeds["seed_policy"] == "multi_seed"
    assert seeds["seed_policy_seeds"] == [13, 17, 23]
    assert seeds["report_mean_std"] is True
    assert seeds["seed_runs"] == [
        {"seed": 13, "dataloader_seed": 13},
        {"seed": 17, "dataloader_seed": 17},
        {"seed": 23, "dataloader_seed": 23},
    ]


def test_frozen_policy_pairs_the_same_seeds_on_both_arms(tmp_path):
    random_contracts = materialize_seed_bound_contracts(_frozen_treatment_fields(tmp_path / "random", ARM_RANDOM))
    quality_contracts = materialize_seed_bound_contracts(_frozen_treatment_fields(tmp_path / "quality", ARM_QUALITY))
    assert [contract["active_seed"] for contract in random_contracts] == FROZEN_TRAINING_SEEDS
    assert [contract["active_seed"] for contract in quality_contracts] == FROZEN_TRAINING_SEEDS
    assert 42 not in FROZEN_TRAINING_SEEDS
    assert [7, 11, 13] != FROZEN_TRAINING_SEEDS
    shared = (
        "gold_pseudo_mix_policy",
        "gold_slots",
        "pseudo_slots",
        "seed_policy",
        "seed_policy_seeds",
    )
    for left, right in zip(random_contracts, quality_contracts):
        assert left["arm"] == ARM_RANDOM
        assert right["arm"] == ARM_QUALITY
        assert left["active_seed"] == right["active_seed"]
        for key in shared:
            assert left[key] == right[key]
        assert left["gold_pseudo_mix_policy"] == "configured_gold_pseudo_slot_ratio"
        assert left["gold_slots"] == 5
        assert left["pseudo_slots"] == 1
        assert left["seed_policy"] == "multi_seed"
        assert left["seed_policy_seeds"] == FROZEN_TRAINING_SEEDS
        assert left["data_manifest_sha256"] != right["data_manifest_sha256"]
        assert left["pseudo_ordered_uid_hash"] != right["pseudo_ordered_uid_hash"]
        assert left["supervised_ordered_uid_hash"] == right["supervised_ordered_uid_hash"]

