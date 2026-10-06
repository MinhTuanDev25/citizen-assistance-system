"""NB14 contract, upstream gate, G_test firewall, and success-gate tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rq2_final_contract import (
    ARM_QUALITY,
    ARM_RANDOM,
    D0_POLICY,
    GTestFirewallError,
    Nb14Flags,
    RUNTIME_ONLY_KEYS,
    STATUS_FAIL,
    STATUS_SUCCESS,
    TrainingContractError,
    UpstreamGateError,
    assert_augmentation_fairness,
    assert_g_test_blocked,
    assert_nb14_output_dir,
    assert_scientific_value_portable,
    bind_frozen_d0_identity,
    build_arm_training_contract,
    build_final_contract,
    compute_nb14_source_fingerprint,
    default_nb14_flags,
    derive_final_status,
    experiment_root_for_arm,
    locked_architecture,
    unlock_g_test,
    verify_arm_training_contract,
    verify_upstream_rq2,
)
from src.rq2_final_train import build_hparams_from_yaml, merge_training_fields
from tests.rq2_nb14_fixtures import all_ready, supervised_rows, world


ROOT = Path(__file__).resolve().parents[1]


def test_default_flags_are_safe():
    flags = default_nb14_flags()
    assert flags.run_real_training is False
    assert flags.allow_g_test_evaluation is False
    assert flags.rq2_final_frozen is False


def test_missing_nb11_fails_preflight(tmp_path):
    flags = Nb14Flags(direct_state_dir=str(tmp_path / "missing"))
    with pytest.raises(UpstreamGateError):
        verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=flags)


def test_stale_nb12_fails_preflight(tmp_path):
    from tests.rq2_nb12_fixtures import build_nb11_generation

    build_nb11_generation(tmp_path)
    flags = Nb14Flags(direct_state_dir=str(tmp_path / "direct"))
    with pytest.raises(UpstreamGateError):
        verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=flags)


def test_unfrozen_nb13_fails_preflight(tmp_path):
    from tests.rq2_nb13_fixtures import seal_nb12_generation, u_prime_row

    seal_nb12_generation(tmp_path, [
        u_prime_row("a", 5, 0.1),
        u_prime_row("b", 4, 0.9),
        u_prime_row("c", 3, 0.4),
        u_prime_row("d", 1, 0.8),
    ])
    d0 = tmp_path / "direct_state"
    from tests.rq2_nb14_fixtures import write_frozen_d0_state

    write_frozen_d0_state(d0)
    flags = Nb14Flags(direct_state_dir=str(d0))
    with pytest.raises(UpstreamGateError):
        verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=flags)


def test_upstream_gate_succeeds_on_synthetic_world(tmp_path):
    env = world(tmp_path)
    payload = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    assert payload["d0"]["d0_policy"] == D0_POLICY
    assert len(payload["nb11_input_contract_sha256"]) == 64
    assert len(payload["nb12_contract_sha256"]) == 64
    assert len(payload["nb13_selection_contract_sha256"]) == 64


def _base_fields(tmp_path, arm):
    env = world(tmp_path)
    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    from src.rq2_final_data import (
        compose_arm_training_rows,
        data_contract_payload,
        load_pinned_nb13_arm_manifest,
        validation_identity,
        verify_equal_budget_from_nb13,
        verify_pinned_nb13_selection,
        write_arm_manifest_csv,
    )

    selection = verify_pinned_nb13_selection(tmp_path, generation_id=upstream["nb13_generation_id"])
    budget = verify_equal_budget_from_nb13(selection)
    pseudo = [] if arm == "d0" else load_pinned_nb13_arm_manifest(tmp_path, arm=arm, upstream=upstream)
    composition = compose_arm_training_rows(
        arm=arm,
        supervised_rows=supervised_rows(),
        pseudo_rows=pseudo,
        selection_contract_sha256=upstream["nb13_selection_contract_sha256"],
        nb12_contract_sha256=upstream["nb12_contract_sha256"],
    )
    sha = write_arm_manifest_csv(tmp_path / f"{arm}.csv", composition["rows"])
    data = data_contract_payload(
        composition,
        manifest_sha256=sha,
        validation=validation_identity(validation_frame_safe()),
        upstream=upstream,
        budget=budget,
    )
    hparams = build_hparams_from_yaml(ROOT)
    from src.rq2_final_contract import resolve_frozen_d0_init
    from tests.rq2_nb14_fixtures import explicit_test_mix_policy, explicit_test_seed_policy

    d0_init = resolve_frozen_d0_init(env["flags"], project_root=tmp_path, identity=upstream["d0"])
    return merge_training_fields(
        arm=arm,
        data_contract=data,
        hparams=hparams,
        source_fingerprint_sha256=compute_nb14_source_fingerprint()["aggregate_sha256"],
        rq1_direct_training_contract_hash=upstream["d0"]["direct_training_contract_hash"],
        d0_init=d0_init,
        mix_policy=explicit_test_mix_policy(),
        seed_policy=explicit_test_seed_policy(),
        runtime_workers=8,
    )


def validation_frame_safe():
    from tests.rq2_nb14_fixtures import validation_frame

    return validation_frame()


def test_training_contract_deterministic_and_ignores_workers(tmp_path):
    fields = _base_fields(tmp_path, ARM_RANDOM)
    left = build_arm_training_contract(fields)
    right = build_arm_training_contract(fields)
    assert left["arm_training_contract_sha256"] == right["arm_training_contract_sha256"]
    verify_arm_training_contract(left)
    for key in RUNTIME_ONLY_KEYS:
        assert key not in left
    changed = dict(fields)
    changed["learning_rate"] = 1e-4
    other = build_arm_training_contract(changed)
    assert other["arm_training_contract_sha256"] != left["arm_training_contract_sha256"]


def test_augmentation_fairness_and_arm_swap_fails(tmp_path):
    random_fields = _base_fields(tmp_path, ARM_RANDOM)
    quality_fields = _base_fields(tmp_path, ARM_QUALITY)
    random_c = build_arm_training_contract(random_fields)
    quality_c = build_arm_training_contract(quality_fields)
    proof = assert_augmentation_fairness(random_c, quality_c)
    assert proof["supervised_ordered_uid_hash"] == random_c["supervised_ordered_uid_hash"]
    assert proof["supervised_pair_hash"] == random_c["supervised_pair_hash"]
    assert proof["supervised_audio_identity_hash"] == random_c["supervised_audio_identity_hash"]
    assert proof["supervised_uid_set_hash"] == random_c["supervised_uid_set_hash"]
    assert random_c["supervised_ordered_uid_hash"] == quality_c["supervised_ordered_uid_hash"]
    assert random_c["pseudo_ordered_uid_hash"] != quality_c["pseudo_ordered_uid_hash"]
    swapped = dict(quality_c)
    swapped["arm"] = ARM_RANDOM
    from src.rq2_final_contract import TrainingContractError

    with pytest.raises(TrainingContractError):
        assert_augmentation_fairness(swapped, quality_c)


def test_g_test_firewall_default_and_unlock(tmp_path):
    flags = default_nb14_flags()
    with pytest.raises(GTestFirewallError):
        assert_g_test_blocked(tmp_path / "data" / "manifests" / "rq1_test.csv", allow=False)
    with pytest.raises(GTestFirewallError):
        unlock_g_test(flags, all_ready())
    frozen = Nb14Flags(allow_g_test_evaluation=True, rq2_final_frozen=True)
    incomplete = dict(all_ready())
    incomplete["best_checkpoints_frozen"] = False
    with pytest.raises(GTestFirewallError):
        unlock_g_test(frozen, incomplete)
    unlock_g_test(frozen, all_ready())
    assert_g_test_blocked(tmp_path / "data" / "manifests" / "rq1_test.csv", allow=True)


def test_success_gate_fails_closed_and_complete_passes():
    empty = derive_final_status({})
    assert empty["status"] == STATUS_FAIL
    assert "current_committed" in empty["failed_checks"]
    ok = {name: True for name in empty["checks"]}
    done = derive_final_status(ok)
    assert done["status"] == STATUS_SUCCESS
    assert done["failed_checks"] == []


def test_final_contract_excludes_timestamps_and_paths():
    payload = build_final_contract({
        "nb11_input_contract_sha256": "11" * 32,
        "created_at_utc": "should-drop",
        "architecture": locked_architecture(),
    })
    assert "created_at_utc" not in payload
    assert "rq2_final_contract_sha256" in payload
    with pytest.raises(Exception):
        assert_scientific_value_portable("/workspace/bahnar", label="path")


def test_checkpoint_roots_are_isolated():
    h = "ab" * 32
    d0 = experiment_root_for_arm(durable_root="/tmp/x", contract_hash=h, arm="d0")
    rnd = experiment_root_for_arm(durable_root="/tmp/x", contract_hash=h, arm="d_random")
    q = experiment_root_for_arm(durable_root="/tmp/x", contract_hash=h, arm="d_quality")
    assert d0 != rnd != q
    assert d0.name == "d0" and rnd.name == "d_random" and q.name == "d_quality"


def test_d0_is_frozen_rq1_checkpoint(tmp_path):
    env = world(tmp_path)
    identity = bind_frozen_d0_identity(env["d0_dir"])
    assert identity["d0_policy"] == D0_POLICY
    assert identity["arm"] == "d0"
    assert len(identity["checkpoint_fingerprint_sha256"]) == 64
    assert identity["best_checkpoint_name"] == "checkpoint-100"


def test_d0_same_name_wrong_fingerprint_fails(tmp_path):
    env = world(tmp_path)
    ckpt = env["d0_dir"] / "checkpoint-100"
    payload = json.loads((ckpt / "full_experiment_fingerprint.json").read_text(encoding="utf-8"))
    payload["direct_training_contract_hash"] = "00" * 32
    (ckpt / "full_experiment_fingerprint.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UpstreamGateError, match="checkpoint identity"):
        bind_frozen_d0_identity(env["d0_dir"])


def test_notebook_default_flags_and_cells_compile():
    nb = json.loads((ROOT / "notebooks" / "14_RQ2_Final_Train_Evaluate.ipynb").read_text())
    source = "".join("".join(c.get("source") or []) for c in nb["cells"] if c["cell_type"] == "code")
    assert "RUN_REAL_TRAINING = False" in source
    assert "ALLOW_G_TEST_EVALUATION = False" in source
    assert "RQ2_FINAL_FROZEN = False" in source
    assert 'GOLD_PSEUDO_MIX_POLICY = "configured_gold_pseudo_slot_ratio"' in source
    assert "GOLD_SLOTS = 5" in source
    assert "PSEUDO_SLOTS = 1" in source
    assert 'SEED_POLICY = "multi_seed"' in source
    assert "SEED_POLICY_SEEDS = [13, 17, 23]" in source
    assert "public_pretrained_xlsr_mbart50_same_as_rq1_d0" not in source
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile("".join(cell.get("source") or []), f"nb14_cell_{i}", "exec")


def test_output_dir_is_isolated(tmp_path):
    with pytest.raises(Exception):
        assert_nb14_output_dir(tmp_path / "artifacts" / "rq2" / "u_clean", tmp_path)
    out = tmp_path / "artifacts" / "rq2" / "final"
    out.mkdir(parents=True)
    assert assert_nb14_output_dir(out, tmp_path, durable_root=tmp_path) == out.resolve()


def test_both_arms_resolve_same_frozen_d0_and_reject_public_init(tmp_path):
    from src.rq2_final_contract import (
        REJECTED_PUBLIC_PRETRAINED_INIT_POLICY,
        TrainingContractError,
        resolve_frozen_d0_init,
    )

    random_fields = _base_fields(tmp_path, ARM_RANDOM)
    quality_fields = _base_fields(tmp_path, ARM_QUALITY)
    env = world(tmp_path)
    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    d0 = resolve_frozen_d0_init(env["flags"], project_root=tmp_path, identity=upstream["d0"])
    random_c = build_arm_training_contract(random_fields)
    quality_c = build_arm_training_contract(quality_fields)
    assert random_c["d0_init_model_state_sha256"] == quality_c["d0_init_model_state_sha256"] == d0["model_state_sha256"]
    assert random_c["d0_checkpoint_fingerprint_sha256"] == quality_c["d0_checkpoint_fingerprint_sha256"]
    assert random_c["init_policy"] != REJECTED_PUBLIC_PRETRAINED_INIT_POLICY
    wrong = dict(random_fields)
    wrong["init_policy"] = REJECTED_PUBLIC_PRETRAINED_INIT_POLICY
    with pytest.raises(TrainingContractError, match="public pretrained"):
        build_arm_training_contract(wrong)
    stale = dict(random_fields)
    stale["rq1_direct_training_contract_hash"] = "00" * 32
    stale_c = build_arm_training_contract(stale)
    assert stale_c["arm_training_contract_sha256"] != random_c["arm_training_contract_sha256"]
    mutated = dict(random_fields)
    mutated["d0_init_model_state_sha256"] = "ff" * 32
    mutated_c = build_arm_training_contract(mutated)
    assert mutated_c["arm_training_contract_sha256"] != random_c["arm_training_contract_sha256"]
    missing_mix = dict(random_fields)
    missing_mix["gold_pseudo_mix_policy"] = "UNSET_REQUIRE_EXPLICIT_CONFIG"
    missing_mix["gold_slots"] = None
    missing_mix["pseudo_slots"] = None
    with pytest.raises(TrainingContractError, match="GOLD_PSEUDO_MIX_POLICY"):
        build_arm_training_contract(missing_mix)
    swapped_mix = dict(quality_fields)
    swapped_mix["gold_slots"] = 2
    swapped_mix["pseudo_slots"] = 1
    quality_other = build_arm_training_contract(swapped_mix)
    with pytest.raises(TrainingContractError):
        assert_augmentation_fairness(random_c, quality_other)


def test_wrong_d0_checkpoint_fails_closed(tmp_path):
    from src.rq2_final_contract import resolve_frozen_d0_init

    env = world(tmp_path)
    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    d0 = resolve_frozen_d0_init(env["flags"], project_root=tmp_path, identity=upstream["d0"])
    (d0["checkpoint_dir"] / "model.safetensors").write_bytes(b"tampered-d0")
    with pytest.raises(Exception, match="fingerprint|model-state|identity|Direct checkpoint"):
        resolve_frozen_d0_init(env["flags"], project_root=tmp_path, identity=upstream["d0"])
    env2 = world(tmp_path / "stale")
    contract_path = env2["d0_dir"] / "direct_training_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["direct_training_contract_hash"] = "00" * 32
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="training-contract|stale|self-consistent|hash"):
        resolve_frozen_d0_init(env2["flags"], project_root=tmp_path / "stale")


def test_multi_seed_materializes_every_declared_seed(tmp_path):
    from src.rq2_final_contract import SEED_POLICY_MULTI, materialize_seed_bound_contracts

    fields = _base_fields(tmp_path, ARM_RANDOM)
    fields["seed_policy"] = SEED_POLICY_MULTI
    fields["seed_policy_seeds"] = [7, 11, 13]
    fields.pop("seed", None)
    fields.pop("dataloader_seed", None)
    with pytest.raises(TrainingContractError, match="active_seed"):
        build_arm_training_contract(fields)
    built = materialize_seed_bound_contracts(fields)
    assert [int(contract["seed"]) for contract in built] == [7, 11, 13]
    assert len({contract["arm_training_contract_sha256"] for contract in built}) == 3
    single = materialize_seed_bound_contracts(_base_fields(tmp_path, ARM_RANDOM))
    assert len(single) == 1
    assert int(single[0]["seed"]) == 42


def test_seed_policy_mean_std_only_for_multi_seed():
    from src.rq2_final_contract import SEED_POLICY_MULTI, SEED_POLICY_SINGLE, summarize_seed_runs

    single = summarize_seed_runs(
        [{"sacrebleu": 10.0, "chrfpp": 20.0}],
        seed_policy={"seed_policy": SEED_POLICY_SINGLE, "seed_policy_seeds": [42], "report_mean_std": False},
    )
    assert single["mean_std_reported"] is False
    assert "sacrebleu_mean" not in single
    assert "compute-constrained" in single["protocol"]
    multi = summarize_seed_runs(
        [{"sacrebleu": 10.0, "chrfpp": 20.0}, {"sacrebleu": 12.0, "chrfpp": 22.0}],
        seed_policy={"seed_policy": SEED_POLICY_MULTI, "seed_policy_seeds": [1, 2], "report_mean_std": True},
    )
    assert multi["mean_std_reported"] is True
    assert "sacrebleu_mean" in multi
    assert multi["protocol"] == "multi-seed protocol"
