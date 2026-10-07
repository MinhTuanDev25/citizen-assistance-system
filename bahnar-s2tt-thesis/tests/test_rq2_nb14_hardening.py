"""NB14 durable-path, pin, resume, freeze, truncation, and audio gates."""
from __future__ import annotations

import json

import pytest

from src.rq2_final_contract import (
    Nb14Flags,
    Rq2FinalError,
    SEED_POLICY_MULTI,
    SEED_POLICY_SINGLE,
    TrainingContractError,
    UpstreamGateError,
    assert_nb14_output_dir,
    normalize_seed_policy,
    verify_upstream_rq2,
)
from src.rq2_final_evaluate import EvaluationError, assert_recorded_best_fingerprint, canonical_references
from src.rq2_final_train import (
    audit_split_target_truncation,
    freeze_speech_feature_encoder,
    ARM_QUALITY,
    ARM_RANDOM,
    ArmIsolationError,
    discover_latest_valid_checkpoint,
    enforce_target_truncation_gate,
    verify_pseudo_audio_rows,
)
from src.rq2_selection import read_manifest_csv
from tests.rq2_nb14_fixtures import seal_nb13, write_complete_checkpoint, write_frozen_d0_state, write_locked_rq1_manifests


def _flags(tmp_path):
    manifests = write_locked_rq1_manifests(tmp_path)
    d0 = tmp_path / "direct_state"
    write_frozen_d0_state(d0, manifests=manifests)
    return Nb14Flags(direct_state_dir=str(d0))


def test_durable_root_ignores_code_checkout_current(tmp_path):
    project = tmp_path / "code"
    durable = tmp_path / "durable"
    project.mkdir()
    decoy = project / "artifacts" / "rq2" / "selection"
    decoy.mkdir(parents=True)
    (decoy / "CURRENT").write_text("code-root-current\n", encoding="utf-8")
    sealed = seal_nb13(durable)
    generation_id = sealed["nb13"]["summary"]["generation_id"]
    (durable / "artifacts" / "rq2" / "selection" / "CURRENT").write_text("other-current\n", encoding="utf-8")
    payload = verify_upstream_rq2(
        project,
        flags=_flags(project),
        artifact_root=durable,
        expected_nb13_generation_id=generation_id,
    )
    assert payload["resolved_nb13_generation_id"] == generation_id
    assert payload["nb13_current_generation_id"] == "other-current"
    assert payload["expected_nb13_generation_id"] == generation_id
    with pytest.raises(Rq2FinalError, match="must be"):
        assert_nb14_output_dir(project / "artifacts" / "rq2" / "final", project, durable_root=durable)
    accepted = assert_nb14_output_dir(durable / "artifacts" / "rq2" / "final", project, durable_root=durable)
    assert accepted == (durable / "artifacts" / "rq2" / "final").resolve()

    # Regression: PROJECT_ROOT/artifacts/rq2 may symlink to durable RQ2.
    project_symlink = tmp_path / "code-symlink"
    project_symlink.mkdir()
    project_rq2 = project_symlink / "artifacts" / "rq2"
    project_rq2.parent.mkdir(parents=True, exist_ok=True)
    durable_rq2 = durable / "artifacts" / "rq2"
    durable_rq2.mkdir(parents=True, exist_ok=True)
    project_rq2.symlink_to(durable_rq2, target_is_directory=True)

    accepted_symlink = assert_nb14_output_dir(
        durable / "artifacts" / "rq2" / "final",
        project_symlink,
        durable_root=durable,
    )
    assert accepted_symlink == (durable / "artifacts" / "rq2" / "final").resolve()

    with pytest.raises(Rq2FinalError, match="PROJECT_ROOT"):
        assert_nb14_output_dir(
            project_symlink / "artifacts" / "rq2" / "final",
            project_symlink,
            durable_root=durable,
        )


def test_missing_and_wrong_nb13_pin_fail(tmp_path):
    durable = tmp_path / "durable"
    sealed = seal_nb13(durable)
    flags = _flags(tmp_path)
    with pytest.raises(UpstreamGateError):
        verify_upstream_rq2(
            tmp_path,
            flags=flags,
            artifact_root=durable,
            expected_nb13_generation_id="missing-nb13-gen",
        )
    with pytest.raises(RuntimeError, match="selection_budget_hours"):
        verify_upstream_rq2(
            tmp_path,
            flags=flags,
            artifact_root=durable,
            expected_nb13_generation_id=sealed["nb13"]["summary"]["generation_id"],
            expected_budget_hours=20.0,
        )


def test_final_identity_fingerprints_fail_closed():
    good = "ab" * 32
    assert_recorded_best_fingerprint({"d_random_best_checkpoint_fingerprint": good}, "d_random", good)
    with pytest.raises(EvaluationError, match="missing"):
        assert_recorded_best_fingerprint({}, "d_random", good)
    with pytest.raises(EvaluationError, match="missing"):
        assert_recorded_best_fingerprint({}, "d_quality", good)
    with pytest.raises(EvaluationError, match="does not match"):
        assert_recorded_best_fingerprint({"d_quality_best_checkpoint_fingerprint": good}, "d_quality", "cd" * 32)


def test_seed_policy_enumerates_every_declared_seed():
    single = normalize_seed_policy({"seed_policy": SEED_POLICY_SINGLE, "seed_policy_seeds": [42]})
    assert single["seed_runs"] == [{"seed": 42, "dataloader_seed": 42}]
    multi = normalize_seed_policy({"seed_policy": SEED_POLICY_MULTI, "seed_policy_seeds": [7, 11, 13]})
    assert [item["seed"] for item in multi["seed_runs"]] == [7, 11, 13]
    assert "seed" not in multi


def test_resume_is_isolated_per_arm(tmp_path):
    layout = {"checkpoints": tmp_path / "checkpoints"}
    extra = {
        "nb13_generation_id": "nb13-pin",
        "manifest_sha256": "aa" * 32,
        "d0_init_model_state_sha256": "bb" * 32,
    }
    write_complete_checkpoint(
        layout["checkpoints"] / "random-step",
        arm=ARM_RANDOM,
        contract_hash="11" * 32,
        data_hash="aa" * 32,
        step=4,
        fingerprint_extra=extra,
    )
    found = discover_latest_valid_checkpoint(
        layout,
        arm=ARM_RANDOM,
        expected_contract_hash="11" * 32,
        expected_data_hash="aa" * 32,
        expected_nb13_generation_id="nb13-pin",
        expected_manifest_sha256="aa" * 32,
        expected_d0_init_sha256="bb" * 32,
    )
    assert found.name == "random-step"
    with pytest.raises(ArmIsolationError):
        discover_latest_valid_checkpoint(
            layout,
            arm=ARM_QUALITY,
            expected_contract_hash="11" * 32,
            expected_data_hash="aa" * 32,
        )
    with pytest.raises(TrainingContractError, match="NB13 generation"):
        discover_latest_valid_checkpoint(
            layout,
            arm=ARM_RANDOM,
            expected_contract_hash="11" * 32,
            expected_data_hash="aa" * 32,
            expected_nb13_generation_id="other-gen",
        )
    empty = {"checkpoints": tmp_path / "empty"}
    assert discover_latest_valid_checkpoint(
        empty, arm=ARM_RANDOM, expected_contract_hash="11" * 32, expected_data_hash="aa" * 32,
    ) is None


def test_feature_encoder_freeze_is_enforced():
    class Param:
        def __init__(self):
            self.requires_grad = True

        def numel(self):
            return 1

    class Feature:
        def __init__(self):
            self.param = Param()

        def named_parameters(self):
            return [("conv.weight", self.param)]

        def _freeze_parameters(self):
            self.param.requires_grad = False

    class Encoder:
        def __init__(self):
            self.feature_extractor = Feature()

    class Model:
        def __init__(self):
            self.encoder = Encoder()

    model = Model()
    freeze_speech_feature_encoder(model)
    assert model.encoder.feature_extractor.param.requires_grad is False
    model.encoder.feature_extractor._freeze_parameters = lambda: None
    model.encoder.feature_extractor.param.requires_grad = True
    with pytest.raises(TrainingContractError, match="trainable"):
        freeze_speech_feature_encoder(model)


def test_tokenizer_audit_counts_rows_before_the_gate():
    class Tok:
        unk_token_id = 1

        def __call__(self, text_target, add_special_tokens, truncation, return_attention_mask):
            assert truncation is False
            return {"input_ids": [[1] * (3 if text != "LONG" else 12) for text in text_target]}

    report = audit_split_target_truncation(Tok(), ["xin chao", "LONG"], split="gold_train", max_target_length=5)
    assert report["n_rows"] == 2
    assert report["n_truncated"] == 1
    assert report["max_target_token_length"] == 12
    gate = enforce_target_truncation_gate([report], max_truncation_rate=0.6)
    assert gate["passed"] is True


def test_truncation_gate_and_untruncated_references():
    below = enforce_target_truncation_gate(
        [{"split": "g_validation", "n_rows": 10, "n_truncated": 0, "truncation_rate": 0.0, "max_target_token_length": 20}],
        max_truncation_rate=0.01,
    )
    assert below["passed"] is True
    with pytest.raises(TrainingContractError, match="truncation_rate"):
        enforce_target_truncation_gate(
            [{"split": "d_random", "n_rows": 10, "n_truncated": 2, "truncation_rate": 0.2, "max_target_token_length": 400}],
            max_truncation_rate=0.01,
        )
    import pandas as pd

    frame = pd.DataFrame({"text_vi_norm": ["câu đầy đủ không bị cắt"]})
    assert canonical_references(frame) == ["câu đầy đủ không bị cắt"]


def test_pseudo_audio_resolves_under_frozen_u_clean(tmp_path):
    durable = tmp_path / "durable"
    sealed = seal_nb13(durable)
    generation_id = sealed["nb13"]["summary"]["generation_id"]
    manifest = durable / "artifacts" / "rq2" / "selection" / "generations" / generation_id / "d_random_manifest.csv"
    rows = read_manifest_csv(manifest)
    u_clean = durable / "artifacts" / "rq2" / "u_clean"
    report = verify_pseudo_audio_rows(rows, u_clean_dir=u_clean)
    assert report["n_rows"] == len(rows)
    with pytest.raises(Exception):
        verify_pseudo_audio_rows([{"segment_uid": "x", "segment_local_path": "../outside.wav"}], u_clean_dir=u_clean)
    missing = dict(rows[0])
    missing["segment_local_path"] = "segments/src-a/missing.wav"
    with pytest.raises(TrainingContractError, match="missing"):
        verify_pseudo_audio_rows([missing], u_clean_dir=u_clean)
    with pytest.raises(TrainingContractError, match="missing"):
        verify_pseudo_audio_rows(rows, u_clean_dir=tmp_path / "wrong-root")


def test_d0_checkpoint_resolver_accepts_durable_ckpts_layout(tmp_path):
    from src.rq2_final_contract import _resolve_d0_checkpoint_dir

    state = tmp_path / "contract_deadbeef"
    experiment_id = "direct_xlsr300m_mbart50_vi_v1"
    best_name = "checkpoint-35000"

    expected = (
        state
        / "checkpoints"
        / "full_train"
        / experiment_id
        / "ckpts"
        / best_name
    )
    expected.mkdir(parents=True)

    resolved = _resolve_d0_checkpoint_dir(
        state,
        experiment_id=experiment_id,
        best_name=best_name,
    )

    assert resolved == expected
