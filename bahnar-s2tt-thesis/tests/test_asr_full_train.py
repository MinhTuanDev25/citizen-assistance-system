"""Tests for full-train / resume-test helpers (Notebook 03)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.asr_full_data import STATUS_FULL_PREPARE, SUMMARY_JSON
from src.asr_full_train import (
    FULL_TRAIN_MARKER,
    RESUME_TEST_MARKER,
    STATUS_FAILED,
    STATUS_FULL_RESUME_TEST,
    STATUS_FULL_TRAINING,
    assert_checkpoint_allowed_for_full_train,
    assert_checkpoint_complete_for_resume,
    assert_no_frozen_test_access,
    assert_ready_for_full_evaluate,
    assert_ready_for_full_train,
    build_data_contract,
    build_full_training_hparams,
    build_training_contract,
    build_resume_test_subset,
    compute_steps_per_epoch,
    derive_full_resume_test_status,
    derive_full_training_status,
    experiment_checkpoint_dir,
    find_latest_valid_checkpoint,
    is_forbidden_init_checkpoint,
    load_full_train_success,
    make_drive_checkpoint_sync_callback,
    resolve_full_stage_status,
    resolve_resume_checkpoint,
    restore_experiment_checkpoints_from_drive,
    sync_experiment_checkpoints_to_drive,
    write_checkpoint_fingerprint,
    write_full_train_summary,
    write_resume_test_summary,
)


def _eligible_df(n=20):
    rows = []
    for i in range(n):
        rows.append({
            "record_uid": f"u{i}",
            "record_id": str(i),
            "group_id": f"g{i % 5}",
            "recording_group_id": f"rg{i % 5}",
            "pair_key": f"pk{i}",
            "text_bahnar": "ab",
            "duration_seconds": 1.0,
        })
    return pd.DataFrame(rows)


def _canonical_contract(*, with_hparams=False, experiment_id="exp1", learning_rate=1e-4, **overrides):
    base = dict(
        dataset_id="ds", dataset_revision="rev", parquet_revision="a" * 40,
        train_manifest_content_hash="th", validation_manifest_content_hash="vh",
        vocab_fp="vfp", processing_version="notebook02_audio_v3",
        min_duration=0.5, max_duration=30.0, target_sr=16000,
        pretrained_model_id="facebook/wav2vec2-xls-r-300m",
        pretrained_model_revision="mrev",
    )
    base.update(overrides)
    if not with_hparams:
        return build_data_contract(**base)
    return build_training_contract(
        **base,
        experiment_id=experiment_id,
        hparams=build_full_training_hparams(n_train=1024, learning_rate=learning_rate),
    )


def _write_prepare_summary(state_dir: Path, contract: dict) -> Path:
    """Prepare summary the way finalize_full_prepare writes it: contract embedded."""
    from src.asr_full_data import PREPARE_STATE_SCHEMA_VERSION

    path = Path(state_dir) / SUMMARY_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": STATUS_FULL_PREPARE,
        "data_contract": contract,
        "contract_hash": contract["contract_hash"],
        "prepare_state_schema_version": PREPARE_STATE_SCHEMA_VERSION,
    }), encoding="utf-8")
    return path


def _complete_ckpt(path: Path, step: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": 0.1}), encoding="utf-8"
    )
    for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_bytes(b"x")
    return path


class TestStepsAndHparams:
    def test_steps_per_epoch(self):
        assert compute_steps_per_epoch(200, per_device_train_batch_size=1, gradient_accumulation_steps=8) == 25

    def test_hparams_t4_defaults_and_epoch_cap(self):
        hp = build_full_training_hparams(n_train=1024, num_train_epochs=1.0)
        assert hp["per_device_train_batch_size"] == 1
        assert hp["gradient_accumulation_steps"] == 8
        assert hp["fp16"] is True
        assert hp["gradient_checkpointing"] is True
        assert hp["learning_rate"] == 3e-4
        assert hp["warmup_ratio"] == 0.05
        assert hp["seed"] == 42
        assert hp["save_steps"] == 500
        assert hp["save_total_limit"] == 2
        assert hp["save_only_model"] is False
        with pytest.raises(ValueError, match="<= 3"):
            build_full_training_hparams(n_train=100, num_train_epochs=4)


class TestCheckpointGuards:
    def test_forbidden_pilot_and_resume_test_paths(self, tmp_path: Path):
        pilot = tmp_path / "pilot" / "exp" / "checkpoint-20"
        pilot.mkdir(parents=True)
        assert is_forbidden_init_checkpoint(pilot)
        rt = tmp_path / "resume_test" / "exp" / "checkpoint-100"
        rt.mkdir(parents=True)
        write_checkpoint_fingerprint(rt.parent, experiment_id="e1", kind=RESUME_TEST_MARKER, global_step=100)
        assert is_forbidden_init_checkpoint(rt)
        with pytest.raises(RuntimeError, match="full_train|pilot or resume_test"):
            assert_checkpoint_allowed_for_full_train(rt, experiment_id="e1")

    def test_full_train_checkpoint_must_match_experiment(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        ckpt = _complete_ckpt(root / "checkpoint-500", 500)
        write_checkpoint_fingerprint(root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=500)
        assert assert_checkpoint_allowed_for_full_train(ckpt, experiment_id="expA")
        with pytest.raises(RuntimeError, match="experiment_id"):
            assert_checkpoint_allowed_for_full_train(ckpt, experiment_id="expB")

    def test_incomplete_checkpoint_rejected(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        ckpt = root / "checkpoint-100"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 100}), encoding="utf-8")
        (ckpt / "pytorch_model.bin").write_bytes(b"x")
        write_checkpoint_fingerprint(root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=100)
        with pytest.raises(RuntimeError, match="incomplete"):
            assert_checkpoint_complete_for_resume(ckpt)
        with pytest.raises(RuntimeError, match="incomplete"):
            assert_checkpoint_allowed_for_full_train(ckpt, experiment_id="expA")

    def test_auto_resume_picks_latest_valid(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=0)
        for step in (100, 500, 1000):
            _complete_ckpt(root / f"checkpoint-{step}", step)
        latest = find_latest_valid_checkpoint(root, experiment_id="expA")
        assert latest is not None and latest.name == "checkpoint-1000"
        auto = resolve_resume_checkpoint(
            resume_policy="auto",
            experiment_dir=root,
            experiment_id="expA",
        )
        assert auto.endswith("checkpoint-1000")
        assert resolve_resume_checkpoint(
            resume_policy="never", experiment_dir=root, experiment_id="expA"
        ) is None

    def test_auto_resume_skips_mixed_kind_and_incomplete(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=0)
        incomplete = root / "checkpoint-2000"
        incomplete.mkdir()
        (incomplete / "trainer_state.json").write_text(json.dumps({"global_step": 2000}), encoding="utf-8")
        _complete_ckpt(root / "checkpoint-500", 500)
        latest = find_latest_valid_checkpoint(root, experiment_id="expA")
        assert latest is not None and latest.name == "checkpoint-500"

        rt = experiment_checkpoint_dir(tmp_path, "expA", kind=RESUME_TEST_MARKER)
        write_checkpoint_fingerprint(rt, experiment_id="expA", kind=RESUME_TEST_MARKER, global_step=200)
        _complete_ckpt(rt / "checkpoint-200", 200)
        assert find_latest_valid_checkpoint(rt, experiment_id="expA", kind=FULL_TRAIN_MARKER) is None

    def test_fingerprint_overwrite_false_preserves(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path, "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=0)
        write_checkpoint_fingerprint(
            root, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=99, overwrite=False
        )
        data = json.loads((root / "full_experiment_fingerprint.json").read_text(encoding="utf-8"))
        assert data["global_step"] == 0

    def test_checkpoint_must_live_under_full_train_dir(self, tmp_path: Path):
        # Same fingerprint kind but path lacks full_train/ segment — reject
        orphan = tmp_path / "orphan_exp" / "checkpoint-100"
        _complete_ckpt(orphan, 100)
        write_checkpoint_fingerprint(
            orphan.parent, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=100
        )
        with pytest.raises(RuntimeError, match="full_train"):
            assert_checkpoint_allowed_for_full_train(orphan, experiment_id="expA")

    def test_checkpoint_isolation_rejects_sibling_kinds(self, tmp_path: Path):
        pilot = experiment_checkpoint_dir(tmp_path, "expA", kind="pilot")
        ck = _complete_ckpt(pilot / "checkpoint-50", 50)
        write_checkpoint_fingerprint(pilot, experiment_id="expA", kind="pilot", global_step=50)
        with pytest.raises(RuntimeError, match="full_train"):
            assert_checkpoint_allowed_for_full_train(ck, experiment_id="expA")


class TestResumeTestStatus:
    def test_subset_group_aware_deterministic(self):
        df = _eligible_df(30)
        a = build_resume_test_subset(df, n_samples=10, seed=42)
        b = build_resume_test_subset(df, n_samples=10, seed=42)
        assert list(a["record_uid"]) == list(b["record_uid"])
        assert a["group_id"].nunique() >= 2

    def test_status_requires_step_200_and_invariants(self):
        assert derive_full_resume_test_status(
            phase_a_reached_100=True,
            phase_b_global_step=200,
            optimizer_restored=True,
            scheduler_restored=True,
            rng_restored=True,
            model_restored=True,
            data_position_ok=True,
            used_separate_experiment_dir=True,
        ) == STATUS_FULL_RESUME_TEST
        assert derive_full_resume_test_status(
            phase_a_reached_100=True,
            phase_b_global_step=199,
            optimizer_restored=True,
            scheduler_restored=True,
            rng_restored=True,
            model_restored=True,
            data_position_ok=True,
            used_separate_experiment_dir=True,
        ) != STATUS_FULL_RESUME_TEST


class TestFullTrainGates:
    def test_ready_requires_prepare_and_resume_test(self, tmp_path: Path):
        contract = _canonical_contract()
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_PREPARE"):
            assert_ready_for_full_train(tmp_path, expected_contract=contract)

        # A status-only summary carries no contract, so it proves nothing.
        (tmp_path / SUMMARY_JSON).write_text(
            json.dumps({"status": STATUS_FULL_PREPARE, "dataset_id": "ds"}), encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_PREPARE"):
            assert_ready_for_full_train(tmp_path, expected_contract=contract)

        _write_prepare_summary(tmp_path, contract)
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_RESUME_TEST"):
            assert_ready_for_full_train(tmp_path, expected_contract=contract)

        write_resume_test_summary(tmp_path, {"status": STATUS_FULL_RESUME_TEST, "experiment_id": "exp1"})
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_RESUME_TEST"):
            assert_ready_for_full_train(tmp_path, expected_contract=contract)

        write_resume_test_summary(tmp_path, {**contract, "status": STATUS_FULL_RESUME_TEST})
        assert_ready_for_full_train(tmp_path, expected_contract=contract)

    def test_stale_summary_does_not_satisfy_ready(self, tmp_path: Path):
        """A prepare summary from a different dataset must not unlock training."""
        _write_prepare_summary(tmp_path, _canonical_contract(dataset_id="old"))
        write_resume_test_summary(
            tmp_path, {**_canonical_contract(), "status": STATUS_FULL_RESUME_TEST},
        )
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_PREPARE"):
            assert_ready_for_full_train(tmp_path, expected_contract=_canonical_contract())

    def test_resolve_stage_status_prefers_pipeline_error(self, tmp_path: Path):
        (tmp_path / SUMMARY_JSON).write_text(
            json.dumps({"status": STATUS_FULL_PREPARE, "dataset_id": "ds", "dataset_revision": "rev"}),
            encoding="utf-8",
        )
        status = resolve_full_stage_status(
            pipeline_error={"message": "boom"},
            current_full_status=None,
            full_stage="prepare",
            state_dir=tmp_path,
            experiment_id="exp1",
            expected_contract=_canonical_contract(),
        )
        assert status == STATUS_FAILED

    def test_training_status_and_frozen_guard(self):
        assert derive_full_training_status(
            reached_target_steps=True,
            full_validation_ok=True,
            best_checkpoint_reload_ok=True,
            frozen_test_accessed=False,
            started_from_base_model=True,
        ) == STATUS_FULL_TRAINING
        assert derive_full_training_status(
            reached_target_steps=True,
            full_validation_ok=True,
            best_checkpoint_reload_ok=True,
            frozen_test_accessed=True,
            started_from_base_model=True,
        ) != STATUS_FULL_TRAINING
        with pytest.raises(RuntimeError, match="Frozen-test"):
            assert_no_frozen_test_access(["data/manifests/rq1_test.csv"], [])
        with pytest.raises(RuntimeError, match="Frozen-test"):
            assert_no_frozen_test_access([], ["rq1_test"])


class TestDriveSyncRestore:
    def test_sync_and_restore_roundtrip(self, tmp_path: Path):
        local_root = tmp_path / "local_ckpts"
        drive_state = tmp_path / "drive_state"
        drive_state.mkdir()
        exp = experiment_checkpoint_dir(local_root, "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(exp, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=100)
        _complete_ckpt(exp / "checkpoint-100", 100)
        synced = sync_experiment_checkpoints_to_drive(
            exp, drive_state, experiment_id="expA", kind=FULL_TRAIN_MARKER
        )
        assert synced is not None and synced.is_dir()

        fresh_local = experiment_checkpoint_dir(tmp_path / "fresh", "expA", kind=FULL_TRAIN_MARKER)
        restored = restore_experiment_checkpoints_from_drive(
            fresh_local, drive_state, experiment_id="expA", kind=FULL_TRAIN_MARKER
        )
        assert restored is not None
        assert (restored / "checkpoint-100" / "optimizer.pt").is_file()

    def test_on_save_callback_syncs_to_drive(self, tmp_path: Path):
        local_root = tmp_path / "local_ckpts"
        drive_state = tmp_path / "drive_state"
        drive_state.mkdir()
        exp = experiment_checkpoint_dir(local_root, "expA", kind=FULL_TRAIN_MARKER)
        write_checkpoint_fingerprint(exp, experiment_id="expA", kind=FULL_TRAIN_MARKER, global_step=0)
        _complete_ckpt(exp / "checkpoint-500", 500)
        cb = make_drive_checkpoint_sync_callback(
            local_experiment_dir=exp,
            full_state_dir=drive_state,
            experiment_id="expA",
            kind=FULL_TRAIN_MARKER,
        )
        state = type("S", (), {"global_step": 500})()
        control = type("C", (), {})()
        cb.on_save(None, state, control)
        drive_exp = drive_state / "checkpoints" / FULL_TRAIN_MARKER / "expA"
        # Checkpoints land once in the immutable store; snapshots reference them.
        assert (drive_exp / "ckpts" / "checkpoint-500" / "optimizer.pt").is_file()
        assert (drive_exp / "LATEST").read_text(encoding="utf-8").strip() == "v1"
        manifest = json.loads((drive_exp / "snapshots" / "v1" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["checkpoints"] == ["checkpoint-500"]
        assert manifest["uploaded_now"] == ["checkpoint-500"]
        fp = json.loads((exp / "full_experiment_fingerprint.json").read_text(encoding="utf-8"))
        assert fp["global_step"] == 500

        # A second save must not re-upload the checkpoint Drive already holds.
        _complete_ckpt(exp / "checkpoint-1000", 1000)
        cb.on_save(None, type("S", (), {"global_step": 1000})(), control)
        manifest2 = json.loads((drive_exp / "snapshots" / "v2" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest2["uploaded_now"] == ["checkpoint-1000"]
        # Ordered by step, not lexicographically ("checkpoint-1000" < "-500" as text).
        assert manifest2["checkpoints"] == ["checkpoint-500", "checkpoint-1000"]
        assert manifest2["retention"]["latest"] == 1000
        assert manifest2["retention"]["previous_latest"] == 500
        assert (drive_exp / "LATEST").read_text(encoding="utf-8").strip() == "v2"


class TestEvaluateAndContractBinds:
    def test_evaluate_requires_full_train_success(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_TRAINING"):
            assert_ready_for_full_evaluate(tmp_path, expected_experiment_id="exp1")
        write_full_train_summary(
            tmp_path,
            {
                "status": STATUS_FULL_TRAINING,
                "experiment_id": "exp1",
                "dataset_id": "ds",
                "dataset_revision": "rev",
                "vocab_fp": "vfp",
                "pretrained_model_id": "m",
                "pretrained_model_revision": "mrev",
            },
        )
        assert_ready_for_full_evaluate(
            tmp_path,
            expected_experiment_id="exp1",
            expected_dataset_id="ds",
            expected_dataset_revision="rev",
            expected_vocab_fp="vfp",
            expected_model_id="m",
            expected_model_revision="mrev",
        )
        with pytest.raises(RuntimeError, match="SUCCESS_FULL_TRAINING"):
            assert_ready_for_full_evaluate(
                tmp_path,
                expected_experiment_id="exp1",
                expected_dataset_id="other",
            )

    def test_stale_full_train_contract_rejected(self, tmp_path: Path):
        write_full_train_summary(
            tmp_path,
            {
                "status": STATUS_FULL_TRAINING,
                "experiment_id": "exp_old",
                "dataset_id": "ds",
                "vocab_fp": "old_vocab",
            },
        )
        assert not load_full_train_success(tmp_path, expected_experiment_id="exp_new")
        assert not load_full_train_success(
            tmp_path, expected_experiment_id="exp_old", expected_vocab_fp="new_vocab"
        )
        assert load_full_train_success(tmp_path, expected_experiment_id="exp_old", expected_vocab_fp="old_vocab")

    def test_resolve_evaluate_fallback_needs_full_train(self, tmp_path: Path):
        status = resolve_full_stage_status(
            pipeline_error=None,
            current_full_status=None,
            full_stage="evaluate",
            state_dir=tmp_path,
            experiment_id="exp1",
        )
        assert status == STATUS_FAILED
        contract = _canonical_contract(experiment_id="exp1", with_hparams=True)
        write_full_train_summary(
            tmp_path, {**contract, "status": STATUS_FULL_TRAINING, "experiment_id": "exp1"},
        )
        (tmp_path / "full_evaluate_summary.json").write_text(
            json.dumps({**contract, "status": "SUCCESS_FULL_EVALUATE", "experiment_id": "exp1"}),
            encoding="utf-8",
        )
        status2 = resolve_full_stage_status(
            pipeline_error=None,
            current_full_status=None,
            full_stage="evaluate",
            state_dir=tmp_path,
            experiment_id="exp1",
            expected_contract=contract,
        )
        assert status2 == "SUCCESS_FULL_EVALUATE"

        # An evaluate summary bound to different hparams is not this run's success.
        status3 = resolve_full_stage_status(
            pipeline_error=None,
            current_full_status=None,
            full_stage="evaluate",
            state_dir=tmp_path,
            experiment_id="exp1",
            expected_contract=_canonical_contract(
                experiment_id="exp1", with_hparams=True, learning_rate=9e-9,
            ),
        )
        assert status3 == STATUS_FAILED


class TestEligibleAudioAndMonitor:
    def test_resolve_and_assert_audio(self, tmp_path: Path):
        from src.asr_full_train import (
            assert_eligible_audio_available,
            build_validation_monitor_subset,
            resolve_eligible_audio_path,
            select_best_checkpoint_by_cer,
        )
        from src.data_utils import safe_cache_filename

        df = _eligible_df(5)
        df["local_cache_relpath"] = [safe_cache_filename(u) for u in df["record_uid"]]
        root = tmp_path / "cache"
        root.mkdir()
        for rel in df["local_cache_relpath"]:
            (root / rel).write_bytes(b"RIFF")
        assert resolve_eligible_audio_path(df.iloc[0], [root]) is not None
        # existence-only path for unit smoke (full verify needs real wav+sidecar)
        assert_eligible_audio_available(df, [root], dataset_revision="rev", verify=False)
        with pytest.raises(RuntimeError, match="missing local audio"):
            assert_eligible_audio_available(
                df, [tmp_path / "empty"], dataset_revision="rev", verify=False
            )
        # corrupt / unverifiable when verify=True
        with pytest.raises(RuntimeError, match="failed verification"):
            assert_eligible_audio_available(df, [root], dataset_revision="rev", verify=True)
        mon = build_validation_monitor_subset(df, n_samples=3, seed=42)
        assert len(mon) == 3
        best = select_best_checkpoint_by_cer([
            {"checkpoint": "a", "cer": 0.4},
            {"checkpoint": "b", "cer": 0.2},
            {"checkpoint": "c", "cer": None},
        ])
        assert best["checkpoint"] == "b"

    def test_inspect_trainer_checkpoint(self, tmp_path: Path):
        from src.asr_full_train import inspect_trainer_checkpoint

        ck = _complete_ckpt(tmp_path / "checkpoint-100", 100)
        info = inspect_trainer_checkpoint(ck)
        assert info["global_step"] == 100
        assert info["model_ok"] and info["optimizer_ok"] and info["scheduler_ok"] and info["rng_ok"]
