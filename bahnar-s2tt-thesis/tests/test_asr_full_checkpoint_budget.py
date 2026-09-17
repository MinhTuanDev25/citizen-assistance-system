"""
Drive checkpoint retention, budget and last-known-good semantics.

Drive is a quota-limited FUSE mount, so space is tracked by explicit accounting
rather than ``shutil.disk_usage``, and the snapshot LATEST points at may never be
sacrificed to make room.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.asr_full_train import (
    FULL_TRAIN_MARKER,
    assert_drive_checkpoint_budget,
    drive_experiment_dir,
    drive_experiment_protected_bytes,
    estimate_checkpoint_bytes,
    experiment_checkpoint_dir,
    measure_dir_bytes,
    plan_checkpoint_retention,
    read_snapshot_manifest,
    resolve_drive_latest_snapshot,
    resolve_snapshot_checkpoints,
    sync_experiment_checkpoints_to_drive,
    write_checkpoint_fingerprint,
)

EXP = "expA"


def _complete_ckpt(path: Path, step: int, *, payload_bytes: int = 512) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (path / name).write_bytes(b"\0" * payload_bytes)
    return path


def _local(tmp_path: Path, *steps: int, payload_bytes: int = 512) -> Path:
    local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
    write_checkpoint_fingerprint(
        local, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        global_step=max(steps), overwrite=True,
    )
    for step in steps:
        _complete_ckpt(local / f"checkpoint-{step}", step, payload_bytes=payload_bytes)
    return local


def _sync(local: Path, drive: Path, **kwargs) -> Path:
    return sync_experiment_checkpoints_to_drive(
        local, drive, experiment_id=EXP, kind=FULL_TRAIN_MARKER, **kwargs,
    )


class TestRetentionPlan:
    def test_latest_and_previous_are_kept_by_default(self):
        plan = plan_checkpoint_retention(candidate_steps=[100, 200, 300], save_total_limit=2)
        assert plan["latest"] == 300
        assert plan["keep"] == [200, 300]
        assert plan["drop"] == [100]

    def test_best_survives_even_when_it_is_not_recent(self):
        """Step-ordered pruning would delete best and silently break evaluate."""
        plan = plan_checkpoint_retention(
            candidate_steps=[100, 200, 300], best_step=100, save_total_limit=2,
        )
        assert set(plan["keep"]) == {100, 300}
        assert plan["reasons"][100] == "best"
        assert plan["reasons"][300] == "latest"

    def test_best_equal_to_latest_is_not_double_counted(self):
        plan = plan_checkpoint_retention(
            candidate_steps=[100, 200], best_step=200, save_total_limit=2,
        )
        assert plan["keep"] == [100, 200]
        assert plan["reasons"][200] == "latest"

    def test_previous_latest_is_dropped_when_the_limit_is_one(self):
        plan = plan_checkpoint_retention(
            candidate_steps=[100, 200, 300], best_step=100, save_total_limit=1,
        )
        assert set(plan["keep"]) == {100, 300}  # mandatory pair still survives

    def test_explicit_previous_latest_beats_the_inferred_one(self):
        plan = plan_checkpoint_retention(
            candidate_steps=[100, 200, 300], previous_latest_step=100, save_total_limit=3,
        )
        assert plan["previous_latest"] == 100 and 100 in plan["keep"]

    def test_no_candidates_yields_nothing_to_keep(self):
        assert plan_checkpoint_retention(candidate_steps=[])["keep"] == []


class TestCheckpointSizeAccounting:
    def test_estimate_includes_optimizer_states(self):
        est = estimate_checkpoint_bytes(n_parameters=1_000_000, overhead_bytes=0)
        assert est["weight_bytes"] == 4_000_000
        assert est["optimizer_bytes"] == 8_000_000
        assert est["checkpoint_bytes"] == 12_000_000

    def test_save_only_model_drops_the_optimizer_cost(self):
        est = estimate_checkpoint_bytes(
            n_parameters=1_000_000, save_only_model=True, overhead_bytes=0,
        )
        assert est["optimizer_bytes"] == 0 and est["checkpoint_bytes"] == 4_000_000

    def test_measured_size_is_recursive_and_not_assumed(self, tmp_path: Path):
        ck = _complete_ckpt(tmp_path / "checkpoint-100", 100, payload_bytes=1000)
        (ck / "nested").mkdir()
        (ck / "nested" / "extra.bin").write_bytes(b"\0" * 250)
        assert measure_dir_bytes(ck) == 4 * 1000 + 250 + len(
            json.dumps({"global_step": 100})
        )

    def test_absent_directory_measures_zero(self, tmp_path: Path):
        assert measure_dir_bytes(tmp_path / "nope") == 0


class TestBudgetGate:
    def test_within_budget_returns_the_accounting(self):
        out = assert_drive_checkpoint_budget(
            protected_bytes=1000, upload_bytes=500, budget_bytes=2000,
        )
        assert out["peak_bytes"] == 1500 and out["ok"] is True

    def test_exceeding_budget_fails_and_refuses_to_drop_the_lkg(self):
        with pytest.raises(RuntimeError) as exc:
            assert_drive_checkpoint_budget(
                protected_bytes=14 * 1024 ** 3, upload_bytes=4 * 1024 ** 3,
                budget_bytes=15 * 1024 ** 3, label="full_train/expA",
            )
        message = str(exc.value)
        assert "budget exceeded" in message
        assert "last-known-good" in message
        assert "DRIVE_CHECKPOINT_BUDGET_BYTES" in message

    def test_temporary_upload_peak_counts_towards_the_budget(self):
        """Protected alone fits, but the copy needs room at the same time."""
        assert_drive_checkpoint_budget(
            protected_bytes=900, upload_bytes=100, budget_bytes=1000,
        )
        with pytest.raises(RuntimeError):
            assert_drive_checkpoint_budget(
                protected_bytes=900, upload_bytes=200, budget_bytes=1000,
            )


class TestSyncRetentionAndLkg:
    def test_budget_failure_happens_before_anything_is_uploaded(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            _sync(local, drive, budget_bytes=1)
        store = drive_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        assert not store.exists() or list(store.glob("checkpoint-*")) == []
        assert resolve_drive_latest_snapshot(drive, EXP, kind=FULL_TRAIN_MARKER) is None

    def test_old_checkpoints_are_collected_only_after_latest_moves(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        snap1 = _sync(local, drive, save_total_limit=2)
        assert (drive_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER) / "LATEST").is_file()

        _complete_ckpt(local / "checkpoint-200", 200)
        _complete_ckpt(local / "checkpoint-300", 300)
        snap2 = _sync(local, drive, save_total_limit=2)
        assert snap2 != snap1

        store = drive_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        kept = sorted(p.name for p in store.glob("checkpoint-*"))
        # The rollback target is the snapshot LATEST used to point at (step 100),
        # not merely the second-newest local checkpoint: step 200 was never a
        # committed LATEST, so it is not a known-good state to fall back to.
        assert kept == ["checkpoint-100", "checkpoint-300"]
        assert read_snapshot_manifest(snap2)["retention"]["previous_latest"] == 100
        # Everything the committed snapshot references still resolves.
        assert len(resolve_snapshot_checkpoints(snap2)) == 2

    def test_best_checkpoint_is_never_collected(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100, 200, 300)
        snap = _sync(local, drive, best_checkpoint_name="checkpoint-100", save_total_limit=2)
        manifest = read_snapshot_manifest(snap)
        assert "checkpoint-100" in manifest["checkpoints"]
        assert "checkpoint-300" in manifest["checkpoints"]
        assert manifest["retention"]["reasons"]["100"] == "best"

    def test_latest_pointer_survives_an_interrupted_second_sync(self, tmp_path: Path):
        """An aborted sync must leave the previous LATEST usable."""
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        _sync(local, drive)
        dest = drive_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER)
        before = (dest / "LATEST").read_text(encoding="utf-8")

        # A half-written checkpoint cannot be committed.
        broken = local / "checkpoint-200"
        broken.mkdir()
        (broken / "trainer_state.json").write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Checkpoint incomplete"):
            _sync(local, drive)
        assert (dest / "LATEST").read_text(encoding="utf-8") == before
        lkg = resolve_drive_latest_snapshot(drive, EXP, kind=FULL_TRAIN_MARKER)
        assert [p.name for p in resolve_snapshot_checkpoints(lkg)] == ["checkpoint-100"]

    def test_protected_bytes_reflect_the_current_lkg(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        assert drive_experiment_protected_bytes(drive, EXP, kind=FULL_TRAIN_MARKER) == 0
        local = _local(tmp_path, 100, payload_bytes=1024)
        _sync(local, drive)
        assert drive_experiment_protected_bytes(drive, EXP, kind=FULL_TRAIN_MARKER) > 4096

    def test_existing_checkpoints_are_not_re_uploaded(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        _sync(local, drive)
        _complete_ckpt(local / "checkpoint-200", 200)
        snap = _sync(local, drive)
        assert read_snapshot_manifest(snap)["uploaded_now"] == ["checkpoint-200"]
