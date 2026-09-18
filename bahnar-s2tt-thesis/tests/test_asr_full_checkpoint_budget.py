"""
Durable checkpoint retention, budget and last-known-good semantics.

Durable volumes often report misleading ``shutil.disk_usage``, so space is tracked
by explicit accounting, and the snapshot LATEST points at may never be sacrificed
to make room.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.asr_full_train import (
    FULL_TRAIN_MARKER,
    assert_durable_checkpoint_budget,
    durable_experiment_dir,
    durable_experiment_protected_bytes,
    estimate_checkpoint_bytes,
    experiment_checkpoint_dir,
    measure_dir_bytes,
    plan_checkpoint_retention,
    plan_local_checkpoint_disk_peak,
    read_snapshot_manifest,
    resolve_durable_latest_snapshot,
    resolve_snapshot_checkpoints,
    sync_experiment_checkpoints_to_durable,
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
    kwargs.setdefault("budget_bytes", 10 ** 12)
    return sync_experiment_checkpoints_to_durable(
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
        # latest + best + previous-latest are all mandatory.
        assert set(plan["keep"]) == {100, 200, 300}
        assert plan["reasons"][100] == "best"
        assert plan["reasons"][300] == "latest"
        assert plan["reasons"][200] == "previous_latest"

    def test_best_equal_to_latest_is_not_double_counted(self):
        plan = plan_checkpoint_retention(
            candidate_steps=[100, 200], best_step=200, save_total_limit=2,
        )
        assert plan["keep"] == [100, 200]
        assert plan["reasons"][200] == "latest"

    def test_limit_one_still_keeps_latest_best_and_lkg(self):
        plan = plan_checkpoint_retention(
            candidate_steps=[100, 200, 300], best_step=100, save_total_limit=1,
        )
        assert set(plan["keep"]) == {100, 200, 300}

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
        assert est["optimizer_bytes"] == 2 * est["weight_bytes"]
        assert est["checkpoint_bytes"] == est["weight_bytes"] + est["optimizer_bytes"]

    def test_save_only_model_drops_optimizer(self):
        est = estimate_checkpoint_bytes(n_parameters=1000, save_only_model=True, overhead_bytes=0)
        assert est["optimizer_bytes"] == 0

    def test_local_peak_is_existing_plus_new(self):
        peak = plan_local_checkpoint_disk_peak(
            existing_checkpoint_bytes=1000,
            new_checkpoint_bytes=400,
            hydrated_wav_bytes=50,
        )
        assert peak["checkpoint_peak_bytes"] == 1400
        assert peak["total_peak_bytes"] == 1450

    def test_measure_dir_bytes_counts_files(self, tmp_path: Path):
        d = tmp_path / "x"
        d.mkdir()
        (d / "a.bin").write_bytes(b"12345")
        assert measure_dir_bytes(d) == 5


class TestDurableBudgetGate:
    def test_peak_must_fit(self):
        with pytest.raises(RuntimeError, match="budget exceeded"):
            assert_durable_checkpoint_budget(
                protected_bytes=8, upload_bytes=5, budget_bytes=10,
            )
        ok = assert_durable_checkpoint_budget(
            protected_bytes=3, upload_bytes=5, budget_bytes=10,
        )
        assert ok["peak_bytes"] == 8 and ok["ok"] is True


class TestSyncRetentionAndLkg:
    def test_budget_failure_happens_before_anything_is_uploaded(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            _sync(local, drive, budget_bytes=1)
        store = durable_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        assert not store.exists() or list(store.glob("checkpoint-*")) == []
        assert resolve_durable_latest_snapshot(drive, EXP, kind=FULL_TRAIN_MARKER) is None

    def test_old_checkpoints_are_collected_only_after_latest_moves(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        snap1 = _sync(local, drive, save_total_limit=2)
        assert (durable_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER) / "LATEST").is_file()

        _complete_ckpt(local / "checkpoint-200", 200)
        _complete_ckpt(local / "checkpoint-300", 300)
        snap2 = _sync(local, drive, save_total_limit=2)
        assert snap2 != snap1

        store = durable_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        kept = sorted(p.name for p in store.glob("checkpoint-*"))
        assert kept == ["checkpoint-100", "checkpoint-300"]
        assert read_snapshot_manifest(snap2)["retention"]["previous_latest"] == 100
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

    def test_rollback_snapshot_refs_survive_store_prune(self, tmp_path: Path):
        """Committed previous snapshot must keep resolving after a newer sync."""
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        snap1 = _sync(local, drive)
        _complete_ckpt(local / "checkpoint-200", 200)
        _complete_ckpt(local / "checkpoint-300", 300)
        snap2 = _sync(local, drive, best_checkpoint_name="checkpoint-300", save_total_limit=2)
        assert resolve_snapshot_checkpoints(snap1)
        assert all(p.is_dir() for p in resolve_snapshot_checkpoints(snap1))
        assert resolve_durable_latest_snapshot(drive, EXP, kind=FULL_TRAIN_MARKER) == snap2

    def test_latest_pointer_survives_an_interrupted_second_sync(self, tmp_path: Path):
        """An aborted sync must leave the previous LATEST usable."""
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        _sync(local, drive)
        dest = durable_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER)
        before = (dest / "LATEST").read_text(encoding="utf-8")

        broken = local / "checkpoint-200"
        broken.mkdir()
        (broken / "trainer_state.json").write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Checkpoint incomplete"):
            _sync(local, drive)
        assert (dest / "LATEST").read_text(encoding="utf-8") == before
        lkg = resolve_durable_latest_snapshot(drive, EXP, kind=FULL_TRAIN_MARKER)
        assert [p.name for p in resolve_snapshot_checkpoints(lkg)] == ["checkpoint-100"]

    def test_protected_bytes_reflect_the_current_lkg(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        assert durable_experiment_protected_bytes(drive, EXP, kind=FULL_TRAIN_MARKER) == 0
        local = _local(tmp_path, 100, payload_bytes=1024)
        _sync(local, drive)
        assert durable_experiment_protected_bytes(drive, EXP, kind=FULL_TRAIN_MARKER) > 4096

    def test_existing_checkpoints_are_not_re_uploaded(self, tmp_path: Path):
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        _sync(local, drive)
        _complete_ckpt(local / "checkpoint-200", 200)
        snap = _sync(local, drive)
        assert read_snapshot_manifest(snap)["uploaded_now"] == ["checkpoint-200"]


class TestUniqueBudgetAccounting:
    def test_unique_bytes_include_rollback_only_without_double_count(self, tmp_path: Path):
        from src.asr_full_train import (
            collect_referenced_checkpoint_names,
            measure_unique_checkpoint_bytes,
        )
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100, payload_bytes=1000)
        snap1 = _sync(local, drive)
        _complete_ckpt(local / "checkpoint-200", 200, payload_bytes=1000)
        _complete_ckpt(local / "checkpoint-300", 300, payload_bytes=1000)
        snap2 = _sync(local, drive, best_checkpoint_name="checkpoint-300", save_total_limit=2)
        store = durable_experiment_dir(drive, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        # LATEST keep + previous snap refs
        planned = read_snapshot_manifest(snap2)["checkpoints"]
        lkg = [p.name for p in resolve_snapshot_checkpoints(snap1)]
        names = collect_referenced_checkpoint_names(planned_names=planned, lkg_names=lkg)
        report = measure_unique_checkpoint_bytes(store, names)
        # Each checkpoint counted once even if in both planned and lkg.
        assert len(report["names"]) == len(set(report["names"]))
        assert report["total_bytes"] == sum(report["per_checkpoint_bytes"].values())
        assert "checkpoint-100" in report["names"]  # rollback-only still present

    def test_preflight_does_not_assume_save_total_limit_uploads(self):
        from src.asr_full_train import plan_local_checkpoint_disk_peak
        peak = plan_local_checkpoint_disk_peak(
            existing_checkpoint_bytes=0,
            new_checkpoint_bytes=100,
            hydrated_wav_bytes=0,
        )
        assert peak["checkpoint_peak_bytes"] == 100  # one new checkpoint, not * limit

    def test_missing_budget_fails_closed_on_sync(self, tmp_path: Path, monkeypatch):
        from src.asr_full_train import ENV_DURABLE_CHECKPOINT_BUDGET_BYTES
        monkeypatch.delenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, raising=False)
        drive = tmp_path / "drive"
        drive.mkdir()
        local = _local(tmp_path, 100)
        with pytest.raises(RuntimeError, match="budget required|Invalid durable|missing/invalid"):
            sync_experiment_checkpoints_to_durable(
                local, drive, experiment_id=EXP, kind=FULL_TRAIN_MARKER, budget_bytes=None,
            )
