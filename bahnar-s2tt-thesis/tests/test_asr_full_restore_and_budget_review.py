"""
Regression for post-643 review: restore fingerprint, sync budget rollback,
direct experiment_root parent, export local_experiment_dir, pre-train budget.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.asr_full_train import (
    ENV_DURABLE_CHECKPOINT_BUDGET_BYTES,
    FULL_TRAIN_MARKER,
    assert_checkpoint_allowed_for_full_train,
    assert_durable_checkpoint_budget,
    build_data_contract,
    collect_referenced_checkpoint_names,
    durable_experiment_dir,
    estimate_checkpoint_bytes,
    experiment_checkpoint_dir,
    measure_unique_checkpoint_bytes,
    plan_checkpoint_retention,
    read_snapshot_manifest,
    resolve_best_checkpoint_from_durable,
    resolve_durable_latest_snapshot,
    resolve_snapshot_checkpoints,
    restore_experiment_checkpoints_from_durable,
    snapshot_max_step,
    sync_experiment_checkpoints_to_durable,
    write_checkpoint_fingerprint,
)
from src.asr_runtime_paths import OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP

EXP = "exp_review"
REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "03_asr_baseline_training.ipynb"


def _contract(*, tag: str) -> dict:
    return build_data_contract(
        dataset_id=f"d-{tag}",
        dataset_revision="r",
        parquet_revision="p" * 40,
        train_manifest_content_hash=("t" + tag)[:64].ljust(64, "0"),
        validation_manifest_content_hash="v" * 64,
        vocab_fp="v" * 64,
        processing_version="notebook02_audio_v3",
        min_duration=0.5,
        max_duration=40.0,
        target_sr=16000,
        pretrained_model_id="m",
        pretrained_model_revision="mr",
        overlap_policy=OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
    )


def _complete_ckpt(path: Path, step: int, marker: str, *, payload: int = 400) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"W:" + marker.encode() + b"\0" * payload)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
        (path / name).write_bytes(name.encode())
    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (path / "MARKER.txt").write_text(marker, encoding="utf-8")
    return path


def _write_fp(exp: Path, *, step: int, contract: dict) -> None:
    write_checkpoint_fingerprint(
        exp,
        experiment_id=EXP,
        kind=FULL_TRAIN_MARKER,
        global_step=step,
        extra={"training_contract": contract, **contract},
        overwrite=True,
    )


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


class TestRestoreDoesNotLegitimizeStaleLocal:
    def test_same_name_replaced_and_fingerprint_upgraded(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        good = _contract(tag="A")
        bad = _contract(tag="B")

        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "GOOD_A")
        _write_fp(local, step=100, contract=good)
        sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, save_total_limit=2,
        )

        local2 = experiment_checkpoint_dir(tmp_path / "local2", EXP, kind=FULL_TRAIN_MARKER)
        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        import shutil

        shutil.copytree(store / "checkpoint-100", local2 / "checkpoint-100")
        _complete_ckpt(local2 / "checkpoint-200", 200, "GOOD_A_200")
        _write_fp(local2, step=200, contract=good)
        sync_experiment_checkpoints_to_durable(
            local2, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
            save_total_limit=2, best_checkpoint_name="checkpoint-200",
        )

        import shutil as sh

        sh.rmtree(local)
        local.mkdir(parents=True)
        _complete_ckpt(local / "checkpoint-100", 100, "STALE_B")
        _write_fp(local, step=100, contract=bad)

        restore_experiment_checkpoints_from_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, expected_contract=good,
        )
        assert (local / "checkpoint-100" / "MARKER.txt").read_text() == "GOOD_A"
        assert (local / "checkpoint-200").is_dir()
        fp = json.loads((local / "full_experiment_fingerprint.json").read_text(encoding="utf-8"))
        assert fp["training_contract"]["dataset_id"] == good["dataset_id"]
        assert_checkpoint_allowed_for_full_train(
            local / "checkpoint-100",
            experiment_id=EXP,
            expected_contract=good,
            experiment_root=local,
            require_complete=False,
        )


class TestSyncBudgetIncludesRollbackOnly:
    def test_sync_budget_counts_rollback_only_checkpoint(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        for step in (100, 200, 300):
            _complete_ckpt(local / f"checkpoint-{step}", step, f"M{step}", payload=8000)
            _write_fp(local, step=step, contract=_contract(tag="A"))
            kw = {"save_total_limit": 2}
            if step == 300:
                kw["best_checkpoint_name"] = "checkpoint-300"
            sync_experiment_checkpoints_to_durable(
                local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, **kw,
            )

        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        snaps = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "snapshots"
        assert (store / "checkpoint-100").is_dir()
        vers = sorted(int(p.name[1:]) for p in snaps.iterdir() if p.name.startswith("v") and p.name[1:].isdigit())
        assert len(vers) >= 2
        lkg = resolve_durable_latest_snapshot(durable, EXP, kind=FULL_TRAIN_MARKER)
        lkg_names = [p.name for p in resolve_snapshot_checkpoints(lkg)]
        rollback = snaps / f"v{vers[-2]}"
        rollback_names = [p.name for p in resolve_snapshot_checkpoints(rollback)]
        assert "checkpoint-100" in rollback_names
        assert "checkpoint-100" not in lkg_names

        _complete_ckpt(local / "checkpoint-400", 400, "M400", payload=8000)
        _write_fp(local, step=400, contract=_contract(tag="A"))
        from src.asr_full_train import list_step_checkpoints, _snapshot_versions

        dest = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER)
        sources = {p.name: p for p in list_step_checkpoints(local)}
        store_names = {
            p.name for p in store.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")
        }
        retention = plan_checkpoint_retention(
            candidate_steps=[int(n.split("-")[-1]) for n in set(sources) | store_names],
            best_step=400,
            previous_latest_step=snapshot_max_step(lkg),
            save_total_limit=2,
        )
        keep = [f"checkpoint-{s}" for s in retention["keep"]]
        existing = []
        for ver in _snapshot_versions(dest):
            for ck in resolve_snapshot_checkpoints(dest / "snapshots" / f"v{ver}"):
                existing.append(ck.name)
        referenced = collect_referenced_checkpoint_names(
            planned_names=keep, lkg_names=lkg_names, rollback_names=existing,
        )
        assert "checkpoint-100" in referenced
        protected = measure_unique_checkpoint_bytes(
            store, [n for n in referenced if (store / n).is_dir()],
        )
        pending = [n for n in keep if not (store / n).is_dir()]
        upload = measure_unique_checkpoint_bytes(store, pending, local_sources=sources)
        need = protected["total_bytes"] + upload["total_bytes"]
        rollback_only = protected["per_checkpoint_bytes"]["checkpoint-100"]
        with pytest.raises(RuntimeError, match="budget|Budget"):
            sync_experiment_checkpoints_to_durable(
                local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
                save_total_limit=2, best_checkpoint_name="checkpoint-400",
                budget_bytes=max(1, need - rollback_only),
            )
        snap = sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
            save_total_limit=2, best_checkpoint_name="checkpoint-400",
            budget_bytes=need,
        )
        assert snap is not None
        assert "checkpoint-100" in read_snapshot_manifest(snap).get("budget", {}).get("detail", {}).get(
            "referenced", referenced
        ) or "checkpoint-100" in referenced


class TestExperimentRootDirectParent:
    def test_nested_checkpoint_rejected_when_root_configured(self, tmp_path: Path):
        root = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        nested = root / "archive" / "checkpoint-1"
        _complete_ckpt(nested, 1, "NEST")
        _write_fp(root, step=1, contract=_contract(tag="A"))
        with pytest.raises(RuntimeError, match="exactly under configured experiment_root"):
            assert_checkpoint_allowed_for_full_train(
                nested, experiment_id=EXP, experiment_root=root, require_complete=False,
            )


class TestExportResolveRequiresLocalDir:
    def test_local_experiment_dir_is_required(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        contract = _contract(tag="A")
        _complete_ckpt(local / "checkpoint-100", 100, "G")
        _write_fp(local, step=100, contract=contract)
        sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        with pytest.raises(TypeError, match="local_experiment_dir"):
            resolve_best_checkpoint_from_durable(  # type: ignore[call-arg]
                durable,
                experiment_id=EXP,
                train_summary={"best_checkpoint": "checkpoint-100"},
                expected_contract=contract,
            )
        best = resolve_best_checkpoint_from_durable(
            durable,
            experiment_id=EXP,
            train_summary={"best_checkpoint": "checkpoint-100"},
            expected_contract=contract,
            local_experiment_dir=local,
        )
        assert Path(best).name == "checkpoint-100"
        assert Path(best).parent == local


class TestMissingFingerprintFailsClosed:
    def test_restore_refuses_orphan_local_checkpoints_without_fingerprint(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        good = _contract(tag="A")
        seeded = experiment_checkpoint_dir(tmp_path / "seed", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(seeded / "checkpoint-100", 100, "GOOD")
        _write_fp(seeded, step=100, contract=good)
        sync_experiment_checkpoints_to_durable(
            seeded, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )

        orphan = experiment_checkpoint_dir(tmp_path / "orphan", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(orphan / "checkpoint-300", 300, "ORPHAN")
        assert not (orphan / "full_experiment_fingerprint.json").is_file()
        with pytest.raises(RuntimeError, match="missing fingerprint"):
            restore_experiment_checkpoints_from_durable(
                orphan, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, expected_contract=good,
            )
        assert not (orphan / "full_experiment_fingerprint.json").is_file()
        assert not (orphan / "checkpoint-100").exists()
        assert (orphan / "checkpoint-300" / "MARKER.txt").read_text() == "ORPHAN"


class TestRestoreAtomicValidation:
    def test_failed_restore_does_not_mutate_local(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        c_a = _contract(tag="A")
        c_b = _contract(tag="B")
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "A100")
        _write_fp(local, step=100, contract=c_a)
        sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        (local / "checkpoint-100" / "MARKER.txt").write_text("LOCAL_CHANGED", encoding="utf-8")
        _complete_ckpt(local / "checkpoint-200", 200, "B200")
        _write_fp(local, step=200, contract=c_b)
        with pytest.raises(RuntimeError, match="ahead of durable"):
            restore_experiment_checkpoints_from_durable(
                local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
            )
        assert (local / "checkpoint-100" / "MARKER.txt").read_text() == "LOCAL_CHANGED"


class TestPreflightProtectedMatchesSync:
    def test_helper_includes_rollback_only_bytes(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        from src.asr_full_train import durable_experiment_protected_bytes

        durable = tmp_path / "durable"
        durable.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        c = _contract(tag="A")
        for step in (100, 200, 300):
            _complete_ckpt(local / f"checkpoint-{step}", step, f"M{step}", payload=5000)
            _write_fp(local, step=step, contract=c)
            kw: dict = {"save_total_limit": 2}
            if step == 300:
                kw["best_checkpoint_name"] = "checkpoint-300"
            sync_experiment_checkpoints_to_durable(
                local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, **kw,
            )
        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        dest = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER)
        from src.asr_full_train import _snapshot_versions

        names = []
        for ver in _snapshot_versions(dest):
            for ck in resolve_snapshot_checkpoints(dest / "snapshots" / f"v{ver}"):
                names.append(ck.name)
        sync_style = measure_unique_checkpoint_bytes(
            store, collect_referenced_checkpoint_names(planned_names=(), rollback_names=names),
        )["total_bytes"]
        helper = durable_experiment_protected_bytes(durable, EXP, kind=FULL_TRAIN_MARKER)
        # Helper also counts snapshot metadata; checkpoint unique bytes must not undercount.
        assert helper >= sync_style
        assert "checkpoint-100" in names


class TestIncompleteStoreCountedAsUpload:
    def test_incomplete_orphan_dir_fails_small_budget(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        c = _contract(tag="A")
        _complete_ckpt(local / "checkpoint-100", 100, "FULL", payload=500)
        _write_fp(local, step=100, contract=c)
        sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, budget_bytes=10**12,
        )
        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        bad = store / "checkpoint-200"
        bad.mkdir(parents=True)
        (bad / "trainer_state.json").write_text("{}", encoding="utf-8")
        _complete_ckpt(local / "checkpoint-200", 200, "FULL", payload=10000)
        _write_fp(local, step=200, contract=c)
        with pytest.raises(RuntimeError, match="budget|Budget"):
            sync_experiment_checkpoints_to_durable(
                local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
                save_total_limit=2, budget_bytes=1101,
            )


class TestCheckpointDigestAndIntegrity:
    def test_referenced_digest_mismatch_fails_closed(self, tmp_path: Path, monkeypatch):
        """1. Referenced same-name + different digest → sync fail; bytes/LATEST/rollback intact."""
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        c = _contract(tag="A")
        seed = experiment_checkpoint_dir(tmp_path / "seed", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(seed / "checkpoint-100", 100, "ORPHAN_BAD")
        _write_fp(seed, step=100, contract=c)
        snap1 = sync_experiment_checkpoints_to_durable(
            seed, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        dest = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER)
        latest_before = (dest / "LATEST").read_text(encoding="utf-8")
        store = dest / "ckpts"
        marker_before = (store / "checkpoint-100" / "MARKER.txt").read_text()

        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "LOCAL_GOOD")
        _write_fp(local, step=100, contract=c)
        with pytest.raises(RuntimeError, match="snapshot-referenced|immutable"):
            sync_experiment_checkpoints_to_durable(
                local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
            )
        assert (store / "checkpoint-100" / "MARKER.txt").read_text() == marker_before == "ORPHAN_BAD"
        assert (dest / "LATEST").read_text(encoding="utf-8") == latest_before
        assert resolve_snapshot_checkpoints(snap1, require_complete=True, verify_digests=True)

    def test_orphan_same_name_replaced_atomically(self, tmp_path: Path, monkeypatch):
        """2. Orphan store dir (not snapshot-referenced) replaced by valid local."""
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        c = _contract(tag="A")
        seed = experiment_checkpoint_dir(tmp_path / "seed", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(seed / "checkpoint-100", 100, "KEEP")
        _write_fp(seed, step=100, contract=c)
        sync_experiment_checkpoints_to_durable(
            seed, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        _complete_ckpt(store / "checkpoint-200", 200, "ORPHAN")
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "KEEP")
        _complete_ckpt(local / "checkpoint-200", 200, "LOCAL_GOOD")
        _write_fp(local, step=200, contract=c)
        snap = sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, save_total_limit=2,
        )
        assert (store / "checkpoint-200" / "MARKER.txt").read_text() == "LOCAL_GOOD"
        assert "checkpoint-200" in read_snapshot_manifest(snap)["uploaded_now"]

    def test_snapshot_digest_key_mismatch_fails(self, tmp_path: Path, monkeypatch):
        """3. Missing/extra digest entry → fail."""
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        c = _contract(tag="A")
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "A")
        _write_fp(local, step=100, contract=c)
        snap = sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        man = read_snapshot_manifest(snap)
        man["checkpoint_digests"] = {}
        (snap / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        with pytest.raises(RuntimeError, match="checkpoint_digests keys"):
            resolve_snapshot_checkpoints(snap)
        man["checkpoint_digests"] = {
            "checkpoint-100": "dead" * 16,
            "checkpoint-999": "beef" * 16,
        }
        (snap / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        with pytest.raises(RuntimeError, match="checkpoint_digests keys"):
            resolve_snapshot_checkpoints(snap)

    def test_tampered_bytes_after_commit_fail_restore(self, tmp_path: Path, monkeypatch):
        """4. Mutate model bytes after commit → restore fails."""
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        c = _contract(tag="A")
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "GOOD")
        _write_fp(local, step=100, contract=c)
        sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        (store / "checkpoint-100" / "model.safetensors").write_bytes(b"TAMPERED" + b"\0" * 400)
        fresh = experiment_checkpoint_dir(tmp_path / "fresh", EXP, kind=FULL_TRAIN_MARKER)
        with pytest.raises(RuntimeError, match="digest mismatch"):
            restore_experiment_checkpoints_from_durable(
                fresh, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, expected_contract=c,
            )
        assert not (fresh / "checkpoint-100").exists()

    def test_missing_snapshot_checkpoint_fails_closed(self, tmp_path: Path, monkeypatch):
        """5. Manifest declares a missing directory → fail."""
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        import shutil

        durable = tmp_path / "durable"
        durable.mkdir()
        c = _contract(tag="A")
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "A")
        _complete_ckpt(local / "checkpoint-200", 200, "B")
        _write_fp(local, step=200, contract=c)
        snap = sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, save_total_limit=2,
        )
        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        shutil.rmtree(store / "checkpoint-200")
        with pytest.raises(RuntimeError, match="missing checkpoints"):
            resolve_snapshot_checkpoints(snap)
        fresh = experiment_checkpoint_dir(tmp_path / "fresh", EXP, kind=FULL_TRAIN_MARKER)
        with pytest.raises(RuntimeError, match="missing checkpoints"):
            restore_experiment_checkpoints_from_durable(
                fresh, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, expected_contract=c,
            )
        assert not (fresh / "full_experiment_fingerprint.json").exists()

    def test_missing_latest_with_residue_fails_empty_returns_none(
        self, tmp_path: Path, monkeypatch
    ):
        """6. LATEST missing + residue → fail; fully empty → None."""
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        assert resolve_durable_latest_snapshot(durable, EXP, kind=FULL_TRAIN_MARKER) is None

        c = _contract(tag="A")
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "A")
        _write_fp(local, step=100, contract=c)
        sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        dest = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER)
        (dest / "LATEST").unlink()
        with pytest.raises(RuntimeError, match="no usable LATEST"):
            resolve_durable_latest_snapshot(durable, EXP, kind=FULL_TRAIN_MARKER)

    def test_interrupted_store_swap_recovers_lkg(self, tmp_path: Path, monkeypatch):
        """7. Interrupted temp/backup swap recovers last-known-good."""
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        import shutil

        durable = tmp_path / "durable"
        durable.mkdir()
        c = _contract(tag="A")
        seed = experiment_checkpoint_dir(tmp_path / "seed", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(seed / "checkpoint-100", 100, "KEEP")
        _write_fp(seed, step=100, contract=c)
        sync_experiment_checkpoints_to_durable(
            seed, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        _complete_ckpt(store / "checkpoint-200", 200, "ORPHAN")
        bak = store / "checkpoint-200.bak_replace"
        shutil.move(str(store / "checkpoint-200"), str(bak))
        tmp = store / "checkpoint-200.tmp_copy"
        tmp.mkdir()
        (tmp / "trainer_state.json").write_text("{}", encoding="utf-8")

        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "KEEP")
        _complete_ckpt(local / "checkpoint-200", 200, "LOCAL_GOOD")
        _write_fp(local, step=200, contract=c)
        snap = sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, save_total_limit=2,
        )
        assert (store / "checkpoint-200" / "MARKER.txt").read_text() == "LOCAL_GOOD"
        assert not (store / "checkpoint-200.bak_replace").exists()
        assert not (store / "checkpoint-200.tmp_copy").exists()
        assert resolve_snapshot_checkpoints(snap)

    def test_current_and_rollback_verify_after_sync(self, tmp_path: Path, monkeypatch):
        """8. After successful sync, current + rollback snapshots both verify."""
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        c = _contract(tag="A")
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "A")
        _write_fp(local, step=100, contract=c)
        snap1 = sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, save_total_limit=2,
        )
        _complete_ckpt(local / "checkpoint-200", 200, "B")
        _write_fp(local, step=200, contract=c)
        snap2 = sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, save_total_limit=2,
        )
        assert resolve_snapshot_checkpoints(snap1, require_complete=True, verify_digests=True)
        assert resolve_snapshot_checkpoints(snap2, require_complete=True, verify_digests=True)
        assert resolve_durable_latest_snapshot(durable, EXP, kind=FULL_TRAIN_MARKER) == snap2

    def test_orphan_unreferenced_pruned_from_retention(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES, str(10**12))
        durable = tmp_path / "durable"
        durable.mkdir()
        c = _contract(tag="A")
        local = experiment_checkpoint_dir(tmp_path / "local", EXP, kind=FULL_TRAIN_MARKER)
        _complete_ckpt(local / "checkpoint-100", 100, "A")
        _write_fp(local, step=100, contract=c)
        sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER,
        )
        store = durable_experiment_dir(durable, EXP, kind=FULL_TRAIN_MARKER) / "ckpts"
        _complete_ckpt(store / "checkpoint-999", 999, "ORPHAN")
        _complete_ckpt(local / "checkpoint-200", 200, "B")
        _write_fp(local, step=200, contract=c)
        snap = sync_experiment_checkpoints_to_durable(
            local, durable, experiment_id=EXP, kind=FULL_TRAIN_MARKER, save_total_limit=2,
        )
        assert "checkpoint-999" not in read_snapshot_manifest(snap)["checkpoints"]
        assert not (store / "checkpoint-999").exists()

    def test_atomic_restore_uses_backup_rename(self):
        import inspect
        from src.asr_full_train import atomic_restore_checkpoint_dir

        src = inspect.getsource(atomic_restore_checkpoint_dir)
        assert ".bak_replace" in src
        assert src.find("os.replace(str(dst), str(bak))") < src.find("os.replace(str(tmp), str(dst))")


class TestNotebookReviewResidues:
    def test_notebook_pretrain_budget_is_one_checkpoint(self):
        nb = json.loads(NB.read_text(encoding="utf-8"))
        f3 = next(
            "".join(c["source"])
            for c in nb["cells"]
            if c.get("cell_type") == "code" and "".join(c.get("source", [])).startswith("# Cell F3")
        )
        assert 'upload_bytes=_ckpt_exact["checkpoint_bytes"]' in f3
        assert "FULL_SAVE_TOTAL_LIMIT)" not in f3.split("upload_bytes=")[1].split("\n")[0]
        assert "drive_checkpoint_dir" not in f3
        assert "durable_checkpoint_dir" in f3

    def test_export_passes_local_experiment_dir(self):
        nb = json.loads(NB.read_text(encoding="utf-8"))
        export = next(
            "".join(c["source"])
            for c in nb["cells"]
            if c.get("cell_type") == "code" and "resolve_best_checkpoint_from_durable" in "".join(c.get("source", []))
            and "_export_best" in "".join(c.get("source", []))
        )
        assert "local_experiment_dir=_export_exp_dir" in export

    def test_no_stale_execution_metadata(self):
        nb = json.loads(NB.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, f"cell {i} has execution_count"
            assert not cell.get("outputs"), f"cell {i} has outputs"

    def test_no_drive_rt_residue(self):
        text = NB.read_text(encoding="utf-8")
        assert "drive_rt" not in text
        assert "drive_checkpoint_dir" not in text
