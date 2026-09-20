"""Retry-safe durable commit for MT resume_test Phase B (no GPU / no training)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.asr_full_train import (
    CHECKPOINT_STORE_DIRNAME,
    checkpoint_content_digest,
    durable_experiment_dir,
    experiment_checkpoint_dir,
    measure_dir_bytes,
    sync_experiment_checkpoints_to_durable,
    write_checkpoint_fingerprint,
)
from src.mt_contract import LOCKED_MODEL_ID, LOCKED_MODEL_REVISION, STATUS_MT_RESUME_TEST, build_mt_data_contract
from src.mt_export import build_notebook04_review_bundle, export_notebook04_run
from src.mt_full_train import (
    PHASE_B_ATTEMPTS_DIRNAME,
    _load_existing_durable_commit,
    assert_ready_for_mt_full_train,
    commit_mt_resume_test_phase_b_durable,
    enforce_mt_resume_test_durable_budget,
    load_mt_resume_test_success,
    measure_mt_resume_test_durable_bytes,
    plan_mt_resume_test_phase_b_attempt_retention,
    prune_mt_resume_test_phase_b_attempts,
    validate_mt_resume_test_durable_commit_references,
    verify_mt_resume_test_durable_commit,
    write_mt_resume_test_summary,
)
from src.mt_prepare import persist_mt_prepare_artifacts
from src.mt_runtime_paths import RESUME_TEST_MARKER

EXP = "mt_bartpho_syllable_v1"
BUDGET = 10**12
PHASE_A = 50
PHASE_B = 100


def _locked_contract(**kwargs):
    base = dict(
        dataset_id="d",
        dataset_revision="r",
        train_manifest_content_hash="a" * 64,
        validation_manifest_content_hash="b" * 64,
        train_uid_set_hash="c" * 64,
        validation_uid_set_hash="d" * 64,
        model_id=LOCKED_MODEL_ID,
        model_revision=LOCKED_MODEL_REVISION,
        tokenizer_fingerprint="e" * 64,
        max_source_length=128,
        max_target_length=128,
    )
    base.update(kwargs)
    return build_mt_data_contract(**base)


def _complete_ckpt(path: Path, step: int, *, tag: bytes = b"x") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": 0.1}), encoding="utf-8"
    )
    for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_bytes(tag)
    return path


def _proof_payload() -> dict:
    # Matches notebook Phase B summary schema: true_restart lives under checks, not proof.
    return {
        "model_restored": True,
        "optimizer_restored": True,
        "scheduler_restored": True,
        "rng_restored": True,
        "data_position_ok": True,
    }


def _success_summary(*, contract_hash: str, commit: dict) -> dict:
    return {
        "status": STATUS_MT_RESUME_TEST,
        "contract_hash": contract_hash,
        "experiment_id": EXP,
        "failed_checks": [],
        "proof": _proof_payload(),
        "checks": {"true_restart": True},
        "cross_session": {"two_sessions": True},
        "durable_commit_ok": True,
        "attempt_id": commit.get("attempt_id"),
        "phase_b_digest": commit.get("phase_b_digest"),
        "phase_b_durable_relpath": commit.get("phase_b_durable_relpath"),
        "durable_commit_mode": commit.get("mode"),
    }


def _local_exp(tmp_path: Path) -> Path:
    return experiment_checkpoint_dir(tmp_path / "local", EXP, kind=RESUME_TEST_MARKER)


def _seed_phase_a_and_b(local: Path, *, a_tag: bytes, b_tag: bytes, global_step: int) -> None:
    write_checkpoint_fingerprint(
        local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=global_step
    )
    _complete_ckpt(local / f"checkpoint-{PHASE_A}", PHASE_A, tag=a_tag)
    _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b_tag)


class TestMtResumeTestPhaseBDurable:
    def test_first_successful_phase_b_sync_mode(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_A)
        _complete_ckpt(local / f"checkpoint-{PHASE_A}", PHASE_A, tag=b"phase-a")
        sync_experiment_checkpoints_to_durable(
            local, state, experiment_id=EXP, kind=RESUME_TEST_MARKER, budget_bytes=BUDGET
        )
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"phase-b-v1")

        commit = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert commit["durable_commit_ok"] is True
        assert commit["mode"] == "sync"
        assert commit["attempt_id"] is None
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        store_b = dest / CHECKPOINT_STORE_DIRNAME / f"checkpoint-{PHASE_B}"
        assert store_b.is_dir()
        assert checkpoint_content_digest(store_b)["digest"] == commit["phase_b_digest"]
        verify_mt_resume_test_durable_commit(state, experiment_id=EXP)

    def test_rerun_reuses_matching_durable_checkpoint(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"same-bytes", global_step=PHASE_B)
        first = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert first["mode"] == "sync"
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        store_b = dest / CHECKPOINT_STORE_DIRNAME / f"checkpoint-{PHASE_B}"
        before = checkpoint_content_digest(store_b)["digest"]

        second = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert second["mode"] == "reuse"
        assert second["phase_b_digest"] == before
        assert checkpoint_content_digest(store_b)["digest"] == before

    def test_immutable_referenced_checkpoint_not_overwritten_on_digest_collision(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"old-phase-b", global_step=PHASE_B)
        first = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert first["mode"] == "sync"
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        store_b = dest / CHECKPOINT_STORE_DIRNAME / f"checkpoint-{PHASE_B}"
        old_digest = checkpoint_content_digest(store_b)["digest"]

        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"new-phase-b-bytes")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        new_digest = checkpoint_content_digest(local / f"checkpoint-{PHASE_B}")["digest"]
        assert new_digest != old_digest

        with pytest.raises(RuntimeError, match="Refusing to mutate snapshot-referenced"):
            sync_experiment_checkpoints_to_durable(
                local, state, experiment_id=EXP, kind=RESUME_TEST_MARKER, budget_bytes=BUDGET
            )
        assert checkpoint_content_digest(store_b)["digest"] == old_digest

        commit = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert commit["mode"] == "attempt"
        assert commit["attempt_id"] == new_digest[:12]
        assert checkpoint_content_digest(store_b)["digest"] == old_digest
        attempt_path = dest / PHASE_B_ATTEMPTS_DIRNAME / commit["attempt_id"]
        assert attempt_path.is_dir()
        assert checkpoint_content_digest(attempt_path)["digest"] == new_digest
        assert (local / f"checkpoint-{PHASE_B}").is_dir()
        verify_mt_resume_test_durable_commit(state, experiment_id=EXP)

    def test_attempt_bytes_counted_in_durable_budget(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"old-b", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        before = measure_mt_resume_test_durable_bytes(dest)
        assert before["attempt_bytes"] == 0

        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"new-b-large" + b"Z" * 4096)
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        commit = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert commit["mode"] == "attempt"
        after = measure_mt_resume_test_durable_bytes(dest)
        assert after["attempt_bytes"] > 0
        assert after["total_bytes"] == after["store_bytes"] + after["attempt_bytes"]
        assert commit["budget"]["attempt_bytes"] == after["attempt_bytes"]

    def test_attempt_over_budget_fails_before_copy(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"old-b", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        store_bytes = measure_mt_resume_test_durable_bytes(dest)["store_bytes"]

        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"new-b" + b"Q" * 2048)
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        new_bytes = measure_dir_bytes(local / f"checkpoint-{PHASE_B}")
        tiny_budget = store_bytes + new_bytes - 1
        assert tiny_budget > store_bytes

        with pytest.raises(RuntimeError, match="budget exceeded|Durable checkpoint budget"):
            commit_mt_resume_test_phase_b_durable(
                local,
                state,
                experiment_id=EXP,
                phase_a_steps=PHASE_A,
                phase_b_steps=PHASE_B,
                budget_bytes=tiny_budget,
            )
        attempts = dest / PHASE_B_ATTEMPTS_DIRNAME
        if attempts.is_dir():
            finalized = [p for p in attempts.iterdir() if p.is_dir() and not p.name.startswith(".")]
            assert finalized == []
        # Prior successful sync commit may remain; must not flip to attempt/SUCCESS path.
        prior = json.loads((state / "mt_resume_test_durable_commit.json").read_text(encoding="utf-8"))
        assert prior.get("mode") == "sync"
        assert prior.get("attempt_id") is None

    def test_stale_orphan_attempt_pruned_current_kept(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"canon-b", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        attempts = dest / PHASE_B_ATTEMPTS_DIRNAME
        orphan = attempts / ("a" * 12)
        _complete_ckpt(orphan, PHASE_B, tag=b"orphan-bytes")
        assert orphan.is_dir()

        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"attempt-1")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        commit1 = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert commit1["mode"] == "attempt"
        assert not orphan.is_dir()
        cur1 = attempts / commit1["attempt_id"]
        assert cur1.is_dir()

        # Second attempt: keep current + previous rollback; prune older orphans.
        older = attempts / ("b" * 12)
        _complete_ckpt(older, PHASE_B, tag=b"older-orphan")
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"attempt-2")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        commit2 = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert commit2["mode"] == "attempt"
        assert (attempts / commit2["attempt_id"]).is_dir()
        assert cur1.is_dir()  # previous rollback retained
        assert not older.is_dir()
        retained = set(commit2["attempt_prune"]["retained"])
        assert commit2["attempt_id"] in retained
        assert commit1["attempt_id"] in retained

    def test_referenced_attempt_not_pruned_by_helper(self, tmp_path: Path):
        dest = tmp_path / "durable"
        attempts = dest / PHASE_B_ATTEMPTS_DIRNAME
        keep_id = "c" * 12
        drop_id = "d" * 12
        _complete_ckpt(attempts / keep_id, PHASE_B, tag=b"keep")
        _complete_ckpt(attempts / drop_id, PHASE_B, tag=b"drop")
        plan = plan_mt_resume_test_phase_b_attempt_retention(
            current_attempt_id=keep_id,
            previous_commit={"attempt_id": keep_id, "phase_b_durable_relpath": f"{PHASE_B_ATTEMPTS_DIRNAME}/{keep_id}"},
        )
        report = prune_mt_resume_test_phase_b_attempts(dest, keep_attempt_ids=plan["keep_attempt_ids"])
        assert keep_id in report["retained"]
        assert drop_id in report["deleted"]
        assert (attempts / keep_id).is_dir()
        assert not (attempts / drop_id).is_dir()

    def test_sync_failure_does_not_leave_success_summary(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"phase-b", global_step=PHASE_B)

        with patch(
            "src.mt_full_train.sync_experiment_checkpoints_to_durable",
            side_effect=RuntimeError("simulated durable sync failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated durable sync failure"):
                commit_mt_resume_test_phase_b_durable(
                    local,
                    state,
                    experiment_id=EXP,
                    phase_a_steps=PHASE_A,
                    phase_b_steps=PHASE_B,
                    budget_bytes=BUDGET,
                )
        assert not (state / "mt_resume_test_durable_commit.json").is_file()
        write_mt_resume_test_summary(
            state,
            {
                "status": "FAILED",
                "contract_hash": "x",
                "experiment_id": EXP,
                "durable_commit_ok": False,
                "proof": _proof_payload(),
                "cross_session": {"two_sessions": True},
            },
        )
        with pytest.raises(RuntimeError, match="not successful|durable_commit_ok|Missing"):
            load_mt_resume_test_success(state, experiment_id=EXP)

    def test_stale_success_summary_rejected_without_durable_commit(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        contract = _locked_contract()
        persist_mt_prepare_artifacts(
            state,
            train_eligible=pd.DataFrame(
                [{"record_uid": "t1", "text_bahnar": "a", "text_vi": "b"}]
            ),
            val_eligible=pd.DataFrame(
                [{"record_uid": "v1", "text_bahnar": "c", "text_vi": "d"}]
            ),
            exclusions=pd.DataFrame(columns=["record_uid", "split", "reason"]),
            summary={"ok": True},
            contract=contract,
        )
        write_mt_resume_test_summary(
            state,
            {
                "status": STATUS_MT_RESUME_TEST,
                "contract_hash": contract["contract_hash"],
                "experiment_id": EXP,
                "failed_checks": [],
                "proof": _proof_payload(),
                "cross_session": {"two_sessions": True},
            },
        )
        with pytest.raises(RuntimeError, match="durable_commit_ok"):
            load_mt_resume_test_success(
                state,
                expected_contract_hash=contract["contract_hash"],
                experiment_id=EXP,
            )
        with pytest.raises(RuntimeError, match="durable_commit_ok"):
            assert_ready_for_mt_full_train(state, contract=contract)

    def test_successful_retry_creates_valid_gate_state(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        contract = _locked_contract()
        persist_mt_prepare_artifacts(
            state,
            train_eligible=pd.DataFrame(
                [{"record_uid": "t1", "text_bahnar": "a", "text_vi": "b"}]
            ),
            val_eligible=pd.DataFrame(
                [{"record_uid": "v1", "text_bahnar": "c", "text_vi": "d"}]
            ),
            exclusions=pd.DataFrame(columns=["record_uid", "split", "reason"]),
            summary={"ok": True},
            contract=contract,
        )
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"old-b", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        write_mt_resume_test_summary(
            state,
            {
                "status": STATUS_MT_RESUME_TEST,
                "contract_hash": contract["contract_hash"],
                "experiment_id": EXP,
                "failed_checks": [],
                "proof": _proof_payload(),
                "cross_session": {"two_sessions": True},
            },
        )
        with pytest.raises(RuntimeError, match="durable_commit_ok"):
            assert_ready_for_mt_full_train(state, contract=contract)

        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"retry-b")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        commit = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert commit["mode"] == "attempt"
        write_mt_resume_test_summary(
            state, _success_summary(contract_hash=contract["contract_hash"], commit=commit)
        )
        loaded = load_mt_resume_test_success(
            state,
            expected_contract_hash=contract["contract_hash"],
            experiment_id=EXP,
        )
        assert loaded["durable_commit_ok"] is True
        assert_ready_for_mt_full_train(state, contract=contract)

    def test_summary_atomic_write_uses_replace(self, tmp_path: Path, monkeypatch):
        state = tmp_path / "state"
        state.mkdir()
        calls = []
        real_replace = os.replace

        def tracking_replace(src, dst):
            calls.append((str(src), str(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", tracking_replace)
        path = write_mt_resume_test_summary(state, {"status": "FAILED", "durable_commit_ok": False})
        assert path.is_file()
        assert any(str(path) == dst and dst.endswith("mt_resume_test_summary.json") for _, dst in calls)
        assert any(src.endswith(".tmp") for src, _ in calls)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "FAILED"

    def test_export_includes_resume_durable_artifacts(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        for name in (
            "mt_resume_test_summary.json",
            "mt_resume_test_phase_a.json",
            "mt_resume_test_contract.json",
            "mt_resume_test_durable_commit.json",
        ):
            (state / name).write_text("{}", encoding="utf-8")
        dest = export_notebook04_run(
            export_root=tmp_path / "exports",
            run_id="r_resume",
            full_state_dir=state,
        )
        for name in (
            "mt_resume_test_summary.json",
            "mt_resume_test_phase_a.json",
            "mt_resume_test_contract.json",
            "mt_resume_test_durable_commit.json",
        ):
            assert (dest / "state" / name).is_file()

    def test_enforce_budget_includes_attempts(self, tmp_path: Path):
        dest = tmp_path / "durable"
        store = dest / CHECKPOINT_STORE_DIRNAME
        _complete_ckpt(store / f"checkpoint-{PHASE_A}", PHASE_A, tag=b"a")
        _complete_ckpt(dest / PHASE_B_ATTEMPTS_DIRNAME / ("e" * 12), PHASE_B, tag=b"attempt")
        measured = measure_mt_resume_test_durable_bytes(dest)
        assert measured["attempt_bytes"] > 0
        with pytest.raises(RuntimeError, match="budget exceeded"):
            enforce_mt_resume_test_durable_budget(
                dest,
                budget_bytes=measured["store_bytes"],
                extra_upload_bytes=1,
            )

    def test_missing_durable_commit_file_is_none(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        dest = tmp_path / "durable"
        dest.mkdir()
        assert _load_existing_durable_commit(state, experiment_id=EXP, dest_root=dest) is None

    def test_corrupt_durable_commit_fail_closed_no_prune(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"canon-b", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        attempts = dest / PHASE_B_ATTEMPTS_DIRNAME
        referenced = attempts / ("f" * 12)
        _complete_ckpt(referenced, PHASE_B, tag=b"must-keep")
        (state / "mt_resume_test_durable_commit.json").write_text(
            '{"durable_commit_ok": true, "attempt_id": "' + ("f" * 12) + '",',
            encoding="utf-8",
        )
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"next-attempt")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        with pytest.raises(RuntimeError, match="Corrupt|unreadable|fail-closed"):
            commit_mt_resume_test_phase_b_durable(
                local,
                state,
                experiment_id=EXP,
                phase_a_steps=PHASE_A,
                phase_b_steps=PHASE_B,
                budget_bytes=BUDGET,
            )
        assert referenced.is_dir()
        with pytest.raises(RuntimeError, match="Corrupt|unreadable|fail-closed"):
            _load_existing_durable_commit(state, experiment_id=EXP, dest_root=dest)

    def test_invalid_durable_commit_object_fail_closed(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        dest = tmp_path / "durable"
        dest.mkdir()
        (state / "mt_resume_test_durable_commit.json").write_text("[]", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Invalid|fail-closed"):
            _load_existing_durable_commit(state, experiment_id=EXP, dest_root=dest)

    def _write_attempt_meta(self, state: Path, dest: Path, *, digest: str, attempt_id: str, **overrides):
        payload = {
            "durable_commit_ok": True,
            "mode": "attempt",
            "experiment_id": EXP,
            "kind": RESUME_TEST_MARKER,
            "phase_b_checkpoint": f"checkpoint-{PHASE_B}",
            "phase_b_digest": digest,
            "attempt_id": attempt_id,
            "phase_b_durable_relpath": f"{PHASE_B_ATTEMPTS_DIRNAME}/{attempt_id}",
        }
        payload.update(overrides)
        (state / "mt_resume_test_durable_commit.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def test_valid_json_wrong_attempt_id_fail_closed(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"a", b_tag=b"b1", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"b2")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        real = commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        keep = dest / PHASE_B_ATTEMPTS_DIRNAME / real["attempt_id"]
        assert keep.is_dir()
        wrong_id = "a" * 12
        assert wrong_id != real["attempt_id"]
        self._write_attempt_meta(
            state, dest, digest=real["phase_b_digest"], attempt_id=wrong_id
        )
        with pytest.raises(RuntimeError, match="attempt_id|prefix|fail-closed|Invalid"):
            commit_mt_resume_test_phase_b_durable(
                local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
            )
        assert keep.is_dir()

    def test_relpath_mismatch_attempt_id_fail_closed(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"a", b_tag=b"old", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"att")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        real = commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        keep = dest / PHASE_B_ATTEMPTS_DIRNAME / real["attempt_id"]
        self._write_attempt_meta(
            state,
            dest,
            digest=real["phase_b_digest"],
            attempt_id=real["attempt_id"],
            phase_b_durable_relpath=f"{PHASE_B_ATTEMPTS_DIRNAME}/{'c' * 12}",
        )
        with pytest.raises(RuntimeError, match="phase_b_durable_relpath|fail-closed|Invalid"):
            _load_existing_durable_commit(state, experiment_id=EXP, dest_root=dest)
        assert keep.is_dir()

    def test_digest_metadata_mismatch_on_disk_fail_closed(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"a", b_tag=b"old", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"att")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        real = commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        keep = dest / PHASE_B_ATTEMPTS_DIRNAME / real["attempt_id"]
        # Same attempt_id prefix, different full digest → on-disk mismatch after path resolve.
        fake_digest = real["attempt_id"] + ("0" * (64 - len(real["attempt_id"])))
        assert fake_digest != real["phase_b_digest"]
        self._write_attempt_meta(
            state,
            dest,
            digest=fake_digest,
            attempt_id=real["attempt_id"],
        )
        with pytest.raises(RuntimeError, match="digest mismatch|fail-closed|Invalid"):
            _load_existing_durable_commit(state, experiment_id=EXP, dest_root=dest)
        assert keep.is_dir()

    def test_referenced_attempt_missing_fail_closed(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        dest = tmp_path / "durable"
        (dest / CHECKPOINT_STORE_DIRNAME).mkdir(parents=True)
        digest = "cd" * 32
        aid = digest[:12]
        self._write_attempt_meta(state, dest, digest=digest, attempt_id=aid)
        with pytest.raises(RuntimeError, match="missing|fail-closed|Invalid"):
            validate_mt_resume_test_durable_commit_references(
                json.loads((state / "mt_resume_test_durable_commit.json").read_text()),
                experiment_id=EXP,
                dest_root=dest,
            )

    def test_referenced_attempt_incomplete_fail_closed(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        dest = tmp_path / "durable"
        digest = "ef" * 32
        aid = digest[:12]
        attempt = dest / PHASE_B_ATTEMPTS_DIRNAME / aid
        attempt.mkdir(parents=True)
        (attempt / "trainer_state.json").write_text("{}", encoding="utf-8")
        self._write_attempt_meta(state, dest, digest=digest, attempt_id=aid)
        with pytest.raises(RuntimeError, match="incomplete|fail-closed|Invalid"):
            _load_existing_durable_commit(state, experiment_id=EXP, dest_root=dest)

    def test_valid_attempt_metadata_allows_orphan_prune(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"a", b_tag=b"old", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"att1")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        real = commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        orphan = dest / PHASE_B_ATTEMPTS_DIRNAME / ("a" * 12)
        _complete_ckpt(orphan, PHASE_B, tag=b"orphan")
        loaded = _load_existing_durable_commit(state, experiment_id=EXP, dest_root=dest)
        assert loaded is not None
        plan = plan_mt_resume_test_phase_b_attempt_retention(
            current_attempt_id="b" * 12, previous_commit=loaded
        )
        report = prune_mt_resume_test_phase_b_attempts(dest, keep_attempt_ids=plan["keep_attempt_ids"])
        assert real["attempt_id"] in report["retained"] or real["attempt_id"] in plan["keep_attempt_ids"]
        assert not orphan.is_dir()
        assert (dest / PHASE_B_ATTEMPTS_DIRNAME / real["attempt_id"]).is_dir()

    def test_valid_sync_metadata_validates(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"a", b_tag=b"sync-b", global_step=PHASE_B)
        commit = commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        assert commit["mode"] == "sync"
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        loaded = _load_existing_durable_commit(state, experiment_id=EXP, dest_root=dest)
        assert loaded is not None
        assert loaded["mode"] == "sync"
        validate_mt_resume_test_durable_commit_references(
            loaded, experiment_id=EXP, dest_root=dest
        )

    def test_invalid_previous_metadata_never_deletes_attempt(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"a", b_tag=b"old", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"keep-me")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        real = commit_mt_resume_test_phase_b_durable(
            local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
        )
        keep = dest / PHASE_B_ATTEMPTS_DIRNAME / real["attempt_id"]
        # Valid JSON, wrong experiment_id
        self._write_attempt_meta(
            state,
            dest,
            digest=real["phase_b_digest"],
            attempt_id=real["attempt_id"],
            experiment_id="wrong_exp",
        )
        with pytest.raises(RuntimeError, match="experiment_id|fail-closed|Invalid"):
            commit_mt_resume_test_phase_b_durable(
                local, state, experiment_id=EXP, phase_a_steps=PHASE_A, phase_b_steps=PHASE_B, budget_bytes=BUDGET
            )
        assert keep.is_dir()

    def test_existing_same_digest_attempt_reuses_without_copytree(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"old-b", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"same-attempt-bytes")
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        first = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        assert first["mode"] == "attempt"
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        attempt_path = dest / PHASE_B_ATTEMPTS_DIRNAME / first["attempt_id"]
        assert attempt_path.is_dir()

        with patch("src.mt_full_train.shutil.copytree") as mock_copy:
            second = commit_mt_resume_test_phase_b_durable(
                local,
                state,
                experiment_id=EXP,
                phase_a_steps=PHASE_A,
                phase_b_steps=PHASE_B,
                budget_bytes=BUDGET,
            )
            mock_copy.assert_not_called()
        assert second["mode"] == "attempt"
        assert second["attempt_id"] == first["attempt_id"]
        assert second["phase_b_digest"] == first["phase_b_digest"]
        assert attempt_path.is_dir()

    def test_existing_same_digest_attempt_no_transient_budget_spike(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"old-b", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"reuse-me" + b"Y" * 1024)
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        first = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        total = measure_mt_resume_test_durable_bytes(dest)["total_bytes"]
        # Budget equals current total: a transient duplicate copy would exceed it.
        second = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=total,
        )
        assert second["mode"] == "attempt"
        assert second["attempt_id"] == first["attempt_id"]
        after = measure_mt_resume_test_durable_bytes(dest)["total_bytes"]
        assert after <= total
        assert second["budget"]["total_bytes"] <= total

    def test_new_attempt_still_enforces_budget(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"old-b", global_step=PHASE_B)
        commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        dest = durable_experiment_dir(state, EXP, kind=RESUME_TEST_MARKER)
        store_bytes = measure_mt_resume_test_durable_bytes(dest)["store_bytes"]
        shutil.rmtree(local / f"checkpoint-{PHASE_B}")
        _complete_ckpt(local / f"checkpoint-{PHASE_B}", PHASE_B, tag=b"brand-new" + b"N" * 2048)
        write_checkpoint_fingerprint(local, experiment_id=EXP, kind=RESUME_TEST_MARKER, global_step=PHASE_B)
        need = measure_dir_bytes(local / f"checkpoint-{PHASE_B}")
        with pytest.raises(RuntimeError, match="budget exceeded|Durable checkpoint budget"):
            commit_mt_resume_test_phase_b_durable(
                local,
                state,
                experiment_id=EXP,
                phase_a_steps=PHASE_A,
                phase_b_steps=PHASE_B,
                budget_bytes=store_bytes + need - 1,
            )


class TestNotebook04ReviewBundleChecksums:
    def test_bundle_checksum_metadata_not_stale(self, tmp_path: Path):
        import zipfile

        # Minimal fake project tree for the packager.
        root = tmp_path / "proj"
        (root / "src").mkdir(parents=True)
        (root / "tests").mkdir(parents=True)
        (root / "notebooks").mkdir(parents=True)
        (root / "configs").mkdir(parents=True)
        (root / "src" / "hello.py").write_text("x = 1\n", encoding="utf-8")
        (root / "tests" / "test_hello.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        (root / "notebooks" / "04_mt_baseline_training.ipynb").write_text(
            json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}),
            encoding="utf-8",
        )
        (root / "configs" / "mt.yaml").write_text("model: x\n", encoding="utf-8")
        (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")

        out = tmp_path / "out"
        info = build_notebook04_review_bundle(root, out_dir=out, stamp="testrun")
        zip_path = Path(info["zip_path"])
        assert zip_path.is_file()
        external = (out / "SHA256SUMS.txt").read_text(encoding="utf-8")
        latest = (out / "LATEST_BUNDLE.txt").read_text(encoding="utf-8")
        assert info["zip_sha256"] in external
        assert f"ZIP_SHA256={info['zip_sha256']}" in latest
        assert "PAYLOAD_SHA256SUMS_SHA256=" in latest

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "artifacts/notebook04/PAYLOAD_SHA256SUMS.txt" in names
            assert "artifacts/notebook04/BUNDLE_META.json" in names
            # Final zip sha must not be self-embedded as SHA256SUMS/LATEST inside.
            assert "artifacts/notebook04/SHA256SUMS.txt" not in names
            assert "artifacts/notebook04/LATEST_BUNDLE.txt" not in names
            meta = json.loads(zf.read("artifacts/notebook04/BUNDLE_META.json"))
            assert meta["checksum_scheme"] == "payload_inside_zip__zip_sha_external_only"
            assert meta["payload_sha256sums_sha256"] == info["payload_sha256sums_sha256"]
            assert info["zip_sha256"] not in zf.read("artifacts/notebook04/PAYLOAD_SHA256SUMS.txt").decode()


class TestLoadMtResumeTestSuccessSchema:
    """Gate must match notebook summary schema: true_restart in checks, not proof."""

    def _ready_state(self, tmp_path: Path):
        state = tmp_path / "state"
        state.mkdir()
        contract = _locked_contract()
        persist_mt_prepare_artifacts(
            state,
            train_eligible=pd.DataFrame(
                [{"record_uid": "t1", "text_bahnar": "a", "text_vi": "b"}]
            ),
            val_eligible=pd.DataFrame(
                [{"record_uid": "v1", "text_bahnar": "c", "text_vi": "d"}]
            ),
            exclusions=pd.DataFrame(columns=["record_uid", "split", "reason"]),
            summary={"ok": True},
            contract=contract,
        )
        local = _local_exp(tmp_path)
        _seed_phase_a_and_b(local, a_tag=b"phase-a", b_tag=b"phase-b", global_step=PHASE_B)
        commit = commit_mt_resume_test_phase_b_durable(
            local,
            state,
            experiment_id=EXP,
            phase_a_steps=PHASE_A,
            phase_b_steps=PHASE_B,
            budget_bytes=BUDGET,
        )
        return state, contract, commit

    def test_real_notebook_schema_passes(self, tmp_path: Path):
        state, contract, commit = self._ready_state(tmp_path)
        summary = _success_summary(contract_hash=contract["contract_hash"], commit=commit)
        assert "true_restart" not in summary["proof"]
        assert summary["checks"]["true_restart"] is True
        write_mt_resume_test_summary(state, summary)
        loaded = load_mt_resume_test_success(
            state,
            expected_contract_hash=contract["contract_hash"],
            experiment_id=EXP,
        )
        assert loaded["checks"]["true_restart"] is True
        assert_ready_for_mt_full_train(state, contract=contract)

    def test_checks_true_restart_false_fails(self, tmp_path: Path):
        state, contract, commit = self._ready_state(tmp_path)
        summary = _success_summary(contract_hash=contract["contract_hash"], commit=commit)
        summary["checks"] = {"true_restart": False}
        write_mt_resume_test_summary(state, summary)
        with pytest.raises(RuntimeError, match="checks.true_restart"):
            load_mt_resume_test_success(
                state,
                expected_contract_hash=contract["contract_hash"],
                experiment_id=EXP,
            )

    def test_checks_true_restart_missing_fails(self, tmp_path: Path):
        state, contract, commit = self._ready_state(tmp_path)
        summary = _success_summary(contract_hash=contract["contract_hash"], commit=commit)
        summary.pop("checks", None)
        summary["proof"] = {**summary["proof"], "true_restart": True}
        write_mt_resume_test_summary(state, summary)
        with pytest.raises(RuntimeError, match="checks.true_restart"):
            load_mt_resume_test_success(
                state,
                expected_contract_hash=contract["contract_hash"],
                experiment_id=EXP,
            )

    def test_cross_session_two_sessions_false_fails(self, tmp_path: Path):
        state, contract, commit = self._ready_state(tmp_path)
        summary = _success_summary(contract_hash=contract["contract_hash"], commit=commit)
        summary["cross_session"] = {"two_sessions": False}
        write_mt_resume_test_summary(state, summary)
        with pytest.raises(RuntimeError, match="two_sessions"):
            load_mt_resume_test_success(
                state,
                expected_contract_hash=contract["contract_hash"],
                experiment_id=EXP,
            )

    def test_cross_session_missing_fails(self, tmp_path: Path):
        state, contract, commit = self._ready_state(tmp_path)
        summary = _success_summary(contract_hash=contract["contract_hash"], commit=commit)
        summary.pop("cross_session", None)
        write_mt_resume_test_summary(state, summary)
        with pytest.raises(RuntimeError, match="two_sessions"):
            load_mt_resume_test_success(
                state,
                expected_contract_hash=contract["contract_hash"],
                experiment_id=EXP,
            )

    def test_proof_flag_false_fails(self, tmp_path: Path):
        state, contract, commit = self._ready_state(tmp_path)
        summary = _success_summary(contract_hash=contract["contract_hash"], commit=commit)
        summary["proof"]["optimizer_restored"] = False
        write_mt_resume_test_summary(state, summary)
        with pytest.raises(RuntimeError, match="proof missing/false: optimizer_restored"):
            load_mt_resume_test_success(
                state,
                expected_contract_hash=contract["contract_hash"],
                experiment_id=EXP,
            )

    def test_does_not_require_proof_true_restart(self, tmp_path: Path):
        state, contract, commit = self._ready_state(tmp_path)
        summary = _success_summary(contract_hash=contract["contract_hash"], commit=commit)
        assert "true_restart" not in summary["proof"]
        write_mt_resume_test_summary(state, summary)
        load_mt_resume_test_success(
            state,
            expected_contract_hash=contract["contract_hash"],
            experiment_id=EXP,
        )
