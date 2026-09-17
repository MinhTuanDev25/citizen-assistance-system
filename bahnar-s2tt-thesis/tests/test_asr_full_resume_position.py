"""
Data-position proof for resume-test: the UID actually consumed after resume.

A matching ``global_step`` is not evidence that the dataloader resumed at the
right sample, and neither is ``Dataset.__getitem__``, because resume also fetches
and collates the batches it skips. Only reaching ``training_step`` counts.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.asr_full_train import (
    derive_resume_test_status_from_proof,
    evaluate_data_position,
    make_uid_tracking_collator,
    make_uid_tracking_trainer_cls,
    new_session_token,
    plan_expected_resume_position,
    summarize_resume_proof,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPlanExpectedPosition:
    def test_offset_follows_batch_size_and_accumulation(self):
        uids = [f"u{i}" for i in range(64)]
        plan = plan_expected_resume_position(
            uids, resume_step=4, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        )
        assert plan["expected_batch_offset"] == 32
        assert plan["expected_sample_index"] == 32
        assert plan["expected_first_uid"] == "u32"
        assert plan["wrapped_epoch"] is False

    def test_larger_batches_advance_further(self):
        uids = [f"u{i}" for i in range(64)]
        plan = plan_expected_resume_position(
            uids, resume_step=2, per_device_train_batch_size=4, gradient_accumulation_steps=2,
        )
        assert plan["expected_sample_index"] == 16 and plan["expected_first_uid"] == "u16"

    def test_position_past_the_epoch_is_flagged(self):
        plan = plan_expected_resume_position(
            ["a", "b"], resume_step=100, per_device_train_batch_size=1,
        )
        assert plan["wrapped_epoch"] is True and plan["expected_first_uid"] is None


class TestTrackingCollator:
    def test_collator_records_uids_and_hides_them_from_the_model(self):
        sink = {}
        seen = []
        collate = make_uid_tracking_collator(lambda feats: seen.append(feats) or "batch", sink)
        assert collate([{"record_uid": "u1", "input_values": 1}]) == "batch"
        collate([{"record_uid": "u2", "input_values": 2}])
        assert sink["collated_batches"] == [["u1"], ["u2"]]
        assert sink["last_collated_uids"] == ["u2"]
        # record_uid is metadata, not a model input.
        assert all("record_uid" not in f for batch in seen for f in batch)

    def test_only_training_step_claims_the_first_consumed_batch(self):
        """Skipped batches reach the collator, so the collator alone can't decide."""
        sink = {}
        collate = make_uid_tracking_collator(lambda feats: feats, sink)

        class Base:
            def training_step(self, model, inputs, *a, **k):
                return "loss"

        tracked = make_uid_tracking_trainer_cls(Base, sink)()
        for uid in ("u0", "u8", "u16", "u24", "u32"):  # resume skips the first four
            collate([{"record_uid": uid}])
        tracked.training_step(None, {})
        collate([{"record_uid": "u40"}])
        tracked.training_step(None, {})
        assert sink["first_consumed_uids"] == ["u32"]
        assert sink["first_consumed_batch_index"] == 4


class TestEvaluateDataPosition:
    def test_matching_uid_and_offset_passes(self):
        sink = {"first_consumed_uids": ["u32"], "first_consumed_batch_index": 4}
        out = evaluate_data_position(sink, expected_uid="u32", expected_offset=4)
        assert out["data_position_ok"] is True and out["offset_matches"] is True

    def test_wrong_uid_fails(self):
        sink = {"first_consumed_uids": ["u0"], "first_consumed_batch_index": 0}
        out = evaluate_data_position(sink, expected_uid="u32", expected_offset=4)
        assert out["data_position_ok"] is False
        assert "expected 'u32'" in out["data_position_detail"]

    def test_restart_from_scratch_is_detected(self):
        """ignore_data_skip=True style restart: right UID count, wrong offset."""
        sink = {"first_consumed_uids": ["u32"], "first_consumed_batch_index": 0}
        out = evaluate_data_position(sink, expected_uid="u32", expected_offset=4)
        assert out["data_position_ok"] is False
        assert "expected 4" in out["data_position_detail"]

    def test_no_uid_reaching_training_step_is_a_failure_not_a_pass(self):
        out = evaluate_data_position({}, expected_uid="u32")
        assert out["data_position_ok"] is False
        assert out["data_position_detail"] == "no_uid_reached_training_step"

    def test_missing_phase_a_plan_cannot_pass(self):
        sink = {"first_consumed_uids": ["u32"]}
        assert evaluate_data_position(sink, expected_uid=None)["data_position_ok"] is False


class TestStatusGateUsesRealPosition:
    def _proof(self, sink_extra, **position):
        restore = {
            "model_restored": True, "optimizer_restored": True, "scheduler_restored": True,
            "step_matches_checkpoint": True, "global_step": 100,
        }
        sink = {"restore": restore, "rng": {"rng_restored": True}, **sink_extra}
        proof = summarize_resume_proof(sink, **position)
        proof["final_global_step"] = 200
        return proof

    def _verdict(self, proof):
        return derive_resume_test_status_from_proof(
            proof, phase_a_reached_target=True,
            cross_session={"two_sessions": True},
            used_separate_experiment_dir=True, expected_phase_b_steps=200,
        )

    def test_correct_position_reaches_success(self):
        proof = self._proof(
            {"first_consumed_uids": ["u32"], "first_consumed_batch_index": 4},
            expected_first_uid="u32", expected_offset=4,
        )
        assert self._verdict(proof)["status"] == "SUCCESS_FULL_RESUME_TEST"

    def test_wrong_position_fails_even_with_perfect_step(self):
        """The whole point: right global_step, wrong data position, still FAILED."""
        proof = self._proof(
            {"first_consumed_uids": ["u0"], "first_consumed_batch_index": 0},
            expected_first_uid="u32", expected_offset=4,
        )
        verdict = self._verdict(proof)
        assert verdict["status"] == "FAILED"
        assert "data_position_ok" in verdict["failed_checks"]

    def test_absent_position_evidence_fails(self):
        proof = self._proof({}, expected_first_uid="u32", expected_offset=4)
        assert "data_position_ok" in self._verdict(proof)["failed_checks"]


_PHASE_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    from src.asr_full_train import new_session_token

    phase, state = sys.argv[1], Path(sys.argv[2])
    token = new_session_token()
    if phase == "a":
        state.write_text(json.dumps({{"session": token, "global_step": 100}}))
    else:
        payload = json.loads(state.read_text())
        from src.asr_full_train import assert_cross_session_resume
        proof = assert_cross_session_resume(payload, current_session=token)
        state.with_suffix(".b.json").write_text(json.dumps(proof))
    print(json.dumps(token))
    """
)


class TestTwoProcessResume:
    """Cross-session proof must come from real distinct processes, not a flag."""

    def _run(self, tmp_path: Path, phase: str, state: Path) -> dict:
        script = tmp_path / "phase.py"
        script.write_text(_PHASE_SCRIPT.format(repo=str(REPO_ROOT)), encoding="utf-8")
        out = subprocess.run(
            [sys.executable, str(script), phase, str(state)],
            capture_output=True, text=True, check=True, cwd=str(REPO_ROOT),
        )
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_two_subprocesses_are_accepted_as_distinct_sessions(self, tmp_path: Path):
        state = tmp_path / "phase_a.json"
        token_a = self._run(tmp_path, "a", state)
        token_b = self._run(tmp_path, "b", state)
        assert token_a["pid"] != token_b["pid"]
        proof = json.loads((tmp_path / "phase_a.b.json").read_text())
        assert proof["two_sessions"] is True
        assert proof["phase_a_pid"] != proof["phase_b_pid"]
        assert proof["phase_a_nonce"] != proof["phase_b_nonce"]

    def test_same_process_is_rejected(self, tmp_path: Path):
        from src.asr_full_train import assert_cross_session_resume

        token = new_session_token()
        with pytest.raises(RuntimeError):
            assert_cross_session_resume({"session": token}, current_session=token)
