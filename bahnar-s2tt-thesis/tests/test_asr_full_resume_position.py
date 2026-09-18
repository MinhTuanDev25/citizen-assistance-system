"""
Data-position proof for resume-test: UID travels on the batch into training_step.

Prefetching collated microbatches must not change which UID is recorded: only the
batch that reaches ``training_step`` counts. Skipped-offset indexes are never a
pass/fail criterion.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections import UserDict
from pathlib import Path

import pytest

from src.asr_full_train import (
    PRIVATE_BATCH_UID_KEY,
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
    def test_offset_uses_epoch_relative_trainer_skip(self):
        uids = [f"u{i}" for i in range(64)]
        plan = plan_expected_resume_position(
            uids, resume_step=4, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        )
        assert plan["expected_first_uid"] == "u32"
        assert plan["expected_batch_offset"] == 32

    def test_multi_epoch_uses_modulo_of_update_steps(self):
        uids = [f"u{i}" for i in range(10)]
        plan = plan_expected_resume_position(
            uids, resume_step=5, per_device_train_batch_size=1, gradient_accumulation_steps=3,
        )
        assert plan["expected_first_uid"] == "u6"


class TestUidTravelsOnBatch:
    def test_collator_attaches_private_uids_and_strips_record_uid(self):
        sink = {}
        seen = []

        def inner(feats):
            seen.append(feats)
            return {"input_values": [1]}

        collate = make_uid_tracking_collator(inner, sink)
        batch = collate([{"record_uid": "u1", "input_values": 1}])
        assert batch[PRIVATE_BATCH_UID_KEY] == ["u1"]
        assert all("record_uid" not in f for f in seen[0])

    def test_collator_accepts_userdict_mapping_batch(self):
        # Regression: HF pad returns UserDict-like BatchEncoding/BatchFeature, not dict.
        sink = {}

        def inner(_feats):
            return UserDict({"input_values": [1, 2], "labels": [3]})

        collate = make_uid_tracking_collator(inner, sink)
        batch = collate([{"record_uid": "u42", "input_values": [0]}])
        assert isinstance(batch, dict)
        assert not isinstance(batch, UserDict)
        assert batch[PRIVATE_BATCH_UID_KEY] == ["u42"]
        assert batch["input_values"] == [1, 2]
        assert batch["labels"] == [3]

    def test_collator_accepts_transformers_batch_feature(self):
        pytest.importorskip("transformers", reason="transformers not installed")
        from transformers.feature_extraction_utils import BatchFeature

        sink = {}

        def inner(_feats):
            return BatchFeature({"input_values": [[0.1]], "attention_mask": [[1]]})

        collate = make_uid_tracking_collator(inner, sink)
        batch = collate([{"record_uid": "uid-bf", "input_values": [0.0]}])
        assert isinstance(batch, dict)
        assert batch[PRIVATE_BATCH_UID_KEY] == ["uid-bf"]
        assert batch["input_values"] == [[0.1]]
        assert batch["attention_mask"] == [[1]]
        assert set(batch) - {PRIVATE_BATCH_UID_KEY} == {"input_values", "attention_mask"}

    def test_trainer_strips_private_uid_before_model_forward(self):
        sink = {}
        collate = make_uid_tracking_collator(
            lambda _f: UserDict({"x": 1, "y": 2}),
            sink,
        )
        seen_inputs = []

        class Base:
            def training_step(self, model, inputs, *a, **k):
                seen_inputs.append(dict(inputs))
                assert PRIVATE_BATCH_UID_KEY not in inputs
                return "loss"

        tracked = make_uid_tracking_trainer_cls(Base, sink)()
        batch = collate([{"record_uid": "u7"}])
        assert PRIVATE_BATCH_UID_KEY in batch
        tracked.training_step(None, batch)
        assert sink["first_consumed_uids"] == ["u7"]
        assert seen_inputs == [{"x": 1, "y": 2}]
        assert PRIVATE_BATCH_UID_KEY not in seen_inputs[0]

    def test_prefetch_eight_microbatches_still_records_first_training_step_uid(self):
        # Collator may run ahead; only the first training_step inputs count.
        sink = {}
        collate = make_uid_tracking_collator(lambda feats: {"x": 1}, sink)

        class Base:
            def training_step(self, model, inputs, *a, **k):
                assert PRIVATE_BATCH_UID_KEY not in inputs
                return "loss"

        tracked = make_uid_tracking_trainer_cls(Base, sink)()
        batches = [collate([{"record_uid": f"u{i}"}]) for i in range(8)]
        for batch in batches[4:5]:
            tracked.training_step(None, batch)
        assert sink["first_consumed_uids"] == ["u4"]
        assert sink["first_consumed_from_inputs"] is True
        tracked.training_step(None, batches[5])
        assert sink["first_consumed_uids"] == ["u4"]

    def test_skipped_batches_do_not_claim_first_consumed(self):
        sink = {}
        collate = make_uid_tracking_collator(lambda feats: {"x": 1}, sink)

        class Base:
            def training_step(self, model, inputs, *a, **k):
                return "loss"

        tracked = make_uid_tracking_trainer_cls(Base, sink)()
        for uid in ("u0", "u1", "u2", "u3", "u4"):
            batch = collate([{"record_uid": uid}])
        tracked.training_step(None, batch)
        assert sink["first_consumed_uids"] == ["u4"]


class TestEvaluateDataPosition:
    def test_matching_uid_passes_without_offset_check(self):
        sink = {"first_consumed_uids": ["u32"], "first_consumed_from_inputs": True}
        out = evaluate_data_position(sink, expected_uid="u32")
        assert out["data_position_ok"] is True
        assert "offset_matches" not in out

    def test_nonzero_skipped_offset_does_not_false_fail(self):
        # Regression: previously comparing collate index to skip offset failed wrongly.
        sink = {
            "first_consumed_uids": ["u32"],
            "first_consumed_from_inputs": True,
            "first_consumed_batch_index": 0,
            "collated_batches": [["u0"], ["u8"], ["u16"], ["u24"], ["u32"]],
        }
        out = evaluate_data_position(sink, expected_uid="u32")
        assert out["data_position_ok"] is True

    def test_wrong_uid_fails(self):
        sink = {"first_consumed_uids": ["u0"]}
        out = evaluate_data_position(sink, expected_uid="u32")
        assert out["data_position_ok"] is False

    def test_restart_from_scratch_wrong_uid_fails(self):
        sink = {"first_consumed_uids": ["u0"], "first_consumed_from_inputs": True}
        out = evaluate_data_position(sink, expected_uid="u32")
        assert out["data_position_ok"] is False

    def test_no_uid_reaching_training_step_fails(self):
        out = evaluate_data_position({}, expected_uid="u32")
        assert out["data_position_ok"] is False
        assert out["data_position_detail"] == "no_uid_reached_training_step"


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
            {"first_consumed_uids": ["u32"], "first_consumed_from_inputs": True},
            expected_first_uid="u32",
        )
        assert self._verdict(proof)["status"] == "SUCCESS_FULL_RESUME_TEST"

    def test_wrong_position_fails_even_with_perfect_step(self):
        proof = self._proof(
            {"first_consumed_uids": ["u0"], "first_consumed_from_inputs": True},
            expected_first_uid="u32",
        )
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

    def test_same_process_is_rejected(self, tmp_path: Path):
        from src.asr_full_train import assert_cross_session_resume

        token = new_session_token()
        with pytest.raises(RuntimeError):
            assert_cross_session_resume({"session": token}, current_session=token)
