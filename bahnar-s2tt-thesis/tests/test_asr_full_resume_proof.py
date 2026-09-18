"""
Resume proof capture: every flag must come from real state, not from a literal.

A tiny model/optimizer/scheduler is checkpointed to disk exactly the way the HF
Trainer does, then compared against live objects that are either correctly
restored or deliberately wrong.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.asr_full_train import (
    capture_resume_proof,
    capture_rng_proof,
    derive_resume_test_status_from_proof,
    summarize_resume_proof,
)

STEP = 100


class TinyModel(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(16, 24)
        self.head = nn.Linear(24, 8)

    def forward(self, x):  # pragma: no cover - not trained in tests
        return self.head(torch.relu(self.encoder(x)))


class FakeTrainer:
    """Only the attributes ``capture_resume_proof`` reads."""

    def __init__(self, model, optimizer, lr_scheduler, global_step):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.state = type("S", (), {"global_step": global_step})()


def _advance(model, optimizer, scheduler, steps: int):
    """Take real optimizer steps so Adam's per-parameter ``step`` counter grows."""
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(torch.ones(2, 16)).pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()


def _make_checkpoint(tmp_path: Path, *, use_safetensors=True):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    _advance(model, optimizer, scheduler, STEP)

    ck = tmp_path / f"checkpoint-{STEP}"
    ck.mkdir(parents=True, exist_ok=True)
    if use_safetensors:
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                  str(ck / "model.safetensors"))
    else:
        torch.save(model.state_dict(), ck / "pytorch_model.bin")
    torch.save(optimizer.state_dict(), ck / "optimizer.pt")
    torch.save(scheduler.state_dict(), ck / "scheduler.pt")
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": STEP}), encoding="utf-8")
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        },
        ck / "rng_state.pth",
    )
    return ck, model, optimizer, scheduler


def _restored_from(ck: Path):
    """Rebuild the objects the way Trainer does when resuming."""
    model = TinyModel(seed=99)  # different init, then load checkpoint weights
    if (ck / "model.safetensors").is_file():
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(ck / "model.safetensors")))
    else:
        model.load_state_dict(torch.load(ck / "pytorch_model.bin", map_location="cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(torch.load(ck / "optimizer.pt", map_location="cpu"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    scheduler.load_state_dict(torch.load(ck / "scheduler.pt", map_location="cpu"))
    return model, optimizer, scheduler


class TestCaptureResumeProof:
    def test_correct_restore_proves_every_component(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        model, optimizer, scheduler = _restored_from(ck)
        proof = capture_resume_proof(
            FakeTrainer(model, optimizer, scheduler, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
            steps_per_epoch=64, gradient_accumulation_steps=8,
        )
        assert proof["model_restored"], proof["model_detail"]
        assert proof["optimizer_restored"], proof["optimizer_detail"]
        assert proof["scheduler_restored"], proof["scheduler_detail"]
        # capture_resume_proof cannot decide the data position: it only knows the
        # step. The verdict comes from the UID the training step consumed.
        assert proof["data_position_ok"] is False
        assert proof["data_position_detail"] == "pending_consumed_uid"
        assert proof["step_offset_consistent"] is True
        assert proof["step_matches_checkpoint"] is True
        assert proof["optimizer_step_counters"] == [STEP]
        assert proof["model_tensors_compared"] >= 2

    def test_pytorch_bin_checkpoint_also_supported(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path, use_safetensors=False)
        model, optimizer, scheduler = _restored_from(ck)
        proof = capture_resume_proof(
            FakeTrainer(model, optimizer, scheduler, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
        )
        assert proof["model_restored"] and proof["optimizer_restored"]

    def test_unrestored_model_is_detected(self, tmp_path: Path):
        """A fresh shell that never loaded the checkpoint must not pass."""
        ck, *_ = _make_checkpoint(tmp_path)
        _model, optimizer, scheduler = _restored_from(ck)
        proof = capture_resume_proof(
            FakeTrainer(TinyModel(seed=1234), optimizer, scheduler, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
        )
        assert proof["model_restored"] is False
        assert proof["model_detail"].startswith("mismatch:")

    def test_fresh_optimizer_is_detected(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        model, _optimizer, scheduler = _restored_from(ck)
        fresh = torch.optim.AdamW(model.parameters(), lr=1e-3)
        proof = capture_resume_proof(
            FakeTrainer(model, fresh, scheduler, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
        )
        assert proof["optimizer_restored"] is False
        assert proof["optimizer_detail"] == "optimizer_state_empty"

    def test_optimizer_from_a_different_step_is_detected(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        model, optimizer, scheduler = _restored_from(ck)
        _advance(model, optimizer, scheduler, 3)  # optimizer now ahead of the checkpoint
        proof = capture_resume_proof(
            FakeTrainer(model, optimizer, scheduler, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
        )
        assert proof["optimizer_restored"] is False
        assert "step_counter" in proof["optimizer_detail"]

    def test_unrestored_scheduler_is_detected(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        model, optimizer, _scheduler = _restored_from(ck)
        fresh_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        proof = capture_resume_proof(
            FakeTrainer(model, optimizer, fresh_sched, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
        )
        assert proof["scheduler_restored"] is False
        assert "last_epoch" in proof["scheduler_detail"]

    def test_missing_scheduler_file_is_reported(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        model, optimizer, scheduler = _restored_from(ck)
        (ck / "scheduler.pt").unlink()
        proof = capture_resume_proof(
            FakeTrainer(model, optimizer, scheduler, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
        )
        assert proof["scheduler_restored"] is False
        assert proof["scheduler_detail"] == "scheduler.pt_missing"

    def test_wrong_global_step_fails_step_consistency(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        model, optimizer, scheduler = _restored_from(ck)
        proof = capture_resume_proof(
            FakeTrainer(model, optimizer, scheduler, 0),
            checkpoint_dir=ck, expected_resume_step=STEP,
            steps_per_epoch=64, gradient_accumulation_steps=8,
        )
        assert proof["step_offset_consistent"] is False
        assert proof["step_matches_checkpoint"] is False

    def test_epoch_offset_is_derived_from_dataloader_length(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        model, optimizer, scheduler = _restored_from(ck)
        proof = capture_resume_proof(
            FakeTrainer(model, optimizer, scheduler, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
            steps_per_epoch=64, gradient_accumulation_steps=8,
        )
        # 64 batches / accum 8 = 8 updates per epoch; step 100 -> epoch 12, offset 4
        assert proof["epochs_trained"] == 12
        assert proof["expected_resume_offset"] == 32


class TestCaptureRngProof:
    def test_unchanged_rng_matches(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        proof = capture_rng_proof(ck)
        assert proof["rng_restored"] is True, proof["rng_detail"]
        assert set(proof["rng_streams"]) >= {"torch_cpu", "python", "numpy"}

    def test_advanced_rng_is_detected(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        torch.rand(5)
        random.random()
        np.random.rand(3)
        proof = capture_rng_proof(ck)
        assert proof["rng_restored"] is False
        assert "mismatch" in proof["rng_detail"]

    def test_missing_rng_file_is_reported(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        (ck / "rng_state.pth").unlink()
        proof = capture_rng_proof(ck)
        assert proof["rng_restored"] is False
        assert proof["rng_detail"] == "rng_state.pth_missing"


class TestProofFeedsTheStatusGate:
    def test_end_to_end_capture_yields_success(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        model, optimizer, scheduler = _restored_from(ck)
        restore = capture_resume_proof(
            FakeTrainer(model, optimizer, scheduler, STEP),
            checkpoint_dir=ck, expected_resume_step=STEP,
            steps_per_epoch=64, gradient_accumulation_steps=8,
        )
        # The Trainer reloads RNG lazily at the first step of the resumed epoch;
        # emulate that before the RNG proof is captured.
        rng_state = torch.load(ck / "rng_state.pth", map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(torch.as_tensor(rng_state["cpu"]).cpu())
        sink = {
            "restore": restore,
            "rng": capture_rng_proof(ck),
            "first_step_global_step": STEP,
            "collated_uids_placeholder": None,
            "first_consumed_uids": ["uid-32"],
            "first_consumed_batch_index": 32,
        }
        proof = summarize_resume_proof(
            sink, expected_first_uid="uid-32",
        )
        proof["final_global_step"] = 200
        verdict = derive_resume_test_status_from_proof(
            proof,
            phase_a_reached_target=True,
            cross_session={"two_sessions": True},
            used_separate_experiment_dir=True,
            expected_phase_b_steps=200,
        )
        assert verdict["status"] == "SUCCESS_FULL_RESUME_TEST", verdict["failed_checks"]

    def test_broken_restore_propagates_to_failure(self, tmp_path: Path):
        ck, *_ = _make_checkpoint(tmp_path)
        _model, optimizer, scheduler = _restored_from(ck)
        sink = {
            "restore": capture_resume_proof(
                FakeTrainer(TinyModel(seed=7), optimizer, scheduler, STEP),
                checkpoint_dir=ck, expected_resume_step=STEP,
            ),
            "rng": capture_rng_proof(ck),
            "first_step_global_step": STEP,
            "first_consumed_uids": ["uid-32"],
            "first_consumed_batch_index": 32,
        }
        proof = summarize_resume_proof(
            sink, expected_first_uid="uid-32",
        )
        proof["final_global_step"] = 200
        verdict = derive_resume_test_status_from_proof(
            proof, phase_a_reached_target=True,
            cross_session={"two_sessions": True},
            used_separate_experiment_dir=True, expected_phase_b_steps=200,
        )
        assert verdict["status"] == "FAILED"
        assert "model_restored" in verdict["failed_checks"]
