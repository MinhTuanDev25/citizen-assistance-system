"""Regression: RQ2 train-time eval must use the frozen RQ1 Direct monitor, not full G_validation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.direct_data import DIRECT_MONITOR_CSV
from src.direct_full_train import monitor_manifest
from src.rq1_contract import sha256_file
from src.rq2_final_contract import (
    ARM_QUALITY,
    ARM_RANDOM,
    FROZEN_RQ1_DIRECT_MONITOR_FILE_SHA256,
    FROZEN_RQ1_DIRECT_MONITOR_SIZE,
    LOCKED_RQ2_EVAL_STEPS,
    LOCKED_RQ2_SAVE_STEPS,
    Nb14Flags,
    STATUS_FAILED_TRAINING,
    STATUS_TRAINING_COMPLETE,
    TrainingContractError,
    UpstreamGateError,
    assert_augmentation_fairness,
    build_arm_training_contract,
    resolve_frozen_d0_init,
    verify_upstream_rq2,
)
from src.rq2_final_data import load_frozen_supervised_splits, load_frozen_validation_monitor
from src.rq2_final_train import (
    EVAL_SEMANTICS_NOTE,
    DurableProgressCallback,
    STATUS_EVALUATING,
    STATUS_SAVING,
    assert_monitor_frame_matches_contract,
    build_trainer,
    ensure_arm_root,
    resume_or_train_arm,
)
from src.rq2_pseudo_contract import write_json
from tests.rq2_nb14_fixtures import (
    frozen_monitor_subset_frame,
    validation_frame,
    validation_population_frame,
    world,
    write_complete_checkpoint,
)
from tests.test_rq2_final_train import _arm_bundle


def test_frozen_monitor_loaded_and_full_validation_retained(tmp_path):
    env = world(tmp_path)
    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    supervised = load_frozen_supervised_splits(
        tmp_path,
        d0_identity=upstream["d0"],
        direct_state_dir=env["flags"].direct_state_dir,
    )
    monitor = supervised["monitor_frame"]
    validation = supervised["validation"]
    assert monitor is not None
    assert len(monitor) == len(validation)  # fixture monitor == full tiny validation
    assert len(monitor) < 11112
    assert len(validation) >= len(monitor)
    assert set(monitor["record_uid"].astype(str)) <= set(validation["record_uid"].astype(str))
    meta = supervised["monitor"]
    assert int(meta["monitor_size"]) == len(monitor)
    assert meta["monitor_file_sha256"] == sha256_file(Path(env["flags"].direct_state_dir) / DIRECT_MONITOR_CSV)


def test_monitor_missing_and_hash_mismatch_fail_closed(tmp_path):
    env = world(tmp_path)
    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    state = Path(env["flags"].direct_state_dir)
    monitor_path = state / DIRECT_MONITOR_CSV
    assert monitor_path.is_file()
    val = validation_frame()
    good_sha = sha256_file(monitor_path)
    with pytest.raises(UpstreamGateError, match="does not match|mismatch|SHA"):
        load_frozen_validation_monitor(
            state,
            val,
            expected_monitor_file_sha256="ab" * 32,
            expected_monitor_size=len(val),
            expected_monitor_uid_set_hash=upstream["d0"]["monitor_uid_set_hash"],
            expected_monitor_pair_hash=upstream["d0"]["monitor_pair_hash"],
            expected_monitor_ordered_row_hash=upstream["d0"]["monitor_ordered_row_hash"],
        )
    monitor_path.unlink()
    with pytest.raises(UpstreamGateError, match="missing"):
        load_frozen_validation_monitor(
            state,
            val,
            expected_monitor_file_sha256=good_sha,
            expected_monitor_size=len(val),
            expected_monitor_uid_set_hash=upstream["d0"]["monitor_uid_set_hash"],
            expected_monitor_pair_hash=upstream["d0"]["monitor_pair_hash"],
            expected_monitor_ordered_row_hash=upstream["d0"]["monitor_ordered_row_hash"],
        )


def test_training_contract_locks_eval_save_and_monitor_identity(tmp_path):
    env, random_c, _, _ = _arm_bundle(tmp_path, ARM_RANDOM)
    _, quality_c, _, _ = _arm_bundle(tmp_path, ARM_QUALITY)
    for contract in (random_c, quality_c):
        assert int(contract["save_steps"]) == LOCKED_RQ2_SAVE_STEPS == 1000
        assert int(contract["eval_steps"]) == LOCKED_RQ2_EVAL_STEPS == 1000
        assert contract["eval_strategy"] == "steps"
        assert contract["save_strategy"] == "steps"
        assert int(contract["monitor_size"]) == len(validation_frame())
        assert len(contract["monitor_file_sha256"]) == 64
        assert len(contract["monitor_uid_set_hash"]) == 64
        assert len(contract["monitor_pair_hash"]) == 64
        assert len(contract["monitor_ordered_row_hash"]) == 64
    assert random_c["monitor_file_sha256"] == quality_c["monitor_file_sha256"]
    assert random_c["monitor_uid_set_hash"] == quality_c["monitor_uid_set_hash"]
    assert random_c["monitor_ordered_row_hash"] == quality_c["monitor_ordered_row_hash"]
    assert random_c["eval_steps"] == quality_c["eval_steps"]
    assert random_c["save_steps"] == quality_c["save_steps"]
    proof = assert_augmentation_fairness(random_c, quality_c)
    assert proof["fairness_proof_sha256"]


def test_contract_hash_changes_when_monitor_or_cadence_changes(tmp_path):
    _, contract, _, _ = _arm_bundle(tmp_path, ARM_RANDOM)
    base = contract["arm_training_contract_sha256"]
    for key, value in (
        ("monitor_file_sha256", "ff" * 32),
        ("monitor_uid_set_hash", "ee" * 32),
        ("eval_steps", 500),
        ("save_steps", 500),
        ("monitor_size", 1),
    ):
        mutated = dict(contract)
        mutated.pop("arm_training_contract_sha256", None)
        mutated[key] = value
        if key in ("eval_steps", "save_steps"):
            with pytest.raises(TrainingContractError):
                build_arm_training_contract(mutated)
            continue
        if key == "monitor_size":
            # size=1 with fixture hashes fails content checks only if SHA fields stay;
            # still must change the scientific hash when rebuilt with valid cadence.
            mutated["monitor_size"] = 1
            # Keep SHA fields valid hex but different size → still builds if hashes present.
            rebuilt = build_arm_training_contract(mutated)
            assert rebuilt["arm_training_contract_sha256"] != base
            continue
        rebuilt = build_arm_training_contract(mutated)
        assert rebuilt["arm_training_contract_sha256"] != base


def test_production_monitor_size_256_requires_frozen_rq1_sha(tmp_path):
    _, contract, _, _ = _arm_bundle(tmp_path, ARM_RANDOM)
    mutated = dict(contract)
    mutated.pop("arm_training_contract_sha256", None)
    mutated["monitor_size"] = FROZEN_RQ1_DIRECT_MONITOR_SIZE
    mutated["monitor_file_sha256"] = "ab" * 32
    with pytest.raises(TrainingContractError, match="frozen RQ1 Direct monitor_file_sha256"):
        build_arm_training_contract(mutated)
    mutated["monitor_file_sha256"] = FROZEN_RQ1_DIRECT_MONITOR_FILE_SHA256
    # Still fails closed on UID/content hashes unless they are valid hex (they are);
    # size 256 with production SHA is accepted by assert_trainable_eval_monitor_contract.
    rebuilt = build_arm_training_contract(mutated)
    assert int(rebuilt["monitor_size"]) == 256
    assert rebuilt["monitor_file_sha256"] == FROZEN_RQ1_DIRECT_MONITOR_FILE_SHA256


def test_build_trainer_rejects_full_validation_as_monitor(tmp_path):
    env, contract, fingerprint, _ = _arm_bundle(tmp_path, ARM_RANDOM)
    layout = ensure_arm_root(tmp_path / "full-val-reject")
    full = validation_frame()
    # Pretend full validation is huge by forcing contract size mismatch.
    contract = dict(contract)
    contract["monitor_size"] = 256
    contract["monitor_file_sha256"] = FROZEN_RQ1_DIRECT_MONITOR_FILE_SHA256
    with pytest.raises(TrainingContractError, match="monitor|frozen"):
        assert_monitor_frame_matches_contract(full, contract, validation_frame=full)


def test_assert_monitor_rejects_uid_outside_population(tmp_path):
    _, contract, _, _ = _arm_bundle(tmp_path, ARM_RANDOM)
    monitor = validation_frame().copy()
    population = validation_frame().copy()
    bad = monitor.copy()
    bad.loc[0, "record_uid"] = "not-in-population"
    with pytest.raises(TrainingContractError, match="outside frozen G_validation|does not match the training contract"):
        assert_monitor_frame_matches_contract(bad, contract, validation_frame=population)


def _progress(layout):
    return json.loads(layout["progress"].read_text(encoding="utf-8"))


def test_heartbeat_atomic_json_and_resume_step(tmp_path):
    layout = ensure_arm_root(tmp_path / "hb")
    cb = DurableProgressCallback(
        layout,
        arm=ARM_RANDOM,
        seed=13,
        every_n_steps=1,
        eval_steps=1000,
        save_steps=1000,
    )
    state = SimpleNamespace(global_step=42, max_steps=100, epoch=0.5)
    control = SimpleNamespace(should_evaluate=False, should_save=False)
    cb.on_train_begin(None, state, control)
    payload = _progress(layout)
    assert payload["arm"] == ARM_RANDOM
    assert payload["seed"] == 13
    assert payload["global_step"] == 42
    assert payload["status"] == "TRAINING_RUNNING"
    assert payload["eval_steps"] == 1000
    assert payload["save_steps"] == 1000
    assert "last_update_utc" in payload
    write_json(layout["progress"], {**payload, "global_step": 99})
    again = _progress(layout)
    assert again["global_step"] == 99
    assert state.global_step == 42


def test_heartbeat_eval_save_status_order(tmp_path):
    layout = ensure_arm_root(tmp_path / "hb-order")
    cb = DurableProgressCallback(
        layout,
        arm=ARM_RANDOM,
        seed=13,
        every_n_steps=50,
        eval_steps=1000,
        save_steps=1000,
    )
    state = SimpleNamespace(global_step=1000, max_steps=5000, epoch=1.0)
    control = SimpleNamespace(should_evaluate=False, should_save=False)
    cb.on_train_begin(None, state, control)
    assert _progress(layout)["status"] == "TRAINING_RUNNING"

    control.should_evaluate = True
    control.should_save = True
    cb.on_step_end(None, state, control)
    assert _progress(layout)["status"] == STATUS_EVALUATING

    control.should_evaluate = False
    cb.on_evaluate(None, state, control, metrics={"eval_sacrebleu": 12.5, "eval_loss": 1.0})
    mid = _progress(layout)
    assert mid["status"] == STATUS_SAVING
    assert mid["last_eval_step"] == 1000
    assert mid["last_eval_metrics"]["eval_sacrebleu"] == 12.5

    control.should_save = False
    cb.on_save(None, state, control)
    after_save = _progress(layout)
    assert after_save["status"] == "TRAINING_RUNNING"
    assert after_save["last_save_step"] == 1000

    before_end = _progress(layout)
    cb.on_train_end(None, state, control)
    assert _progress(layout)["status"] == before_end["status"]
    assert _progress(layout)["status"] != STATUS_TRAINING_COMPLETE


def test_heartbeat_resume_throughput_uses_session_delta(tmp_path):
    import time

    layout = ensure_arm_root(tmp_path / "hb-resume-eta")
    cb = DurableProgressCallback(
        layout,
        arm=ARM_QUALITY,
        seed=13,
        every_n_steps=1,
        eval_steps=1000,
        save_steps=1000,
    )
    state = SimpleNamespace(global_step=10000, max_steps=20000, epoch=2.0)
    control = SimpleNamespace(should_evaluate=False, should_save=False)
    cb.on_train_begin(None, state, control)
    assert cb._base_global_step == 10000
    assert _progress(layout)["global_step"] == 10000

    cb._started = time.monotonic() - 10.0
    state.global_step = 10050
    cb.on_step_end(None, state, control)
    payload = _progress(layout)
    assert payload["global_step"] == 10050
    # 50 new steps over ~10s => ~5 steps/s; must NOT use 10050/elapsed.
    assert payload["steps_per_second"] == pytest.approx(5.0, abs=0.2)
    assert payload["estimated_remaining_seconds"] == pytest.approx(1990.0, abs=80.0)


def test_build_trainer_eval_dataset_is_proper_monitor_subset(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")

    import torch
    from transformers import PretrainedConfig, PreTrainedModel

    env, contract, fingerprint, _ = _arm_bundle(tmp_path, ARM_RANDOM)
    full = validation_population_frame(n=4)
    monitor = frozen_monitor_subset_frame(full, uids=["val-1", "val-3"])
    assert len(full) == 4
    assert len(monitor) == 2
    assert monitor["record_uid"].tolist() != full["record_uid"].tolist()

    monitor_path = tmp_path / "monitor_subset.csv"
    monitor.to_csv(monitor_path, index=False)
    live = monitor_manifest(monitor, path=monitor_path)
    contract = dict(contract)
    contract["fp16"] = False
    contract["bf16"] = False
    contract["monitor_size"] = int(len(monitor))
    contract["monitor_file_sha256"] = sha256_file(monitor_path)
    contract["monitor_uid_set_hash"] = str(live["uid_set_hash"])
    contract["monitor_pair_hash"] = str(live["pair_hash"])
    contract["monitor_ordered_row_hash"] = str(live["ordered_row_hash"])

    train_frame = pd.DataFrame(
        [
            {"record_uid": "sup-1", "text_vi_norm": "ngan", "text_vi": "ngan"},
            {"record_uid": "sup-2", "text_vi_norm": "ngan hai", "text_vi": "ngan hai"},
        ]
    )

    class DummyFE:
        def __call__(self, wav, sampling_rate=16000, return_attention_mask=True):
            import numpy as np

            arr = np.zeros((1, 16), dtype=np.float32)
            return {"input_values": arr, "attention_mask": np.ones_like(arr, dtype=np.int64)}

        def pad(self, features, padding=True, return_tensors="pt"):
            import numpy as np

            batch = len(features)
            return {
                "input_values": torch.zeros(batch, 16),
                "attention_mask": torch.ones(batch, 16, dtype=torch.long),
            }

    class DummyTok:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"
        lang_code_to_id = {"vi_VN": 2}

        def __call__(self, text_target=None, max_length=256, truncation=True, add_special_tokens=True, **kwargs):
            return {"input_ids": [2, 3, 4]}

        def pad(self, features, padding=True, return_tensors="pt"):
            batch = len(features)
            return {
                "input_ids": torch.ones(batch, 3, dtype=torch.long),
                "attention_mask": torch.ones(batch, 3, dtype=torch.long),
            }

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["hyp"] * len(ids)

    class TinyCfg(PretrainedConfig):
        model_type = "tiny-nb14-monitor"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.is_encoder_decoder = True
            self.vocab_size = 32
            self.hidden_size = 8
            self.decoder_start_token_id = 2
            self.pad_token_id = 0
            self.eos_token_id = 1

    class TinyModel(PreTrainedModel):
        config_class = TinyCfg
        main_input_name = "input_values"

        def __init__(self, config):
            super().__init__(config)
            self.proj = torch.nn.Linear(1, 8)
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig(
                decoder_start_token_id=2,
                pad_token_id=0,
                eos_token_id=1,
                max_length=8,
                _from_model_config=False,
            )

        def forward(self, input_values=None, attention_mask=None, labels=None, **kwargs):
            pooled = input_values.mean(dim=-1, keepdim=True)
            hidden = self.proj(pooled)
            loss = hidden.sum() * 0
            return {"loss": loss, "logits": hidden}

        def generate(self, input_values=None, **kwargs):
            batch = input_values.shape[0]
            return torch.ones(batch, 4, dtype=torch.long)

    layout = ensure_arm_root(tmp_path / "monitor-subset-trainer")
    trainer = build_trainer(
        arm=ARM_RANDOM,
        layout=layout,
        training_contract=contract,
        fingerprint=fingerprint,
        train_frame=train_frame,
        validation_frame=full,
        monitor_frame=monitor,
        processors=(DummyFE(), DummyTok()),
        model=TinyModel(TinyCfg()),
    )
    assert len(full) == 4
    assert len(monitor) == 2
    assert len(trainer.eval_dataset) == 2
    assert trainer._rq2_eval_monitor_n == 2
    assert trainer._rq2_full_validation_n == 4
    assert trainer._rq2_eval_semantics == EVAL_SEMANTICS_NOTE
    eval_uids = trainer.eval_dataset.frame["record_uid"].astype(str).tolist()
    assert eval_uids == monitor["record_uid"].astype(str).tolist()
    assert eval_uids != full["record_uid"].astype(str).tolist()


def test_post_train_finalization_failure_marks_progress_failed(tmp_path, monkeypatch):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    layout = ensure_arm_root(tmp_path / "finalize-fail")
    monitor = validation_frame()

    class FakeTrainer:
        def __init__(self):
            self.state = SimpleNamespace(global_step=7, max_steps=7, epoch=1.0, best_model_checkpoint=None)

        def train(self, resume_from_checkpoint=None):
            write_complete_checkpoint(
                layout["checkpoints"] / "checkpoint-7",
                arm=ARM_RANDOM,
                contract_hash=contract["arm_training_contract_sha256"],
                data_hash=sha,
                step=7,
            )
            return SimpleNamespace(global_step=7)

    def factory(**kwargs):
        return FakeTrainer()

    import src.rq2_final_train as train_mod

    def boom(*args, **kwargs):
        raise TrainingContractError("forced post-train proof failure")

    monkeypatch.setattr(train_mod, "write_training_complete_proof", boom)
    with pytest.raises(TrainingContractError, match="forced post-train proof failure"):
        resume_or_train_arm(
            arm=ARM_RANDOM,
            flags=Nb14Flags(run_real_training=True, direct_state_dir=env["flags"].direct_state_dir),
            layout=layout,
            training_contract=contract,
            fingerprint=fingerprint,
            train_frame=monitor,
            validation_frame=monitor,
            monitor_frame=monitor,
            trainer_factory=factory,
        )
    progress = _progress(layout)
    assert progress["status"] == STATUS_FAILED_TRAINING
    assert progress["status"] != STATUS_TRAINING_COMPLETE
    assert progress["global_step"] == 7


def test_d0_init_surfaces_monitor_fields(tmp_path):
    env = world(tmp_path)
    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    d0_init = resolve_frozen_d0_init(env["flags"], project_root=tmp_path, identity=upstream["d0"])
    assert int(d0_init["monitor_size"]) == len(validation_frame())
    assert len(d0_init["monitor_file_sha256"]) == 64
    assert d0_init["monitor_file_sha256"] == upstream["d0"]["monitor_file_sha256"]
    _, contract, _, _ = _arm_bundle(tmp_path, ARM_RANDOM)
    assert contract["monitor_file_sha256"] == d0_init["monitor_file_sha256"]
    assert int(contract["monitor_size"]) == int(d0_init["monitor_size"])
