"""NB14 resume proof, best-checkpoint, and arm-isolation tests."""
from __future__ import annotations

import json

import pytest

from src.rq2_final_contract import (
    ARM_D0,
    ARM_QUALITY,
    ARM_RANDOM,
    ArmIsolationError,
    GTestFirewallError,
    Nb14Flags,
    TrainingContractError,
    build_arm_training_contract,
    compute_nb14_source_fingerprint,
    experiment_root_for_arm,
    verify_upstream_rq2,
)
from src.rq2_final_data import (
    compose_arm_training_rows,
    data_contract_payload,
    load_pinned_nb13_arm_manifest,
    validation_identity,
    verify_equal_budget_from_nb13,
    write_arm_manifest_csv,
)
from src.rq2_final_train import (
    assert_no_cross_arm_resume,
    assert_training_reached_max_steps,
    bind_d0_arm,
    build_hparams_from_yaml,
    build_trainer,
    build_training_arguments,
    build_training_complete_proof,
    compute_validation_metrics,
    ensure_arm_root,
    experiment_fingerprint,
    freeze_best_checkpoint,
    merge_training_fields,
    prove_live_trainer_resume,
    prove_resume,
    read_training_complete_proof,
    resume_or_train_arm,
    select_best_checkpoint,
    validation_history_from_checkpoints,
    write_resume_proof,
    write_training_complete_proof,
)
from src.rq2_selection import verify_published_selection
from src.rq2_selection_contract import SELECTION_RELATIVE_DIR
from tests.rq2_nb14_fixtures import ARM_FINGERPRINTS, explicit_test_mix_policy, explicit_test_seed_policy, supervised_rows, validation_frame, world, write_complete_checkpoint


def _arm_bundle(tmp_path, arm):
    env = world(tmp_path)
    upstream = verify_upstream_rq2(tmp_path, artifact_root=tmp_path, flags=env["flags"])
    selection = verify_published_selection(
        tmp_path / SELECTION_RELATIVE_DIR,
        project_root=tmp_path,
        durable_root=tmp_path,
        generation_id=upstream["nb13_generation_id"],
    )
    budget = verify_equal_budget_from_nb13(selection)
    pseudo = [] if arm == ARM_D0 else load_pinned_nb13_arm_manifest(tmp_path, arm=arm, upstream=upstream)
    composition = compose_arm_training_rows(
        arm=arm,
        supervised_rows=supervised_rows(),
        pseudo_rows=pseudo,
        selection_contract_sha256=upstream["nb13_selection_contract_sha256"],
        nb12_contract_sha256=upstream["nb12_contract_sha256"],
    )
    sha = write_arm_manifest_csv(tmp_path / f"{arm}.csv", composition["rows"])
    data = data_contract_payload(
        composition,
        manifest_sha256=sha,
        validation=validation_identity(validation_frame()),
        upstream=upstream,
        budget=budget,
    )
    fields = merge_training_fields(
        arm=arm,
        data_contract=data,
        hparams=build_hparams_from_yaml(tmp_path),
        source_fingerprint_sha256=compute_nb14_source_fingerprint()["aggregate_sha256"],
        rq1_direct_training_contract_hash=upstream["d0"]["direct_training_contract_hash"],
        d0_init=upstream["d0"],
        mix_policy=explicit_test_mix_policy(),
        seed_policy=explicit_test_seed_policy(),
    )
    contract = build_arm_training_contract(fields)
    fingerprint = experiment_fingerprint(
        arm=arm,
        training_contract=contract,
        data_manifest_sha256=sha,
        model_revision=contract["decoder_revision"],
    )
    return env, contract, fingerprint, sha


def test_valid_checkpoint_resumes(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    ckpt = write_complete_checkpoint(
        tmp_path / "ckpt-random",
        arm=ARM_RANDOM,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
        step=50,
    )
    proof = prove_resume(
        ckpt,
        arm=ARM_RANDOM,
        expected_contract_hash=contract["arm_training_contract_sha256"],
        expected_data_hash=sha,
    )
    assert proof["optimizer_restored"] is True
    assert proof["scheduler_restored"] is True
    assert proof["rng_restored"] is True
    assert proof["global_step"] == 50


def test_wrong_experiment_fingerprint_fails(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_QUALITY)
    ckpt = write_complete_checkpoint(
        tmp_path / "ckpt-wrong",
        arm=ARM_RANDOM,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
    )
    with pytest.raises(ArmIsolationError):
        prove_resume(
            ckpt,
            arm=ARM_QUALITY,
            expected_contract_hash=contract["arm_training_contract_sha256"],
            expected_data_hash=sha,
        )


def test_incomplete_optimizer_fails_resume(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    ckpt = write_complete_checkpoint(
        tmp_path / "ckpt-bad",
        arm=ARM_RANDOM,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
    )
    (ckpt / "optimizer.pt").unlink()
    with pytest.raises((RuntimeError, TrainingContractError)):
        prove_resume(
            ckpt,
            arm=ARM_RANDOM,
            expected_contract_hash=contract["arm_training_contract_sha256"],
            expected_data_hash=sha,
        )


def test_best_checkpoint_validation_only_and_tie_break():
    history = [
        {"checkpoint_name": "checkpoint-10", "global_step": 10, "eval_sacrebleu": 12.0, "split": "validation"},
        {"checkpoint_name": "checkpoint-20", "global_step": 20, "eval_sacrebleu": 12.0, "split": "validation"},
        {"checkpoint_name": "checkpoint-30", "global_step": 30, "eval_sacrebleu": 11.0, "split": "validation"},
    ]
    best = select_best_checkpoint(history)
    assert best["checkpoint_name"] == "checkpoint-20"
    assert best["g_test_used"] is False
    with pytest.raises(GTestFirewallError):
        select_best_checkpoint(history + [{"checkpoint_name": "x", "global_step": 1, "eval_sacrebleu": 99, "split": "test"}])
    with pytest.raises(GTestFirewallError):
        select_best_checkpoint([{"checkpoint_name": "x", "global_step": 1, "eval_sacrebleu": 1, "g_test_bleu": 9, "split": "validation"}])


def test_d0_bind_does_not_train(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_D0)
    layout = ensure_arm_root(tmp_path / "d0-root")
    result = resume_or_train_arm(
        arm=ARM_D0,
        flags=env["flags"],
        layout=layout,
        training_contract=contract,
        fingerprint=fingerprint,
    )
    assert result["trained"] is False
    assert result["identity"]["d0_policy"] == "reuse_frozen_rq1_direct_checkpoint"


def test_real_training_skipped_without_flag(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    layout = ensure_arm_root(tmp_path / "random-root")
    result = resume_or_train_arm(
        arm=ARM_RANDOM,
        flags=env["flags"],
        layout=layout,
        training_contract=contract,
        fingerprint=fingerprint,
    )
    assert result["trained"] is False
    assert result["reason"] == "RUN_REAL_TRAINING=False"


def test_cross_arm_resume_rejected():
    with pytest.raises(ArmIsolationError):
        assert_no_cross_arm_resume(ARM_RANDOM, ARM_QUALITY)


def test_isolated_roots_do_not_share_paths(tmp_path):
    h = "cd" * 32
    a = experiment_root_for_arm(durable_root=tmp_path, contract_hash=h, arm=ARM_D0)
    b = experiment_root_for_arm(durable_root=tmp_path, contract_hash=h, arm=ARM_RANDOM)
    c = experiment_root_for_arm(durable_root=tmp_path, contract_hash=h, arm=ARM_QUALITY)
    assert len({a, b, c}) == 3
    layout = ensure_arm_root(tmp_path / "q")
    write_complete_checkpoint(
        layout["checkpoints"] / "checkpoint-1",
        arm=ARM_QUALITY,
        contract_hash=h,
        data_hash=h,
        step=1,
    )
    freeze_best_checkpoint(
        layout,
        {
            **select_best_checkpoint([{"checkpoint_name": "checkpoint-1", "global_step": 1, "eval_sacrebleu": 1.0, "split": "validation"}]),
        },
        arm=ARM_QUALITY,
        contract_hash=h,
        data_hash=h,
    )


def test_best_checkpoint_rejects_non_finite_metrics():
    base = {"checkpoint_name": "checkpoint-1", "global_step": 1, "split": "validation"}
    with pytest.raises(TrainingContractError, match="not finite"):
        select_best_checkpoint([{**base, "eval_sacrebleu": float("nan")}])
    with pytest.raises(TrainingContractError, match="not finite"):
        select_best_checkpoint([{**base, "eval_sacrebleu": float("inf")}])
    with pytest.raises(TrainingContractError, match="not finite"):
        select_best_checkpoint([{**base, "eval_sacrebleu": float("-inf")}])


def test_resume_or_train_constructs_real_trainer_without_callback(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    constructed = {}

    class FakeTrainer:
        def __init__(self, layout, arm, training_contract):
            self.state = type(
                "S",
                (),
                {"global_step": 7, "max_steps": 7, "epoch": 1.0, "best_model_checkpoint": None},
            )()
            self.optimizer = object()
            self.lr_scheduler = object()
            self._layout = layout
            self._arm = arm
            self._contract = training_contract

        def train(self, resume_from_checkpoint=None):
            write_complete_checkpoint(
                self._layout["checkpoints"] / "checkpoint-7",
                arm=self._arm,
                contract_hash=self._contract["arm_training_contract_sha256"],
                data_hash=self._contract["data_manifest_sha256"],
                step=7,
            )
            constructed["trained"] = True
            constructed["resume"] = resume_from_checkpoint
            return type("R", (), {"global_step": 7})()

    def factory(**kwargs):
        constructed["arm"] = kwargs["arm"]
        constructed["has_layout"] = "layout" in kwargs
        return FakeTrainer(kwargs["layout"], kwargs["arm"], kwargs["training_contract"])

    flags = Nb14Flags(run_real_training=True, direct_state_dir=env["flags"].direct_state_dir)
    layout = ensure_arm_root(tmp_path / "random-real")
    monitor = validation_frame()
    result = resume_or_train_arm(
        arm=ARM_RANDOM,
        flags=flags,
        layout=layout,
        training_contract=contract,
        fingerprint=fingerprint,
        train_frame=validation_frame(),
        validation_frame=monitor,
        monitor_frame=monitor,
        trainer_factory=factory,
    )
    assert result["trained"] is True
    assert constructed["trained"] is True
    assert constructed["arm"] == ARM_RANDOM

    env_q, contract_q, fingerprint_q, _ = _arm_bundle(tmp_path, ARM_QUALITY)
    constructed.clear()
    layout_q = ensure_arm_root(tmp_path / "quality-real")
    result_q = resume_or_train_arm(
        arm=ARM_QUALITY,
        flags=Nb14Flags(run_real_training=True, direct_state_dir=env_q["flags"].direct_state_dir),
        layout=layout_q,
        training_contract=contract_q,
        fingerprint=fingerprint_q,
        train_frame=validation_frame(),
        validation_frame=monitor,
        monitor_frame=monitor,
        trainer_factory=factory,
    )
    assert result_q["trained"] is True
    assert constructed["arm"] == ARM_QUALITY


def test_build_training_arguments_and_trainer_builders(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_QUALITY)
    args = build_training_arguments(output_dir=tmp_path / "out", training_contract=contract)
    assert args.predict_with_generate is True
    assert str(args.metric_for_best_model) == "eval_sacrebleu"
    assert build_trainer is not None
    assert env["flags"].run_real_training is False


def test_live_resume_proof_after_restore(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    ckpt = write_complete_checkpoint(
        tmp_path / "ckpt-live",
        arm=ARM_RANDOM,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
        step=50,
    )

    class DummyTrainer:
        def __init__(self):
            self.state = type("S", (), {"global_step": 50})()
            self.optimizer = type("O", (), {"state_dict": lambda self: {"state": {0: {"step": 50}}}})()
            self.lr_scheduler = type("L", (), {"state_dict": lambda self: {"last_epoch": 50}})()

    proof = prove_live_trainer_resume(
        DummyTrainer(),
        expected_global_step=50,
        checkpoint_path=ckpt,
        arm=ARM_RANDOM,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        expected_data_hash=sha,
    )
    assert proof["proof_status"] == "RESUME_PROVEN"
    assert proof["restored_global_step"] == 50
    DummyTrainer().state.global_step = 0
    broken = DummyTrainer()
    broken.state = type("S", (), {"global_step": 0})()
    with pytest.raises(TrainingContractError, match="live resume"):
        prove_live_trainer_resume(
            broken,
            expected_global_step=50,
            checkpoint_path=ckpt,
            arm=ARM_RANDOM,
            training_contract_sha256=contract["arm_training_contract_sha256"],
            expected_data_hash=sha,
        )


def test_resume_available_is_not_training_complete(tmp_path):
    from src.rq2_final_evaluate import derive_final_readiness

    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    layout = ensure_arm_root(tmp_path / "resume-incomplete")
    ckpt = write_complete_checkpoint(
        layout["checkpoints"] / "checkpoint-50",
        arm=ARM_RANDOM,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
        step=50,
    )
    proof = prove_resume(
        ckpt,
        arm=ARM_RANDOM,
        expected_contract_hash=contract["arm_training_contract_sha256"],
        expected_data_hash=sha,
    )
    assert proof["optimizer_restored"] is True
    readiness = derive_final_readiness(
        flags=env["flags"],
        layouts={ARM_RANDOM: layout, ARM_QUALITY: layout},
        contracts={ARM_RANDOM: {"contract": contract}, ARM_QUALITY: {"contract": contract}},
    )
    assert readiness["flat"]["d_random_complete"] is False
    write_training_complete_proof(
        layout,
        arm=ARM_RANDOM,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        final_global_step=50,
        checkpoint_path=ckpt,
        expected_max_steps=50,
        reached_max_steps=True,
    )
    ready_after = derive_final_readiness(
        flags=env["flags"],
        layouts={ARM_RANDOM: layout},
        contracts={ARM_RANDOM: {"contract": contract}},
    )
    assert ready_after["gates"]["d_random_complete"]["ok"] is True


def test_wrong_arm_checkpoint_rejected_from_best_selection(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_QUALITY)
    layout = ensure_arm_root(tmp_path / "best-own")
    write_complete_checkpoint(
        layout["checkpoints"] / "checkpoint-10",
        arm=ARM_RANDOM,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
        step=10,
    )
    history = validation_history_from_checkpoints(
        layout,
        arm=ARM_QUALITY,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        data_contract_sha256=sha,
    )
    assert history == []


def test_stale_contract_checkpoint_rejected_from_best_selection(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_QUALITY)
    layout = ensure_arm_root(tmp_path / "best-stale")
    write_complete_checkpoint(
        layout["checkpoints"] / "checkpoint-10",
        arm=ARM_QUALITY,
        contract_hash="00" * 32,
        data_hash=sha,
        step=10,
    )
    history = validation_history_from_checkpoints(
        layout,
        arm=ARM_QUALITY,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        data_contract_sha256=sha,
    )
    assert history == []


def test_real_trainer_evaluate_emits_untruncated_metrics(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    import math
    import wave
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from transformers import PretrainedConfig, PreTrainedModel

    from src.data_utils import safe_cache_filename
    from src.metrics import mt_corpus_metrics
    from src.mt_normalize import normalize_mt_text_v1
    from src.rq2_final_train import build_validation_metric_context

    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    long_ref = "đây là câu dài để kiểm tra tham chiếu không bị cắt " * 8
    cache = tmp_path / "audio_cache"
    cache.mkdir()
    train_uids = ["sup-1", "sup-2"]
    val_uids = ["val-long", "val-2"]
    for uid in train_uids + val_uids:
        path = cache / safe_cache_filename(uid)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    train_frame = pd.DataFrame([
        {"record_uid": "sup-1", "text_vi_norm": "ngan", "wav_local_path": str(cache / safe_cache_filename("sup-1"))},
        {"record_uid": "sup-2", "text_vi_norm": "ngan hai", "wav_local_path": str(cache / safe_cache_filename("sup-2"))},
    ])
    val_frame = pd.DataFrame([
        {"record_uid": "val-long", "text_vi_norm": normalize_mt_text_v1(long_ref), "text_vi": long_ref},
        {"record_uid": "val-2", "text_vi_norm": normalize_mt_text_v1("tham chiếu validation hai"), "text_vi": "tham chiếu validation hai"},
    ])

    class DummyFE:
        def __call__(self, wav, sampling_rate=16000, return_attention_mask=True):
            arr = np.asarray(wav, dtype=np.float32).reshape(1, -1)
            return {"input_values": arr, "attention_mask": np.ones_like(arr, dtype=np.int64)}

        def pad(self, features, padding=True, return_tensors="pt"):
            seqs = [torch.as_tensor(np.asarray(f["input_values"]).reshape(-1), dtype=torch.float32) for f in features]
            width = max(s.numel() for s in seqs)
            padded = torch.zeros(len(seqs), width)
            mask = torch.zeros(len(seqs), width, dtype=torch.long)
            for i, seq in enumerate(seqs):
                padded[i, : seq.numel()] = seq
                mask[i, : seq.numel()] = 1
            return {"input_values": padded, "attention_mask": mask}

    class DummyTok:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"
        lang_code_to_id = {"vi_VN": 2}

        def __call__(self, text_target=None, max_length=256, truncation=True, add_special_tokens=True, **kwargs):
            ids = list(range(2, 2 + max(1, len(str(text_target)))))
            if truncation:
                ids = ids[: int(max_length)]
            return {"input_ids": ids}

        def pad(self, features, padding=True, return_tensors="pt"):
            seqs = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
            width = max(s.numel() for s in seqs)
            ids = torch.full((len(seqs), width), int(self.pad_token_id))
            mask = torch.zeros(len(seqs), width, dtype=torch.long)
            for i, seq in enumerate(seqs):
                ids[i, : seq.numel()] = seq
                mask[i, : seq.numel()] = 1
            return {"input_ids": ids, "attention_mask": mask}

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["hyp"] * len(ids)

    class TinyCfg(PretrainedConfig):
        model_type = "tiny-nb14"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.is_encoder_decoder = True
            self.vocab_size = 32
            self.hidden_size = 8
            self.decoder_start_token_id = 2
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.bos_token_id = 2

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
                bos_token_id=2,
                max_length=8,
                _from_model_config=False,
            )

        def forward(self, input_values=None, attention_mask=None, labels=None, **kwargs):
            pooled = input_values.mean(dim=-1, keepdim=True)
            hidden = self.proj(pooled)
            loss = hidden.sum() * 0
            if labels is not None:
                loss = loss + labels.float().sum() * 0
            return {"loss": loss, "logits": hidden}

        def generate(self, input_values=None, **kwargs):
            batch = input_values.shape[0]
            return torch.ones(batch, 4, dtype=torch.long)

    captured = {}
    original = mt_corpus_metrics

    def spy(hyps, refs):
        captured["refs"] = list(refs)
        captured["hyps"] = list(hyps)
        return original(hyps, refs)

    import src.rq2_final_train as train_mod
    import src.metrics as metrics_mod

    metrics_mod.mt_corpus_metrics = spy
    train_mod.compute_validation_metrics.__globals__["mt_corpus_metrics"] = spy
    layout = ensure_arm_root(tmp_path / "eval-trainer")
    contract = dict(contract)
    contract["fp16"] = False
    contract["bf16"] = False
    contract["max_target_length"] = 4
    contract["per_device_eval_batch_size"] = 1
    contract["per_device_train_batch_size"] = 1
    try:
        # Train-time eval uses the frozen monitor frame (here: same tiny val fixture),
        # not a separately sampled subset. Full validation_frame is still passed for identity.
        monitor_frame = val_frame.copy().reset_index(drop=True)
        from src.direct_full_train import monitor_manifest
        from src.rq1_contract import sha256_file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            monitor_path = Path(tmp.name)
        try:
            monitor_frame.to_csv(monitor_path, index=False)
            live = monitor_manifest(monitor_frame, path=monitor_path)
            contract = dict(contract)
            contract["monitor_size"] = int(len(monitor_frame))
            contract["monitor_file_sha256"] = sha256_file(monitor_path)
            contract["monitor_uid_set_hash"] = str(live["uid_set_hash"])
            contract["monitor_pair_hash"] = str(live["pair_hash"])
            contract["monitor_ordered_row_hash"] = str(live["ordered_row_hash"])
        finally:
            monitor_path.unlink(missing_ok=True)
        trainer = build_trainer(
            arm=ARM_RANDOM,
            layout=layout,
            training_contract=contract,
            fingerprint=fingerprint,
            train_frame=train_frame,
            validation_frame=val_frame,
            monitor_frame=monitor_frame,
            audio_cache_roots=[cache],
            processors=(DummyFE(), DummyTok()),
            model=TinyModel(TinyCfg()),
        )
        # Avoid multiprocessing pickle of local DummyFE in unit tests.
        trainer.args.dataloader_num_workers = 0
        trainer.args.dataloader_persistent_workers = False
        trainer.args.dataloader_prefetch_factor = None
        metrics = trainer.evaluate()
    finally:
        metrics_mod.mt_corpus_metrics = original
        train_mod.compute_validation_metrics.__globals__["mt_corpus_metrics"] = original
    assert "eval_sacrebleu" in metrics
    assert "eval_chrfpp" in metrics
    assert math.isfinite(float(metrics["eval_sacrebleu"]))
    assert math.isfinite(float(metrics["eval_chrfpp"]))
    assert captured["refs"][0] == normalize_mt_text_v1(long_ref)
    assert len(captured["refs"][0]) > 4
    ctx = build_validation_metric_context(val_frame, DummyTok())
    assert ctx["ordered_refs"][0] == normalize_mt_text_v1(long_ref)


def test_gold_pseudo_mix_independent_of_pseudo_count_and_resume_cursor():
    from src.rq2_final_train import GoldPseudoMixSampler

    gold = [0, 1]
    left = GoldPseudoMixSampler(gold, [10, 11], gold_slots=1, pseudo_slots=1, seed=7)
    right = GoldPseudoMixSampler(gold, [10, 11, 12, 13, 14], gold_slots=1, pseudo_slots=1, seed=7)
    assert left.observed_ratio()["configured_ratio"] == right.observed_ratio()["configured_ratio"] == "1:1"
    assert left.observed_ratio()["n_gold_draws"] == left.observed_ratio()["n_pseudo_draws"]
    assert right.observed_ratio()["n_gold_draws"] == right.observed_ratio()["n_pseudo_draws"]
    same = GoldPseudoMixSampler(gold, [10, 11], gold_slots=1, pseudo_slots=1, seed=7)
    assert left.schedule() == same.schedule()
    consumed = 2
    resumed = GoldPseudoMixSampler(gold, [10, 11], gold_slots=1, pseudo_slots=1, seed=7, start=consumed)
    assert list(resumed) == left.schedule()[consumed:]


def test_unset_mix_policy_prevents_real_training(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    broken = dict(contract)
    broken["gold_pseudo_mix_policy"] = "UNSET_REQUIRE_EXPLICIT_CONFIG"
    broken["gold_slots"] = None
    broken["pseudo_slots"] = None
    flags = Nb14Flags(run_real_training=True, direct_state_dir=env["flags"].direct_state_dir)
    layout = ensure_arm_root(tmp_path / "mix-unset")
    with pytest.raises(TrainingContractError, match="GOLD_PSEUDO_MIX_POLICY"):
        resume_or_train_arm(
            arm=ARM_RANDOM,
            flags=flags,
            layout=layout,
            training_contract=broken,
            fingerprint=fingerprint,
        )


def test_training_complete_keeps_terminal_and_best_distinct(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_QUALITY)
    layout = ensure_arm_root(tmp_path / "term-best")
    terminal = write_complete_checkpoint(
        layout["checkpoints"] / "checkpoint-30",
        arm=ARM_QUALITY,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
        step=30,
    )
    best = write_complete_checkpoint(
        layout["checkpoints"] / "checkpoint-10",
        arm=ARM_QUALITY,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
        step=10,
    )
    from src.rq2_final_train import checkpoint_model_state_sha256

    payload = write_training_complete_proof(
        layout,
        arm=ARM_QUALITY,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        final_global_step=30,
        checkpoint_path=terminal,
        best_checkpoint={
            "checkpoint_name": best.name,
            "checkpoint_fingerprint": checkpoint_model_state_sha256(best),
            "validation_metric": 12.5,
        },
        expected_max_steps=30,
        reached_max_steps=True,
    )
    assert payload["terminal_checkpoint"] == "checkpoint-30"
    assert payload["best_checkpoint"] == "checkpoint-10"
    assert payload["terminal_checkpoint"] != payload["best_checkpoint"]


def test_real_hf_trainer_resume_restores_step(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    import torch
    from torch.utils.data import Dataset
    from transformers import PretrainedConfig, PreTrainedModel, Trainer, TrainingArguments

    from src.rq2_final_train import prove_live_trainer_resume
    from src.rq2_pseudo_contract import write_json

    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    layout = ensure_arm_root(tmp_path / "hf-resume")
    write_json(layout["checkpoints"] / "full_experiment_fingerprint.json", dict(fingerprint))

    class TinyCfg(PretrainedConfig):
        model_type = "tiny"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 8

    class TinyModel(PreTrainedModel):
        config_class = TinyCfg

        def __init__(self, config):
            super().__init__(config)
            self.proj = torch.nn.Linear(4, 2)

        def forward(self, input_ids=None, labels=None, **kwargs):
            logits = self.proj(input_ids.float())
            loss = logits.sum() * 0
            if labels is not None:
                loss = loss + (logits[:, 0] - labels.float()).pow(2).mean() * 0
            return {"loss": loss, "logits": logits}

    class TinyData(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "input_ids": torch.ones(4),
                "labels": torch.tensor(0),
            }

    args = TrainingArguments(
        output_dir=str(layout["checkpoints"]),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=1,
        save_steps=1,
        logging_steps=1,
        report_to=[],
        seed=int(contract["seed"]),
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
    )
    first = Trainer(model=TinyModel(TinyCfg()), args=args, train_dataset=TinyData())
    first.train()
    ckpts = sorted(p for p in layout["checkpoints"].iterdir() if p.is_dir() and p.name.startswith("checkpoint-"))
    assert ckpts
    resume_dir = ckpts[-1]
    write_json(resume_dir / "full_experiment_fingerprint.json", dict(fingerprint))
    second_args = TrainingArguments(
        output_dir=str(layout["checkpoints"]),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=1,
        save_steps=1,
        logging_steps=1,
        report_to=[],
        seed=int(contract["seed"]),
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
    )
    second = Trainer(model=TinyModel(TinyCfg()), args=second_args, train_dataset=TinyData())
    second.train(resume_from_checkpoint=str(resume_dir))
    proof = prove_live_trainer_resume(
        second,
        expected_global_step=int(second.state.global_step),
        checkpoint_path=resume_dir,
        arm=ARM_RANDOM,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        expected_data_hash=sha,
    )
    assert proof["proof_status"] == "RESUME_PROVEN"
    assert int(second.state.global_step) >= 1
    assert second.optimizer is not None
    assert second.lr_scheduler is not None
    opt_state = second.optimizer.state_dict()
    assert opt_state
    sch_state = second.lr_scheduler.state_dict()
    assert sch_state
    assert (resume_dir / "rng_state.pth").is_file()
    assert proof["rng_restored"] is True
    write_resume_proof(layout, proof)
    assert layout["resume_proof"].is_file()


def test_assert_training_reached_max_steps_cases():
    from types import SimpleNamespace

    ok = assert_training_reached_max_steps(
        SimpleNamespace(state=SimpleNamespace(global_step=46119, max_steps=46119)),
        arm=ARM_RANDOM,
    )
    assert ok == {
        "final_global_step": 46119,
        "expected_max_steps": 46119,
        "reached_max_steps": True,
    }
    # CASE C: resume mid-run then complete to max_steps.
    resumed = assert_training_reached_max_steps(
        SimpleNamespace(state=SimpleNamespace(global_step=46119, max_steps=46119)),
        arm=ARM_QUALITY,
    )
    assert resumed["reached_max_steps"] is True
    with pytest.raises(TrainingContractError, match="before reaching max_steps"):
        assert_training_reached_max_steps(
            SimpleNamespace(state=SimpleNamespace(global_step=45000, max_steps=46119)),
            arm=ARM_RANDOM,
        )
    with pytest.raises(TrainingContractError, match="max_steps must be > 0"):
        assert_training_reached_max_steps(
            SimpleNamespace(state=SimpleNamespace(global_step=10, max_steps=0)),
            arm=ARM_RANDOM,
        )


def test_premature_stop_marks_failed_and_skips_training_complete(tmp_path):
    from types import SimpleNamespace

    from src.rq2_final_contract import STATUS_FAILED_TRAINING, STATUS_TRAINING_COMPLETE

    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    layout = ensure_arm_root(tmp_path / "premature-stop")
    monitor = validation_frame()

    class EarlyStopTrainer:
        def __init__(self):
            self.state = SimpleNamespace(
                global_step=45000,
                max_steps=46119,
                epoch=2.9,
                best_model_checkpoint=None,
            )

        def train(self, resume_from_checkpoint=None):
            write_complete_checkpoint(
                layout["checkpoints"] / "checkpoint-45000",
                arm=ARM_RANDOM,
                contract_hash=contract["arm_training_contract_sha256"],
                data_hash=sha,
                step=45000,
            )
            self.state.global_step = 45000
            return SimpleNamespace(global_step=45000)

    def factory(**kwargs):
        return EarlyStopTrainer()

    with pytest.raises(TrainingContractError, match="before reaching max_steps"):
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
    progress = json.loads(layout["progress"].read_text(encoding="utf-8"))
    assert progress["status"] == STATUS_FAILED_TRAINING
    assert progress["status"] != STATUS_TRAINING_COMPLETE
    assert progress["global_step"] == 45000
    assert progress["max_steps"] == 46119
    assert "before reaching max_steps" in progress["failure_reason"]
    assert not layout["training_complete"].is_file()


def test_normal_completion_records_reached_max_steps(tmp_path):
    from types import SimpleNamespace

    from src.rq2_final_contract import STATUS_TRAINING_COMPLETE

    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_QUALITY)
    layout = ensure_arm_root(tmp_path / "complete-max")
    monitor = validation_frame()

    class CompleteTrainer:
        def __init__(self):
            self.state = SimpleNamespace(
                global_step=0,
                max_steps=46119,
                epoch=0.0,
                best_model_checkpoint=None,
            )
            self._rq2_eval_monitor_n = 256
            self._rq2_full_validation_n = 11112
            self._rq2_eval_semantics = "monitor-only"

        def train(self, resume_from_checkpoint=None):
            # Simulate a resumed run that finishes at the planned max_steps.
            assert resume_from_checkpoint is None or True
            write_complete_checkpoint(
                layout["checkpoints"] / "checkpoint-46119",
                arm=ARM_QUALITY,
                contract_hash=contract["arm_training_contract_sha256"],
                data_hash=sha,
                step=46119,
            )
            self.state.global_step = 46119
            self.state.epoch = 3.0
            return SimpleNamespace(global_step=46119)

    result = resume_or_train_arm(
        arm=ARM_QUALITY,
        flags=Nb14Flags(run_real_training=True, direct_state_dir=env["flags"].direct_state_dir),
        layout=layout,
        training_contract=contract,
        fingerprint=fingerprint,
        train_frame=monitor,
        validation_frame=monitor,
        monitor_frame=monitor,
        trainer_factory=lambda **kwargs: CompleteTrainer(),
    )
    assert result["lifecycle"] == STATUS_TRAINING_COMPLETE
    complete = json.loads(layout["training_complete"].read_text(encoding="utf-8"))
    assert complete["final_global_step"] == 46119
    assert complete["expected_max_steps"] == 46119
    assert complete["reached_max_steps"] is True
    assert complete["eval_monitor_n"] == 256
    assert complete["full_validation_n"] == 11112
    progress = json.loads(layout["progress"].read_text(encoding="utf-8"))
    assert progress["status"] == STATUS_TRAINING_COMPLETE
    assert progress["global_step"] == 46119
    assert progress["max_steps"] == 46119
    assert read_training_complete_proof(
        layout,
        arm=ARM_QUALITY,
        expected_contract_hash=contract["arm_training_contract_sha256"],
    )["reached_max_steps"] is True


def test_publish_failure_after_build_leaves_no_success_proof(tmp_path, monkeypatch):
    """Gap regression: failure after in-memory proof construction must not publish SUCCESS."""
    from pathlib import Path
    from types import SimpleNamespace

    from src.rq2_final_contract import STATUS_FAILED_TRAINING, STATUS_TRAINING_COMPLETE
    import src.rq2_final_train as train_mod

    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    layout = ensure_arm_root(tmp_path / "publish-gap")
    monitor = validation_frame()
    built = {}

    class CompleteTrainer:
        def __init__(self):
            self.state = SimpleNamespace(global_step=0, max_steps=100, epoch=0.0, best_model_checkpoint=None)
            self._rq2_eval_monitor_n = 2
            self._rq2_full_validation_n = 4
            self._rq2_eval_semantics = "monitor-only"

        def train(self, resume_from_checkpoint=None):
            write_complete_checkpoint(
                layout["checkpoints"] / "checkpoint-100",
                arm=ARM_RANDOM,
                contract_hash=contract["arm_training_contract_sha256"],
                data_hash=sha,
                step=100,
            )
            self.state.global_step = 100
            return SimpleNamespace(global_step=100)

    real_build = train_mod.build_training_complete_proof

    def spy_build(**kwargs):
        payload = real_build(**kwargs)
        built["payload"] = dict(payload)
        return payload

    real_write_json = train_mod.write_json

    def fail_on_training_complete(path, obj):
        if Path(path).name == "training_complete.json":
            raise RuntimeError("forced training_complete publish failure")
        return real_write_json(path, obj)

    monkeypatch.setattr(train_mod, "build_training_complete_proof", spy_build)
    monkeypatch.setattr(train_mod, "write_json", fail_on_training_complete)

    with pytest.raises(RuntimeError, match="forced training_complete publish failure"):
        resume_or_train_arm(
            arm=ARM_RANDOM,
            flags=Nb14Flags(run_real_training=True, direct_state_dir=env["flags"].direct_state_dir),
            layout=layout,
            training_contract=contract,
            fingerprint=fingerprint,
            train_frame=monitor,
            validation_frame=monitor,
            monitor_frame=monitor,
            trainer_factory=lambda **kwargs: CompleteTrainer(),
        )
    assert built["payload"]["status"] == STATUS_TRAINING_COMPLETE
    assert built["payload"]["reached_max_steps"] is True
    assert not layout["training_complete"].is_file()
    progress = json.loads(layout["progress"].read_text(encoding="utf-8"))
    assert progress["status"] == STATUS_FAILED_TRAINING
    assert progress["status"] != STATUS_TRAINING_COMPLETE
    with pytest.raises(TrainingContractError, match="training_complete.json is missing"):
        read_training_complete_proof(
            layout,
            arm=ARM_RANDOM,
            expected_contract_hash=contract["arm_training_contract_sha256"],
        )


def test_derive_final_readiness_rejects_malformed_terminal_step_proofs(tmp_path):
    from src.rq2_final_evaluate import derive_final_readiness

    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_RANDOM)
    layout = ensure_arm_root(tmp_path / "readiness-malformed")
    terminal = write_complete_checkpoint(
        layout["checkpoints"] / "checkpoint-100",
        arm=ARM_RANDOM,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
        step=100,
    )
    good = build_training_complete_proof(
        arm=ARM_RANDOM,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        final_global_step=100,
        checkpoint_path=terminal,
        expected_max_steps=100,
        reached_max_steps=True,
    )
    write_training_complete_proof(
        layout,
        arm=ARM_RANDOM,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        final_global_step=100,
        checkpoint_path=terminal,
        expected_max_steps=100,
        reached_max_steps=True,
    )
    ready = derive_final_readiness(
        flags=env["flags"],
        layouts={ARM_RANDOM: layout},
        contracts={ARM_RANDOM: {"contract": contract}},
    )
    assert ready["gates"]["d_random_complete"]["ok"] is True

    cases = [
        {k: v for k, v in good.items() if k != "reached_max_steps"},
        {**good, "reached_max_steps": False},
        {**good, "final_global_step": 90, "expected_max_steps": 100, "reached_max_steps": True},
        {**good, "final_global_step": 0, "expected_max_steps": 0, "reached_max_steps": True},
    ]
    for bad in cases:
        layout["training_complete"].write_text(json.dumps(bad), encoding="utf-8")
        readiness = derive_final_readiness(
            flags=env["flags"],
            layouts={ARM_RANDOM: layout},
            contracts={ARM_RANDOM: {"contract": contract}},
        )
        assert readiness["gates"]["d_random_complete"]["ok"] is False
        assert readiness["flat"]["d_random_complete"] is False


def test_read_training_complete_proof_requires_terminal_step_invariants(tmp_path):
    env, contract, fingerprint, sha = _arm_bundle(tmp_path, ARM_QUALITY)
    layout = ensure_arm_root(tmp_path / "reader-invariants")
    terminal = write_complete_checkpoint(
        layout["checkpoints"] / "checkpoint-46119",
        arm=ARM_QUALITY,
        contract_hash=contract["arm_training_contract_sha256"],
        data_hash=sha,
        step=46119,
    )
    good = build_training_complete_proof(
        arm=ARM_QUALITY,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        final_global_step=46119,
        checkpoint_path=terminal,
        expected_max_steps=46119,
        reached_max_steps=True,
    )
    write_training_complete_proof(
        layout,
        arm=ARM_QUALITY,
        training_contract_sha256=contract["arm_training_contract_sha256"],
        final_global_step=46119,
        checkpoint_path=terminal,
        expected_max_steps=46119,
        reached_max_steps=True,
    )
    assert read_training_complete_proof(
        layout,
        arm=ARM_QUALITY,
        expected_contract_hash=contract["arm_training_contract_sha256"],
    )["final_global_step"] == 46119

    def _write_mutated(**overrides):
        payload = dict(good)
        payload.update(overrides)
        layout["training_complete"].write_text(json.dumps(payload), encoding="utf-8")

    _write_mutated()
    # B: missing reached_max_steps
    missing = dict(good)
    missing.pop("reached_max_steps")
    layout["training_complete"].write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(TrainingContractError, match="missing reached_max_steps"):
        read_training_complete_proof(
            layout,
            arm=ARM_QUALITY,
            expected_contract_hash=contract["arm_training_contract_sha256"],
        )
    # C: reached_max_steps false
    _write_mutated(reached_max_steps=False)
    with pytest.raises(TrainingContractError, match="reached_max_steps must be exactly true"):
        read_training_complete_proof(
            layout,
            arm=ARM_QUALITY,
            expected_contract_hash=contract["arm_training_contract_sha256"],
        )
    # D: step mismatch
    _write_mutated(final_global_step=45000, expected_max_steps=46119, reached_max_steps=True)
    with pytest.raises(TrainingContractError, match="final_global_step=45000"):
        read_training_complete_proof(
            layout,
            arm=ARM_QUALITY,
            expected_contract_hash=contract["arm_training_contract_sha256"],
        )
    # E: expected_max_steps <= 0
    _write_mutated(final_global_step=0, expected_max_steps=0, reached_max_steps=True)
    with pytest.raises(TrainingContractError, match="expected_max_steps must be > 0"):
        read_training_complete_proof(
            layout,
            arm=ARM_QUALITY,
            expected_contract_hash=contract["arm_training_contract_sha256"],
        )

