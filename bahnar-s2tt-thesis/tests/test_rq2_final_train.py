"""NB14 resume proof, best-checkpoint, and arm-isolation tests."""
from __future__ import annotations

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
    bind_d0_arm,
    build_hparams_from_yaml,
    build_trainer,
    build_training_arguments,
    compute_validation_metrics,
    ensure_arm_root,
    experiment_fingerprint,
    freeze_best_checkpoint,
    merge_training_fields,
    prove_live_trainer_resume,
    prove_resume,
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
    upstream = verify_upstream_rq2(tmp_path, flags=env["flags"])
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
            self.state = type("S", (), {"global_step": 7, "best_model_checkpoint": None})()
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
    result = resume_or_train_arm(
        arm=ARM_RANDOM,
        flags=flags,
        layout=layout,
        training_contract=contract,
        fingerprint=fingerprint,
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
        trainer = build_trainer(
            arm=ARM_RANDOM,
            layout=layout,
            training_contract=contract,
            fingerprint=fingerprint,
            train_frame=train_frame,
            validation_frame=val_frame,
            audio_cache_roots=[cache],
            processors=(DummyFE(), DummyTok()),
            model=TinyModel(TinyCfg()),
        )
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


