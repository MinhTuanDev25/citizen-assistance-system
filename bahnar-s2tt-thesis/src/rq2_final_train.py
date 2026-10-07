"""RQ2 Notebook 14 arm-isolated training, resume proof, and validation selection.

D0 is bound from the frozen RQ1 Direct checkpoint and never starts a Trainer.
D-Random and D-Quality each own an isolated experiment root and CONTINUE FROM
that same frozen D0 model state using the RQ1 Direct stack:

    SpeechEncoderDecoderModel.from_pretrained(frozen_d0_checkpoint)
    src.direct_dataset.DirectSpeechTranslationDataset / DirectDataCollator
    src.direct_full_train.make_resume_safe_seq2seq_trainer_cls
    src.asr_full_train.assert_checkpoint_complete_for_resume

HuggingFace Trainer.train() is gated by RUN_REAL_TRAINING. Tests may inject a
trainer_factory; production uses build_trainer() and does not need trainer_fn.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from src.asr_full_train import assert_checkpoint_complete_for_resume
from src.direct_contract import load_direct_yaml_config
from src.rq1_contract import sha256_json
from src.rq2_final_contract import (
    ARM_D0,
    ARM_QUALITY,
    ARM_RANDOM,
    AUGMENTATION_INIT_POLICY,
    BEST_CHECKPOINT_METRIC,
    BEST_CHECKPOINT_TIE_BREAK,
    D0_POLICY,
    FROZEN_RQ1_DIRECT_MONITOR_FILE_SHA256,
    FROZEN_RQ1_DIRECT_MONITOR_SIZE,
    GOLD_PSEUDO_MIX_POLICY_SLOTTED,
    GOLD_PSEUDO_MIX_POLICY_UNSET,
    LOCKED_RQ2_EVAL_STEPS,
    LOCKED_RQ2_EVAL_STRATEGY,
    LOCKED_RQ2_SAVE_STEPS,
    LOCKED_RQ2_SAVE_STRATEGY,
    REJECTED_PUBLIC_PRETRAINED_INIT_POLICY,
    GTestFirewallError,
    RUNTIME_ONLY_KEYS,
    SEED_POLICY_MULTI,
    SEED_POLICY_SINGLE,
    SEED_POLICY_UNSET,
    STATUS_FAILED_TRAINING,
    STATUS_NOT_STARTED,
    STATUS_RESUME_AVAILABLE,
    STATUS_TRAINING_COMPLETE,
    STATUS_TRAINING_RUNNING,
    TRAINABLE_ARMS,
    ArmIsolationError,
    Nb14Flags,
    Rq2FinalError,
    TrainingContractError,
    assert_checkpoint_arm_isolation,
    assert_g_test_blocked,
    bind_frozen_d0_identity,
    build_arm_training_contract,
    code_root,
    experiment_root_for_arm,
    is_sha256,
    locked_architecture,
    normalize_gold_pseudo_mix_policy,
    normalize_seed_policy,
    require_configured_mix_policy,
    require_configured_seed_policy,
    resolve_frozen_d0_init,
    resolve_d0_best_checkpoint_dir,
)
from src.rq2_pseudo_contract import atomic_write_text, write_json


TrainerFn = Callable[..., Dict[str, Any]]


def arm_state_layout(root: Union[str, Path]) -> Dict[str, Path]:
    base = Path(root)
    return {
        "root": base,
        "training_contract": base / "training_contract.json",
        "experiment_fingerprint": base / "experiment_fingerprint.json",
        "checkpoints": base / "checkpoints",
        "latest": base / "LATEST",
        "best_checkpoint": base / "best_checkpoint.json",
        "resume_proof": base / "resume_proof.json",
        "training_complete": base / "training_complete.json",
        "validation_predictions": base / "validation_predictions",
        "logs": base / "logs",
        "progress": base / "logs" / "progress.json",
    }


def ensure_arm_root(root: Union[str, Path]) -> Dict[str, Path]:
    layout = arm_state_layout(root)
    for key in ("root", "checkpoints", "validation_predictions", "logs"):
        layout[key].mkdir(parents=True, exist_ok=True)
    return layout


def write_progress(layout: Mapping[str, Path], payload: Mapping[str, Any]) -> None:
    write_json(layout["progress"], dict(payload))


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


STATUS_EVALUATING = "EVALUATING"
STATUS_SAVING = "SAVING"

EVAL_SEMANTICS_NOTE = (
    "eval_* is the fixed frozen RQ1 Direct validation monitor; "
    "full G_validation is retained for explicit full-validation evaluation "
    "and is never attached to periodic train-time evaluation."
)


class DurableProgressCallback:
    """Runtime-only durable heartbeat. Does not alter scientific Trainer behavior."""

    def __init__(
        self,
        layout: Mapping[str, Path],
        *,
        arm: str,
        seed: Optional[int],
        every_n_steps: int = 50,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
    ):
        self.layout = layout
        self.arm = str(arm)
        self.seed = None if seed is None else int(seed)
        self.every_n_steps = max(1, int(every_n_steps))
        self.eval_steps = None if eval_steps is None else int(eval_steps)
        self.save_steps = None if save_steps is None else int(save_steps)
        self._started = None
        self._base_global_step = 0
        self._last_eval_step = None
        self._last_save_step = None
        self._last_eval_metrics = None

    def _write(self, state: Any, status: str) -> None:
        import time

        if self._started is None:
            self._started = time.monotonic()
        global_step = int(getattr(state, "global_step", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        epoch = float(getattr(state, "epoch", 0.0) or 0.0)
        elapsed = max(0.0, time.monotonic() - float(self._started))
        # Throughput is session-local so resume does not inflate steps/s.
        completed = max(0, global_step - int(self._base_global_step or 0))
        sps = float(completed) / elapsed if elapsed > 0 and completed > 0 else 0.0
        remaining = None
        if sps > 0 and max_steps > global_step:
            remaining = float(max_steps - global_step) / sps
        percent = None
        if max_steps > 0:
            percent = min(100.0, 100.0 * float(global_step) / float(max_steps))
        payload = {
            "arm": self.arm,
            "seed": self.seed,
            "status": status,
            "global_step": global_step,
            "max_steps": max_steps,
            "percent_complete": percent,
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": remaining,
            "steps_per_second": sps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "last_eval_step": self._last_eval_step,
            "last_save_step": self._last_save_step,
            "last_eval_metrics": self._last_eval_metrics,
            "last_update_utc": _utc_now(),
        }
        write_progress(self.layout, payload)

    def on_train_begin(self, args, state, control, **kwargs):
        import time

        self._base_global_step = int(getattr(state, "global_step", 0) or 0)
        self._started = time.monotonic()
        self._write(state, STATUS_TRAINING_RUNNING)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # DefaultFlowCallback sets should_evaluate/should_save before later callbacks.
        # Mark live status immediately before those operations begin.
        if getattr(control, "should_evaluate", False):
            self._write(state, STATUS_EVALUATING)
            return control
        if getattr(control, "should_save", False):
            self._write(state, STATUS_SAVING)
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        if step > 0 and step % self.every_n_steps == 0:
            self._write(state, STATUS_TRAINING_RUNNING)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._last_eval_step = int(getattr(state, "global_step", 0) or 0)
        if isinstance(metrics, Mapping):
            self._last_eval_metrics = {
                key: metrics[key]
                for key in ("eval_sacrebleu", "eval_chrfpp", "eval_monitor_n", "eval_loss")
                if key in metrics
            }
        if getattr(control, "should_save", False):
            self._write(state, STATUS_SAVING)
        else:
            self._write(state, STATUS_TRAINING_RUNNING)
        return control

    def on_save(self, args, state, control, **kwargs):
        self._last_save_step = int(getattr(state, "global_step", 0) or 0)
        self._write(state, STATUS_TRAINING_RUNNING)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # TRAINING_COMPLETE is owned by train_or_resume_arm after proof succeeds.
        return control


def _as_trainer_callback(callback: DurableProgressCallback):
    from transformers import TrainerCallback

    class _Bound(TrainerCallback):  # type: ignore[misc,valid-type]
        def on_train_begin(self, args, state, control, **kwargs):
            return callback.on_train_begin(args, state, control, **kwargs)

        def on_step_end(self, args, state, control, **kwargs):
            return callback.on_step_end(args, state, control, **kwargs)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            return callback.on_evaluate(args, state, control, metrics=metrics, **kwargs)

        def on_save(self, args, state, control, **kwargs):
            return callback.on_save(args, state, control, **kwargs)

        def on_train_end(self, args, state, control, **kwargs):
            return callback.on_train_end(args, state, control, **kwargs)

    return _Bound()


def build_hparams_from_yaml(project_root: Union[str, Path]) -> Dict[str, Any]:
    cfg = load_direct_yaml_config(code_root() / "configs" / "direct.yaml")
    arch = locked_architecture()
    return {
        **arch,
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "warmup_ratio": float(cfg["warmup_ratio"]),
        "weight_decay": float(cfg["weight_decay"]),
        "num_train_epochs": float(cfg["num_train_epochs"]),
        "save_steps": int(cfg["save_steps"]),
        "eval_steps": int(cfg["eval_steps"]),
        "eval_strategy": LOCKED_RQ2_EVAL_STRATEGY,
        "save_strategy": LOCKED_RQ2_SAVE_STRATEGY,
        "monitor_size": int(cfg["monitor_size"]),
        "fp16": bool(cfg["fp16"]),
        "bf16": bool(cfg["bf16"]),
        "gradient_checkpointing": bool(cfg["gradient_checkpointing"]),
        "freeze_feature_encoder": bool(cfg["freeze_feature_encoder"]),
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "generation_max_length": int(cfg["generation_max_length"]),
        "num_beams": int(cfg["num_beams"]),
        "save_total_limit": int(cfg["save_total_limit"]),
        "gold_pseudo_mix_policy": GOLD_PSEUDO_MIX_POLICY_UNSET,
        "gold_slots": None,
        "pseudo_slots": None,
        "seed_policy": SEED_POLICY_UNSET,
        "seed_policy_seeds": [],
    }


def merge_training_fields(
    *,
    arm: str,
    data_contract: Mapping[str, Any],
    hparams: Mapping[str, Any],
    source_fingerprint_sha256: str,
    rq1_direct_training_contract_hash: str,
    d0_init: Optional[Mapping[str, Any]] = None,
    mix_policy: Optional[Mapping[str, Any]] = None,
    seed_policy: Optional[Mapping[str, Any]] = None,
    runtime_workers: Optional[int] = None,
) -> Dict[str, Any]:
    init_policy = D0_POLICY if arm == ARM_D0 else AUGMENTATION_INIT_POLICY
    d0 = dict(d0_init or {})
    d0_identity = d0.get("identity") if isinstance(d0.get("identity"), Mapping) else {}
    mix = normalize_gold_pseudo_mix_policy(mix_policy or hparams)
    seeds = normalize_seed_policy(seed_policy or hparams)

    def _d0_field(key: str, default: Any = "") -> Any:
        if d0.get(key) not in (None, ""):
            return d0.get(key)
        if d0_identity.get(key) not in (None, ""):
            return d0_identity.get(key)
        return hparams.get(key, default)

    fields = {
        "arm": arm,
        "d0_policy": D0_POLICY if arm == ARM_D0 else "",
        "init_policy": init_policy,
        "d0_init_model_state_sha256": str(d0.get("model_state_sha256") or hparams.get("d0_init_model_state_sha256") or ""),
        "d0_checkpoint_fingerprint_sha256": str(
            d0.get("checkpoint_fingerprint_sha256") or hparams.get("d0_checkpoint_fingerprint_sha256") or ""
        ),
        "d0_best_checkpoint_name": str(d0.get("best_checkpoint_name") or hparams.get("d0_best_checkpoint_name") or ""),
        "gold_pseudo_mix_policy": mix["gold_pseudo_mix_policy"],
        "gold_slots": mix["gold_slots"],
        "pseudo_slots": mix["pseudo_slots"],
        "seed_policy": seeds["seed_policy"],
        "seed_policy_seeds": list(seeds.get("seed_policy_seeds") or []),
        "supervised_base_policy": data_contract["supervised_base_policy"],
        "data_manifest_sha256": data_contract["data_manifest_sha256"],
        "ordered_training_uid_hash": data_contract["ordered_training_uid_hash"],
        "training_pair_hash": data_contract["training_pair_hash"],
        "supervised_ordered_uid_hash": data_contract["supervised_ordered_uid_hash"],
        "supervised_pair_hash": data_contract["supervised_pair_hash"],
        "supervised_uid_set_hash": data_contract["supervised_uid_set_hash"],
        "supervised_audio_identity_hash": data_contract["supervised_audio_identity_hash"],
        "pseudo_ordered_uid_hash": data_contract["pseudo_ordered_uid_hash"],
        "pseudo_pair_hash": data_contract["pseudo_pair_hash"],
        "validation_ordered_uid_hash": data_contract["validation_ordered_uid_hash"],
        "validation_pair_hash": data_contract["validation_pair_hash"],
        "validation_uid_set_hash": data_contract["validation_uid_set_hash"],
        "validation_audio_identity_hash": data_contract["validation_audio_identity_hash"],
        "nb11_input_contract_sha256": data_contract["nb11_input_contract_sha256"],
        "nb12_contract_sha256": data_contract["nb12_contract_sha256"],
        "nb13_selection_contract_sha256": data_contract["nb13_selection_contract_sha256"],
        "selection_budget_hours": data_contract["selection_budget_hours"],
        "selection_budget_seconds": data_contract["selection_budget_seconds"],
        "realized_pseudo_duration_seconds": data_contract["realized_pseudo_duration_seconds"],
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "rq1_direct_training_contract_hash": rq1_direct_training_contract_hash,
        "save_steps": LOCKED_RQ2_SAVE_STEPS,
        "eval_steps": LOCKED_RQ2_EVAL_STEPS,
        "eval_strategy": LOCKED_RQ2_EVAL_STRATEGY,
        "save_strategy": LOCKED_RQ2_SAVE_STRATEGY,
        "monitor_size": int(_d0_field("monitor_size", 0) or 0),
        "monitor_file_sha256": str(_d0_field("monitor_file_sha256", "") or ""),
        "monitor_uid_set_hash": str(_d0_field("monitor_uid_set_hash", "") or ""),
        "monitor_pair_hash": str(_d0_field("monitor_pair_hash", "") or ""),
        "monitor_ordered_row_hash": str(_d0_field("monitor_ordered_row_hash", "") or ""),
        **{k: hparams[k] for k in hparams if k not in RUNTIME_ONLY_KEYS and k not in {
            "gold_pseudo_mix_policy", "gold_slots", "pseudo_slots", "seed_policy", "seed_policy_seeds", "seed", "dataloader_seed",
            "save_steps", "eval_steps", "eval_strategy", "save_strategy",
            "monitor_size", "monitor_file_sha256", "monitor_uid_set_hash", "monitor_pair_hash", "monitor_ordered_row_hash",
        }},
    }
    if seeds.get("configured") and seeds.get("seed_policy") == SEED_POLICY_SINGLE:
        only_seed = int(seeds["active_seed"])
        requested = (seed_policy or {}).get("active_seed")
        fields["active_seed"] = int(requested) if requested is not None else only_seed
        if int(fields["active_seed"]) != only_seed:
            raise TrainingContractError("single-seed active_seed must be the only declared seed")
        fields["seed"] = int(fields["active_seed"])
        fields["dataloader_seed"] = int(fields["active_seed"])
        fields["seed_runs"] = list(seeds.get("seed_runs") or [])
        fields["seed_policy_record"] = dict(seeds["seed_policy_record"])
    elif seeds.get("seed_policy") == SEED_POLICY_MULTI:
        fields["seed_runs"] = list(seeds.get("seed_runs") or [])
        fields["seed_policy_record"] = dict(seeds["seed_policy_record"])
        if (seed_policy or {}).get("active_seed") is not None:
            fields["active_seed"] = int(seed_policy["active_seed"])
    if runtime_workers is not None:
        # Accepted as a runtime argument, never copied into the scientific contract.
        fields["_runtime_dataloader_num_workers"] = int(runtime_workers)
        fields.pop("_runtime_dataloader_num_workers")
    return fields


def experiment_fingerprint(
    *,
    arm: str,
    training_contract: Mapping[str, Any],
    data_manifest_sha256: str,
    model_revision: str,
    selected_manifest_sha256: str = "",
    nb13_generation_id: str = "",
) -> Dict[str, Any]:
    payload = {
        "arm": arm,
        "arm_training_contract_sha256": training_contract["arm_training_contract_sha256"],
        "data_manifest_sha256": data_manifest_sha256,
        "encoder_revision": training_contract["encoder_revision"],
        "decoder_revision": training_contract["decoder_revision"],
        "model_revision": model_revision,
        "init_policy": training_contract["init_policy"],
        "nb13_generation_id": str(nb13_generation_id or training_contract.get("nb13_generation_id") or ""),
        "manifest_sha256": str(training_contract.get("data_manifest_sha256") or data_manifest_sha256),
        "selected_manifest_sha256": str(selected_manifest_sha256 or ""),
        "d0_init_model_state_sha256": str(training_contract.get("d0_init_model_state_sha256") or ""),
        "training_seed": training_contract.get("seed"),
    }
    payload["experiment_fingerprint_sha256"] = sha256_json(
        {k: v for k, v in payload.items() if k != "experiment_fingerprint_sha256"}
    )
    return payload


def write_arm_contracts(layout: Mapping[str, Path], training_contract: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> None:
    write_json(layout["training_contract"], dict(training_contract))
    write_json(layout["experiment_fingerprint"], dict(fingerprint))


def bind_d0_arm(
    *,
    layout: Mapping[str, Path],
    flags: Nb14Flags,
    training_contract: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
) -> Dict[str, Any]:
    if training_contract.get("arm") != ARM_D0:
        raise TrainingContractError("bind_d0_arm requires the D0 contract")
    identity = bind_frozen_d0_identity(flags.direct_state_dir, project_root=None)
    write_arm_contracts(layout, training_contract, fingerprint)
    best = {
        "arm": ARM_D0,
        "policy": D0_POLICY,
        "selection_rule": "frozen_rq1_direct_best_checkpoint",
        "tie_break": list(BEST_CHECKPOINT_TIE_BREAK),
        "metric_for_best_model": BEST_CHECKPOINT_METRIC,
        "checkpoint_name": identity["best_checkpoint_name"],
        "checkpoint_fingerprint": identity.get("checkpoint_fingerprint_sha256") or "",
        "model_state_sha256": identity.get("model_state_sha256") or "",
        "direct_training_contract_hash": identity["direct_training_contract_hash"],
        "validation_metric": None,
        "global_step": None,
        "g_test_used": False,
    }
    write_json(layout["best_checkpoint"], best)
    atomic_write_text(layout["latest"], identity["best_checkpoint_name"] + "\n")
    write_progress(layout, {"status": "d0_bound", "arm": ARM_D0, "records_processed": 0})
    return {"identity": identity, "best": best, "trained": False}


def discover_latest_valid_checkpoint(
    layout: Mapping[str, Path],
    *,
    arm: str,
    expected_contract_hash: str,
    expected_data_hash: str,
    expected_nb13_generation_id: Optional[str] = None,
    expected_manifest_sha256: Optional[str] = None,
    expected_selected_manifest_sha256: Optional[str] = None,
    expected_d0_init_sha256: Optional[str] = None,
    expected_active_seed: Optional[int] = None,
) -> Optional[Path]:
    """Return the newest valid checkpoint for this arm, or None when none exist.

    A checkpoint that belongs to another arm, contract, NB13 generation, or
    manifest is rejected. Corrupt files among existing checkpoints fail closed
    instead of starting from D0.
    """
    root = Path(layout["checkpoints"])
    if not root.is_dir():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / "trainer_state.json").is_file()]
    if not candidates:
        return None
    valid: List[tuple] = []
    for path in candidates:
        fingerprint = assert_checkpoint_arm_isolation(
            path,
            arm=arm,
            expected_contract_hash=expected_contract_hash,
            expected_data_hash=expected_data_hash,
        )
        if expected_nb13_generation_id and str(fingerprint.get("nb13_generation_id") or "") != str(expected_nb13_generation_id):
            raise TrainingContractError(f"{arm} checkpoint NB13 generation does not match the pinned selection")
        if expected_manifest_sha256 and str(fingerprint.get("manifest_sha256") or "") != str(expected_manifest_sha256):
            raise TrainingContractError(f"{arm} checkpoint manifest hash does not match the pinned selection")
        if expected_selected_manifest_sha256 and str(fingerprint.get("selected_manifest_sha256") or "") != str(expected_selected_manifest_sha256):
            raise TrainingContractError(f"{arm} checkpoint selected-manifest hash does not match the pinned NB13 arm")
        if expected_d0_init_sha256 and str(fingerprint.get("d0_init_model_state_sha256") or "") != str(expected_d0_init_sha256):
            raise TrainingContractError(f"{arm} checkpoint D0 initialization does not match the frozen D0")
        if expected_active_seed is not None:
            recorded_seed = fingerprint.get("training_seed")
            if recorded_seed is None or int(recorded_seed) != int(expected_active_seed):
                raise TrainingContractError(f"{arm} checkpoint active_seed does not match this seed run")
        proof = prove_resume(
            path,
            arm=arm,
            expected_contract_hash=expected_contract_hash,
            expected_data_hash=expected_data_hash,
        )
        valid.append((int(proof["global_step"]), path.name, path))
    valid.sort()
    return valid[-1][2]


def enforce_target_truncation_gate(
    reports: Sequence[Mapping[str, Any]],
    *,
    max_truncation_rate: float,
) -> Dict[str, Any]:
    """Fail before training when any audited split exceeds the locked truncation rate."""
    from src.rq1_contract import sha256_json

    checked = []
    for report in reports:
        rate = float(report["truncation_rate"])
        threshold = float(max_truncation_rate)
        if rate > threshold:
            raise TrainingContractError(
                f"{report.get('split')} truncation_rate {rate} exceeds hard_max_truncation_rate {threshold}"
            )
        checked.append({
            "split": str(report.get("split")),
            "n_rows": int(report["n_rows"]),
            "n_truncated": int(report["n_truncated"]),
            "truncation_rate": rate,
            "max_target_token_length": int(report["max_target_token_length"]),
            "hard_max_truncation_rate": threshold,
        })
    return {"passed": True, "reports": checked, "audit_sha256": sha256_json(checked)}


def audit_split_target_truncation(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    split: str,
    max_target_length: int,
) -> Dict[str, Any]:
    """Count target tokens with truncation disabled, using the training tokenizer."""
    from src.direct_model import audit_target_tokenizer

    report = audit_target_tokenizer(
        tokenizer,
        list(texts),
        max_target_length=int(max_target_length),
        max_unk_rate=1.0,
        max_truncation_rate=1.0,
        split=str(split),
    )
    target = report["target"]
    return {
        "split": str(split),
        "n_rows": int(report["n"]),
        "n_truncated": int(target["truncation_count"]),
        "truncation_rate": float(target["truncation_rate"]),
        "max_target_token_length": int(target["max"]),
        "tokenizer_fingerprint": report.get("tokenizer_fingerprint"),
    }


def verify_pseudo_audio_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    u_clean_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Resolve every selected pseudo segment under the frozen NB11 root."""
    from src.rq2_pseudo_contract import resolve_nb11_segment_path

    resolved = []
    for row in rows:
        uid = str(row.get("segment_uid") or row.get("example_uid") or "")
        rel = str(row.get("segment_local_path") or row.get("source_path") or "")
        path = resolve_nb11_segment_path(rel, u_clean_dir)
        if not path.is_file():
            raise TrainingContractError(f"pseudo audio is missing for {uid or rel}")
        resolved.append(str(path))
    return {"n_rows": len(resolved), "u_clean_dir": str(Path(u_clean_dir).resolve())}


def prove_resume(
    checkpoint_path: Union[str, Path],
    *,
    arm: str,
    expected_contract_hash: str,
    expected_data_hash: str,
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    assert_g_test_blocked(path, allow=False)
    try:
        inspect = assert_checkpoint_complete_for_resume(path)
    except RuntimeError as exc:
        raise TrainingContractError(str(exc)) from exc
    fingerprint = assert_checkpoint_arm_isolation(
        path,
        arm=arm,
        expected_contract_hash=expected_contract_hash,
        expected_data_hash=expected_data_hash,
    )
    state = inspect.get("trainer_state") or {}
    if int(state.get("global_step") or 0) <= 0:
        raise TrainingContractError(f"{arm} resume refused: global_step is not restored")
    if not inspect.get("optimizer_ok"):
        raise TrainingContractError(f"{arm} resume refused: optimizer state missing")
    if not inspect.get("scheduler_ok"):
        raise TrainingContractError(f"{arm} resume refused: scheduler state missing")
    if not inspect.get("rng_ok"):
        raise TrainingContractError(f"{arm} resume refused: RNG state missing")
    return {
        "arm": arm,
        "checkpoint_name": path.name,
        "checkpoint_path": str(path),
        "checkpoint_fingerprint": checkpoint_model_state_sha256(path),
        "global_step": int(state.get("global_step") or 0),
        "expected_global_step": int(state.get("global_step") or 0),
        "optimizer_restored": True,
        "scheduler_restored": True,
        "rng_restored": True,
        "model_restored": True,
        "experiment_fingerprint_sha256": fingerprint.get("experiment_fingerprint_sha256"),
        "training_contract_sha256": expected_contract_hash,
    }


def prove_live_trainer_resume(
    trainer: Any,
    *,
    expected_global_step: int,
    checkpoint_path: Union[str, Path],
    arm: str,
    training_contract_sha256: str,
    expected_data_hash: str,
) -> Dict[str, Any]:
    """Prove the live Trainer actually restored the intended checkpoint state."""
    path = Path(checkpoint_path)
    file_proof = prove_resume(
        path,
        arm=arm,
        expected_contract_hash=training_contract_sha256,
        expected_data_hash=expected_data_hash,
    )
    state = getattr(trainer, "state", None)
    live_step = int(getattr(state, "global_step", -1) or -1) if state is not None else -1
    if live_step != int(expected_global_step):
        raise TrainingContractError(
            f"{arm} live resume proof failed: trainer.state.global_step={live_step} "
            f"expected={expected_global_step}"
        )
    optimizer = getattr(trainer, "optimizer", None)
    scheduler = getattr(trainer, "lr_scheduler", None)
    optimizer_restored = optimizer is not None
    scheduler_restored = scheduler is not None
    rng_restored = bool(file_proof.get("rng_restored")) and (path / "rng_state.pth").is_file()
    if hasattr(optimizer, "state_dict"):
        try:
            opt_state = optimizer.state_dict()
            optimizer_restored = bool(opt_state.get("state") or opt_state)
        except Exception as exc:
            raise TrainingContractError(f"{arm} optimizer state is not readable after resume: {exc}") from exc
    if hasattr(scheduler, "state_dict"):
        try:
            sch_state = scheduler.state_dict()
            scheduler_restored = "last_epoch" in sch_state or bool(sch_state)
        except Exception as exc:
            raise TrainingContractError(f"{arm} scheduler state is not readable after resume: {exc}") from exc
    if not optimizer_restored or not scheduler_restored or not rng_restored:
        raise TrainingContractError(
            f"{arm} live resume proof failed: optimizer={optimizer_restored} "
            f"scheduler={scheduler_restored} rng={rng_restored}"
        )
    proof = {
        "arm": arm,
        "checkpoint_path": str(path),
        "checkpoint_fingerprint": file_proof["checkpoint_fingerprint"],
        "restored_global_step": live_step,
        "expected_global_step": int(expected_global_step),
        "optimizer_restored": True,
        "scheduler_restored": True,
        "rng_restored": True,
        "training_contract_sha256": training_contract_sha256,
        "proof_status": "RESUME_PROVEN",
    }
    return proof


def write_resume_proof(layout: Mapping[str, Path], proof: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(proof)
    write_json(layout["resume_proof"], payload)
    return payload


def assert_training_reached_max_steps(trainer: Any, *, arm: str) -> Dict[str, Any]:
    """Fail closed unless Trainer returned only after reaching the planned max_steps."""
    state = getattr(trainer, "state", None)
    if state is None:
        raise TrainingContractError(f"{arm} trainer.state is missing after train()")
    try:
        final_global_step = int(getattr(state, "global_step", 0) or 0)
        expected_max_steps = int(getattr(state, "max_steps", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise TrainingContractError(f"{arm} trainer.state step fields are unreadable after train()") from exc
    if expected_max_steps <= 0:
        raise TrainingContractError(
            f"{arm} trainer.state.max_steps must be > 0 after train(); got {expected_max_steps}"
        )
    if final_global_step != expected_max_steps:
        raise TrainingContractError(
            f"{arm} training returned before reaching max_steps: "
            f"final_global_step={final_global_step} expected_max_steps={expected_max_steps}"
        )
    return {
        "final_global_step": final_global_step,
        "expected_max_steps": expected_max_steps,
        "reached_max_steps": True,
    }


def assert_training_complete_terminal_steps(payload: Mapping[str, Any], *, arm: str) -> None:
    """Trust-boundary step invariants for a TRAINING_COMPLETE proof payload."""
    if "expected_max_steps" not in payload:
        raise TrainingContractError(f"{arm} training_complete is missing expected_max_steps")
    if "reached_max_steps" not in payload:
        raise TrainingContractError(f"{arm} training_complete is missing reached_max_steps")
    if "final_global_step" not in payload:
        raise TrainingContractError(f"{arm} training_complete is missing final_global_step")
    try:
        final_global_step = int(payload["final_global_step"])
        expected_max_steps = int(payload["expected_max_steps"])
    except (TypeError, ValueError) as exc:
        raise TrainingContractError(f"{arm} training_complete step fields are invalid") from exc
    if expected_max_steps <= 0:
        raise TrainingContractError(
            f"{arm} training_complete expected_max_steps must be > 0; got {expected_max_steps}"
        )
    if payload.get("reached_max_steps") is not True:
        raise TrainingContractError(f"{arm} training_complete reached_max_steps must be exactly true")
    if final_global_step != expected_max_steps:
        raise TrainingContractError(
            f"{arm} training_complete final_global_step={final_global_step} "
            f"!= expected_max_steps={expected_max_steps}"
        )


def build_training_complete_proof(
    *,
    arm: str,
    training_contract_sha256: str,
    final_global_step: int,
    checkpoint_path: Union[str, Path],
    best_checkpoint: Optional[Mapping[str, Any]] = None,
    expected_max_steps: int,
    reached_max_steps: bool = True,
    mix_observed: Optional[Mapping[str, Any]] = None,
    eval_monitor_n: Optional[int] = None,
    full_validation_n: Optional[int] = None,
    eval_semantics: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct and validate the final completion payload. Does not publish."""
    path = Path(checkpoint_path)
    fingerprint = checkpoint_model_state_sha256(path)
    best = dict(best_checkpoint or {})
    best_name = str(best.get("checkpoint_name") or "")
    best_fp = str(best.get("checkpoint_fingerprint") or best.get("checkpoint_fingerprint_sha256") or "")
    payload: Dict[str, Any] = {
        "arm": arm,
        "training_contract_sha256": training_contract_sha256,
        "final_global_step": int(final_global_step),
        "expected_max_steps": int(expected_max_steps),
        "reached_max_steps": bool(reached_max_steps),
        "terminal_checkpoint": path.name,
        "terminal_checkpoint_fingerprint": fingerprint,
        "best_checkpoint": best_name,
        "best_checkpoint_fingerprint": best_fp,
        "best_validation_metric": best.get("validation_metric"),
        "status": STATUS_TRAINING_COMPLETE,
    }
    if not is_sha256(fingerprint):
        raise TrainingContractError(f"{arm} terminal checkpoint fingerprint is empty or invalid")
    if best_name and not is_sha256(best_fp):
        raise TrainingContractError(f"{arm} best checkpoint fingerprint is empty or invalid")
    if best_name and best_name == path.name and best_fp and best_fp != fingerprint:
        raise TrainingContractError("best checkpoint fingerprint collided with a different terminal checkpoint")
    for key in (
        "active_seed",
        "g_test_used",
        "nb13_generation_id",
        "d0_init_model_state_sha256",
        "selected_manifest_sha256",
        "global_step",
        "checkpoint_name",
    ):
        if key in best and best.get(key) not in (None, ""):
            payload[key] = best[key]
    if best.get("validation_metric") is not None:
        payload["validation_metric"] = best.get("validation_metric")
        payload["validation_metric_value"] = best.get("validation_metric")
    if best.get("g_test_used") is not None:
        payload["g_test_used"] = best.get("g_test_used")
    if mix_observed:
        payload["mix_observed"] = dict(mix_observed)
    if eval_monitor_n is not None:
        payload["eval_monitor_n"] = int(eval_monitor_n)
    if full_validation_n is not None:
        payload["full_validation_n"] = int(full_validation_n)
    if eval_semantics is not None:
        payload["eval_semantics"] = str(eval_semantics)
    assert_training_complete_terminal_steps(payload, arm=arm)
    return payload


def write_training_complete_proof(
    layout: Mapping[str, Path],
    *,
    arm: str,
    training_contract_sha256: str,
    final_global_step: int,
    checkpoint_path: Union[str, Path],
    best_checkpoint: Optional[Mapping[str, Any]] = None,
    expected_max_steps: int,
    reached_max_steps: bool = True,
    mix_observed: Optional[Mapping[str, Any]] = None,
    eval_monitor_n: Optional[int] = None,
    full_validation_n: Optional[int] = None,
    eval_semantics: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the final completion payload, then publish it exactly once."""
    payload = build_training_complete_proof(
        arm=arm,
        training_contract_sha256=training_contract_sha256,
        final_global_step=final_global_step,
        checkpoint_path=checkpoint_path,
        best_checkpoint=best_checkpoint,
        expected_max_steps=expected_max_steps,
        reached_max_steps=reached_max_steps,
        mix_observed=mix_observed,
        eval_monitor_n=eval_monitor_n,
        full_validation_n=full_validation_n,
        eval_semantics=eval_semantics,
    )
    write_json(layout["training_complete"], payload)
    return payload


def read_training_complete_proof(layout: Mapping[str, Path], *, arm: str, expected_contract_hash: str) -> Dict[str, Any]:
    path = Path(layout["training_complete"])
    if not path.is_file():
        raise TrainingContractError(f"{arm} training_complete.json is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != STATUS_TRAINING_COMPLETE:
        raise TrainingContractError(f"{arm} is not TRAINING_COMPLETE")
    if str(payload.get("arm") or "") != arm:
        raise TrainingContractError(f"{arm} training_complete arm mismatch")
    if str(payload.get("training_contract_sha256") or "") != expected_contract_hash:
        raise TrainingContractError(f"{arm} training_complete contract mismatch")
    if not is_sha256(payload.get("terminal_checkpoint_fingerprint")):
        raise TrainingContractError(f"{arm} training_complete terminal fingerprint is invalid")
    if payload.get("best_checkpoint") and not is_sha256(payload.get("best_checkpoint_fingerprint")):
        raise TrainingContractError(f"{arm} training_complete best fingerprint is invalid")
    terminal_name = str(payload.get("terminal_checkpoint") or "")
    best_name = str(payload.get("best_checkpoint") or "")
    if terminal_name and best_name and terminal_name == best_name:
        if str(payload.get("terminal_checkpoint_fingerprint") or "") != str(payload.get("best_checkpoint_fingerprint") or ""):
            raise TrainingContractError(f"{arm} terminal/best share a name but not a fingerprint")
    assert_training_complete_terminal_steps(payload, arm=arm)
    return payload


def select_best_checkpoint(
    history: Sequence[Mapping[str, Any]],
    *,
    metric: str = BEST_CHECKPOINT_METRIC,
    greater_is_better: bool = True,
) -> Dict[str, Any]:
    ranked = []
    for row in history:
        if any("g_test" in str(k).lower() for k in row.keys()) or str(row.get("split") or "") == "test":
            raise GTestFirewallError("G_test metrics cannot influence best-checkpoint selection")
        if str(row.get("split") or "validation") != "validation":
            raise GTestFirewallError("best checkpoint must use validation-only rows")
        try:
            score = float(row[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingContractError(f"validation history missing finite {metric}") from exc
        if not math.isfinite(score):
            raise TrainingContractError(f"validation {metric} is not finite: {score!r}")
        ranked.append(dict(row, _score=score, _step=int(row.get("global_step") or 0), _name=str(row.get("checkpoint_name") or "")))
    if not ranked:
        raise TrainingContractError("validation history is empty")
    ranked.sort(key=lambda row: ((-row["_score"] if greater_is_better else row["_score"]), -row["_step"], row["_name"]))
    best = ranked[0]
    return {
        "checkpoint_name": best["_name"],
        "global_step": best["_step"],
        "validation_metric": best["_score"],
        "metric_for_best_model": metric,
        "selection_rule": "validation_only",
        "tie_break": list(BEST_CHECKPOINT_TIE_BREAK),
        "g_test_used": False,
    }


def freeze_best_checkpoint(
    layout: Mapping[str, Path],
    selection: Mapping[str, Any],
    *,
    arm: str,
    contract_hash: str,
    data_hash: str,
) -> Dict[str, Any]:
    payload = dict(selection)
    payload["arm"] = arm
    payload["arm_training_contract_sha256"] = contract_hash
    payload["data_contract_sha256"] = data_hash
    ckpt = Path(layout["checkpoints"]) / str(payload["checkpoint_name"])
    ownership = assert_checkpoint_arm_isolation(
        ckpt,
        arm=arm,
        expected_contract_hash=contract_hash,
        expected_data_hash=data_hash,
    )
    fingerprint = str(payload.get("checkpoint_fingerprint") or payload.get("checkpoint_fingerprint_sha256") or "")
    live_fp = checkpoint_model_state_sha256(ckpt)
    if fingerprint and fingerprint != live_fp:
        raise TrainingContractError("best-checkpoint fingerprint does not match the selected checkpoint")
    fingerprint = live_fp
    if not is_sha256(fingerprint):
        raise TrainingContractError("best-checkpoint fingerprint is not a sha256")
    payload["checkpoint_fingerprint"] = fingerprint
    payload["experiment_fingerprint_sha256"] = ownership.get("experiment_fingerprint_sha256")
    payload["g_test_used"] = False
    write_json(layout["best_checkpoint"], payload)
    atomic_write_text(layout["latest"], str(payload["checkpoint_name"]) + "\n")
    complete_path = Path(layout["training_complete"])
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["best_checkpoint"] = str(payload["checkpoint_name"])
        complete["best_checkpoint_fingerprint"] = fingerprint
        complete["g_test_used"] = False
        complete["validation_metric"] = payload.get("validation_metric")
        complete["validation_metric_value"] = payload.get("validation_metric")
        if payload.get("global_step") is not None:
            complete["global_step"] = payload.get("global_step")
        for key in (
            "active_seed",
            "nb13_generation_id",
            "d0_init_model_state_sha256",
            "selected_manifest_sha256",
        ):
            if payload.get(key) not in (None, ""):
                complete[key] = payload[key]
        write_json(complete_path, complete)
    return payload


def checkpoint_model_state_sha256(checkpoint_path: Union[str, Path]) -> str:
    from src.rq2_final_contract import _model_state_sha256

    return _model_state_sha256(Path(checkpoint_path))


def validation_history_from_checkpoints(
    layout: Mapping[str, Path],
    *,
    arm: str,
    training_contract_sha256: str,
    data_contract_sha256: str,
    metric: str = BEST_CHECKPOINT_METRIC,
) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    ckpt_root = Path(layout["checkpoints"])
    if not ckpt_root.is_dir():
        return history
    for ckpt in sorted(p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")):
        state_path = ckpt / "trainer_state.json"
        if not state_path.is_file():
            continue
        try:
            assert_checkpoint_arm_isolation(
                ckpt,
                arm=arm,
                expected_contract_hash=training_contract_sha256,
                expected_data_hash=data_contract_sha256,
            )
            fingerprint = checkpoint_model_state_sha256(ckpt)
        except Exception:
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        score = None
        for row in reversed(list(state.get("log_history") or [])):
            if metric in row:
                score = row[metric]
                break
        if score is None:
            continue
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        history.append({
            "checkpoint_name": ckpt.name,
            "global_step": int(state.get("global_step") or 0),
            metric: numeric,
            "split": "validation",
            "checkpoint_fingerprint": fingerprint,
            "arm": arm,
            "arm_training_contract_sha256": training_contract_sha256,
            "data_contract_sha256": data_contract_sha256,
        })
    return history


def training_rows_to_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append({
            "record_uid": str(row["example_uid"]),
            "text_vi_norm": str(row["target_text_norm"]),
            "wav_local_path": str(row.get("source_path") or ""),
            "local_cache_relpath": str(row.get("source_path") or ""),
            "pcm16_sha256": str(row.get("audio_sha256") or ""),
            "group_id": str(row.get("group_id") or ""),
            "kind": str(row.get("kind") or ""),
            "split": "train",
        })
    return pd.DataFrame(records)


class GoldPseudoMixSampler:
    """Deterministic gold:pseudo slot sampler independent of unique pseudo count."""

    def __init__(
        self,
        gold_indices: Sequence[int],
        pseudo_indices: Sequence[int],
        *,
        gold_slots: int,
        pseudo_slots: int,
        seed: int,
        start: int = 0,
    ):
        if not gold_indices:
            raise TrainingContractError("gold:pseudo sampler requires supervised gold indices")
        if not pseudo_indices:
            raise TrainingContractError("gold:pseudo sampler requires pseudo-labeled indices")
        if int(gold_slots) <= 0 or int(pseudo_slots) <= 0:
            raise TrainingContractError("gold:pseudo sampler requires a configured positive slot ratio")
        self.gold_indices = [int(i) for i in gold_indices]
        self.pseudo_indices = [int(i) for i in pseudo_indices]
        self.gold_slots = int(gold_slots)
        self.pseudo_slots = int(pseudo_slots)
        self.seed = int(seed)
        self.start = int(start)
        cycle = self.gold_slots + self.pseudo_slots
        self.epoch_length = len(self.gold_indices) * cycle // self.gold_slots

    def __len__(self) -> int:
        return int(self.epoch_length)

    def schedule(self) -> List[int]:
        ordered = []
        gold_pos = 0
        pseudo_pos = 0
        rng = __import__("random").Random(self.seed)
        gold = list(self.gold_indices)
        pseudo = list(self.pseudo_indices)
        rng.shuffle(gold)
        rng.shuffle(pseudo)
        while len(ordered) < self.epoch_length:
            for _ in range(self.gold_slots):
                ordered.append(gold[gold_pos % len(gold)])
                gold_pos += 1
                if len(ordered) >= self.epoch_length:
                    return ordered
            for _ in range(self.pseudo_slots):
                ordered.append(pseudo[pseudo_pos % len(pseudo)])
                pseudo_pos += 1
                if len(ordered) >= self.epoch_length:
                    return ordered
        return ordered

    def __iter__(self):
        seq = self.schedule()[self.start:]
        yield from seq

    def observed_ratio(self) -> Dict[str, Any]:
        seq = self.schedule()
        gold_set = set(self.gold_indices)
        n_gold = sum(1 for idx in seq if idx in gold_set)
        n_pseudo = len(seq) - n_gold
        return {
            "gold_slots": self.gold_slots,
            "pseudo_slots": self.pseudo_slots,
            "configured_ratio": f"{self.gold_slots}:{self.pseudo_slots}",
            "n_gold_draws": n_gold,
            "n_pseudo_draws": n_pseudo,
            "observed_ratio": f"{n_gold}:{n_pseudo}",
            "epoch_length": len(seq),
        }

    def cursor_after(self, consumed: int) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "start": int(consumed),
            "next_index": self.schedule()[int(consumed) % self.epoch_length] if self.epoch_length else None,
            "gold_slots": self.gold_slots,
            "pseudo_slots": self.pseudo_slots,
        }


def mix_indices_from_frame(frame: pd.DataFrame) -> Dict[str, List[int]]:
    kinds = frame["kind"].astype(str).tolist() if "kind" in frame.columns else ["supervised_g_train"] * len(frame)
    gold = [i for i, kind in enumerate(kinds) if kind != "pseudo_nb13"]
    pseudo = [i for i, kind in enumerate(kinds) if kind == "pseudo_nb13"]
    return {"gold": gold, "pseudo": pseudo}


def build_gold_pseudo_sampler(
    train_frame: pd.DataFrame,
    training_contract: Mapping[str, Any],
    *,
    start: int = 0,
) -> GoldPseudoMixSampler:
    mix = require_configured_mix_policy(training_contract, for_real_training=True)
    seed_policy = require_configured_seed_policy(training_contract, for_real_training=True)
    if training_contract.get("active_seed") is None:
        raise TrainingContractError("training contract is missing active_seed; it does not default to seeds[0]")
    active_seed = int(training_contract["active_seed"])
    declared = [int(seed) for seed in seed_policy.get("seed_policy_seeds") or []]
    if active_seed not in declared:
        raise TrainingContractError(f"active_seed {active_seed} is not in the declared seed list")
    parts = mix_indices_from_frame(train_frame)
    return GoldPseudoMixSampler(
        parts["gold"],
        parts["pseudo"],
        gold_slots=int(mix["gold_slots"]),
        pseudo_slots=int(mix["pseudo_slots"]),
        seed=active_seed,
        start=int(start),
    )


def freeze_speech_feature_encoder(model: Any) -> None:
    """Re-apply the speech-encoder freeze after D0 load and after resume."""
    enc = model.encoder
    if hasattr(enc, "freeze_feature_encoder"):
        enc.freeze_feature_encoder()
    elif hasattr(enc, "feature_extractor") and hasattr(enc.feature_extractor, "_freeze_parameters"):
        enc.feature_extractor._freeze_parameters()
    else:
        raise TrainingContractError("cannot locate the speech feature encoder freeze method")
    feature = getattr(enc, "feature_extractor", None)
    if feature is None:
        raise TrainingContractError("speech feature encoder is missing after freeze")
    trainable = [name for name, param in feature.named_parameters() if param.requires_grad]
    if trainable:
        raise TrainingContractError("feature encoder remains trainable: " + ", ".join(trainable[:5]))


def build_arm_model(
    training_contract: Mapping[str, Any],
    *,
    processors: Optional[tuple] = None,
    d0_checkpoint_dir: Optional[Union[str, Path]] = None,
):
    from src.direct_model import load_direct_processors

    if str(training_contract.get("init_policy") or "") == REJECTED_PUBLIC_PRETRAINED_INIT_POLICY:
        raise TrainingContractError("public pretrained XLS-R+mBART reinitialization is rejected")
    if str(training_contract.get("init_policy") or "") != AUGMENTATION_INIT_POLICY:
        raise TrainingContractError("trainable arms must continue from the frozen D0 checkpoint")
    if processors is None:
        feature_extractor, tokenizer = load_direct_processors(
            encoder_id=str(training_contract["encoder_id"]),
            encoder_revision=str(training_contract["encoder_revision"]),
            decoder_id=str(training_contract["decoder_id"]),
            decoder_revision=str(training_contract["decoder_revision"]),
            target_lang=str(training_contract["target_lang"]),
        )
    else:
        feature_extractor, tokenizer = processors
    if d0_checkpoint_dir is None:
        raise TrainingContractError("frozen D0 checkpoint directory is required to initialize RQ2 arms")
    checkpoint = Path(d0_checkpoint_dir)
    if not checkpoint.is_dir():
        raise TrainingContractError(f"frozen D0 checkpoint is missing: {checkpoint.name}")
    live_fp = checkpoint_model_state_sha256(checkpoint)
    expected = str(training_contract.get("d0_init_model_state_sha256") or "")
    if not is_sha256(expected) or live_fp != expected:
        raise TrainingContractError("D0 init model-state fingerprint does not match the training contract")
    from transformers import SpeechEncoderDecoderModel

    model = SpeechEncoderDecoderModel.from_pretrained(str(checkpoint))
    if training_contract.get("freeze_feature_encoder") is True:
        freeze_speech_feature_encoder(model)
    elif training_contract.get("freeze_feature_encoder") is not True:
        raise TrainingContractError("trainable arms require freeze_feature_encoder=True")
    return model, feature_extractor, tokenizer


def build_arm_datasets(
    *,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_extractor: Any,
    tokenizer: Any,
    training_contract: Mapping[str, Any],
    audio_cache_roots: Sequence[Union[str, Path]] = (),
):
    from src.direct_dataset import DirectSpeechTranslationDataset

    sample_rate = int(training_contract.get("sample_rate") or 16000)
    max_target = int(training_contract.get("max_target_length") or 256)
    train_ds = DirectSpeechTranslationDataset(
        train_frame,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        audio_cache_roots=list(audio_cache_roots),
        sample_rate=sample_rate,
        max_target_length=max_target,
    )
    val_ds = DirectSpeechTranslationDataset(
        validation_frame,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        audio_cache_roots=list(audio_cache_roots),
        sample_rate=sample_rate,
        max_target_length=max_target,
    )
    return {"train": train_ds, "validation": val_ds}


def build_validation_metric_context(validation_frame: pd.DataFrame, tokenizer: Any) -> Dict[str, Any]:
    """Canonical untruncated validation references, aligned to ordered UIDs."""
    from src.rq2_final_evaluate import canonical_references

    if "record_uid" not in validation_frame.columns:
        raise TrainingContractError("validation frame missing record_uid")
    work = validation_frame.reset_index(drop=True)
    uids = work["record_uid"].astype(str).tolist()
    if len(uids) != len(set(uids)):
        raise TrainingContractError("validation frame contains duplicate record_uid values")
    if not uids:
        raise TrainingContractError("validation frame is empty")
    refs = canonical_references(work)
    if len(refs) != len(uids):
        raise TrainingContractError("validation UID/reference alignment drifted")
    missing = [uid for uid, ref in zip(uids, refs) if not str(ref or "").strip()]
    if missing:
        raise TrainingContractError(f"validation missing canonical reference_vi for {missing[:5]}")
    return {
        "ordered_uids": uids,
        "ordered_refs": refs,
        "reference_map": dict(zip(uids, refs)),
        "tokenizer": tokenizer,
    }


def compute_validation_metrics(eval_prediction: Any, context: Mapping[str, Any]) -> Dict[str, float]:
    """Decode hypotheses only. References come from canonical untruncated text_vi_norm."""
    import numpy as np

    from src.metrics import mt_corpus_metrics

    tokenizer = context["tokenizer"]
    ordered_uids = list(context["ordered_uids"])
    ordered_refs = list(context["ordered_refs"])
    if hasattr(eval_prediction, "predictions"):
        preds = eval_prediction.predictions
    elif isinstance(eval_prediction, (tuple, list)):
        preds = eval_prediction[0]
    else:
        preds = eval_prediction
    if isinstance(preds, tuple):
        preds = preds[0]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise TrainingContractError("tokenizer.pad_token_id is required for validation metrics")
    preds = np.asarray(preds)
    preds = np.where(preds != -100, preds, int(pad_id))
    hyps = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if len(hyps) != len(ordered_uids):
        raise TrainingContractError(
            f"validation prediction count {len(hyps)} does not match canonical UID count {len(ordered_uids)}"
        )
    metrics = mt_corpus_metrics(list(hyps), ordered_refs)
    if not metrics.get("finite"):
        raise TrainingContractError("validation SacreBLEU/chrF++ is not finite")
    return {
        "sacrebleu": float(metrics["sacrebleu"]),
        "chrfpp": float(metrics["chrfpp"]),
        "monitor_n": float(len(ordered_uids)),
    }


def build_arm_collator(feature_extractor: Any, tokenizer: Any):
    from src.direct_dataset import DirectDataCollator

    return DirectDataCollator(feature_extractor, tokenizer)


def build_training_arguments(
    *,
    output_dir: Union[str, Path],
    training_contract: Mapping[str, Any],
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
):
    from transformers import Seq2SeqTrainingArguments

    fp16 = bool(training_contract.get("fp16"))
    bf16 = bool(training_contract.get("bf16"))
    try:
        import torch

        if fp16 and not torch.cuda.is_available():
            fp16 = False
    except Exception:
        fp16 = False
    try:
        save_steps = int(training_contract["save_steps"])
        eval_steps = int(training_contract["eval_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingContractError("training contract must explicitly set save_steps and eval_steps") from exc
    if save_steps != LOCKED_RQ2_SAVE_STEPS or eval_steps != LOCKED_RQ2_EVAL_STEPS:
        raise TrainingContractError(
            f"training contract must lock save_steps={LOCKED_RQ2_SAVE_STEPS} and "
            f"eval_steps={LOCKED_RQ2_EVAL_STEPS}"
        )
    eval_strategy = str(training_contract.get("eval_strategy") or "")
    save_strategy = str(training_contract.get("save_strategy") or "")
    if eval_strategy != LOCKED_RQ2_EVAL_STRATEGY or save_strategy != LOCKED_RQ2_SAVE_STRATEGY:
        raise TrainingContractError("training contract must lock eval_strategy/save_strategy='steps'")
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_contract["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_contract.get("per_device_eval_batch_size") or 1),
        gradient_accumulation_steps=int(training_contract["gradient_accumulation_steps"]),
        learning_rate=float(training_contract["learning_rate"]),
        warmup_ratio=float(training_contract["warmup_ratio"]),
        weight_decay=float(training_contract["weight_decay"]),
        num_train_epochs=float(training_contract["num_train_epochs"]),
        seed=int(training_contract["seed"]),
        data_seed=int(training_contract.get("dataloader_seed") or training_contract["seed"]),
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=bool(training_contract.get("gradient_checkpointing")),
        optim=str(training_contract.get("optimizer") or "adamw_torch"),
        lr_scheduler_type=str(training_contract.get("lr_scheduler_type") or "linear"),
        max_grad_norm=float(training_contract.get("max_grad_norm") or 1.0),
        report_to=[],
        remove_unused_columns=True,
        predict_with_generate=True,
        generation_max_length=int(training_contract.get("generation_max_length") or 256),
        generation_num_beams=int(training_contract.get("num_beams") or 4),
        metric_for_best_model=str(training_contract.get("metric_for_best_model") or BEST_CHECKPOINT_METRIC),
        greater_is_better=bool(training_contract.get("greater_is_better", True)),
        load_best_model_at_end=True,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=int(training_contract.get("save_total_limit") or 2),
        logging_steps=int(training_contract.get("logging_steps") or 50),

        # Runtime-only input pipeline tuning.
        # These settings do not alter the scientific training contract,
        # data ordering policy, model, optimizer, loss, or seed policy.
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
    )
    if resume_from_checkpoint:
        args.resume_from_checkpoint = str(resume_from_checkpoint)
    return args


def assert_monitor_frame_matches_contract(
    monitor_frame: pd.DataFrame,
    training_contract: Mapping[str, Any],
    *,
    validation_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Fail closed unless the Trainer eval frame is the locked frozen monitor."""
    from src.direct_full_train import monitor_manifest

    if monitor_frame is None or not isinstance(monitor_frame, pd.DataFrame) or monitor_frame.empty:
        raise TrainingContractError("train-time eval requires the frozen validation monitor frame")
    try:
        expected_size = int(training_contract["monitor_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingContractError("training contract is missing monitor_size") from exc
    if len(monitor_frame) != expected_size:
        raise TrainingContractError(
            f"train-time eval monitor has {len(monitor_frame)} rows, contract expects {expected_size}"
        )
    if expected_size >= 11112:
        raise TrainingContractError("train-time eval monitor must not be the full G_validation frame")
    live = monitor_manifest(monitor_frame)
    for key, contract_key in (
        ("n", "monitor_size"),
        ("uid_set_hash", "monitor_uid_set_hash"),
        ("pair_hash", "monitor_pair_hash"),
        ("ordered_row_hash", "monitor_ordered_row_hash"),
    ):
        if key == "n":
            if int(live["n"]) != expected_size:
                raise TrainingContractError("train-time eval monitor size drifted from the training contract")
            continue
        expected = str(training_contract.get(contract_key) or "")
        if not is_sha256(expected) or str(live.get(key) or "") != expected:
            raise TrainingContractError(f"train-time eval monitor {contract_key} does not match the training contract")
    if validation_frame is not None:
        pop = set(validation_frame["record_uid"].astype(str).tolist())
        uids = monitor_frame["record_uid"].astype(str).tolist()
        if len(uids) != len(set(uids)):
            raise TrainingContractError("train-time eval monitor contains duplicate record_uid values")
        missing = [uid for uid in uids if uid not in pop]
        if missing:
            raise TrainingContractError("train-time eval monitor UID is outside frozen G_validation")
    return live


def build_trainer(
    *,
    arm: str,
    layout: Mapping[str, Path],
    training_contract: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    monitor_frame: Optional[pd.DataFrame] = None,
    audio_cache_roots: Sequence[Union[str, Path]] = (),
    processors: Optional[tuple] = None,
    model: Any = None,
    resume_checkpoint: Optional[Union[str, Path]] = None,
    d0_checkpoint_dir: Optional[Union[str, Path]] = None,
):
    """Construct the RQ1 Direct Seq2Seq trainer for one RQ2 arm. Does not call train().

    ``validation_frame`` is the full frozen Direct-eligible G_validation and is
    retained for identity checks. Trainer ``eval_dataset`` uses ``monitor_frame``,
    the frozen RQ1 Direct 256-row validation monitor.
    """
    if arm not in TRAINABLE_ARMS:
        raise TrainingContractError(f"build_trainer cannot train arm {arm!r}")
    if str(training_contract.get("init_policy") or "") == REJECTED_PUBLIC_PRETRAINED_INIT_POLICY:
        raise TrainingContractError("public pretrained XLS-R+mBART reinitialization is rejected")
    if monitor_frame is None:
        raise TrainingContractError(
            "build_trainer requires monitor_frame=frozen RQ1 Direct validation monitor; "
            "full G_validation must not be used as Trainer eval_dataset"
        )
    if validation_frame is None or not isinstance(validation_frame, pd.DataFrame) or validation_frame.empty:
        raise TrainingContractError("build_trainer requires the full frozen G_validation frame for identity checks")
    assert_monitor_frame_matches_contract(
        monitor_frame,
        training_contract,
        validation_frame=validation_frame,
    )
    mix_sampler = None
    if "kind" in train_frame.columns and (train_frame["kind"].astype(str) == "pseudo_nb13").any():
        mix_sampler = build_gold_pseudo_sampler(train_frame, training_contract)
    if processors is None and model is None:
        model, feature_extractor, tokenizer = build_arm_model(
            training_contract, d0_checkpoint_dir=d0_checkpoint_dir,
        )
    elif processors is not None and model is None:
        model, feature_extractor, tokenizer = build_arm_model(
            training_contract, processors=processors, d0_checkpoint_dir=d0_checkpoint_dir,
        )
    else:
        feature_extractor, tokenizer = processors or (None, None)
        if feature_extractor is None or tokenizer is None:
            raise TrainingContractError("build_trainer needs processors when a model is injected")
    datasets = build_arm_datasets(
        train_frame=train_frame,
        validation_frame=monitor_frame,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        training_contract=training_contract,
        audio_cache_roots=audio_cache_roots,
    )
    collator = build_arm_collator(feature_extractor, tokenizer)
    args = build_training_arguments(
        output_dir=layout["checkpoints"],
        training_contract=training_contract,
        resume_from_checkpoint=resume_checkpoint,
    )
    from src.direct_full_train import make_resume_safe_seq2seq_trainer_cls

    metric_context = build_validation_metric_context(monitor_frame, tokenizer)

    def _compute_metrics(eval_prediction):
        return compute_validation_metrics(eval_prediction, metric_context)

    BaseTrainer = make_resume_safe_seq2seq_trainer_cls()

    class Rq2ResumeProofTrainer(BaseTrainer):  # type: ignore[misc,valid-type]
        def _get_train_sampler(self, *args, **kwargs):
            sampler = getattr(self, "_rq2_mix_sampler", None)
            if sampler is not None:
                return sampler
            return super()._get_train_sampler(*args, **kwargs)

        def _load_from_checkpoint(self, resume_from_checkpoint, *args, **kwargs):
            result = super()._load_from_checkpoint(resume_from_checkpoint, *args, **kwargs)
            if training_contract.get("freeze_feature_encoder") is True:
                freeze_speech_feature_encoder(self.model)
            expected = getattr(self, "_rq2_expected_resume_step", None)
            if expected is None or not resume_from_checkpoint:
                return result
            proof = prove_live_trainer_resume(
                self,
                expected_global_step=int(expected),
                checkpoint_path=resume_from_checkpoint,
                arm=str(getattr(self, "_rq2_arm", arm)),
                training_contract_sha256=str(training_contract["arm_training_contract_sha256"]),
                expected_data_hash=str(fingerprint.get("data_manifest_sha256") or training_contract["data_manifest_sha256"]),
            )
            sampler = getattr(self, "_rq2_mix_sampler", None)
            if sampler is not None:
                consumed = int(expected) * int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
                proof["mix_cursor"] = sampler.cursor_after(consumed)
                proof["mix_observed"] = sampler.observed_ratio()
            self._rq2_live_resume_proof = proof
            write_resume_proof(layout, proof)
            return result

    heartbeat = DurableProgressCallback(
        layout,
        arm=arm,
        seed=training_contract.get("active_seed", training_contract.get("seed")),
        every_n_steps=50,
        eval_steps=int(training_contract["eval_steps"]),
        save_steps=int(training_contract["save_steps"]),
    )
    trainer = Rq2ResumeProofTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=[_as_trainer_callback(heartbeat)],
    )
    # transformers 4.57.x treats SpeechEncoderDecoderModel(**kwargs) as accepting
    # loss kwargs and injects num_items_in_batch. That kwarg is forwarded to the
    # MBART decoder, whose forward() does not accept it.
    trainer.model_accepts_loss_kwargs = False
    trainer._rq2_arm = arm
    trainer._rq2_fingerprint = dict(fingerprint)
    trainer._rq2_metric_context = metric_context
    trainer._rq2_mix_sampler = mix_sampler
    trainer._rq2_eval_monitor_n = int(len(monitor_frame))
    trainer._rq2_full_validation_n = int(len(validation_frame))
    trainer._rq2_eval_semantics = EVAL_SEMANTICS_NOTE
    if mix_sampler is not None:
        trainer._rq2_mix_observed = mix_sampler.observed_ratio()
    if resume_checkpoint is not None:
        try:
            inspect = json.loads((Path(resume_checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
            trainer._rq2_expected_resume_step = int(inspect.get("global_step") or 0)
            write_progress(layout, {
                "arm": arm,
                "seed": training_contract.get("active_seed", training_contract.get("seed")),
                "status": STATUS_RESUME_AVAILABLE,
                "global_step": trainer._rq2_expected_resume_step,
                "max_steps": None,
                "percent_complete": None,
                "epoch": inspect.get("epoch"),
                "elapsed_seconds": 0.0,
                "estimated_remaining_seconds": None,
                "steps_per_second": 0.0,
                "eval_steps": int(training_contract["eval_steps"]),
                "save_steps": int(training_contract["save_steps"]),
                "last_eval_step": None,
                "last_save_step": None,
                "last_eval_metrics": None,
                "last_update_utc": _utc_now(),
            })
        except Exception as exc:
            raise TrainingContractError(f"{arm} resume checkpoint trainer_state.json is unreadable") from exc
    return trainer


def train_or_resume_arm(
    *,
    arm: str,
    flags: Nb14Flags,
    layout: Mapping[str, Path],
    training_contract: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    train_frame: Optional[pd.DataFrame] = None,
    validation_frame: Optional[pd.DataFrame] = None,
    monitor_frame: Optional[pd.DataFrame] = None,
    audio_cache_roots: Sequence[Union[str, Path]] = (),
    resume_checkpoint: Optional[Union[str, Path]] = None,
    trainer_factory: Optional[Callable[..., Any]] = None,
    processors: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Real D-Random / D-Quality training entry. D0 is bind-only."""
    ensure_arm_root(layout["root"])
    write_arm_contracts(layout, training_contract, fingerprint)
    write_json(layout["checkpoints"] / "full_experiment_fingerprint.json", dict(fingerprint))
    write_progress(layout, {
        "arm": arm,
        "seed": training_contract.get("active_seed", training_contract.get("seed")),
        "status": "PREPARING",
        "global_step": 0,
        "max_steps": None,
        "percent_complete": 0.0,
        "epoch": 0.0,
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "steps_per_second": 0.0,
        "eval_steps": training_contract.get("eval_steps"),
        "save_steps": training_contract.get("save_steps"),
        "last_eval_step": None,
        "last_save_step": None,
        "last_eval_metrics": None,
        "last_update_utc": _utc_now(),
    })
    if arm == ARM_D0:
        return bind_d0_arm(layout=layout, flags=flags, training_contract=training_contract, fingerprint=fingerprint)
    if arm not in TRAINABLE_ARMS:
        raise TrainingContractError(f"cannot train arm {arm!r}")
    if flags.run_real_training is True:
        require_configured_mix_policy(training_contract, for_real_training=True)
        require_configured_seed_policy(training_contract, for_real_training=True)
        if str(training_contract.get("init_policy") or "") == REJECTED_PUBLIC_PRETRAINED_INIT_POLICY:
            raise TrainingContractError("public pretrained XLS-R+mBART reinitialization is rejected")
        if monitor_frame is None:
            raise TrainingContractError(
                "real training requires monitor_frame=frozen RQ1 Direct validation monitor"
            )
    proof = None
    live_proof = None
    if resume_checkpoint is not None:
        proof = prove_resume(
            resume_checkpoint,
            arm=arm,
            expected_contract_hash=training_contract["arm_training_contract_sha256"],
            expected_data_hash=training_contract["data_manifest_sha256"],
        )
        write_resume_proof(layout, {**proof, "proof_status": STATUS_RESUME_AVAILABLE})
        write_progress(layout, {
            "arm": arm,
            "seed": training_contract.get("active_seed", training_contract.get("seed")),
            "status": STATUS_RESUME_AVAILABLE,
            "global_step": proof["global_step"],
            "max_steps": None,
            "percent_complete": None,
            "epoch": proof.get("epoch"),
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "steps_per_second": 0.0,
            "eval_steps": training_contract.get("eval_steps"),
            "save_steps": training_contract.get("save_steps"),
            "last_eval_step": None,
            "last_save_step": None,
            "last_eval_metrics": None,
            "last_update_utc": _utc_now(),
        })
    if flags.run_real_training is not True:
        write_progress(layout, {"status": "skipped_real_training", "arm": arm})
        return {
            "trained": False,
            "resumed": proof is not None,
            "proof": proof,
            "reason": "RUN_REAL_TRAINING=False",
            "lifecycle": STATUS_RESUME_AVAILABLE if proof is not None else STATUS_NOT_STARTED,
        }
    factory = trainer_factory or build_trainer
    write_progress(layout, {
        "arm": arm,
        "seed": training_contract.get("active_seed", training_contract.get("seed")),
        "status": STATUS_TRAINING_RUNNING,
        "global_step": int((proof or {}).get("global_step") or 0),
        "max_steps": None,
        "percent_complete": None,
        "epoch": (proof or {}).get("epoch"),
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "steps_per_second": 0.0,
        "eval_steps": training_contract.get("eval_steps"),
        "save_steps": training_contract.get("save_steps"),
        "last_eval_step": None,
        "last_save_step": None,
        "last_eval_metrics": None,
        "last_update_utc": _utc_now(),
    })
    d0_init = resolve_frozen_d0_init(flags)
    if d0_init["model_state_sha256"] != str(training_contract.get("d0_init_model_state_sha256") or ""):
        raise TrainingContractError("frozen D0 init fingerprint does not match the arm training contract")
    if d0_init["checkpoint_fingerprint_sha256"] != str(training_contract.get("d0_checkpoint_fingerprint_sha256") or ""):
        raise TrainingContractError("frozen D0 checkpoint fingerprint does not match the arm training contract")
    d0_checkpoint_dir = d0_init["checkpoint_dir"]
    trainer_kwargs = {
        "arm": arm,
        "layout": layout,
        "training_contract": training_contract,
        "fingerprint": fingerprint,
        "train_frame": train_frame,
        "validation_frame": validation_frame,
        "monitor_frame": monitor_frame,
        "audio_cache_roots": audio_cache_roots,
        "processors": processors,
        "resume_checkpoint": resume_checkpoint,
        "d0_checkpoint_dir": d0_checkpoint_dir,
    }
    trainer = None
    try:
        try:
            trainer = factory(**trainer_kwargs)
        except TypeError as exc:
            if "monitor_frame" not in str(exc):
                raise
            # Older injected trainer_factory fixtures may not accept monitor_frame.
            trainer_kwargs.pop("monitor_frame", None)
            trainer = factory(**trainer_kwargs)
        result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
        live_proof = getattr(trainer, "_rq2_live_resume_proof", None)
        if resume_checkpoint is not None:
            expected_step = int((proof or {}).get("global_step") or 0)
            if live_proof is None:
                live_proof = prove_live_trainer_resume(
                    trainer,
                    expected_global_step=expected_step,
                    checkpoint_path=resume_checkpoint,
                    arm=arm,
                    training_contract_sha256=training_contract["arm_training_contract_sha256"],
                    expected_data_hash=training_contract["data_manifest_sha256"],
                )
            write_resume_proof(layout, live_proof)
            if live_proof.get("proof_status") != "RESUME_PROVEN":
                raise TrainingContractError(f"{arm} resume restore was not proven")
        terminal_proof = assert_training_reached_max_steps(trainer, arm=arm)
        final_step = int(terminal_proof["final_global_step"])
        expected_max_steps = int(terminal_proof["expected_max_steps"])
        terminal = _terminal_checkpoint_path(layout, trainer=trainer, resume_checkpoint=resume_checkpoint)
        best_payload = {}
        best_path = layout.get("best_checkpoint")
        if best_path and Path(best_path).is_file():
            best_payload = json.loads(Path(best_path).read_text(encoding="utf-8"))
        mix_observed = getattr(trainer, "_rq2_mix_observed", None)
        # Build+validate the full proof in memory first. Publish training_complete.json
        # exactly once only after every required post-train field is present.
        complete = write_training_complete_proof(
            layout,
            arm=arm,
            training_contract_sha256=training_contract["arm_training_contract_sha256"],
            final_global_step=final_step,
            checkpoint_path=terminal,
            best_checkpoint=best_payload,
            expected_max_steps=expected_max_steps,
            reached_max_steps=True,
            mix_observed=dict(mix_observed) if mix_observed else None,
            eval_monitor_n=int(
                getattr(trainer, "_rq2_eval_monitor_n", training_contract.get("monitor_size") or 0) or 0
            ),
            full_validation_n=int(getattr(trainer, "_rq2_full_validation_n", 0) or 0),
            eval_semantics=str(getattr(trainer, "_rq2_eval_semantics", "") or EVAL_SEMANTICS_NOTE),
        )
        write_progress(layout, {
            "arm": arm,
            "seed": training_contract.get("active_seed", training_contract.get("seed")),
            "status": STATUS_TRAINING_COMPLETE,
            "global_step": final_step,
            "max_steps": expected_max_steps,
            "percent_complete": 100.0,
            "epoch": getattr(getattr(trainer, "state", None), "epoch", None),
            "elapsed_seconds": None,
            "estimated_remaining_seconds": 0.0,
            "steps_per_second": None,
            "eval_steps": training_contract.get("eval_steps"),
            "save_steps": training_contract.get("save_steps"),
            "last_eval_step": None,
            "last_save_step": None,
            "last_eval_metrics": None,
            "last_update_utc": _utc_now(),
        })
        return {
            "trained": True,
            "resumed": proof is not None,
            "proof": proof,
            "live_resume_proof": live_proof,
            "training_complete": complete,
            "result": result,
            "trainer": trainer,
            "lifecycle": STATUS_TRAINING_COMPLETE,
        }
    except Exception as exc:
        failed_step = 0
        failed_max = None
        if trainer is not None:
            state = getattr(trainer, "state", None)
            failed_step = int(getattr(state, "global_step", 0) or 0)
            try:
                failed_max = int(getattr(state, "max_steps", 0) or 0)
            except (TypeError, ValueError):
                failed_max = None
        elif proof is not None:
            failed_step = int(proof.get("global_step") or 0)
        write_progress(layout, {
            "arm": arm,
            "seed": training_contract.get("active_seed", training_contract.get("seed")),
            "status": STATUS_FAILED_TRAINING,
            "global_step": failed_step,
            "max_steps": failed_max,
            "percent_complete": None,
            "epoch": None,
            "elapsed_seconds": None,
            "estimated_remaining_seconds": None,
            "steps_per_second": None,
            "eval_steps": training_contract.get("eval_steps"),
            "save_steps": training_contract.get("save_steps"),
            "last_eval_step": None,
            "last_save_step": None,
            "last_eval_metrics": None,
            "failure_reason": str(exc)[:800],
            "error": type(exc).__name__,
            "last_update_utc": _utc_now(),
        })
        raise


def _latest_checkpoint_path(
    layout: Mapping[str, Path],
    *,
    resume_checkpoint: Optional[Union[str, Path]] = None,
) -> Path:
    ckpt_root = Path(layout["checkpoints"])
    named = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if named:
        return named[-1]
    if resume_checkpoint is not None and Path(resume_checkpoint).is_dir():
        return Path(resume_checkpoint)
    raise TrainingContractError("no terminal checkpoint exists after train()")


def _terminal_checkpoint_path(
    layout: Mapping[str, Path],
    *,
    trainer: Any,
    resume_checkpoint: Optional[Union[str, Path]] = None,
) -> Path:
    """Latest training checkpoint, never the older best-validation checkpoint."""
    del trainer  # best_model_checkpoint is not the terminal checkpoint
    return _latest_checkpoint_path(layout, resume_checkpoint=resume_checkpoint)


def resume_or_train_arm(
    *,
    arm: str,
    flags: Nb14Flags,
    layout: Mapping[str, Path],
    training_contract: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    trainer_fn: Optional[TrainerFn] = None,
    trainer_factory: Optional[Callable[..., Any]] = None,
    resume_checkpoint: Optional[Union[str, Path]] = None,
    train_frame: Optional[pd.DataFrame] = None,
    validation_frame: Optional[pd.DataFrame] = None,
    monitor_frame: Optional[pd.DataFrame] = None,
    audio_cache_roots: Sequence[Union[str, Path]] = (),
    processors: Optional[tuple] = None,
) -> Dict[str, Any]:
    if trainer_fn is not None:
        ensure_arm_root(layout["root"])
        write_arm_contracts(layout, training_contract, fingerprint)
        if flags.run_real_training is not True:
            return {"trained": False, "resumed": False, "reason": "RUN_REAL_TRAINING=False"}
        result = trainer_fn(arm=arm, layout=layout, training_contract=training_contract, fingerprint=fingerprint)
        return {"trained": True, "resumed": bool(resume_checkpoint), "result": result}
    return train_or_resume_arm(
        arm=arm,
        flags=flags,
        layout=layout,
        training_contract=training_contract,
        fingerprint=fingerprint,
        train_frame=train_frame,
        validation_frame=validation_frame,
        monitor_frame=monitor_frame,
        audio_cache_roots=audio_cache_roots,
        resume_checkpoint=resume_checkpoint,
        trainer_factory=trainer_factory,
        processors=processors,
    )


def assert_no_cross_arm_resume(source_arm: str, dest_arm: str) -> None:
    if source_arm != dest_arm:
        raise ArmIsolationError(f"cannot resume {dest_arm} from a {source_arm} checkpoint")
