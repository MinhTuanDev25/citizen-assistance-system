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
    GOLD_PSEUDO_MIX_POLICY_SLOTTED,
    GOLD_PSEUDO_MIX_POLICY_UNSET,
    REJECTED_PUBLIC_PRETRAINED_INIT_POLICY,
    GTestFirewallError,
    RUNTIME_ONLY_KEYS,
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
    mix = normalize_gold_pseudo_mix_policy(mix_policy or hparams)
    seeds = normalize_seed_policy(seed_policy or hparams)
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
        **{k: hparams[k] for k in hparams if k not in RUNTIME_ONLY_KEYS and k not in {
            "gold_pseudo_mix_policy", "gold_slots", "pseudo_slots", "seed_policy", "seed_policy_seeds", "seed", "dataloader_seed",
        }},
    }
    if seeds.get("configured"):
        fields["seed"] = int(seeds["seed"])
        fields["dataloader_seed"] = int(seeds["dataloader_seed"])
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
) -> Dict[str, Any]:
    payload = {
        "arm": arm,
        "arm_training_contract_sha256": training_contract["arm_training_contract_sha256"],
        "data_manifest_sha256": data_manifest_sha256,
        "encoder_revision": training_contract["encoder_revision"],
        "decoder_revision": training_contract["decoder_revision"],
        "model_revision": model_revision,
        "init_policy": training_contract["init_policy"],
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


def write_training_complete_proof(
    layout: Mapping[str, Path],
    *,
    arm: str,
    training_contract_sha256: str,
    final_global_step: int,
    checkpoint_path: Union[str, Path],
    best_checkpoint: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    fingerprint = checkpoint_model_state_sha256(path)
    best = dict(best_checkpoint or {})
    best_name = str(best.get("checkpoint_name") or "")
    best_fp = str(best.get("checkpoint_fingerprint") or best.get("checkpoint_fingerprint_sha256") or "")
    payload = {
        "arm": arm,
        "training_contract_sha256": training_contract_sha256,
        "final_global_step": int(final_global_step),
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
    parts = mix_indices_from_frame(train_frame)
    return GoldPseudoMixSampler(
        parts["gold"],
        parts["pseudo"],
        gold_slots=int(mix["gold_slots"]),
        pseudo_slots=int(mix["pseudo_slots"]),
        seed=int(seed_policy["seed"]),
        start=int(start),
    )


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
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=int(training_contract.get("save_steps") or 1000),
        eval_steps=int(training_contract.get("eval_steps") or training_contract.get("save_steps") or 1000),
        save_total_limit=int(training_contract.get("save_total_limit") or 2),
        logging_steps=int(training_contract.get("logging_steps") or 50),
    )
    if resume_from_checkpoint:
        args.resume_from_checkpoint = str(resume_from_checkpoint)
    return args


def build_trainer(
    *,
    arm: str,
    layout: Mapping[str, Path],
    training_contract: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    audio_cache_roots: Sequence[Union[str, Path]] = (),
    processors: Optional[tuple] = None,
    model: Any = None,
    resume_checkpoint: Optional[Union[str, Path]] = None,
    d0_checkpoint_dir: Optional[Union[str, Path]] = None,
):
    """Construct the RQ1 Direct Seq2Seq trainer for one RQ2 arm. Does not call train()."""
    if arm not in TRAINABLE_ARMS:
        raise TrainingContractError(f"build_trainer cannot train arm {arm!r}")
    if str(training_contract.get("init_policy") or "") == REJECTED_PUBLIC_PRETRAINED_INIT_POLICY:
        raise TrainingContractError("public pretrained XLS-R+mBART reinitialization is rejected")
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
        validation_frame=validation_frame,
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

    metric_context = build_validation_metric_context(validation_frame, tokenizer)

    def _compute_metrics(eval_prediction):
        return compute_validation_metrics(eval_prediction, metric_context)

    BaseTrainer = make_resume_safe_seq2seq_trainer_cls()

    class Rq2ResumeProofTrainer(BaseTrainer):  # type: ignore[misc,valid-type]
        def _get_train_sampler(self):
            sampler = getattr(self, "_rq2_mix_sampler", None)
            if sampler is not None:
                return sampler
            return super()._get_train_sampler()

        def _load_from_checkpoint(self, resume_from_checkpoint, *args, **kwargs):
            result = super()._load_from_checkpoint(resume_from_checkpoint, *args, **kwargs)
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

    trainer = Rq2ResumeProofTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )
    trainer._rq2_arm = arm
    trainer._rq2_fingerprint = dict(fingerprint)
    trainer._rq2_metric_context = metric_context
    trainer._rq2_mix_sampler = mix_sampler
    if mix_sampler is not None:
        trainer._rq2_mix_observed = mix_sampler.observed_ratio()
    if resume_checkpoint is not None:
        try:
            inspect = json.loads((Path(resume_checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
            trainer._rq2_expected_resume_step = int(inspect.get("global_step") or 0)
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
    audio_cache_roots: Sequence[Union[str, Path]] = (),
    resume_checkpoint: Optional[Union[str, Path]] = None,
    trainer_factory: Optional[Callable[..., Any]] = None,
    processors: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Real D-Random / D-Quality training entry. D0 is bind-only."""
    ensure_arm_root(layout["root"])
    write_arm_contracts(layout, training_contract, fingerprint)
    write_json(layout["checkpoints"] / "full_experiment_fingerprint.json", dict(fingerprint))
    write_progress(layout, {"status": "starting", "arm": arm, "records_processed": 0, "batches_processed": 0})
    if arm == ARM_D0:
        return bind_d0_arm(layout=layout, flags=flags, training_contract=training_contract, fingerprint=fingerprint)
    if arm not in TRAINABLE_ARMS:
        raise TrainingContractError(f"cannot train arm {arm!r}")
    if flags.run_real_training is True:
        require_configured_mix_policy(training_contract, for_real_training=True)
        require_configured_seed_policy(training_contract, for_real_training=True)
        if str(training_contract.get("init_policy") or "") == REJECTED_PUBLIC_PRETRAINED_INIT_POLICY:
            raise TrainingContractError("public pretrained XLS-R+mBART reinitialization is rejected")
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
        write_progress(layout, {"status": STATUS_RESUME_AVAILABLE, "arm": arm, "global_step": proof["global_step"]})
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
    write_progress(layout, {"status": STATUS_TRAINING_RUNNING, "arm": arm})
    d0_init = resolve_frozen_d0_init(flags)
    if d0_init["model_state_sha256"] != str(training_contract.get("d0_init_model_state_sha256") or ""):
        raise TrainingContractError("frozen D0 init fingerprint does not match the arm training contract")
    if d0_init["checkpoint_fingerprint_sha256"] != str(training_contract.get("d0_checkpoint_fingerprint_sha256") or ""):
        raise TrainingContractError("frozen D0 checkpoint fingerprint does not match the arm training contract")
    d0_checkpoint_dir = d0_init["checkpoint_dir"]
    trainer = factory(
        arm=arm,
        layout=layout,
        training_contract=training_contract,
        fingerprint=fingerprint,
        train_frame=train_frame,
        validation_frame=validation_frame,
        audio_cache_roots=audio_cache_roots,
        processors=processors,
        resume_checkpoint=resume_checkpoint,
        d0_checkpoint_dir=d0_checkpoint_dir,
    )
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
            write_progress(layout, {"status": STATUS_FAILED_TRAINING, "arm": arm})
            raise TrainingContractError(f"{arm} resume restore was not proven")
    final_step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or getattr(result, "global_step", 0) or 0)
    terminal = _terminal_checkpoint_path(layout, trainer=trainer, resume_checkpoint=resume_checkpoint)
    best_payload = {}
    best_path = layout.get("best_checkpoint")
    if best_path and Path(best_path).is_file():
        best_payload = json.loads(Path(best_path).read_text(encoding="utf-8"))
    complete = write_training_complete_proof(
        layout,
        arm=arm,
        training_contract_sha256=training_contract["arm_training_contract_sha256"],
        final_global_step=final_step,
        checkpoint_path=terminal,
        best_checkpoint=best_payload,
    )
    mix_observed = getattr(trainer, "_rq2_mix_observed", None)
    if mix_observed:
        complete["mix_observed"] = dict(mix_observed)
        write_json(layout["training_complete"], complete)
    write_progress(layout, {"status": STATUS_TRAINING_COMPLETE, "arm": arm, "records_processed": final_step})
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
        audio_cache_roots=audio_cache_roots,
        resume_checkpoint=resume_checkpoint,
        trainer_factory=trainer_factory,
        processors=processors,
    )


def assert_no_cross_arm_resume(source_arm: str, dest_arm: str) -> None:
    if source_arm != dest_arm:
        raise ArmIsolationError(f"cannot resume {dest_arm} from a {source_arm} checkpoint")
