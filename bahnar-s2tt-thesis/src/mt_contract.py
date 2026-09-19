"""MT data contracts and deterministic hashing for Notebook 04."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from src.asr_runtime_paths import OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP
from src.mt_runtime_paths import (
    DEFAULT_MODEL_ID_CANDIDATE,
    MT_NORMALIZATION_VERSION,
    SOURCE_FIELD,
    TARGET_FIELD,
)

STATUS_MT_PREPARE = "SUCCESS_MT_PREPARE"
STATUS_MT_RESUME_TEST = "SUCCESS_MT_RESUME_TEST"
STATUS_MT_TRAINING = "SUCCESS_MT_TRAINING"
STATUS_MT_EVALUATE = "SUCCESS_MT_EVALUATE"
STATUS_FAILED = "FAILED"

STAGE_VERSION_PREPARE = "mt_prepare_v1"
STAGE_VERSION_RESUME_TEST = "mt_resume_test_v1"
STAGE_VERSION_TRAIN = "mt_train_v1"
STAGE_VERSION_EVALUATE = "mt_evaluate_v1"

# Locked scientific choices (Notebook 04 BARTPho baseline).
LOCKED_MODEL_ID = "vinai/bartpho-syllable"
LOCKED_MODEL_REVISION = "36eee8b4d648dd99da56462edcda3c5c97f7f3de"
LOCKED_EXPERIMENT_ID = "mt_bartpho_syllable_v1"
LOCKED_METRIC_FOR_BEST_MODEL = "eval_sacrebleu"
LOCKED_GREATER_IS_BETTER = True
LOCKED_HARD_MAX_UNK_RATE = 0.05
LOCKED_HARD_MAX_TRUNCATION_RATE = 0.05
LOCKED_MT_MONITOR_SIZE = 512

_FORBIDDEN_REVISIONS = frozenset(
    {
        "",
        "main",
        "master",
        "latest",
        "unpinned_probe_only",
        "head",
    }
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)

MT_DATA_CONTRACT_KEYS = (
    "dataset_id",
    "dataset_revision",
    "train_manifest_content_hash",
    "validation_manifest_content_hash",
    "train_uid_set_hash",
    "validation_uid_set_hash",
    "model_id",
    "model_revision",
    "tokenizer_fingerprint",
    "normalization_version",
    "source_field",
    "target_field",
    "max_source_length",
    "max_target_length",
    "truncation_policy",
    "exclusions_policy",
    "overlap_policy",
    "stage_version",
)

GENERATION_CONFIG_KEYS = (
    "generation_length_policy",
    "generation_max_length",
    "num_beams",
    "max_source_length",
    "max_target_length",
    "metric_for_best_model",
    "greater_is_better",
    "normalization_version",
    "model_id",
    "model_revision",
    "tokenizer_fingerprint",
)


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_hash(payload: Mapping[str, Any], *, keys: Sequence[str]) -> str:
    subset = {k: payload[k] for k in keys if k in payload}
    return hashlib.sha256(_stable_json(subset).encode("utf-8")).hexdigest()


def build_mt_data_contract(
    *,
    dataset_id: str,
    dataset_revision: str,
    train_manifest_content_hash: str,
    validation_manifest_content_hash: str,
    train_uid_set_hash: str,
    validation_uid_set_hash: str,
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    truncation_policy: str = "truncate_to_max_length",
    exclusions_policy: str = "empty_after_norm_or_duplicate_uid",
    normalization_version: str = MT_NORMALIZATION_VERSION,
    overlap_policy: str = OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
    source_field: str = SOURCE_FIELD,
    target_field: str = TARGET_FIELD,
    stage_version: str = STAGE_VERSION_PREPARE,
) -> Dict[str, Any]:
    assert_model_revision_pinned(model_id, model_revision)
    payload: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "train_manifest_content_hash": train_manifest_content_hash,
        "validation_manifest_content_hash": validation_manifest_content_hash,
        "train_uid_set_hash": train_uid_set_hash,
        "validation_uid_set_hash": validation_uid_set_hash,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "normalization_version": normalization_version,
        "source_field": source_field,
        "target_field": target_field,
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "truncation_policy": truncation_policy,
        "exclusions_policy": exclusions_policy,
        "overlap_policy": overlap_policy,
        "stage_version": stage_version,
    }
    payload["contract_hash"] = contract_hash(payload, keys=MT_DATA_CONTRACT_KEYS)
    return payload


def assert_model_revision_pinned(model_id: str, model_revision: str) -> None:
    """Fail-closed: every full stage (including prepare) requires a commit pin."""
    mid = str(model_id or "").strip()
    rev = str(model_revision or "").strip()
    if not mid:
        raise RuntimeError("MODEL_ID must be non-empty")
    if rev.lower() in _FORBIDDEN_REVISIONS or "unpinned" in rev.lower():
        raise RuntimeError(
            "MODEL_REVISION must be an explicit HF commit pin "
            f"(got {model_revision!r}). Locked baseline expects "
            f"{LOCKED_MODEL_ID!r} @ {LOCKED_MODEL_REVISION!r}."
        )
    if not _COMMIT_RE.match(rev):
        raise RuntimeError(
            f"MODEL_REVISION must look like a git commit hash (got {model_revision!r})"
        )


def assert_locked_bartpho_baseline(model_id: str, model_revision: str) -> None:
    assert_model_revision_pinned(model_id, model_revision)
    if str(model_id).strip() != LOCKED_MODEL_ID:
        raise RuntimeError(f"MODEL_ID must be {LOCKED_MODEL_ID!r}, got {model_id!r}")
    if str(model_revision).strip() != LOCKED_MODEL_REVISION:
        raise RuntimeError(
            f"MODEL_REVISION must be {LOCKED_MODEL_REVISION!r}, got {model_revision!r}"
        )


def contracts_equal(a: Mapping[str, Any], b: Mapping[str, Any], *, keys: Optional[Sequence[str]] = None) -> bool:
    use = list(keys) if keys is not None else list(MT_DATA_CONTRACT_KEYS)
    for k in use:
        if a.get(k) != b.get(k):
            return False
    return True


def build_generation_config(
    *,
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    generation_max_length: int,
    num_beams: int,
    metric_for_best_model: str = LOCKED_METRIC_FOR_BEST_MODEL,
    greater_is_better: bool = LOCKED_GREATER_IS_BETTER,
    normalization_version: str = MT_NORMALIZATION_VERSION,
) -> Dict[str, Any]:
    return {
        "generation_length_policy": "generation_max_length",
        "generation_max_length": int(generation_max_length),
        "num_beams": int(num_beams),
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "metric_for_best_model": str(metric_for_best_model),
        "greater_is_better": bool(greater_is_better),
        "normalization_version": normalization_version,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }


def assert_generation_config_matches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    mismatches = []
    for k in GENERATION_CONFIG_KEYS:
        if expected.get(k) != actual.get(k):
            mismatches.append(f"{k}: expected={expected.get(k)!r} actual={actual.get(k)!r}")
    if mismatches:
        raise RuntimeError(
            "MT evaluate generation/config mismatch vs training summary: "
            + "; ".join(mismatches)
        )


MT_TRAINING_CONTRACT_KEYS = (
    "mt_data_contract_hash",
    "train_uid_set_hash",
    "validation_uid_set_hash",
    "model_id",
    "model_revision",
    "tokenizer_fingerprint",
    "normalization_version",
    "experiment_id",
    "seed",
    "learning_rate",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "max_steps",
    "warmup_ratio",
    "weight_decay",
    "fp16",
    "bf16",
    "gradient_checkpointing",
    "save_steps",
    "eval_steps",
    "eval_policy",
    "monitor_size",
    "monitor_uid_set_hash",
    "save_total_limit",
    "max_source_length",
    "max_target_length",
    "generation_length_policy",
    "generation_max_length",
    "num_beams",
    "metric_for_best_model",
    "greater_is_better",
    "stage_version",
)


def build_mt_training_contract(
    *,
    mt_data_contract_hash: str,
    train_uid_set_hash: str,
    validation_uid_set_hash: str,
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    experiment_id: str,
    seed: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    max_steps: int,
    warmup_ratio: float,
    weight_decay: float,
    fp16: bool,
    bf16: bool,
    gradient_checkpointing: bool,
    save_steps: int,
    eval_steps: int,
    save_total_limit: int,
    max_source_length: int,
    max_target_length: int,
    generation_max_length: int,
    num_beams: int,
    metric_for_best_model: str = LOCKED_METRIC_FOR_BEST_MODEL,
    greater_is_better: bool = LOCKED_GREATER_IS_BETTER,
    normalization_version: str = MT_NORMALIZATION_VERSION,
    eval_policy: str = "steps_on_fixed_monitor_subset",
    monitor_size: int = LOCKED_MT_MONITOR_SIZE,
    monitor_uid_set_hash: str = "",
    generation_length_policy: str = "generation_max_length",
    stage_version: str = STAGE_VERSION_TRAIN,
) -> Dict[str, Any]:
    """Deterministic MT training contract (data identity + hparams + generation)."""
    assert_locked_bartpho_baseline(model_id, model_revision)
    payload: Dict[str, Any] = {
        "mt_data_contract_hash": str(mt_data_contract_hash),
        "train_uid_set_hash": str(train_uid_set_hash),
        "validation_uid_set_hash": str(validation_uid_set_hash),
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "tokenizer_fingerprint": str(tokenizer_fingerprint),
        "normalization_version": str(normalization_version),
        "experiment_id": str(experiment_id),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "per_device_eval_batch_size": int(per_device_eval_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "num_train_epochs": float(num_train_epochs),
        "max_steps": int(max_steps),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "fp16": bool(fp16),
        "bf16": bool(bf16),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "save_steps": int(save_steps),
        "eval_steps": int(eval_steps),
        "eval_policy": str(eval_policy),
        "monitor_size": int(monitor_size),
        "monitor_uid_set_hash": str(monitor_uid_set_hash),
        "save_total_limit": int(save_total_limit),
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "generation_length_policy": str(generation_length_policy),
        "generation_max_length": int(generation_max_length),
        "num_beams": int(num_beams),
        "metric_for_best_model": str(metric_for_best_model),
        "greater_is_better": bool(greater_is_better),
        "stage_version": str(stage_version),
    }
    payload["mt_training_contract_hash"] = contract_hash(payload, keys=MT_TRAINING_CONTRACT_KEYS)
    return payload


def assert_mt_training_contract_self_consistent(contract: Mapping[str, Any]) -> None:
    """Fail if stored mt_training_contract_hash != recomputed canonical hash."""
    stored = str(contract.get("mt_training_contract_hash") or "")
    if not stored:
        raise RuntimeError("mt_training_contract_hash missing")
    recomputed = contract_hash(contract, keys=MT_TRAINING_CONTRACT_KEYS)
    if stored != recomputed:
        raise RuntimeError(
            "mt_training_contract self-consistency failed: "
            f"stored={stored!r} recomputed={recomputed!r}"
        )


def assert_mt_training_contracts_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str = "MT training contract",
) -> None:
    assert_mt_training_contract_self_consistent(expected)
    assert_mt_training_contract_self_consistent(actual)
    exp_h = str(expected.get("mt_training_contract_hash") or "")
    act_h = str(actual.get("mt_training_contract_hash") or "")
    if not exp_h or not act_h or exp_h != act_h:
        mismatches = []
        for k in MT_TRAINING_CONTRACT_KEYS:
            if expected.get(k) != actual.get(k):
                mismatches.append(f"{k}: expected={expected.get(k)!r} actual={actual.get(k)!r}")
        detail = "; ".join(mismatches[:12]) or f"hash {act_h!r} != {exp_h!r}"
        raise RuntimeError(f"{label} mismatch: {detail}")


def fingerprint_extra_from_mt_training_contract(train_contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Fields written into full_experiment_fingerprint.json for durable resume."""
    return {
        "experiment_id": train_contract.get("experiment_id"),
        "mt_data_contract_hash": train_contract.get("mt_data_contract_hash"),
        "mt_training_contract_hash": train_contract.get("mt_training_contract_hash"),
        "model_id": train_contract.get("model_id"),
        "model_revision": train_contract.get("model_revision"),
        "tokenizer_fingerprint": train_contract.get("tokenizer_fingerprint"),
        "normalization_version": train_contract.get("normalization_version"),
        "hparams": {k: train_contract.get(k) for k in MT_TRAINING_CONTRACT_KEYS},
        "contract_hash": train_contract.get("mt_training_contract_hash"),
        "mt_training_contract": dict(train_contract),
    }


MT_RESUME_TEST_CONTRACT_KEYS = (
    "mt_data_contract_hash",
    "model_id",
    "model_revision",
    "tokenizer_fingerprint",
    "subset_uid_set_hash",
    "seed",
    "subset_size",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "phase_a_target_steps",
    "phase_b_target_steps",
    "stage_version",
)


def build_mt_resume_test_contract(
    *,
    mt_data_contract_hash: str,
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    subset_uid_set_hash: str,
    seed: int,
    subset_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    phase_a_target_steps: int,
    phase_b_target_steps: int,
    stage_version: str = STAGE_VERSION_RESUME_TEST,
) -> Dict[str, Any]:
    assert_model_revision_pinned(model_id, model_revision)
    payload: Dict[str, Any] = {
        "mt_data_contract_hash": str(mt_data_contract_hash),
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "tokenizer_fingerprint": str(tokenizer_fingerprint),
        "subset_uid_set_hash": str(subset_uid_set_hash),
        "seed": int(seed),
        "subset_size": int(subset_size),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "learning_rate": float(learning_rate),
        "phase_a_target_steps": int(phase_a_target_steps),
        "phase_b_target_steps": int(phase_b_target_steps),
        "stage_version": str(stage_version),
    }
    payload["mt_resume_test_contract_hash"] = contract_hash(payload, keys=MT_RESUME_TEST_CONTRACT_KEYS)
    return payload


def assert_mt_resume_test_contract_self_consistent(contract: Mapping[str, Any]) -> None:
    stored = str(contract.get("mt_resume_test_contract_hash") or "")
    if not stored:
        raise RuntimeError("mt_resume_test_contract_hash missing")
    recomputed = contract_hash(contract, keys=MT_RESUME_TEST_CONTRACT_KEYS)
    if stored != recomputed:
        raise RuntimeError(
            "mt_resume_test_contract self-consistency failed: "
            f"stored={stored!r} recomputed={recomputed!r}"
        )


def assert_mt_resume_test_contracts_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str = "MT resume-test contract",
) -> None:
    assert_mt_resume_test_contract_self_consistent(expected)
    assert_mt_resume_test_contract_self_consistent(actual)
    if str(expected.get("mt_resume_test_contract_hash")) != str(actual.get("mt_resume_test_contract_hash")):
        mismatches = [
            f"{k}: expected={expected.get(k)!r} actual={actual.get(k)!r}"
            for k in MT_RESUME_TEST_CONTRACT_KEYS
            if expected.get(k) != actual.get(k)
        ]
        detail = "; ".join(mismatches[:12]) or "hash mismatch"
        raise RuntimeError(f"{label} mismatch: {detail}")
