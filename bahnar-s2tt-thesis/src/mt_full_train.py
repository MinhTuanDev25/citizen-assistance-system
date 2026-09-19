"""
Notebook 04 MT full-stage orchestration: resume proof, train/eval gates, statuses.

Reuses durable checkpoint primitives from ``asr_full_train`` (namespace-isolated via
paths). Does not reuse CTC/audio/ASR contracts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from src.asr_full_train import (
    assert_checkpoint_allowed_for_full_train,
    assert_checkpoint_complete_for_resume,
    assert_cross_session_resume,
    assert_no_frozen_test_access,
    derive_resume_test_status_from_proof,
    durable_experiment_dir,
    ensure_experiment_fingerprint,
    experiment_checkpoint_dir,
    is_forbidden_init_checkpoint,
    make_resume_proof_callback,
    make_sequential_sampler_trainer_cls,
    make_uid_tracking_collator,
    make_uid_tracking_trainer_cls,
    new_session_token,
    plan_expected_resume_position,
    restore_experiment_checkpoints_from_durable,
    resolve_best_checkpoint_from_durable,
    summarize_resume_proof,
    sync_experiment_checkpoints_to_durable,
    write_checkpoint_fingerprint,
)
from src.metrics import mt_corpus_metrics
from src.mt_contract import (
    LOCKED_MT_MONITOR_SIZE,
    STATUS_FAILED,
    STATUS_MT_EVALUATE,
    STATUS_MT_PREPARE,
    STATUS_MT_RESUME_TEST,
    STATUS_MT_TRAINING,
    STAGE_VERSION_EVALUATE,
    STAGE_VERSION_RESUME_TEST,
    STAGE_VERSION_TRAIN,
    assert_generation_config_matches,
    assert_locked_bartpho_baseline,
    assert_model_revision_pinned,
    assert_mt_resume_test_contract_self_consistent,
    assert_mt_resume_test_contracts_match,
    assert_mt_training_contract_self_consistent,
    assert_mt_training_contracts_match,
    build_generation_config,
    build_mt_data_contract,
    build_mt_resume_test_contract,
    build_mt_training_contract,
    contracts_equal,
    fingerprint_extra_from_mt_training_contract,
)
from src.mt_dataset import MtTextDataset, make_seq2seq_collator
from src.mt_normalize import normalize_mt_text_v1
from src.mt_prepare import load_mt_prepare_success
from src.mt_runtime_paths import FULL_TRAIN_MARKER, PILOT_MARKER, RESUME_TEST_MARKER, SOURCE_FIELD, TARGET_FIELD
from src.mt_tokenize import load_mt_tokenizer, tokenizer_fingerprint
from src.data_utils import compute_uid_set_hash

import hashlib

RESUME_TEST_PHASE_A_STEPS = 50
RESUME_TEST_PHASE_B_STEPS = 100
RESUME_TEST_SUBSET_SIZE = 64


def mt_experiment_dir(
    root: Union[str, Path],
    *,
    kind: str,
    experiment_id: str,
) -> Path:
    return experiment_checkpoint_dir(root, experiment_id, kind=kind)


def assert_mt_checkpoint_namespace(path: Union[str, Path], *, allowed_kind: str) -> None:
    p = str(Path(path).resolve())
    if f"/{allowed_kind}/" not in p and not p.endswith(f"/{allowed_kind}"):
        raise RuntimeError(f"Checkpoint path not under namespace {allowed_kind!r}: {path}")
    if allowed_kind == FULL_TRAIN_MARKER and is_forbidden_init_checkpoint(path):
        raise RuntimeError(f"Forbidden init checkpoint for full MT train: {path}")


def build_mt_resume_test_subset(df: pd.DataFrame, *, n_samples: int, seed: int) -> pd.DataFrame:
    if len(df) < n_samples:
        raise RuntimeError(f"Need >= {n_samples} MT rows for resume_test, got {len(df)}")
    return df.sample(n=n_samples, random_state=seed).sort_index().reset_index(drop=True)


def write_mt_resume_test_summary(state_dir: Path, payload: Dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_resume_test_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_mt_resume_test_success(
    state_dir: Union[str, Path],
    *,
    expected_contract_hash: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(state_dir) / "mt_resume_test_summary.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT resume_test summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != STATUS_MT_RESUME_TEST:
        raise RuntimeError(f"MT resume_test not successful: {data.get('status')}")
    if expected_contract_hash and data.get("contract_hash") != expected_contract_hash:
        raise RuntimeError("MT resume_test contract_hash mismatch")
    return data


def write_mt_train_summary(state_dir: Path, payload: Dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_train_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_mt_train_success(
    state_dir: Union[str, Path],
    *,
    expected_contract_hash: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(state_dir) / "mt_train_summary.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT train summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != STATUS_MT_TRAINING:
        raise RuntimeError(f"MT training not successful: {data.get('status')}")
    if expected_contract_hash and data.get("contract_hash") != expected_contract_hash:
        raise RuntimeError("MT train contract_hash mismatch")
    return data


def write_mt_evaluate_summary(state_dir: Path, payload: Dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_evaluate_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def assert_ready_for_mt_full_train(state_dir: Union[str, Path], *, contract: Mapping[str, Any]) -> None:
    assert_locked_bartpho_baseline(str(contract.get("model_id")), str(contract.get("model_revision")))
    load_mt_prepare_success(state_dir, expected_contract=dict(contract))
    load_mt_resume_test_success(state_dir, expected_contract_hash=str(contract.get("contract_hash")))


def assert_ready_for_mt_evaluate(state_dir: Union[str, Path], *, contract: Mapping[str, Any]) -> None:
    assert_ready_for_mt_full_train(state_dir, contract=contract)
    load_mt_train_success(state_dir, expected_contract_hash=str(contract.get("contract_hash")))


def derive_mt_training_status(
    *,
    global_step: int,
    expected_max_steps: int,
    best_checkpoint_exists: bool,
    metrics_finite: bool,
    frozen_test_accessed: bool,
    started_from_base_or_same_experiment: bool,
) -> str:
    ok = (
        int(global_step) >= int(expected_max_steps)
        and best_checkpoint_exists
        and bool(metrics_finite)
        and frozen_test_accessed is False
        and started_from_base_or_same_experiment
    )
    return STATUS_MT_TRAINING if ok else STATUS_FAILED


def derive_mt_evaluate_status(
    *,
    prediction_count: int,
    validation_count: int,
    metrics_finite: bool,
    frozen_test_accessed: bool,
    contract_ok: bool,
) -> str:
    ok = (
        int(prediction_count) == int(validation_count)
        and metrics_finite
        and frozen_test_accessed is False
        and contract_ok
    )
    return STATUS_MT_EVALUATE if ok else STATUS_FAILED


def resolve_mt_stage_status(
    *,
    full_stage: str,
    pipeline_error: Optional[Any],
    full_status: Optional[str],
) -> str:
    if pipeline_error is not None:
        return STATUS_FAILED
    if full_status:
        return str(full_status)
    return STATUS_FAILED


def derive_notebook04_handoff(
    status: str,
    *,
    frozen_test_accessed: bool,
) -> Dict[str, bool]:
    """Cascaded C0 handoff only after independent SUCCESS_MT_EVALUATE."""
    return {
        "ready_for_cascaded_c0": status == STATUS_MT_EVALUATE and frozen_test_accessed is False,
        "ready_for_rq1_final": False,
        "frozen_test_accessed": bool(frozen_test_accessed),
    }


def build_mt_training_hparams(
    *,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    seed: int,
    max_steps: int,
    num_train_epochs: float,
    save_steps: int,
    eval_steps: int,
    save_total_limit: int,
    fp16: bool,
    bf16: bool,
    gradient_checkpointing: bool,
    max_source_length: int,
    max_target_length: int,
    generation_max_length: int,
    num_beams: int,
    metric_for_best_model: str,
    greater_is_better: bool,
) -> Dict[str, Any]:
    return {
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "per_device_eval_batch_size": int(per_device_eval_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "learning_rate": float(learning_rate),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "num_train_epochs": float(num_train_epochs),
        "save_steps": int(save_steps),
        "eval_steps": int(eval_steps),
        "save_total_limit": int(save_total_limit),
        "fp16": bool(fp16),
        "bf16": bool(bf16),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "generation_max_length": int(generation_max_length),
        "num_beams": int(num_beams),
        "metric_for_best_model": str(metric_for_best_model),
        "greater_is_better": bool(greater_is_better),
        "stage_version": STAGE_VERSION_TRAIN,
    }


def load_mt_seq2seq_model(model_id: str, model_revision: str) -> Any:
    from transformers import AutoModelForSeq2SeqLM

    assert_model_revision_pinned(model_id, model_revision)
    return AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=model_revision)


def count_tokens_no_special(tokenizer: Any, text: str) -> int:
    enc = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
    return int(len(list(enc["input_ids"])))


def count_generated_tokens(token_ids: Sequence[int], *, pad_token_id: Optional[int], eos_token_id: Optional[int]) -> int:
    ids = list(int(x) for x in token_ids)
    if pad_token_id is not None:
        while ids and ids[-1] == int(pad_token_id):
            ids.pop()
    if eos_token_id is not None and ids and ids[-1] == int(eos_token_id):
        ids = ids[:-1]
    return int(len(ids))


def build_mt_validation_monitor_subset(
    eligible_val_df: pd.DataFrame,
    *,
    n_samples: int = LOCKED_MT_MONITOR_SIZE,
    salt: str = "mt_monitor_v1",
) -> pd.DataFrame:
    """Deterministic hash-based UID selection (not first-N)."""
    if len(eligible_val_df) == 0:
        raise RuntimeError("Cannot build MT validation-monitor subset from empty validation")
    n = min(int(n_samples), int(len(eligible_val_df)))
    scored = []
    for uid in eligible_val_df["record_uid"].astype(str).tolist():
        h = hashlib.sha256(f"{salt}:{uid}".encode("utf-8")).hexdigest()
        scored.append((h, uid))
    scored.sort(key=lambda x: (x[0], x[1]))
    selected = {uid for _, uid in scored[:n]}
    out = eligible_val_df.loc[eligible_val_df["record_uid"].astype(str).isin(selected)].copy()
    out = out.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    if len(out) != n:
        raise RuntimeError(f"MT monitor subset size mismatch: got {len(out)} expected {n}")
    return out


def persist_mt_monitor_manifest(
    state_dir: Union[str, Path],
    monitor_df: pd.DataFrame,
    *,
    n_samples: int,
    salt: str = "mt_monitor_v1",
) -> Dict[str, Any]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    uids = monitor_df["record_uid"].astype(str).tolist()
    payload = {
        "n_samples": int(n_samples),
        "actual_size": int(len(uids)),
        "salt": salt,
        "selection_policy": "sha256_salt_uid_sorted",
        "ordered_uids": uids,
        "uid_set_hash": compute_uid_set_hash(monitor_df),
        "note": (
            "Training checkpoint selection uses monitor SacreBLEU only; "
            "final FULL_STAGE=evaluate reports full-validation SacreBLEU."
        ),
    }
    path = state_dir / "mt_validation_monitor_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_mt_training_contract(state_dir: Union[str, Path], contract: Mapping[str, Any]) -> Path:
    """Atomic write; refuses overwrite when an existing contract hash differs."""
    assert_mt_training_contract_self_consistent(contract)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_training_contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        assert_mt_training_contracts_match(
            existing, dict(contract), label="persisted mt_training_contract.json"
        )
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_mt_training_contract(
    state_dir: Union[str, Path],
    *,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(state_dir) / "mt_training_contract.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT training contract: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert_mt_training_contract_self_consistent(data)
    if expected_hash and str(data.get("mt_training_contract_hash")) != str(expected_hash):
        raise RuntimeError("MT training contract hash mismatch on load")
    return data


def write_mt_resume_test_contract(state_dir: Union[str, Path], contract: Mapping[str, Any]) -> Path:
    assert_mt_resume_test_contract_self_consistent(contract)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "mt_resume_test_contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        assert_mt_resume_test_contracts_match(
            existing, dict(contract), label="persisted mt_resume_test_contract.json"
        )
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(contract), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_mt_resume_test_contract(
    state_dir: Union[str, Path],
    *,
    expected: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    path = Path(state_dir) / "mt_resume_test_contract.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT resume-test contract: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert_mt_resume_test_contract_self_consistent(data)
    if expected is not None:
        assert_mt_resume_test_contracts_match(data, dict(expected), label="resume-test contract")
    return data


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_canonical_json(obj: Any) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def persist_durable_tokenizer_audit(
    state_dir: Union[str, Path],
    audit: Mapping[str, Any],
    *,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    mt_data_contract_hash: str,
    train_uid_set_hash: Optional[str] = None,
    validation_uid_set_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist tokenizer_audit.json + sidecar checksum into durable contract state."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(audit)
    payload["tokenizer_fingerprint"] = str(tokenizer_fingerprint)
    payload["max_source_length"] = int(max_source_length)
    payload["max_target_length"] = int(max_target_length)
    payload["mt_data_contract_hash"] = str(mt_data_contract_hash)
    if train_uid_set_hash is not None:
        payload["train_uid_set_hash"] = str(train_uid_set_hash)
    if validation_uid_set_hash is not None:
        payload["validation_uid_set_hash"] = str(validation_uid_set_hash)
    digest = sha256_canonical_json(payload)
    payload["audit_sha256"] = digest
    path = state_dir / "tokenizer_audit.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Pretty file for humans; checksum is over canonical form of identity+audit body
    # Recompute digest excluding audit_sha256 itself for stable verify.
    body = {k: v for k, v in payload.items() if k != "audit_sha256"}
    digest = sha256_canonical_json(body)
    payload["audit_sha256"] = digest
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    meta = {
        "path": str(path),
        "audit_sha256": digest,
        "tokenizer_fingerprint": str(tokenizer_fingerprint),
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "mt_data_contract_hash": str(mt_data_contract_hash),
    }
    (state_dir / "tokenizer_audit.sha256").write_text(digest + "\n", encoding="utf-8")
    return meta


def load_durable_tokenizer_audit(
    state_dir: Union[str, Path],
    *,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    mt_data_contract_hash: str,
    train_uid_set_hash: Optional[str] = None,
    validation_uid_set_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load durable tokenizer_audit.json if identity+checksum match.
    Raises RuntimeError on missing/mismatch (caller may fall back to fresh audit).
    """
    path = Path(state_dir) / "tokenizer_audit.json"
    if not path.is_file():
        raise RuntimeError(f"Missing durable tokenizer_audit.json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in data.items() if k != "audit_sha256"}
    digest = sha256_canonical_json(body)
    stored = str(data.get("audit_sha256") or "")
    if stored != digest:
        raise RuntimeError(
            f"tokenizer_audit SHA256 mismatch: stored={stored!r} recomputed={digest!r}"
        )
    if str(data.get("tokenizer_fingerprint") or "") != str(tokenizer_fingerprint):
        raise RuntimeError("tokenizer_audit tokenizer_fingerprint mismatch")
    if int(data.get("max_source_length") or -1) != int(max_source_length):
        raise RuntimeError("tokenizer_audit max_source_length mismatch")
    if int(data.get("max_target_length") or -1) != int(max_target_length):
        raise RuntimeError("tokenizer_audit max_target_length mismatch")
    if str(data.get("mt_data_contract_hash") or "") != str(mt_data_contract_hash):
        raise RuntimeError("tokenizer_audit mt_data_contract_hash mismatch")
    if train_uid_set_hash is not None and data.get("train_uid_set_hash") is not None:
        if str(data.get("train_uid_set_hash")) != str(train_uid_set_hash):
            raise RuntimeError("tokenizer_audit train_uid_set_hash mismatch")
    if validation_uid_set_hash is not None and data.get("validation_uid_set_hash") is not None:
        if str(data.get("validation_uid_set_hash")) != str(validation_uid_set_hash):
            raise RuntimeError("tokenizer_audit validation_uid_set_hash mismatch")
    if data.get("passed") is not True:
        raise RuntimeError("durable tokenizer_audit did not pass")
    return data


def try_load_durable_tokenizer_audit(
    state_dir: Optional[Union[str, Path]],
    *,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    mt_data_contract_hash: str,
    train_uid_set_hash: Optional[str] = None,
    validation_uid_set_hash: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if state_dir is None:
        return None
    try:
        return load_durable_tokenizer_audit(
            state_dir,
            tokenizer_fingerprint=tokenizer_fingerprint,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            mt_data_contract_hash=mt_data_contract_hash,
            train_uid_set_hash=train_uid_set_hash,
            validation_uid_set_hash=validation_uid_set_hash,
        )
    except Exception:
        return None


def strip_record_uid_features(features: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate HF Trainer remove_unused_columns for record_uid before Seq2Seq collator."""
    return [{k: v for k, v in dict(f).items() if k != "record_uid"} for f in features]


def ensure_mt_full_train_fingerprint(
    experiment_dir: Union[str, Path],
    *,
    train_contract: Mapping[str, Any],
    global_step: int = 0,
    update_global_step: bool = False,
) -> Path:
    """
    Fail-closed fingerprint policy for full_train:

    - If step checkpoints exist: existing fingerprint must match current training
      contract BEFORE any rewrite. Never stamp a new identity onto old weights.
    - If directory empty: create fingerprint for the current contract.
    - If update_global_step=True and identity already matches: merge global_step only.
    """
    from src.asr_full_train import list_step_checkpoints, read_checkpoint_fingerprint

    root = Path(experiment_dir)
    root.mkdir(parents=True, exist_ok=True)
    assert_mt_training_contract_self_consistent(train_contract)
    steps = list_step_checkpoints(root)
    fp = read_checkpoint_fingerprint(root)
    expected_hash = str(train_contract.get("mt_training_contract_hash") or "")
    if steps:
        if not fp:
            raise RuntimeError(
                f"Local full_train dir has checkpoints but missing fingerprint "
                f"(refuse to stamp new identity): {root}"
            )
        got = str(
            fp.get("mt_training_contract_hash")
            or (fp.get("mt_training_contract") or {}).get("mt_training_contract_hash")
            or ""
        )
        if got != expected_hash:
            raise RuntimeError(
                "Local full_train fingerprint mt_training_contract_hash mismatch "
                f"(existing={got!r} current={expected_hash!r}); refuse resume/stamp"
            )
        if str(fp.get("experiment_id")) != str(train_contract.get("experiment_id")):
            raise RuntimeError("Local full_train fingerprint experiment_id mismatch")
        if str(fp.get("kind")) != FULL_TRAIN_MARKER:
            raise RuntimeError(f"Local fingerprint kind mismatch: {fp.get('kind')}")
        if update_global_step:
            return write_checkpoint_fingerprint(
                root,
                experiment_id=str(train_contract["experiment_id"]),
                kind=FULL_TRAIN_MARKER,
                global_step=int(global_step),
                extra=fingerprint_extra_from_mt_training_contract(train_contract),
                overwrite=True,
                preserve_existing=True,
            )
        # Identity already valid — do not restamp before resume/train.
        return root / "full_experiment_fingerprint.json"
    # Empty dir: create fresh fingerprint for current contract
    return write_checkpoint_fingerprint(
        root,
        experiment_id=str(train_contract["experiment_id"]),
        kind=FULL_TRAIN_MARKER,
        global_step=int(global_step),
        extra=fingerprint_extra_from_mt_training_contract(train_contract),
        overwrite=True,
        preserve_existing=False,
    )


def write_mt_resume_test_fingerprint(
    experiment_dir: Union[str, Path],
    *,
    experiment_id: str,
    mt_data_contract: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    global_step: int,
) -> Path:
    """Create a real full_experiment_fingerprint.json before durable sync (resume_test)."""
    extra = {
        "mt_data_contract_hash": mt_data_contract.get("contract_hash"),
        "mt_data_contract": dict(mt_data_contract),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "contract_hash": mt_data_contract.get("contract_hash"),
    }
    return write_checkpoint_fingerprint(
        experiment_dir,
        experiment_id=experiment_id,
        kind=RESUME_TEST_MARKER,
        global_step=int(global_step),
        extra=extra,
        overwrite=True,
        preserve_existing=False,
    )


def write_mt_full_train_fingerprint(
    experiment_dir: Union[str, Path],
    *,
    train_contract: Mapping[str, Any],
    global_step: int = 0,
    update_global_step: bool = False,
) -> Path:
    """Backward-compatible alias — prefer ensure_mt_full_train_fingerprint."""
    return ensure_mt_full_train_fingerprint(
        experiment_dir,
        train_contract=train_contract,
        global_step=global_step,
        update_global_step=update_global_step,
    )


def derive_started_from_base_or_same_experiment(
    *,
    resume_ckpt: Optional[str],
    experiment_id: str,
    train_contract: Mapping[str, Any],
    local_experiment_dir: Union[str, Path],
    loaded_from_pinned_base: bool,
) -> bool:
    """
    Provenance gate:
    - no resume → True only if model was loaded from pinned base weights
    - resume → True only if checkpoint is allowed for this full_train experiment
      and matches the current MT training contract hash
    """
    if not resume_ckpt:
        return bool(loaded_from_pinned_base)
    expected = {
        "experiment_id": experiment_id,
        "hparams": fingerprint_extra_from_mt_training_contract(train_contract).get("hparams"),
        "mt_training_contract_hash": train_contract.get("mt_training_contract_hash"),
        "contract_hash": train_contract.get("mt_training_contract_hash"),
    }
    assert_checkpoint_allowed_for_full_train(
        resume_ckpt,
        experiment_id=experiment_id,
        expected_contract=expected,
        experiment_root=local_experiment_dir,
    )
    assert_checkpoint_complete_for_resume(resume_ckpt)
    return True


def derive_phase_a_reached_target(
    phase_a_payload: Mapping[str, Any],
    *,
    target_steps: int,
    checkpoint_path: Union[str, Path],
) -> bool:
    if int(phase_a_payload.get("global_step") or -1) != int(target_steps):
        return False
    try:
        assert_checkpoint_complete_for_resume(checkpoint_path)
    except Exception:
        return False
    return True


def derive_used_separate_experiment_dir(
    experiment_dir: Union[str, Path],
    *,
    allowed_kind: str = RESUME_TEST_MARKER,
) -> bool:
    p = str(Path(experiment_dir).resolve())
    if f"/{allowed_kind}/" not in p and not p.endswith(f"/{allowed_kind}"):
        return False
    if f"/{FULL_TRAIN_MARKER}/" in p or f"/{PILOT_MARKER}/" in p:
        return False
    return True


def resolve_latest_checkpoint_dir(experiment_dir: Union[str, Path]) -> Optional[Path]:
    from src.asr_full_train import list_step_checkpoints

    steps = list_step_checkpoints(experiment_dir)
    return steps[-1] if steps else None


def resolve_mt_best_checkpoint_for_evaluate(
    *,
    full_state_dir: Union[str, Path],
    experiment_id: str,
    train_summary: Mapping[str, Any],
    local_experiment_dir: Union[str, Path],
    expected_training_contract: Optional[Mapping[str, Any]] = None,
) -> str:
    """Correct wrapper around resolve_best_checkpoint_from_durable + MT contract checks."""
    expected = None
    if expected_training_contract is not None:
        expected = {
            "experiment_id": expected_training_contract.get("experiment_id"),
            "hparams": fingerprint_extra_from_mt_training_contract(expected_training_contract).get("hparams"),
            "mt_training_contract_hash": expected_training_contract.get("mt_training_contract_hash"),
            "contract_hash": expected_training_contract.get("mt_training_contract_hash"),
        }
    best = resolve_best_checkpoint_from_durable(
        full_state_dir,
        experiment_id=experiment_id,
        train_summary=dict(train_summary),
        local_experiment_dir=local_experiment_dir,
        expected_contract=expected,
    )
    best_path = Path(best)
    if not best_path.is_dir():
        raise RuntimeError(f"Resolved best checkpoint missing: {best_path}")
    assert_checkpoint_complete_for_resume(best_path)
    assert_checkpoint_allowed_for_full_train(
        best_path,
        experiment_id=experiment_id,
        expected_contract=expected,
        experiment_root=local_experiment_dir,
    )
    if expected_training_contract is not None:
        # Fingerprint may live at experiment root
        from src.asr_full_train import read_checkpoint_fingerprint

        fp = read_checkpoint_fingerprint(local_experiment_dir) or {}
        got_hash = fp.get("mt_training_contract_hash") or (fp.get("mt_training_contract") or {}).get(
            "mt_training_contract_hash"
        )
        if got_hash and str(got_hash) != str(expected_training_contract.get("mt_training_contract_hash")):
            raise RuntimeError(
                "Restored experiment fingerprint mt_training_contract_hash mismatch: "
                f"{got_hash} != {expected_training_contract.get('mt_training_contract_hash')}"
            )
    return str(best_path)


def assert_evaluate_config_consistency(
    *,
    train_summary: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    current_training_contract: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
) -> None:
    assert_mt_training_contracts_match(
        training_contract, current_training_contract, label="evaluate vs persisted training contract"
    )
    if str(train_summary.get("mt_training_contract_hash")) != str(
        training_contract.get("mt_training_contract_hash")
    ):
        raise RuntimeError("train_summary mt_training_contract_hash != training contract")
    data_h = train_summary.get("mt_data_contract_hash") or train_summary.get("contract_hash")
    if str(data_h) != str(training_contract.get("mt_data_contract_hash")):
        raise RuntimeError("evaluate data contract hash mismatch vs training contract")
    assert_evaluate_matches_train_summary(
        train_summary,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_fingerprint=tokenizer_fingerprint,
        generation_config=generation_config,
    )


def derive_frozen_test_accessed(opened_paths: Sequence[str], loaded_splits: Sequence[str]) -> bool:
    """Return True if any opened path/split looks like frozen test (should already have raised)."""
    from src.asr_utils import is_forbidden_test_path
    from src.asr_full_data import is_frozen_split_label

    for p in opened_paths:
        if is_forbidden_test_path(p):
            return True
    for s in loaded_splits:
        if is_frozen_split_label(str(s)):
            return True
    return False


def generate_mt_predictions(
    *,
    model: Any,
    tokenizer: Any,
    dataset: MtTextDataset,
    generation_max_length: int,
    num_beams: int,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Greedy/beam generate; uses generation_max_length (same semantics as Trainer)."""
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    model.eval()
    collator = make_seq2seq_collator(tokenizer, model)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    def _collate(feats):
        uids = [f["record_uid"] for f in feats]
        batch = collator([{k: v for k, v in f.items() if k != "record_uid"} for f in feats])
        batch["_uids"] = uids
        return batch

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    src_col = f"{SOURCE_FIELD}_norm" if f"{SOURCE_FIELD}_norm" in dataset.df.columns else SOURCE_FIELD
    tgt_col = f"{TARGET_FIELD}_norm" if f"{TARGET_FIELD}_norm" in dataset.df.columns else TARGET_FIELD
    uid_to_idx = {
        str(uid): i for i, uid in enumerate(dataset.df["record_uid"].astype(str).tolist())
    }
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            uids = batch.pop("_uids")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=int(generation_max_length),
                num_beams=int(num_beams),
            )
            preds = tokenizer.batch_decode(outs, skip_special_tokens=True)
            for i, uid in enumerate(uids):
                idx = uid_to_idx.get(str(uid))
                if idx is None:
                    raise RuntimeError(f"UID not found in dataset index: {uid}")
                row = dataset.df.iloc[idx]
                src_text = normalize_mt_text_v1(row[src_col])
                tgt_text = normalize_mt_text_v1(row[tgt_col])
                pred_ids = outs[i].detach().cpu().tolist()
                rows.append(
                    {
                        "record_uid": str(uid),
                        "text_bahnar": src_text,
                        "text_vi_reference": tgt_text,
                        "text_vi_prediction": normalize_mt_text_v1(preds[i]),
                        "source_token_length": count_tokens_no_special(tokenizer, src_text),
                        "target_token_length": count_tokens_no_special(tokenizer, tgt_text),
                        "prediction_token_length": count_generated_tokens(
                            pred_ids, pad_token_id=pad_id, eos_token_id=eos_id
                        ),
                        "target_char_length": int(len(tgt_text)),
                        "source_char_length": int(len(src_text)),
                    }
                )
    return rows


def evaluate_mt_predictions(pred_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hyps = [str(r["text_vi_prediction"]) for r in pred_rows]
    refs = [str(r["text_vi_reference"]) for r in pred_rows]
    metrics = mt_corpus_metrics(hyps, refs)
    metrics["prediction_count"] = len(pred_rows)
    return metrics


def assert_evaluate_matches_train_summary(
    train_summary: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    generation_config: Mapping[str, Any],
) -> None:
    """Fail-closed: evaluate must reuse train generation/model settings."""
    from src.mt_contract import assert_generation_config_matches, build_generation_config

    expected = train_summary.get("generation_config") or {}
    if not expected:
        expected = build_generation_config(
            model_id=str(train_summary.get("model_id")),
            model_revision=str(train_summary.get("model_revision")),
            tokenizer_fingerprint=str(train_summary.get("tokenizer_fingerprint")),
            max_source_length=int((train_summary.get("hparams") or {}).get("max_source_length") or generation_config["max_source_length"]),
            max_target_length=int((train_summary.get("hparams") or {}).get("max_target_length") or generation_config["max_target_length"]),
            generation_max_length=int((train_summary.get("hparams") or {}).get("generation_max_length") or generation_config["generation_max_length"]),
            num_beams=int((train_summary.get("hparams") or {}).get("num_beams") or generation_config["num_beams"]),
            metric_for_best_model=str((train_summary.get("hparams") or {}).get("metric_for_best_model") or generation_config["metric_for_best_model"]),
            greater_is_better=bool((train_summary.get("hparams") or {}).get("greater_is_better", generation_config["greater_is_better"])),
        )
    if str(train_summary.get("model_id")) != str(model_id):
        raise RuntimeError("evaluate MODEL_ID != train summary")
    if str(train_summary.get("model_revision")) != str(model_revision):
        raise RuntimeError("evaluate MODEL_REVISION != train summary")
    if str(train_summary.get("tokenizer_fingerprint")) != str(tokenizer_fingerprint):
        raise RuntimeError("evaluate tokenizer_fingerprint != train summary")
    assert_generation_config_matches(expected, generation_config)


def select_best_checkpoint_by_metric(
    history: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    greater_is_better: bool,
) -> Optional[str]:
    best_name = None
    best_val = None
    for row in history:
        if metric_key not in row:
            continue
        val = row[metric_key]
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            continue
        if best_val is None:
            best_val, best_name = val, row.get("checkpoint")
            continue
        better = val > best_val if greater_is_better else val < best_val
        if better:
            best_val, best_name = val, row.get("checkpoint")
    return str(best_name) if best_name else None
