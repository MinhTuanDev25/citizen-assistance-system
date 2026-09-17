"""
Full-training / resume-test orchestration helpers for Notebook 03.

Pilot and resume-test checkpoints must NEVER initialize full training.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.asr_utils import checkpoint_belongs_to_run, is_forbidden_test_path, sample_pilot_data
from src.asr_full_data import load_prepare_success

STATUS_FULL_RESUME_TEST = "SUCCESS_FULL_RESUME_TEST"
STATUS_FULL_TRAINING = "SUCCESS_FULL_TRAINING"
STATUS_FAILED = "FAILED"

RESUME_TEST_MARKER = "resume_test"
PILOT_MARKER = "pilot"
FULL_TRAIN_MARKER = "full_train"

RESUME_TEST_PHASE_A_STEPS = 100
RESUME_TEST_PHASE_B_STEPS = 200

STAGE_VERSION_PREPARE = "full_prepare_v1"
STAGE_VERSION_RESUME_TEST = "full_resume_test_v1"
STAGE_VERSION_TRAIN = "full_train_v1"
STAGE_VERSION_EVALUATE = "full_evaluate_v1"

DATA_CONTRACT_KEYS = (
    "dataset_id",
    "dataset_revision",
    # Immutable parquet snapshot sha: audio bytes are bound to it, and it moves
    # independently of the dataset revision when HF re-converts the dataset.
    "parquet_revision",
    "train_manifest_content_hash",
    "validation_manifest_content_hash",
    "vocab_fp",
    "processing_version",
    "min_duration",
    "max_duration",
    "target_sr",
    "pretrained_model_id",
    "pretrained_model_revision",
    "stage_version",
)

# Keep legacy alias for older call sites
TRAINING_CONTRACT_KEYS = DATA_CONTRACT_KEYS + ("experiment_id", "hparams")


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def contract_hash(payload: Dict[str, Any], *, keys: Sequence[str]) -> str:
    import hashlib

    subset = {k: payload.get(k) for k in keys if k in payload}
    return hashlib.sha256(_stable_json(subset).encode("utf-8")).hexdigest()


def build_data_contract(
    *,
    dataset_id: str,
    dataset_revision: str,
    parquet_revision: str,
    train_manifest_content_hash: str,
    validation_manifest_content_hash: str,
    vocab_fp: str,
    processing_version: str,
    min_duration: float,
    max_duration: float,
    target_sr: int,
    pretrained_model_id: str,
    pretrained_model_revision: str,
    stage_version: str = STAGE_VERSION_PREPARE,
) -> Dict[str, Any]:
    """Prepare-stage contract — no full-train hparams required."""
    if not str(parquet_revision or "").strip():
        raise ValueError("parquet_revision is required in the data contract")
    payload = {
        "dataset_id": str(dataset_id),
        "dataset_revision": str(dataset_revision),
        "parquet_revision": str(parquet_revision),
        "train_manifest_content_hash": str(train_manifest_content_hash),
        "validation_manifest_content_hash": str(validation_manifest_content_hash),
        "vocab_fp": str(vocab_fp),
        "processing_version": str(processing_version),
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "target_sr": int(target_sr),
        "pretrained_model_id": str(pretrained_model_id),
        "pretrained_model_revision": str(pretrained_model_revision),
        "stage_version": str(stage_version),
    }
    payload["contract_hash"] = contract_hash(payload, keys=DATA_CONTRACT_KEYS)
    return payload


def build_resume_test_contract(
    *,
    experiment_id: str,
    data_contract: Dict[str, Any],
    subset_size: int,
    seed: int = 42,
) -> Dict[str, Any]:
    payload = {
        **{k: data_contract[k] for k in DATA_CONTRACT_KEYS if k in data_contract},
        "experiment_id": str(experiment_id),
        "stage_version": STAGE_VERSION_RESUME_TEST,
        "subset_size": int(subset_size),
        "seed": int(seed),
        "data_contract_hash": data_contract.get("contract_hash"),
    }
    keys = list(DATA_CONTRACT_KEYS) + ["experiment_id", "subset_size", "seed", "data_contract_hash"]
    payload["contract_hash"] = contract_hash(payload, keys=keys)
    return payload


def build_train_contract(
    *,
    experiment_id: str,
    data_contract: Dict[str, Any],
    hparams: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        **{k: data_contract[k] for k in DATA_CONTRACT_KEYS if k in data_contract},
        "experiment_id": str(experiment_id),
        "stage_version": STAGE_VERSION_TRAIN,
        "hparams": dict(hparams),
        "data_contract_hash": data_contract.get("contract_hash"),
    }
    keys = list(DATA_CONTRACT_KEYS) + ["experiment_id", "hparams", "data_contract_hash"]
    payload["contract_hash"] = contract_hash(payload, keys=keys)
    return payload


def build_evaluate_contract(*, train_contract: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate binds the exact train contract (including hparams)."""
    payload = dict(train_contract)
    payload["stage_version"] = STAGE_VERSION_EVALUATE
    payload["train_contract_hash"] = train_contract.get("contract_hash")
    keys = list(DATA_CONTRACT_KEYS) + ["experiment_id", "hparams", "train_contract_hash"]
    payload["contract_hash"] = contract_hash(payload, keys=keys)
    return payload


def build_training_contract(
    *,
    experiment_id: str,
    dataset_id: str,
    dataset_revision: str,
    parquet_revision: str,
    train_manifest_content_hash: str,
    validation_manifest_content_hash: str,
    vocab_fp: str,
    processing_version: str,
    min_duration: float,
    max_duration: float,
    target_sr: int,
    pretrained_model_id: str,
    pretrained_model_revision: str,
    hparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper → train_contract when hparams provided, else data_contract+experiment."""
    data = build_data_contract(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        parquet_revision=parquet_revision,
        train_manifest_content_hash=train_manifest_content_hash,
        validation_manifest_content_hash=validation_manifest_content_hash,
        vocab_fp=vocab_fp,
        processing_version=processing_version,
        min_duration=min_duration,
        max_duration=max_duration,
        target_sr=target_sr,
        pretrained_model_id=pretrained_model_id,
        pretrained_model_revision=pretrained_model_revision,
    )
    if hparams:
        return build_train_contract(experiment_id=experiment_id, data_contract=data, hparams=hparams)
    out = dict(data)
    out["experiment_id"] = str(experiment_id)
    out["hparams"] = {}
    return out


def training_contract_matches(
    actual: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
    *,
    require_hparams: bool = False,
) -> bool:
    if not actual:
        return False
    # Field comparison is the authority. A self-declared ``contract_hash`` is
    # never a shortcut: a payload whose fields were edited after hashing would
    # otherwise match on a stale hash and let a wrong dataset/vocab through.
    keys = list(DATA_CONTRACT_KEYS)
    if "experiment_id" in expected:
        keys.append("experiment_id")
    for key in keys:
        if key not in expected:
            continue
        if key == "stage_version":
            continue  # stage wrappers may differ; hash/fields carry identity
        if str(actual.get(key)) != str(expected.get(key)):
            return False
    if require_hparams:
        if dict(actual.get("hparams") or {}) != dict(expected.get("hparams") or {}):
            return False
    return True


def assert_training_contract(
    actual: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
    *,
    label: str,
    require_hparams: bool = False,
) -> None:
    if not training_contract_matches(actual, expected, require_hparams=require_hparams):
        raise RuntimeError(
            f"{label}: training_contract mismatch "
            f"(expected experiment={expected.get('experiment_id')}, "
            f"dataset={expected.get('dataset_id')}@{expected.get('dataset_revision')}, "
            f"hash={expected.get('contract_hash')})"
        )


def extract_training_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pull contract fields from a summary / fingerprint payload."""
    out = {}
    for key in list(DATA_CONTRACT_KEYS) + ["experiment_id", "hparams", "contract_hash", "data_contract_hash", "train_contract_hash"]:
        if key in payload:
            out[key] = payload[key]
    for nested_key in ("training_contract", "data_contract", "resume_test_contract", "train_contract", "evaluate_contract"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            for key, val in nested.items():
                if key not in out:
                    out[key] = val
    return out


# ---------------------------------------------------------------------------
# Steps / TrainingArguments builders
# ---------------------------------------------------------------------------

def compute_steps_per_epoch(
    n_train: int,
    *,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    n_devices: int = 1,
) -> int:
    """HF-style steps per epoch from eligible train size and effective batch."""
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    eff = max(1, int(per_device_train_batch_size) * int(gradient_accumulation_steps) * int(n_devices))
    return int(math.ceil(int(n_train) / eff))


def build_full_training_hparams(
    *,
    n_train: int,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    warmup_ratio: float = 0.05,
    seed: int = 42,
    save_steps: int = 500,
    save_total_limit: int = 2,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    n_devices: int = 1,
) -> Dict[str, Any]:
    if num_train_epochs > 3:
        raise ValueError("num_train_epochs must be <= 3 for this thesis stage")
    spe = compute_steps_per_epoch(
        n_train,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        n_devices=n_devices,
    )
    max_steps = int(math.ceil(spe * float(num_train_epochs)))
    return {
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "save_only_model": False,
        "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
        "steps_per_epoch": spe,
        "num_train_epochs": float(num_train_epochs),
        "max_steps": max_steps,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
    }


# ---------------------------------------------------------------------------
# Checkpoint identity / auto-resume
# ---------------------------------------------------------------------------

def experiment_checkpoint_dir(
    root: Union[str, Path],
    experiment_id: str,
    *,
    kind: str,
) -> Path:
    """kind in {full_train, resume_test, pilot} — separate directories."""
    return Path(root) / str(kind) / str(experiment_id)


def write_checkpoint_fingerprint(
    checkpoint_dir: Union[str, Path],
    *,
    experiment_id: str,
    kind: str,
    global_step: int,
    extra: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
    preserve_existing: bool = True,
) -> Path:
    """
    Write experiment fingerprint at experiment root.

    Set overwrite=False to refuse clobbering an existing fingerprint
    (use after validation — never stamp a new identity before checks).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "full_experiment_fingerprint.json"
    if path.is_file() and not overwrite:
        return path
    payload: Dict[str, Any] = {}
    if preserve_existing and path.is_file():
        try:
            payload.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            payload = {}
    payload.update({
        "experiment_id": experiment_id,
        "kind": kind,
        "global_step": int(global_step),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def ensure_experiment_fingerprint(
    experiment_dir: Union[str, Path],
    *,
    experiment_id: str,
    kind: str,
    expected_contract: Optional[Dict[str, Any]] = None,
    allow_create_if_empty: bool = True,
) -> Dict[str, Any]:
    """
    Fail-closed fingerprint policy:
      - If step checkpoints exist but fingerprint missing/wrong → raise (do NOT auto-stamp).
      - If directory empty and allow_create_if_empty → may create after caller validates.
    """
    root = Path(experiment_dir)
    fp = read_checkpoint_fingerprint(root)
    steps = list_step_checkpoints(root) if root.is_dir() else []
    if steps and not fp:
        raise RuntimeError(
            f"Experiment dir has checkpoints but missing fingerprint (fail-closed): {root}"
        )
    if steps and fp:
        if fp.get("kind") != kind or str(fp.get("experiment_id")) != str(experiment_id):
            raise RuntimeError(
                f"Experiment fingerprint mismatch (fail-closed): {root} fp={fp}"
            )
        if expected_contract is not None:
            assert_training_contract(
                extract_training_contract(fp),
                expected_contract,
                label=f"fingerprint {root}",
                require_hparams=bool(expected_contract.get("hparams")),
            )
        return fp
    if not steps and not fp and not allow_create_if_empty:
        raise RuntimeError(f"No fingerprint and create forbidden: {root}")
    return fp or {}



def assert_checkpoint_complete_for_resume(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Fail if model/optimizer/scheduler/RNG artifacts are incomplete."""
    info = inspect_trainer_checkpoint(checkpoint_path)
    missing = [
        name for name, ok in [
            ("model", info["model_ok"]),
            ("optimizer", info["optimizer_ok"]),
            ("scheduler", info["scheduler_ok"]),
            ("rng", info["rng_ok"]),
        ] if not ok
    ]
    if missing:
        raise RuntimeError(
            f"Checkpoint incomplete for resume (missing {missing}): {checkpoint_path}"
        )
    return info


def read_checkpoint_fingerprint(checkpoint_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = Path(checkpoint_dir) / "full_experiment_fingerprint.json"
    if not path.is_file():
        # Also accept fingerprint one level up (trainer checkpoint-NNN/)
        parent = Path(checkpoint_dir).parent / "full_experiment_fingerprint.json"
        if parent.is_file():
            path = parent
        else:
            # look inside checkpoint dir only
            return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_for(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    Resolve the fingerprint for a checkpoint directory.

    The trainer writes ``checkpoint-N`` under the experiment root, while the
    Drive store nests them one level deeper (``<exp>/ckpts/checkpoint-N``), so
    walk up a bounded number of levels instead of assuming a fixed depth.
    """
    node = Path(checkpoint_path)
    for _ in range(4):
        fp = read_checkpoint_fingerprint(node)
        if fp:
            return fp
        if node.parent == node:
            break
        node = node.parent
    return None


def is_forbidden_init_checkpoint(checkpoint_path: Union[str, Path, None]) -> bool:
    """True if path looks like pilot or resume_test (must not init full training)."""
    if not checkpoint_path:
        return False
    p = str(Path(checkpoint_path)).replace("\\", "/").lower()
    if f"/{PILOT_MARKER}/" in p or p.endswith(f"/{PILOT_MARKER}"):
        return True
    if f"/{RESUME_TEST_MARKER}/" in p or p.endswith(f"/{RESUME_TEST_MARKER}"):
        return True
    fp = _fingerprint_for(Path(checkpoint_path))
    if fp and fp.get("kind") in {PILOT_MARKER, RESUME_TEST_MARKER}:
        return True
    return False


def assert_checkpoint_allowed_for_full_train(
    checkpoint_path: Union[str, Path, None],
    *,
    experiment_id: str,
    require_complete: bool = True,
    expected_contract: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Validate resume checkpoint for FULL_STAGE=train.
    Raises if missing fingerprint, wrong experiment, or pilot/resume_test origin.
    Path must be under configured ``full_train/<experiment_id>/``.
    """
    if not checkpoint_path:
        raise RuntimeError("Resume checkpoint path is empty")
    cp = Path(checkpoint_path)
    if not cp.exists():
        raise RuntimeError(f"Resume checkpoint does not exist: {cp}")
    parts = [p.lower() for p in cp.parts]
    if FULL_TRAIN_MARKER not in parts:
        raise RuntimeError(
            f"Checkpoint must live under a '{FULL_TRAIN_MARKER}/' directory: {cp}"
        )
    # Exact experiment directory segment after full_train
    try:
        ft_idx = parts.index(FULL_TRAIN_MARKER)
        exp_part = parts[ft_idx + 1] if ft_idx + 1 < len(parts) else ""
    except ValueError:
        exp_part = ""
    if exp_part != str(experiment_id).lower():
        raise RuntimeError(
            f"Checkpoint experiment_id mismatch: must live under the configured root "
            f"'{FULL_TRAIN_MARKER}/{experiment_id}/' but found segment {exp_part!r}: {cp}"
        )
    if PILOT_MARKER in parts or RESUME_TEST_MARKER in parts:
        raise RuntimeError(
            "Refusing to initialize full training from pilot or resume_test checkpoint path: "
            f"{cp}"
        )
    if is_forbidden_init_checkpoint(cp):
        raise RuntimeError(
            "Refusing to initialize full training from pilot or resume_test checkpoint: "
            f"{cp}"
        )
    fp = _fingerprint_for(cp)
    if not fp:
        raise RuntimeError(
            f"Checkpoint lacks full_experiment_fingerprint.json — refusing resume: {cp}"
        )
    if fp.get("kind") != FULL_TRAIN_MARKER:
        raise RuntimeError(
            f"Checkpoint kind={fp.get('kind')!r} is not '{FULL_TRAIN_MARKER}': {cp}"
        )
    if str(fp.get("experiment_id")) != str(experiment_id):
        raise RuntimeError(
            f"Checkpoint experiment_id={fp.get('experiment_id')!r} != {experiment_id!r}"
        )
    if expected_contract is not None:
        assert_training_contract(
            extract_training_contract(fp),
            expected_contract,
            label=f"checkpoint {cp}",
            require_hparams=bool(expected_contract.get("hparams")),
        )
    if require_complete and cp.name.startswith("checkpoint-"):
        assert_checkpoint_complete_for_resume(cp)
    return str(cp)


def list_step_checkpoints(experiment_dir: Union[str, Path]) -> List[Path]:
    root = Path(experiment_dir)
    if not root.is_dir():
        return []
    out = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-", 1)[1])
            except ValueError:
                continue
            out.append((step, p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def find_latest_valid_checkpoint(
    experiment_dir: Union[str, Path],
    *,
    experiment_id: str,
    kind: str = FULL_TRAIN_MARKER,
) -> Optional[Path]:
    """RESUME_POLICY=auto — latest checkpoint with matching fingerprint."""
    root = Path(experiment_dir)
    # Ensure root fingerprint matches
    root_fp = read_checkpoint_fingerprint(root)
    if root_fp:
        if str(root_fp.get("experiment_id")) != str(experiment_id):
            return None
        if root_fp.get("kind") != kind:
            return None
    candidates = list_step_checkpoints(root)
    for cp in reversed(candidates):
        try:
            if kind == FULL_TRAIN_MARKER:
                assert_checkpoint_allowed_for_full_train(cp, experiment_id=experiment_id)
            else:
                fp = _fingerprint_for(cp)
                if not fp or fp.get("kind") != kind or str(fp.get("experiment_id")) != str(experiment_id):
                    continue
            return cp
        except RuntimeError:
            continue
    return None


def resolve_resume_checkpoint(
    *,
    resume_policy: str,
    experiment_dir: Path,
    experiment_id: str,
    explicit_path: Optional[Union[str, Path]] = None,
    kind: str = FULL_TRAIN_MARKER,
) -> Optional[str]:
    """
    resume_policy:
      - 'auto': latest valid checkpoint for this experiment
      - 'never': None
      - 'explicit': require explicit_path
    """
    policy = str(resume_policy or "auto").lower()
    if policy == "never":
        return None
    if policy == "explicit":
        if not explicit_path:
            raise RuntimeError("RESUME_POLICY=explicit requires a checkpoint path")
        if kind == FULL_TRAIN_MARKER:
            return assert_checkpoint_allowed_for_full_train(
                explicit_path, experiment_id=experiment_id
            )
        return str(explicit_path)
    if policy == "auto":
        found = find_latest_valid_checkpoint(
            experiment_dir, experiment_id=experiment_id, kind=kind
        )
        return str(found) if found else None
    raise ValueError(f"Unknown RESUME_POLICY: {resume_policy!r}")


# ---------------------------------------------------------------------------
# Atomic checkpoint copy (local → Drive FULL_STATE_DIR)
# ---------------------------------------------------------------------------

def atomic_restore_checkpoint_dir(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Copy one checkpoint/snapshot directory into place atomically.

    Scoped to a single checkpoint on purpose: pointing this at an experiment root
    would replace the whole Drive tree and destroy the versioned snapshot store.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp_copy")
    if tmp.exists():
        shutil.rmtree(tmp)
    if src.is_dir():
        shutil.copytree(src, tmp)
    else:
        shutil.copy2(src, tmp)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.replace(str(tmp), str(dst))
    return dst


def drive_experiment_dir(
    full_state_dir: Union[str, Path],
    experiment_id: str,
    *,
    kind: str,
) -> Path:
    return experiment_checkpoint_dir(Path(full_state_dir) / "checkpoints", experiment_id, kind=kind)


CHECKPOINT_STORE_DIRNAME = "ckpts"
SNAPSHOT_MANIFEST_NAME = "manifest.json"
DEFAULT_DRIVE_CHECKPOINT_BUDGET_BYTES = 15 * 1024 ** 3


def measure_dir_bytes(path: Union[str, Path]) -> int:
    """Recursive on-disk size of a directory (0 when absent)."""
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return int(root.stat().st_size)
    total = 0
    for entry in root.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += int(entry.stat().st_size)
        except OSError:
            continue
    return total


def estimate_checkpoint_bytes(
    *,
    n_parameters: int,
    save_only_model: bool = False,
    bytes_per_param: int = 4,
    optimizer_state_multiplier: int = 2,
    overhead_bytes: int = 64 * 1024 ** 2,
) -> Dict[str, Any]:
    """
    Pre-training estimate of one checkpoint, derived from the model itself.

    Trainer keeps fp32 master weights even under fp16, and AdamW adds two state
    tensors per parameter, so the optimizer dominates the footprint.
    """
    params = int(n_parameters)
    weights = params * int(bytes_per_param)
    optimizer = 0 if save_only_model else weights * int(optimizer_state_multiplier)
    total = weights + optimizer + int(overhead_bytes)
    return {
        "n_parameters": params,
        "weight_bytes": weights,
        "optimizer_bytes": optimizer,
        "overhead_bytes": int(overhead_bytes),
        "checkpoint_bytes": total,
    }


def plan_checkpoint_retention(
    *,
    candidate_steps: Sequence[int],
    best_step: Optional[int] = None,
    previous_latest_step: Optional[int] = None,
    save_total_limit: int = 2,
) -> Dict[str, Any]:
    """
    Which checkpoint steps Drive should persist, in priority order.

    Latest is needed to resume, best is what evaluate loads, previous-latest is
    the rollback. Ordinary step-ordered pruning would delete ``best`` as soon as
    two newer checkpoints exist, which silently breaks evaluate.
    """
    steps = sorted({int(s) for s in candidate_steps})
    if not steps:
        return {"keep": [], "latest": None, "best": None, "previous_latest": None, "drop": []}
    latest = steps[-1]
    priority: List[int] = [latest]
    reasons = {latest: "latest"}
    if best_step is not None and int(best_step) in steps and int(best_step) not in reasons:
        priority.append(int(best_step))
        reasons[int(best_step)] = "best"
    prev = int(previous_latest_step) if previous_latest_step is not None else None
    if prev is None:
        older = [s for s in steps if s not in reasons]
        prev = older[-1] if older else None
    if prev is not None and prev in steps and prev not in reasons:
        priority.append(prev)
        reasons[prev] = "previous_latest"
    limit = max(1, int(save_total_limit))
    # Latest and best are mandatory; previous-latest is a bonus if the limit allows.
    mandatory = [s for s in priority if reasons[s] in ("latest", "best")]
    keep = mandatory[:]
    for step in priority:
        if step in keep:
            continue
        if len(keep) >= max(limit, len(mandatory)):
            break
        keep.append(step)
    keep = sorted(set(keep))
    return {
        "keep": keep,
        "latest": latest,
        "best": int(best_step) if best_step is not None else None,
        "previous_latest": prev,
        "reasons": {int(s): reasons[s] for s in keep if s in reasons},
        "drop": [s for s in steps if s not in keep],
    }


def drive_experiment_protected_bytes(
    full_state_dir: Union[str, Path],
    experiment_id: str,
    *,
    kind: str,
) -> int:
    """Bytes the current LATEST snapshot (the last-known-good) already occupies."""
    snapshot = resolve_drive_latest_snapshot(full_state_dir, experiment_id, kind=kind)
    if snapshot is None:
        return 0
    total = measure_dir_bytes(snapshot)
    for ck in resolve_snapshot_checkpoints(snapshot):
        total += measure_dir_bytes(ck)
    return total


def assert_drive_checkpoint_budget(
    *,
    protected_bytes: int,
    upload_bytes: int,
    budget_bytes: int = DEFAULT_DRIVE_CHECKPOINT_BUDGET_BYTES,
    label: str = "drive checkpoints",
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Budget gate for Drive using explicit accounting.

    ``/content/drive`` is a FUSE mount whose ``shutil.disk_usage`` does not
    reflect the account quota, so the ceiling has to be configured and the bytes
    counted by us. Never resolved by deleting the last-known-good.
    """
    protected = int(protected_bytes)
    upload = int(upload_bytes)
    peak = protected + upload
    report = {
        "label": label,
        "protected_bytes": protected,
        "upload_bytes": upload,
        "peak_bytes": peak,
        "budget_bytes": int(budget_bytes),
        "ok": peak <= int(budget_bytes),
        "detail": dict(detail or {}),
    }
    if not report["ok"]:
        raise RuntimeError(
            f"Drive checkpoint budget exceeded for {label}: "
            f"protected={protected / 1e9:.1f}GB + upload={upload / 1e9:.1f}GB "
            f"= {peak / 1e9:.1f}GB > budget={int(budget_bytes) / 1e9:.1f}GB. "
            f"Refusing to drop the last-known-good to make room; raise "
            f"DRIVE_CHECKPOINT_BUDGET_BYTES or lower save_total_limit. "
            f"detail={report['detail']}"
        )
    return report
REQUIRED_CHECKPOINT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
MODEL_FILE_CANDIDATES = ("model.safetensors", "pytorch_model.bin")


def _verify_checkpoint_dir_complete(path: Path) -> None:
    """A checkpoint counts as uploaded only when every resume artifact is there."""
    missing = [n for n in REQUIRED_CHECKPOINT_FILES if not (path / n).is_file()]
    if not any((path / n).is_file() for n in MODEL_FILE_CANDIDATES):
        missing.append("|".join(MODEL_FILE_CANDIDATES))
    if missing:
        raise RuntimeError(f"Checkpoint incomplete after copy (missing {missing}): {path}")


def _verify_checkpoint_tree(root: Path, *, experiment_id: str, kind: str) -> None:
    fp = read_checkpoint_fingerprint(root)
    if not fp or fp.get("kind") != kind or str(fp.get("experiment_id")) != str(experiment_id):
        raise RuntimeError(f"Snapshot fingerprint invalid under {root}: {fp}")
    steps = list_step_checkpoints(root)
    if not steps:
        # Allow empty freshly-initialized tree only if fingerprint present with step 0
        if int(fp.get("global_step") or 0) > 0:
            raise RuntimeError(f"Snapshot claims global_step>0 but has no checkpoints: {root}")


def _snapshot_versions(dest_root: Path) -> List[int]:
    snap_root = dest_root / "snapshots"
    if not snap_root.is_dir():
        return []
    return sorted(
        int(p.name[1:]) for p in snap_root.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    )


def read_snapshot_manifest(snapshot_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = Path(snapshot_dir) / SNAPSHOT_MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sync_experiment_checkpoints_to_drive(
    local_experiment_dir: Union[str, Path],
    full_state_dir: Union[str, Path],
    *,
    experiment_id: str,
    kind: str,
    require_drive: bool = True,
    best_checkpoint_name: Optional[str] = None,
    save_total_limit: int = 2,
    budget_bytes: int = DEFAULT_DRIVE_CHECKPOINT_BUDGET_BYTES,
) -> Optional[Path]:
    """
    Incremental versioned sync: upload only checkpoints Drive does not have yet,
    verify them, then commit a new ``snapshots/vN`` manifest and flip ``LATEST``.

    Copying the whole experiment tree on every save re-uploads gigabytes of
    already-synced checkpoints, so each ``checkpoint-N`` is written once into an
    immutable ``ckpts/`` store and snapshots merely reference it by name. The
    previous LATEST snapshot and every checkpoint it references stay untouched,
    so an interrupted sync always leaves a usable last-known-good.
    """
    local = Path(local_experiment_dir)
    if not local.is_dir():
        if require_drive:
            raise RuntimeError(f"Local experiment checkpoint dir missing for Drive sync: {local}")
        return None
    drive_root = Path(full_state_dir)
    if not drive_root.exists():
        if require_drive:
            raise RuntimeError(
                f"Drive FULL_STATE_DIR missing for checkpoint sync (fail-closed): {drive_root}"
            )
        return None
    fp_name = "full_experiment_fingerprint.json"
    if not (local / fp_name).is_file():
        raise RuntimeError(f"Refusing Drive sync without local fingerprint: {local / fp_name}")

    dest_root = drive_experiment_dir(drive_root, experiment_id, kind=kind)
    store = dest_root / CHECKPOINT_STORE_DIRNAME
    store.mkdir(parents=True, exist_ok=True)

    # The snapshot about to be replaced is the last-known-good until LATEST flips.
    lkg_snapshot = resolve_drive_latest_snapshot(drive_root, experiment_id, kind=kind)
    lkg_step = snapshot_max_step(lkg_snapshot) if lkg_snapshot else None

    local_steps = list_step_checkpoints(local)
    # Uploads can only come from local; the store is the destination, and a
    # partial copy there must never be mistaken for a usable source.
    sources: Dict[str, Path] = {p.name: p for p in local_steps}
    store_names = {
        p.name for p in store.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")
    }

    def _step_of(name: str) -> int:
        return int(str(name).split("-")[-1])

    retention = plan_checkpoint_retention(
        candidate_steps=[_step_of(n) for n in set(sources) | store_names],
        best_step=_step_of(best_checkpoint_name) if best_checkpoint_name else None,
        previous_latest_step=lkg_step if lkg_step and lkg_step >= 0 else None,
        save_total_limit=save_total_limit,
    )
    keep_names = [f"checkpoint-{s}" for s in retention["keep"]]

    # Budget is checked before any upload: protected bytes already on Drive plus
    # the temporary peak of what we are about to copy.
    pending = [n for n in keep_names if not (store / n).is_dir()]
    protected_bytes = sum(measure_dir_bytes(store / n) for n in keep_names if (store / n).is_dir())
    upload_bytes = sum(measure_dir_bytes(sources[n]) for n in pending if n in sources)
    budget_report = assert_drive_checkpoint_budget(
        protected_bytes=protected_bytes,
        upload_bytes=upload_bytes,
        budget_bytes=budget_bytes,
        label=f"{kind}/{experiment_id}",
        detail={"keep": keep_names, "uploading": pending, "lkg_step": lkg_step},
    )

    uploaded: List[str] = []
    for name in keep_names:
        source = sources.get(name)
        target = store / name
        if target.is_dir():
            try:
                _verify_checkpoint_dir_complete(target)
                continue  # already durable, do not re-upload
            except RuntimeError:
                shutil.rmtree(target, ignore_errors=True)  # partial from an interrupted run
        if source is None or not source.is_dir():
            continue
        tmp = store / f"{name}.tmp_copy"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(source, tmp)
        _verify_checkpoint_dir_complete(tmp)
        os.replace(str(tmp), str(target))
        uploaded.append(name)

    names = sorted(
        (n for n in keep_names if (store / n).is_dir()),
        key=_step_of,
    )
    if not names:
        raise RuntimeError(
            f"No complete checkpoint available to snapshot under {store} "
            f"(retention wanted {keep_names})"
        )
    for name in names:
        _verify_checkpoint_dir_complete(store / name)

    version = (max(_snapshot_versions(dest_root)) + 1) if _snapshot_versions(dest_root) else 1
    snap_dir = dest_root / "snapshots" / f"v{version}"
    tmp_snap = dest_root / "snapshots" / f"v{version}.tmp_copy"
    if tmp_snap.exists():
        shutil.rmtree(tmp_snap)
    tmp_snap.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local / fp_name, tmp_snap / fp_name)
    fp = json.loads((local / fp_name).read_text(encoding="utf-8"))
    manifest = {
        "version": f"v{version}",
        "experiment_id": str(experiment_id),
        "kind": str(kind),
        "global_step": int(fp.get("global_step") or 0),
        "checkpoints": names,
        "uploaded_now": uploaded,
        "store_dir": CHECKPOINT_STORE_DIRNAME,
        "retention": retention,
        "budget": budget_report,
        "previous_latest_snapshot": lkg_snapshot.name if lkg_snapshot else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (tmp_snap / SNAPSHOT_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(str(tmp_snap), str(snap_dir))

    # Verify the committed snapshot resolves before advertising it as LATEST.
    resolved = resolve_snapshot_checkpoints(snap_dir)
    if len(resolved) != len(names):
        raise RuntimeError(
            f"Snapshot v{version} references {len(names)} checkpoints but only "
            f"{len(resolved)} resolve under {store}"
        )
    latest = dest_root / "LATEST"
    tmp_latest = dest_root / "LATEST.tmp"
    tmp_latest.write_text(f"v{version}\n", encoding="utf-8")
    os.replace(str(tmp_latest), str(latest))
    shutil.copy2(local / fp_name, dest_root / fp_name)

    # Only now is the new snapshot the last-known-good, so only now may old data
    # go. Anything the committed snapshot still references is untouchable.
    keep_set = set(names)
    for entry in sorted(store.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.endswith(".tmp_copy"):
            shutil.rmtree(entry, ignore_errors=True)
            continue
        if entry.name.startswith("checkpoint-") and entry.name not in keep_set:
            shutil.rmtree(entry, ignore_errors=True)
    for old in _snapshot_versions(dest_root):
        if old >= version - 1:
            continue  # keep the committed snapshot and one rollback pointer
        shutil.rmtree(dest_root / "snapshots" / f"v{old}", ignore_errors=True)
    return snap_dir


def resolve_snapshot_checkpoints(snapshot_dir: Union[str, Path]) -> List[Path]:
    """Checkpoint directories a snapshot points at (manifest-based or inline)."""
    snap = Path(snapshot_dir)
    manifest = read_snapshot_manifest(snap)
    if manifest is None:
        return list_step_checkpoints(snap)  # legacy inline snapshot
    store = snap.parent.parent / str(manifest.get("store_dir") or CHECKPOINT_STORE_DIRNAME)
    out = []
    for name in manifest.get("checkpoints") or []:
        p = store / str(name)
        if p.is_dir():
            out.append(p)
    return sorted(out, key=lambda p: int(p.name.split("-")[-1]))


def resolve_drive_latest_snapshot(
    full_state_dir: Union[str, Path],
    experiment_id: str,
    *,
    kind: str,
) -> Optional[Path]:
    dest_root = drive_experiment_dir(full_state_dir, experiment_id, kind=kind)
    latest = dest_root / "LATEST"
    if latest.is_file():
        version = latest.read_text(encoding="utf-8").strip()
        snap = dest_root / "snapshots" / version
        if snap.is_dir():
            return snap
    # Legacy flat layout fallback
    if dest_root.is_dir() and list_step_checkpoints(dest_root):
        return dest_root
    return None


def snapshot_max_step(snapshot_dir: Union[str, Path]) -> int:
    steps = [int(p.name.split("-")[-1]) for p in resolve_snapshot_checkpoints(snapshot_dir)]
    return max(steps) if steps else -1


def make_drive_checkpoint_sync_callback(
    *,
    local_experiment_dir: Union[str, Path],
    full_state_dir: Union[str, Path],
    experiment_id: str,
    kind: str = FULL_TRAIN_MARKER,
    training_contract: Optional[Dict[str, Any]] = None,
):
    """HF TrainerCallback: versioned Drive sync on every save."""
    try:
        from transformers import TrainerCallback
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"transformers TrainerCallback unavailable: {exc}") from exc

    class DriveCheckpointSyncCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            try:
                extra = dict(training_contract or {})
                write_checkpoint_fingerprint(
                    local_experiment_dir,
                    experiment_id=experiment_id,
                    kind=kind,
                    global_step=int(getattr(state, "global_step", 0) or 0),
                    extra=extra or None,
                    overwrite=True,
                    preserve_existing=True,
                )
                sync_experiment_checkpoints_to_drive(
                    local_experiment_dir,
                    full_state_dir,
                    experiment_id=experiment_id,
                    kind=kind,
                    require_drive=True,
                )
            except Exception as sync_exc:
                raise RuntimeError(
                    f"Drive checkpoint sync failed at step={getattr(state, 'global_step', None)}: "
                    f"{type(sync_exc).__name__}: {sync_exc}"
                ) from sync_exc
            return control

    return DriveCheckpointSyncCallback()


def restore_experiment_checkpoints_from_drive(
    local_experiment_dir: Union[str, Path],
    full_state_dir: Union[str, Path],
    *,
    experiment_id: str,
    kind: str,
) -> Optional[Path]:
    """
    Merge the LATEST Drive snapshot into the local experiment dir.

    Local is not automatically preferred: a Colab session may hold a truncated
    local tree while Drive has newer steps (or vice versa), so both sides are
    validated and every checkpoint the snapshot references is materialised
    locally. Existing local checkpoints are kept, never overwritten blindly.
    """
    local = Path(local_experiment_dir)
    snap = resolve_drive_latest_snapshot(full_state_dir, experiment_id, kind=kind)
    local_steps = list_step_checkpoints(local) if local.is_dir() else []
    if snap is None:
        if local_steps:
            ensure_experiment_fingerprint(
                local, experiment_id=experiment_id, kind=kind, allow_create_if_empty=False
            )
            return local
        return None
    fp = read_checkpoint_fingerprint(snap)
    if not fp or fp.get("kind") != kind or str(fp.get("experiment_id")) != str(experiment_id):
        raise RuntimeError(
            f"Drive checkpoint fingerprint mismatch for restore: {snap} fp={fp}"
        )
    if local_steps:
        ensure_experiment_fingerprint(
            local, experiment_id=experiment_id, kind=kind, allow_create_if_empty=False
        )
    local.mkdir(parents=True, exist_ok=True)
    fp_name = "full_experiment_fingerprint.json"
    local_max = max((int(p.name.split("-")[-1]) for p in local_steps), default=-1)
    if not (local / fp_name).is_file() or snapshot_max_step(snap) > local_max:
        src_fp = snap / fp_name
        if src_fp.is_file():
            shutil.copy2(src_fp, local / fp_name)
    have = {p.name for p in local_steps}
    for ck in resolve_snapshot_checkpoints(snap):
        if ck.name in have:
            continue
        atomic_restore_checkpoint_dir(ck, local / ck.name)
    return local


def resolve_best_checkpoint_from_drive(
    full_state_dir: Union[str, Path],
    *,
    experiment_id: str,
    train_summary: Dict[str, Any],
    expected_contract: Optional[Dict[str, Any]] = None,
    local_experiment_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Restore LATEST snapshot then resolve the *best* checkpoint from train summary.
    Never silently substitutes latest step checkpoint for best.
    """
    preferred = train_summary.get("best_checkpoint")
    if not preferred:
        raise RuntimeError("train summary missing best_checkpoint — refuse latest fallback")
    name = Path(str(preferred)).name
    if not name.startswith("checkpoint-"):
        raise RuntimeError(f"train summary best_checkpoint is not a step checkpoint: {preferred!r}")

    if local_experiment_dir is not None:
        restored = restore_experiment_checkpoints_from_drive(
            local_experiment_dir, full_state_dir,
            experiment_id=experiment_id, kind=FULL_TRAIN_MARKER,
        )
        if restored is None:
            raise RuntimeError("No Drive LATEST snapshot available for evaluate")
        root = Path(local_experiment_dir)
        candidate = root / name
    else:
        snap = resolve_drive_latest_snapshot(
            full_state_dir, experiment_id, kind=FULL_TRAIN_MARKER
        )
        if snap is None:
            raise RuntimeError("No Drive LATEST snapshot available for evaluate")
        root = snap
        by_name = {p.name: p for p in resolve_snapshot_checkpoints(snap)}
        candidate = by_name.get(name, snap / name)
    if not candidate.exists():
        raise RuntimeError(
            f"Best checkpoint {name} not found under restored tree {root} — "
            "refusing to substitute the latest checkpoint"
        )
    assert_checkpoint_allowed_for_full_train(
        candidate,
        experiment_id=experiment_id,
        expected_contract=expected_contract,
    )
    if expected_contract is not None:
        assert_training_contract(
            extract_training_contract(_fingerprint_for(candidate) or {}),
            expected_contract,
            label="evaluate best checkpoint",
            require_hparams=True,
        )
    return str(candidate)


def new_session_token() -> Dict[str, Any]:
    """
    Identity of the current Python process, used to prove Phase B is a new session.

    Boot-relative process start time plus PID plus a random nonce cannot be
    reproduced by a second Trainer inside the same process.
    """
    import uuid

    try:
        started = os.stat(f"/proc/{os.getpid()}").st_mtime
    except Exception:
        started = None
    return {
        "pid": int(os.getpid()),
        "nonce": uuid.uuid4().hex,
        "process_started": started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def assert_cross_session_resume(
    phase_a_payload: Dict[str, Any],
    *,
    current_session: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fail-closed proof that Phase A and Phase B ran in two different processes.

    Same-process resume can pass every state check while still reusing live
    weights, so the resume test would prove nothing about a Colab reset.
    """
    a = dict((phase_a_payload or {}).get("session") or {})
    if not a:
        raise RuntimeError(
            "Phase A artifact carries no session token — cannot prove two-session resume"
        )
    b = dict(current_session or {})
    if not b:
        raise RuntimeError("Phase B session token missing")
    if str(a.get("nonce")) == str(b.get("nonce")):
        raise RuntimeError(
            "Phase B reused the Phase A session token — resume test must run in a new process"
        )
    same_pid = int(a.get("pid") or -1) == int(b.get("pid") or -2)
    same_start = (
        a.get("process_started") is not None
        and a.get("process_started") == b.get("process_started")
    )
    if same_pid and same_start:
        raise RuntimeError(
            f"Phase A and Phase B ran in the same process (pid={b.get('pid')}) — "
            "run Phase B after a runtime restart"
        )
    return {
        "two_sessions": True,
        "phase_a_pid": a.get("pid"),
        "phase_b_pid": b.get("pid"),
        "phase_a_nonce": a.get("nonce"),
        "phase_b_nonce": b.get("nonce"),
    }


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(str(path), map_location="cpu")


def _model_param_probe(model: Any, *, n_tensors: int = 6, n_values: int = 2048) -> Dict[str, str]:
    """
    Hash a deterministic slice of a few parameters instead of the full state dict.

    Loading 1.2GB of weights twice inside a Colab session to compare them is
    wasteful; evenly spaced slices are enough to detect a non-restored model.
    """
    import hashlib

    import torch

    sd = model.state_dict()
    names = sorted(k for k, v in sd.items() if hasattr(v, "numel") and v.numel() >= 8)
    if not names:
        return {}
    step = max(1, len(names) // int(n_tensors))
    picked = names[::step][:n_tensors]
    out: Dict[str, str] = {}
    for name in picked:
        flat = sd[name].detach().reshape(-1)[: int(n_values)].to(torch.float32).cpu().numpy()
        out[name] = hashlib.sha256(flat.tobytes()).hexdigest()
    return out


def _checkpoint_param_probe(
    checkpoint_dir: Path, *, names: Sequence[str], n_values: int = 2048
) -> Dict[str, str]:
    """Same probe computed straight from the checkpoint file, without loading it fully."""
    import hashlib

    import numpy as np

    safe = Path(checkpoint_dir) / "model.safetensors"
    out: Dict[str, str] = {}
    if safe.is_file():
        from safetensors import safe_open

        with safe_open(str(safe), framework="np") as handle:
            available = set(handle.keys())
            for name in names:
                if name not in available:
                    continue
                sliced = handle.get_slice(name)
                shape = sliced.get_shape()
                total = int(np.prod(shape)) if shape else 0
                if total == 0:
                    continue
                # slice along dim 0 then flatten, enough rows to cover n_values
                rows = shape[0]
                per_row = max(1, total // max(1, rows))
                take = min(rows, max(1, -(-int(n_values) // per_row)))
                chunk = np.asarray(sliced[0:take]).reshape(-1)[: int(n_values)]
                out[name] = hashlib.sha256(
                    chunk.astype(np.float32, copy=False).tobytes()
                ).hexdigest()
        return out
    bin_path = Path(checkpoint_dir) / "pytorch_model.bin"
    if bin_path.is_file():
        import torch

        sd = _torch_load(bin_path)
        for name in names:
            if name not in sd:
                continue
            flat = sd[name].detach().reshape(-1)[: int(n_values)].to(torch.float32).cpu().numpy()
            out[name] = hashlib.sha256(flat.tobytes()).hexdigest()
    return out


def capture_resume_proof(
    trainer: Any,
    *,
    checkpoint_dir: Union[str, Path],
    expected_resume_step: int,
    steps_per_epoch: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """
    Compare live Trainer state against the checkpoint, right after restore.

    Must be called from ``on_train_begin`` — before any new step mutates the
    optimizer/scheduler — otherwise the values being compared are post-training
    state and prove nothing about the restore.

    Each flag is derived from real state:
      * model      — parameter slice hashes equal the checkpoint's
      * optimizer  — Adam per-parameter ``step`` counter equals ``global_step``
      * scheduler  — live ``state_dict()`` equals ``scheduler.pt``
      * data pos.  — ``global_step`` and derived epoch/batch offset match
    """
    ck = Path(checkpoint_dir)
    state = getattr(trainer, "state", None)
    global_step = int(getattr(state, "global_step", -1) or -1)
    expected = int(expected_resume_step)
    proof: Dict[str, Any] = {
        "checkpoint_dir": str(ck),
        "global_step": global_step,
        "expected_global_step": expected,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    trainer_state_step = None
    ts_path = ck / "trainer_state.json"
    if ts_path.is_file():
        trainer_state_step = int(json.loads(ts_path.read_text(encoding="utf-8")).get("global_step") or -1)
    proof["checkpoint_global_step"] = trainer_state_step
    proof["step_matches_checkpoint"] = (
        trainer_state_step is not None and trainer_state_step == global_step == expected
    )

    # --- model
    model = getattr(trainer, "model", None)
    model_restored, model_detail = False, "no_model"
    if model is not None:
        try:
            live = _model_param_probe(model)
            disk = _checkpoint_param_probe(ck, names=list(live))
            shared = [n for n in live if n in disk]
            if not shared:
                model_detail = "no_comparable_tensors"
            else:
                mismatched = [n for n in shared if live[n] != disk[n]]
                model_restored = not mismatched
                model_detail = "ok" if model_restored else f"mismatch:{mismatched[:3]}"
            proof["model_tensors_compared"] = len(shared)
        except Exception as exc:  # noqa: BLE001
            model_detail = f"error:{type(exc).__name__}:{exc}"
    proof["model_restored"] = bool(model_restored)
    proof["model_detail"] = model_detail

    # --- optimizer (Adam step counter is the ground truth, no 2.4GB reload)
    opt = getattr(trainer, "optimizer", None)
    opt_restored, opt_detail, opt_steps = False, "no_optimizer", None
    if opt is not None and hasattr(opt, "state_dict"):
        sd = opt.state_dict()
        entries = list((sd.get("state") or {}).values())
        if not entries:
            opt_detail = "optimizer_state_empty"
        else:
            steps = []
            for entry in entries:
                raw = entry.get("step")
                if raw is None:
                    continue
                steps.append(int(raw.item()) if hasattr(raw, "item") else int(raw))
            opt_steps = sorted(set(steps))
            if not opt_steps:
                opt_detail = "no_step_counter"
            elif opt_steps == [expected]:
                opt_restored, opt_detail = True, "ok"
            else:
                opt_detail = f"step_counter={opt_steps[:3]} != {expected}"
    proof["optimizer_restored"] = bool(opt_restored)
    proof["optimizer_detail"] = opt_detail
    proof["optimizer_step_counters"] = opt_steps

    # --- scheduler
    sched = getattr(trainer, "lr_scheduler", None)
    sch_restored, sch_detail = False, "no_scheduler"
    sch_path = ck / "scheduler.pt"
    if sched is not None and hasattr(sched, "state_dict"):
        live_sd = {k: v for k, v in sched.state_dict().items() if k != "lr_lambdas"}
        proof["scheduler_last_epoch"] = live_sd.get("last_epoch")
        if not sch_path.is_file():
            sch_detail = "scheduler.pt_missing"
        else:
            try:
                disk_sd = {k: v for k, v in (_torch_load(sch_path) or {}).items() if k != "lr_lambdas"}
                if live_sd == disk_sd:
                    sch_restored, sch_detail = True, "ok"
                else:
                    diff = sorted({k for k in set(live_sd) | set(disk_sd) if live_sd.get(k) != disk_sd.get(k)})
                    sch_detail = f"state_dict_diff={diff[:4]}"
            except Exception as exc:  # noqa: BLE001
                sch_detail = f"error:{type(exc).__name__}"
    proof["scheduler_restored"] = bool(sch_restored)
    proof["scheduler_detail"] = sch_detail

    # --- data position
    # Only the expected *offset* can be computed here; whether the dataloader
    # actually resumed there is decided later from the UID the training step
    # consumed. A matching global_step proves nothing about data order.
    if steps_per_epoch:
        updates_per_epoch = max(1, int(steps_per_epoch) // max(1, int(gradient_accumulation_steps)))
        proof["epochs_trained"] = int(global_step // updates_per_epoch)
        proof["expected_resume_offset"] = int(
            (global_step % updates_per_epoch) * int(gradient_accumulation_steps)
        )
    proof["step_offset_consistent"] = global_step == expected
    proof["data_position_ok"] = False
    proof["data_position_detail"] = "pending_consumed_uid"
    return proof


def capture_rng_proof(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Byte-compare the live RNG state against ``rng_state.pth``.

    The Trainer loads RNG lazily at the first step of the resumed epoch, so this
    must run from ``on_step_begin`` of the first step — not ``on_train_begin``.
    """
    ck = Path(checkpoint_dir)
    path = ck / "rng_state.pth"
    out: Dict[str, Any] = {"rng_file": str(path)}
    if not path.is_file():
        out.update({"rng_restored": False, "rng_detail": "rng_state.pth_missing"})
        return out
    try:
        import numpy as np
        import torch

        saved = _torch_load(path)
        checks: Dict[str, bool] = {}
        if "cpu" in saved:
            checks["torch_cpu"] = bool(
                torch.equal(torch.as_tensor(saved["cpu"]).cpu(), torch.random.get_rng_state())
            )
        if "python" in saved:
            import random

            checks["python"] = tuple(saved["python"][1]) == tuple(random.getstate()[1])
        if "numpy" in saved:
            live = np.random.get_state()
            checks["numpy"] = bool(np.array_equal(saved["numpy"][1], live[1]))
        if not checks:
            out.update({"rng_restored": False, "rng_detail": "no_comparable_rng_streams"})
            return out
        bad = sorted(k for k, ok in checks.items() if not ok)
        out.update({
            "rng_restored": not bad,
            "rng_detail": "ok" if not bad else f"mismatch:{bad}",
            "rng_streams": checks,
        })
    except Exception as exc:  # noqa: BLE001
        out.update({"rng_restored": False, "rng_detail": f"error:{type(exc).__name__}:{exc}"})
    return out


def make_resume_proof_callback(
    *,
    checkpoint_dir: Union[str, Path],
    expected_resume_step: int,
    sink: Dict[str, Any],
    steps_per_epoch: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
):
    """
    TrainerCallback that captures the restore proof at the only valid moments.

    ``on_train_begin`` runs after Trainer loaded model/optimizer/scheduler but
    before the first step; RNG is captured at the first ``on_step_begin``.
    """
    try:
        from transformers import TrainerCallback
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"transformers TrainerCallback unavailable: {exc}") from exc

    class ResumeProofCallback(TrainerCallback):
        def __init__(self) -> None:
            self._captured_step = False

        def on_train_begin(self, args, state, control, **kwargs):
            trainer_like = type("T", (), {
                "state": state,
                "model": kwargs.get("model"),
                "optimizer": kwargs.get("optimizer"),
                "lr_scheduler": kwargs.get("lr_scheduler"),
            })()
            spe = steps_per_epoch
            if spe is None:
                loader = kwargs.get("train_dataloader")
                try:
                    spe = len(loader) if loader is not None else None
                except TypeError:
                    spe = None
            sink["restore"] = capture_resume_proof(
                trainer_like,
                checkpoint_dir=checkpoint_dir,
                expected_resume_step=expected_resume_step,
                steps_per_epoch=spe,
                gradient_accumulation_steps=int(
                    getattr(args, "gradient_accumulation_steps", gradient_accumulation_steps) or 1
                ),
            )
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            if not self._captured_step:
                self._captured_step = True
                sink["rng"] = capture_rng_proof(checkpoint_dir)
                sink["first_step_global_step"] = int(getattr(state, "global_step", -1) or -1)
            return control

    return ResumeProofCallback()


def make_sequential_sampler_trainer_cls(base_cls: Any) -> Any:
    """
    Trainer subclass whose train sampler is strictly in-order.

    Resume-test needs a reproducible sample sequence across two processes;
    Trainer's default RandomSampler would make "the first sample after resume"
    unknowable.
    """

    class SequentialSamplerTrainer(base_cls):  # type: ignore[misc,valid-type]
        def _get_train_sampler(self, *args, **kwargs):
            from torch.utils.data import SequentialSampler

            dataset = args[0] if args else self.train_dataset
            return SequentialSampler(dataset)

    return SequentialSamplerTrainer


def make_uid_tracking_collator(inner_collator: Any, sink: Dict[str, Any]) -> Any:
    """
    Wrap a data collator so every collated batch records its record UIDs.

    Requires ``dataloader_num_workers=0`` (and therefore no prefetching) so that
    the most recently collated batch is the one about to be trained on.
    """

    def collate(features):
        uids = [
            str(f.get("record_uid"))
            for f in features
            if isinstance(f, dict) and f.get("record_uid") is not None
        ]
        sink.setdefault("collated_batches", []).append(uids)
        sink["last_collated_uids"] = uids
        clean = [
            {k: v for k, v in f.items() if k != "record_uid"} if isinstance(f, dict) else f
            for f in features
        ]
        return inner_collator(clean)

    return collate


def make_uid_tracking_trainer_cls(base_cls: Any, sink: Dict[str, Any]) -> Any:
    """
    Trainer subclass that records the first microbatch ``training_step`` consumes.

    ``Dataset.__getitem__`` and the collator both run for batches that resume
    *skips*, so neither can identify the first consumed sample on its own; only
    reaching ``training_step`` proves consumption.
    """

    class UidTrackingTrainer(base_cls):  # type: ignore[misc,valid-type]
        def training_step(self, model, inputs, *args, **kwargs):
            if "first_consumed_uids" not in sink:
                sink["first_consumed_uids"] = list(sink.get("last_collated_uids") or [])
                sink["first_consumed_batch_index"] = max(
                    0, len(sink.get("collated_batches") or []) - 1
                )
            return super().training_step(model, inputs, *args, **kwargs)

    return UidTrackingTrainer


def evaluate_data_position(
    sink: Dict[str, Any],
    *,
    expected_uid: Optional[str],
    expected_offset: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare the first UID actually trained on after resume with Phase A's plan.

    Fail-closed: no recorded UID means no proof, which is a failure, not a pass.
    """
    consumed = list(sink.get("first_consumed_uids") or [])
    actual = consumed[0] if consumed else None
    if not expected_uid:
        return {
            "data_position_ok": False,
            "data_position_detail": "no_expected_uid_from_phase_a",
            "consumed_first_uid": actual,
        }
    if actual is None:
        return {
            "data_position_ok": False,
            "data_position_detail": "no_uid_reached_training_step",
            "consumed_first_uid": None,
        }
    ok = str(actual) == str(expected_uid)
    detail = "ok" if ok else f"consumed {actual!r} but expected {expected_uid!r}"
    out = {
        "data_position_ok": bool(ok),
        "data_position_detail": detail,
        "consumed_first_uid": str(actual),
        "expected_first_uid": str(expected_uid),
        "consumed_batch_uids": consumed,
        "batches_collated_before_first_step": int(
            sink.get("first_consumed_batch_index") or 0
        ),
    }
    if expected_offset is not None:
        out["expected_offset"] = int(expected_offset)
        actual_offset = int(sink.get("first_consumed_batch_index") or 0)
        out["offset_matches"] = actual_offset == int(expected_offset)
        if not out["offset_matches"]:
            out["data_position_ok"] = False
            out["data_position_detail"] = (
                f"{detail}; resumed at batch {actual_offset}, expected {int(expected_offset)}"
            )
    return out


def plan_expected_resume_position(
    ordered_uids: Sequence[str],
    *,
    resume_step: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """
    Phase A's record of where Phase B must pick up.

    With a sequential sampler and a fixed seed the sample order is deterministic,
    so the resume point is an exact UID rather than an approximation.
    """
    uids = [str(u) for u in ordered_uids]
    per_batch = max(1, int(per_device_train_batch_size))
    microbatches_consumed = int(resume_step) * max(1, int(gradient_accumulation_steps))
    index = microbatches_consumed * per_batch
    return {
        "ordered_uids": uids,
        "resume_step": int(resume_step),
        "expected_batch_offset": microbatches_consumed,
        "expected_sample_index": index,
        "expected_first_uid": uids[index] if 0 <= index < len(uids) else None,
        "wrapped_epoch": index >= len(uids),
    }


def summarize_resume_proof(
    sink: Dict[str, Any],
    *,
    expected_first_uid: Optional[str] = None,
    expected_offset: Optional[int] = None,
) -> Dict[str, Any]:
    """Flatten the callback sink into the flags the status gate consumes."""
    restore = dict(sink.get("restore") or {})
    rng = dict(sink.get("rng") or {})
    if not restore:
        raise RuntimeError(
            "Resume proof missing: on_train_begin never fired (no restore captured)"
        )
    if not rng:
        raise RuntimeError(
            "RNG proof missing: on_step_begin never fired (no step ran after resume)"
        )
    position = evaluate_data_position(
        sink, expected_uid=expected_first_uid, expected_offset=expected_offset
    )
    return {
        **restore,
        **position,
        "rng_restored": bool(rng.get("rng_restored")),
        "rng_detail": rng.get("rng_detail"),
        "rng_streams": rng.get("rng_streams"),
        "first_step_global_step": sink.get("first_step_global_step"),
    }


# ---------------------------------------------------------------------------
# Resume-test subset + status
# ---------------------------------------------------------------------------

def build_resume_test_subset(
    eligible_train_df: pd.DataFrame,
    *,
    n_samples: int = 64,
    seed: int = 42,
) -> pd.DataFrame:
    """Fixed group-aware subset for resume_test only (not full training data)."""
    if len(eligible_train_df) == 0:
        raise RuntimeError("Cannot build resume_test subset from empty eligible train")
    return sample_pilot_data(
        eligible_train_df,
        min(int(n_samples), len(eligible_train_df)),
        seed=seed,
        group_col="group_id",
        uid_col="record_uid",
    )


def derive_full_resume_test_status(
    *,
    phase_a_reached_100: bool,
    phase_b_global_step: int,
    optimizer_restored: bool,
    scheduler_restored: bool,
    rng_restored: bool,
    model_restored: bool,
    data_position_ok: bool,
    used_separate_experiment_dir: bool,
) -> str:
    ok = (
        phase_a_reached_100
        and int(phase_b_global_step) == RESUME_TEST_PHASE_B_STEPS
        and optimizer_restored
        and scheduler_restored
        and rng_restored
        and model_restored
        and data_position_ok
        and used_separate_experiment_dir
    )
    return STATUS_FULL_RESUME_TEST if ok else STATUS_FAILED


def derive_resume_test_status_from_proof(
    proof: Dict[str, Any],
    *,
    phase_a_reached_target: bool,
    cross_session: Dict[str, Any],
    used_separate_experiment_dir: bool,
    expected_phase_b_steps: int = RESUME_TEST_PHASE_B_STEPS,
) -> Dict[str, Any]:
    """
    Derive resume-test status purely from captured state — nothing is assumed true.

    ``true_restart`` is the two-session proof, not a literal.
    """
    p = dict(proof or {})
    checks = {
        "phase_a_reached_target": bool(phase_a_reached_target),
        "model_restored": bool(p.get("model_restored")),
        "optimizer_restored": bool(p.get("optimizer_restored")),
        "scheduler_restored": bool(p.get("scheduler_restored")),
        "rng_restored": bool(p.get("rng_restored")),
        "data_position_ok": bool(p.get("data_position_ok")),
        "step_matches_checkpoint": bool(p.get("step_matches_checkpoint")),
        "true_restart": bool((cross_session or {}).get("two_sessions")),
        "used_separate_experiment_dir": bool(used_separate_experiment_dir),
    }
    final_step = int(p.get("final_global_step") or p.get("global_step") or -1)
    checks["reached_phase_b_steps"] = final_step == int(expected_phase_b_steps)
    failed = sorted(k for k, ok in checks.items() if not ok)
    return {
        "status": STATUS_FULL_RESUME_TEST if not failed else STATUS_FAILED,
        "checks": checks,
        "failed_checks": failed,
        "proof": p,
        "cross_session": dict(cross_session or {}),
    }


def derive_full_evaluate_status(
    *,
    full_train_success: bool,
    best_checkpoint_from_drive_valid: bool,
    metrics_finite: bool,
    frozen_test_accessed: bool,
    contract_matches: bool,
) -> Dict[str, Any]:
    """Evaluate succeeds only when every upstream gate holds (fail-closed)."""
    checks = {
        "full_train_success": bool(full_train_success),
        "best_checkpoint_from_drive_valid": bool(best_checkpoint_from_drive_valid),
        "metrics_finite": bool(metrics_finite),
        "no_frozen_test_access": frozen_test_accessed is False,
        "contract_matches": bool(contract_matches),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    return {
        "status": "SUCCESS_FULL_EVALUATE" if not failed else STATUS_FAILED,
        "checks": checks,
        "failed_checks": failed,
    }


CANONICAL_METRIC_KEYS = ("loss", "cer", "wer")


def canonicalize_trainer_metrics(
    metrics: Dict[str, Any],
    *,
    required: Sequence[str] = CANONICAL_METRIC_KEYS,
) -> Dict[str, float]:
    """
    Strip the Trainer metric prefix so gates compare canonical names.

    ``trainer.predict()`` emits ``test_*`` while ``evaluate()`` emits ``eval_*``;
    checking one set of names against the other silently fails every gate. The
    ``test_`` prefix is Transformers' ``metric_key_prefix`` and says nothing about
    the frozen test split.
    """
    raw = dict(metrics or {})
    out: Dict[str, float] = {}
    missing: List[str] = []
    for key in required:
        value = None
        for candidate in (key, f"test_{key}", f"eval_{key}"):
            if candidate in raw:
                value = raw[candidate]
                break
        if value is None:
            missing.append(key)
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            missing.append(key)
    if missing:
        raise RuntimeError(
            f"Trainer metrics lack required keys {missing} "
            f"(available={sorted(raw)}); refusing to score this run"
        )
    return out


def metrics_are_finite(metrics: Dict[str, Any], *, keys: Sequence[str] = ("eval_wer", "eval_cer", "eval_loss")) -> bool:
    """True only when every requested metric exists and is a finite number."""
    for key in keys:
        if key not in (metrics or {}):
            return False
        try:
            val = float(metrics[key])
        except (TypeError, ValueError):
            return False
        if math.isnan(val) or math.isinf(val):
            return False
    return True


def load_resume_test_success(
    state_dir: Path,
    *,
    expected_experiment_id: Optional[str] = None,
    expected_dataset_id: Optional[str] = None,
    expected_dataset_revision: Optional[str] = None,
    expected_vocab_fp: Optional[str] = None,
    expected_model_id: Optional[str] = None,
    expected_model_revision: Optional[str] = None,
    expected_contract: Optional[Dict[str, Any]] = None,
) -> bool:
    path = Path(state_dir) / "full_resume_test_summary.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != STATUS_FULL_RESUME_TEST:
        return False
    if expected_contract is not None:
        return training_contract_matches(
            extract_training_contract(data), expected_contract, require_hparams=False
        )
    if expected_experiment_id is not None and str(data.get("experiment_id")) != str(expected_experiment_id):
        return False
    if expected_dataset_id is not None and str(data.get("dataset_id")) != str(expected_dataset_id):
        return False
    if expected_dataset_revision is not None and str(data.get("dataset_revision")) != str(expected_dataset_revision):
        return False
    if expected_vocab_fp is not None and str(data.get("vocab_fp")) != str(expected_vocab_fp):
        return False
    if expected_model_id is not None and str(data.get("pretrained_model_id")) != str(expected_model_id):
        return False
    if expected_model_revision is not None and str(data.get("pretrained_model_revision")) != str(expected_model_revision):
        return False
    return True


def load_full_train_success(
    state_dir: Path,
    *,
    expected_experiment_id: Optional[str] = None,
    expected_dataset_id: Optional[str] = None,
    expected_dataset_revision: Optional[str] = None,
    expected_vocab_fp: Optional[str] = None,
    expected_model_id: Optional[str] = None,
    expected_model_revision: Optional[str] = None,
    expected_contract: Optional[Dict[str, Any]] = None,
) -> bool:
    path = Path(state_dir) / "full_train_summary.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != STATUS_FULL_TRAINING:
        return False
    if expected_contract is not None:
        return training_contract_matches(
            extract_training_contract(data), expected_contract, require_hparams=bool(expected_contract.get("hparams"))
        )
    checks = [
        (expected_experiment_id, "experiment_id"),
        (expected_dataset_id, "dataset_id"),
        (expected_dataset_revision, "dataset_revision"),
        (expected_vocab_fp, "vocab_fp"),
        (expected_model_id, "pretrained_model_id"),
        (expected_model_revision, "pretrained_model_revision"),
    ]
    for expected, key in checks:
        if expected is not None and str(data.get(key)) != str(expected):
            return False
    return True


def assert_ready_for_full_evaluate(
    state_dir: Path,
    *,
    expected_experiment_id: str,
    expected_dataset_id: Optional[str] = None,
    expected_dataset_revision: Optional[str] = None,
    expected_vocab_fp: Optional[str] = None,
    expected_model_id: Optional[str] = None,
    expected_model_revision: Optional[str] = None,
    expected_contract: Optional[Dict[str, Any]] = None,
) -> None:
    """Evaluate may only succeed after SUCCESS_FULL_TRAINING for this contract."""
    ok = load_full_train_success(
        state_dir,
        expected_experiment_id=expected_experiment_id,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_revision=expected_dataset_revision,
        expected_vocab_fp=expected_vocab_fp,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        expected_contract=expected_contract,
    )
    if not ok:
        raise RuntimeError(
            "FULL_STAGE=evaluate requires SUCCESS_FULL_TRAINING for current "
            "experiment/dataset/model/manifest/vocab contract. Run FULL_STAGE='train' first."
        )


def write_resume_test_summary(state_dir: Path, payload: Dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "full_resume_test_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Full-train gates / status
# ---------------------------------------------------------------------------

def assert_ready_for_full_train(
    state_dir: Path,
    *,
    expected_contract: Dict[str, Any],
) -> None:
    """
    Full train requires SUCCESS_FULL_PREPARE and SUCCESS_FULL_RESUME_TEST, both
    bound to the exact canonical contract of this run.
    """
    if not expected_contract:
        raise ValueError("assert_ready_for_full_train requires the canonical contract")
    if not load_prepare_success(state_dir, expected_contract=expected_contract):
        from src.asr_full_data import SUMMARY_JSON, prepare_contract_mismatches

        summary_path = Path(state_dir) / SUMMARY_JSON
        detail = "no prepare summary"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            detail = (
                f"status={summary.get('status')!r} "
                f"mismatches={prepare_contract_mismatches(summary, expected_contract)[:4]}"
            )
        raise RuntimeError(
            "FULL_STAGE=train requires SUCCESS_FULL_PREPARE for the current contract. "
            f"Run FULL_STAGE='prepare' first ({detail})."
        )
    if not load_resume_test_success(state_dir, expected_contract=expected_contract):
        raise RuntimeError(
            "FULL_STAGE=train requires SUCCESS_FULL_RESUME_TEST for current training_contract. "
            "Run FULL_STAGE='resume_test' first."
        )


def resolve_full_stage_status(
    *,
    pipeline_error: Optional[Dict[str, Any]],
    current_full_status: Optional[str],
    full_stage: str,
    state_dir: Path,
    experiment_id: str,
    expected_contract: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Prefer current-run failure over durable summaries from prior sessions.

    A fallback to a previous session's summary is only allowed when this run
    produced no error and no status of its own, *and* the summary matches the
    exact canonical contract. Without a contract there is no safe fallback.
    """
    if pipeline_error:
        return STATUS_FAILED
    if current_full_status:
        return str(current_full_status)
    if not expected_contract:
        return STATUS_FAILED
    if full_stage == "prepare" and load_prepare_success(
        state_dir, expected_contract=expected_contract
    ):
        return "SUCCESS_FULL_PREPARE"
    if full_stage == "resume_test" and load_resume_test_success(
        state_dir, expected_contract=expected_contract
    ):
        return STATUS_FULL_RESUME_TEST
    if full_stage == "train" and load_full_train_success(
        state_dir, expected_experiment_id=experiment_id, expected_contract=expected_contract
    ):
        return STATUS_FULL_TRAINING
    if full_stage == "evaluate":
        if load_full_train_success(
            state_dir, expected_experiment_id=experiment_id, expected_contract=expected_contract
        ):
            p = Path(state_dir) / "full_evaluate_summary.json"
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                recorded = extract_training_contract(d)
                if (
                    str(d.get("status", "")).startswith("SUCCESS")
                    and str(d.get("experiment_id")) == str(experiment_id)
                    and training_contract_matches(
                        recorded, expected_contract, require_hparams=True
                    )
                ):
                    return str(d["status"])
    return STATUS_FAILED


def derive_full_training_status(
    *,
    reached_target_steps: bool,
    full_validation_ok: bool,
    best_checkpoint_reload_ok: bool,
    frozen_test_accessed: bool,
    started_from_base_model: bool,
) -> str:
    ok = (
        reached_target_steps
        and full_validation_ok
        and best_checkpoint_reload_ok
        and (frozen_test_accessed is False)
        and started_from_base_model
    )
    return STATUS_FULL_TRAINING if ok else STATUS_FAILED


def assert_no_frozen_test_access(paths: Sequence[Any], splits: Sequence[Any]) -> None:
    for p in paths:
        if is_forbidden_test_path(str(p)):
            raise RuntimeError(f"Frozen-test path accessed during full training: {p}")
    forbidden = {"test", "rq1_test", "frozen_test"}
    for s in splits:
        if str(s).strip().lower() in forbidden:
            raise RuntimeError(f"Frozen-test split accessed during full training: {s}")


def pretrained_must_be_base(
    *,
    resume_from: Optional[str],
    pretrained_model_id: str,
    expected_model_id: str = "facebook/wav2vec2-xls-r-300m",
) -> bool:
    """When resume_from is None, training must start from the locked base model id."""
    if resume_from:
        return True  # resume path validated separately
    return str(pretrained_model_id) == str(expected_model_id)


def write_full_train_summary(state_dir: Path, payload: Dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "full_train_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Eligible audio resolution / validation-monitor / checkpoint inspect
# ---------------------------------------------------------------------------

def resolve_eligible_audio_path(
    row: Any,
    cache_roots: Sequence[Union[str, Path]],
) -> Optional[Path]:
    """Resolve WAV under local / NB02 cache roots (never Drive bulk WAV)."""
    from src.data_utils import safe_cache_filename

    uid = str(row["record_uid"] if hasattr(row, "__getitem__") else row.get("record_uid"))
    rel = None
    if hasattr(row, "get"):
        rel = row.get("local_cache_relpath")
    elif hasattr(row, "__getitem__"):
        try:
            rel = row["local_cache_relpath"]
        except Exception:
            rel = None
    candidates = []
    if rel is not None and str(rel).strip():
        candidates.append(str(rel))
    candidates.append(safe_cache_filename(uid))
    for root in cache_roots:
        root_p = Path(root)
        for name in candidates:
            p = root_p / name
            if p.is_file():
                return p
    return None


def assert_eligible_audio_available(
    df: pd.DataFrame,
    cache_roots: Sequence[Union[str, Path]],
    *,
    dataset_revision: str,
    target_sr: int = 16000,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    expected_processing_version: str = "notebook02_audio_v3",
    max_list: int = 8,
    verify: bool = True,
    use_pcm_index: Optional[bool] = None,
) -> None:
    """
    Hard-fail if any eligible row lacks a usable local WAV.

    When verify=True (default), re-check sr/duration/finite/checksum/provenance —
    existence alone is not enough after Colab reset.

    Frames produced by the streaming prepare carry ``sha256_pcm`` per row, so
    verification is a single PCM hash compare instead of re-deriving NB02
    provenance from a per-file sidecar.
    """
    from src.asr_full_data import verify_cached_wav_usable, verify_eligible_audio_with_index

    if use_pcm_index is None:
        use_pcm_index = "sha256_pcm" in getattr(df, "columns", [])
    if use_pcm_index:
        if "sha256_pcm" not in df.columns:
            raise RuntimeError("PCM index verification requested but sha256_pcm column is absent")
        verify_eligible_audio_with_index(df, cache_roots, max_list=max_list)
        return

    missing: List[str] = []
    corrupt: List[str] = []
    for _, row in df.iterrows():
        path = resolve_eligible_audio_path(row, cache_roots)
        if path is None:
            missing.append(str(row["record_uid"]))
            continue
        if not verify:
            continue
        st = verify_cached_wav_usable(
            path,
            record_uid=str(row["record_uid"]),
            dataset_revision=dataset_revision,
            target_sr=target_sr,
            min_duration=min_duration,
            max_duration=max_duration,
            expected_processing_version=expected_processing_version,
        )
        if not st.get("ok"):
            corrupt.append(f"{row['record_uid']}:{st.get('reason')}")
    if missing:
        raise RuntimeError(
            f"{len(missing)} eligible records missing local audio cache "
            f"(examples={missing[:max_list]}). Re-run FULL_STAGE=prepare."
        )
    if corrupt:
        raise RuntimeError(
            f"{len(corrupt)} eligible audio caches failed verification "
            f"(examples={corrupt[:max_list]}). Re-run FULL_STAGE=prepare."
        )


def build_validation_monitor_subset(
    eligible_val_df: pd.DataFrame,
    *,
    n_samples: int = 64,
    seed: int = 42,
) -> pd.DataFrame:
    """Group-aware fixed subset for mid-training validation monitoring."""
    if len(eligible_val_df) == 0:
        raise RuntimeError("Cannot build validation-monitor subset from empty eligible validation")
    return sample_pilot_data(
        eligible_val_df,
        min(int(n_samples), len(eligible_val_df)),
        seed=seed,
        group_col="group_id",
        uid_col="record_uid",
    )


def inspect_trainer_checkpoint(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspect HF Trainer checkpoint artifacts for resume invariants.
    Does not load tensors — presence + trainer_state.json fields only.
    """
    cp = Path(checkpoint_dir)
    if not cp.is_dir():
        raise RuntimeError(f"Checkpoint dir missing: {cp}")
    state_path = cp / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"trainer_state.json missing in {cp}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    global_step = int(state.get("global_step") or 0)

    def _exists(*names: str) -> bool:
        return any((cp / n).exists() for n in names)

    model_ok = _exists(
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors",
    )
    # Also accept sharded / HF weight index
    if not model_ok:
        model_ok = any(cp.glob("pytorch_model*.bin")) or any(cp.glob("model*.safetensors"))
    optimizer_ok = _exists("optimizer.pt", "optimizer.bin")
    scheduler_ok = _exists("scheduler.pt", "scheduler.bin")
    rng_ok = _exists("rng_state.pth", "rng_state_0.pth")
    return {
        "path": str(cp),
        "global_step": global_step,
        "model_ok": model_ok,
        "optimizer_ok": optimizer_ok,
        "scheduler_ok": scheduler_ok,
        "rng_ok": rng_ok,
        "trainer_state": {
            "global_step": global_step,
            "epoch": state.get("epoch"),
            "best_metric": state.get("best_metric"),
            "best_model_checkpoint": state.get("best_model_checkpoint"),
        },
    }


def collect_gpu_memory_snapshot() -> Dict[str, Any]:
    """Best-effort CUDA memory stats (empty if no CUDA)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
    except Exception as exc:
        return {"cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}


def select_best_checkpoint_by_cer(
    history_rows: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick checkpoint entry with lowest CER from eval history rows."""
    best = None
    for row in history_rows:
        cer = row.get("cer")
        if cer is None:
            continue
        try:
            cer_f = float(cer)
        except (TypeError, ValueError):
            continue
        if best is None or cer_f < float(best["cer"]):
            best = dict(row)
            best["cer"] = cer_f
    return best
