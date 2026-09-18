"""
Full-training / resume-test orchestration helpers for Notebook 03.

Pilot and resume-test checkpoints must NEVER initialize full training.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from src.asr_utils import checkpoint_belongs_to_run, is_forbidden_test_path, sample_pilot_data
from src.asr_full_data import is_frozen_split_label, load_prepare_success
from src.asr_full_pcm import AUDIO_PCM_PIPELINE_VERSION
from src.asr_runtime_paths import OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP

STATUS_FULL_RESUME_TEST = "SUCCESS_FULL_RESUME_TEST"
STATUS_FULL_TRAINING = "SUCCESS_FULL_TRAINING"
STATUS_FAILED = "FAILED"

RESUME_TEST_MARKER = "resume_test"
PILOT_MARKER = "pilot"
FULL_TRAIN_MARKER = "full_train"

RESUME_TEST_PHASE_A_STEPS = 100
RESUME_TEST_PHASE_B_STEPS = 200

STAGE_VERSION_PREPARE = "full_prepare_v2"
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
    # Canonical PCM encode/decode pipeline id (distinct from NB02 processing_version).
    "audio_pcm_pipeline_version",
    "min_duration",
    "max_duration",
    "target_sr",
    "pretrained_model_id",
    "pretrained_model_revision",
    "overlap_policy",
    "stage_version",
)

# Keep legacy alias for older call sites
TRAINING_CONTRACT_KEYS = DATA_CONTRACT_KEYS + ("experiment_id", "hparams")


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def contract_hash(payload: Dict[str, Any], *, keys: Sequence[str]) -> str:
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
    overlap_policy: str = OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
    stage_version: str = STAGE_VERSION_PREPARE,
    audio_pcm_pipeline_version: str = AUDIO_PCM_PIPELINE_VERSION,
) -> Dict[str, Any]:
    """Prepare-stage contract — no full-train hparams required."""
    if not str(parquet_revision or "").strip():
        raise ValueError("parquet_revision is required in the data contract")
    if not str(overlap_policy or "").strip():
        raise ValueError("overlap_policy is required in the data contract")
    if not str(audio_pcm_pipeline_version or "").strip():
        raise ValueError("audio_pcm_pipeline_version is required in the data contract")
    payload = {
        "dataset_id": str(dataset_id),
        "dataset_revision": str(dataset_revision),
        "parquet_revision": str(parquet_revision),
        "train_manifest_content_hash": str(train_manifest_content_hash),
        "validation_manifest_content_hash": str(validation_manifest_content_hash),
        "vocab_fp": str(vocab_fp),
        "processing_version": str(processing_version),
        "audio_pcm_pipeline_version": str(audio_pcm_pipeline_version),
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "target_sr": int(target_sr),
        "pretrained_model_id": str(pretrained_model_id),
        "pretrained_model_revision": str(pretrained_model_revision),
        "overlap_policy": str(overlap_policy),
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
    audio_pcm_pipeline_version: str = AUDIO_PCM_PIPELINE_VERSION,
    overlap_policy: str = OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
) -> Dict[str, Any]:
    """Wrapper → train_contract when hparams provided, else data_contract+experiment."""
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
        audio_pcm_pipeline_version=audio_pcm_pipeline_version,
        overlap_policy=overlap_policy,
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
    Durable store nests them one level deeper (``<exp>/ckpts/checkpoint-N``), so
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
    experiment_root: Optional[Union[str, Path]] = None,
) -> str:
    """
    Validate resume checkpoint for FULL_STAGE=train.
    Raises if missing fingerprint, wrong experiment, or pilot/resume_test origin.
    Path must lie under the configured ``full_train/<experiment_id>/`` root
    (exact root when ``experiment_root`` is provided; otherwise the path's own
    ``full_train/<experiment_id>`` parent must match).
    """
    if not checkpoint_path:
        raise RuntimeError("Resume checkpoint path is empty")
    cp = Path(checkpoint_path)
    if not cp.exists():
        raise RuntimeError(f"Resume checkpoint does not exist: {cp}")
    cp_resolved = cp.resolve()
    if experiment_root is not None:
        root = Path(experiment_root).resolve()
        try:
            cp_resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"Checkpoint must lie exactly under configured "
                f"'{FULL_TRAIN_MARKER}/{experiment_id}/' root {root}, got {cp}"
            ) from exc
        if root.name != str(experiment_id) or root.parent.name != FULL_TRAIN_MARKER:
            raise RuntimeError(
                f"Configured experiment_root must be .../{FULL_TRAIN_MARKER}/{experiment_id}, "
                f"got {root}"
            )
        # Direct child only: full_train/<experiment_id>/checkpoint-N (not nested archives).
        parent = cp_resolved.parent if cp_resolved.name.startswith("checkpoint-") else cp_resolved
        if parent != root:
            raise RuntimeError(
                f"Checkpoint must sit exactly under configured experiment_root {root}, "
                f"got parent={parent}"
            )
    else:
        parts = [p.lower() for p in cp.parts]
        if FULL_TRAIN_MARKER not in parts:
            raise RuntimeError(
                f"Checkpoint must live under a '{FULL_TRAIN_MARKER}/' directory: {cp}"
            )
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
        # Exact parent: checkpoint-N must sit directly in full_train/<experiment_id>/.
        parent = cp.parent if cp.name.startswith("checkpoint-") else cp
        if parent.name != str(experiment_id) or parent.parent.name != FULL_TRAIN_MARKER:
            raise RuntimeError(
                f"Checkpoint must sit exactly under '{FULL_TRAIN_MARKER}/{experiment_id}/', "
                f"got parent={parent}"
            )
    parts = [p.lower() for p in cp.parts]
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
# Atomic checkpoint copy (local → durable FULL_STATE_DIR)
# ---------------------------------------------------------------------------

def atomic_restore_checkpoint_dir(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Replace ``dst`` with a copy of ``src`` using rename-over with backup.

    Sequence: copy → ``dst`` to ``.bak_replace`` → ``.tmp_copy`` to ``dst`` →
    drop backup. A crash mid-flight leaves ``.bak_replace`` and/or ``.tmp_copy``
    for the next call to recover or clean.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp_copy")
    bak = dst.parent / (dst.name + ".bak_replace")

    # Recover / clean debris from an interrupted prior replace.
    if bak.exists() and not dst.exists():
        os.replace(str(bak), str(dst))
    elif bak.exists():
        if bak.is_dir():
            shutil.rmtree(bak)
        else:
            bak.unlink()
    if tmp.exists():
        if tmp.is_dir():
            shutil.rmtree(tmp)
        else:
            tmp.unlink()

    if src.is_dir():
        shutil.copytree(src, tmp)
    else:
        shutil.copy2(src, tmp)

    if dst.exists():
        os.replace(str(dst), str(bak))
    os.replace(str(tmp), str(dst))
    if bak.exists():
        if bak.is_dir():
            shutil.rmtree(bak)
        else:
            bak.unlink()
    return dst


def durable_experiment_dir(
    full_state_dir: Union[str, Path],
    experiment_id: str,
    *,
    kind: str,
) -> Path:
    return experiment_checkpoint_dir(Path(full_state_dir) / "checkpoints", experiment_id, kind=kind)


CHECKPOINT_STORE_DIRNAME = "ckpts"
SNAPSHOT_MANIFEST_NAME = "manifest.json"
ENV_DURABLE_CHECKPOINT_BUDGET_BYTES = "BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES"


def resolve_durable_checkpoint_budget_bytes(
    value: Any = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    required: bool = True,
) -> Optional[int]:
    """
    Resolve durable upload budget for checkpoint sync stages only.

    No numeric default (including no 15GiB). When ``required`` is True (sync /
    on_save paths), missing or non-positive values fail closed. Prepare and
    evaluate must not call this with required=True.
    """
    raw: Any = value
    if raw is None:
        source = env if env is not None else os.environ
        raw = source.get(ENV_DURABLE_CHECKPOINT_BUDGET_BYTES)
    if raw is None or str(raw).strip() == "":
        if required:
            raise RuntimeError(
                f"Durable checkpoint budget required for sync stages: set "
                f"{ENV_DURABLE_CHECKPOINT_BUDGET_BYTES} to a positive byte count "
                f"(no default). Prepare/evaluate do not need this."
            )
        return None
    try:
        budget = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid {ENV_DURABLE_CHECKPOINT_BUDGET_BYTES}={raw!r}; expected positive int bytes"
        ) from exc
    if budget <= 0:
        raise RuntimeError(
            f"Invalid durable checkpoint budget {budget}: must be a positive byte count"
        )
    return budget


def collect_referenced_checkpoint_names(
    *,
    planned_names: Sequence[str],
    lkg_names: Optional[Sequence[str]] = None,
    rollback_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Unique checkpoint directory names referenced by planned + LKG/rollback."""
    ordered: List[str] = []
    seen = set()
    for group in (planned_names, lkg_names or (), rollback_names or ()):
        for name in group:
            key = str(name)
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def measure_unique_checkpoint_bytes(
    store: Union[str, Path],
    names: Sequence[str],
    *,
    local_sources: Optional[Dict[str, Path]] = None,
    prefer_local: bool = False,
) -> Dict[str, Any]:
    """
    Sum on-disk bytes for unique checkpoint names without double-counting.

    Prefer durable store copies by default; set ``prefer_local=True`` when sizing
    pending re-uploads so incomplete store debris is not mistaken for the full
    payload that will be copied from local.
    """
    root = Path(store)
    sources = dict(local_sources or {})
    per: Dict[str, int] = {}
    for name in names:
        key = str(name)
        src = sources.get(key)
        if prefer_local and src is not None and Path(src).is_dir():
            per[key] = measure_dir_bytes(src)
            continue
        target = root / key
        if target.is_dir():
            per[key] = measure_dir_bytes(target)
            continue
        if src is not None and Path(src).is_dir():
            per[key] = measure_dir_bytes(src)
        else:
            per[key] = 0
    return {
        "names": list(per.keys()),
        "per_checkpoint_bytes": per,
        "total_bytes": int(sum(per.values())),
    }



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
    Which checkpoint steps durable storage should persist, in priority order.

    Latest is needed to resume, best is what evaluate loads, previous-latest/LKG
    is the rollback. Ordinary step-ordered pruning would delete ``best`` as soon
    as two newer checkpoints exist, which silently breaks evaluate.
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
    # latest + best + previous-latest/LKG are always protected when present.
    mandatory = [s for s in priority if reasons[s] in ("latest", "best", "previous_latest")]
    limit = max(len(mandatory), int(save_total_limit))
    keep = mandatory[:]
    for step in priority:
        if step in keep:
            continue
        if len(keep) >= limit:
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


def durable_experiment_protected_bytes(
    full_state_dir: Union[str, Path],
    experiment_id: str,
    *,
    kind: str,
) -> int:
    """
    Unique durable bytes already occupied by every retained snapshot.

    Matches sync accounting: LATEST ∪ rollback snapshot checkpoint refs (not
    LATEST alone). Snapshot metadata directories are included once each.
    """
    dest_root = durable_experiment_dir(full_state_dir, experiment_id, kind=kind)
    store = dest_root / CHECKPOINT_STORE_DIRNAME
    names: List[str] = []
    meta_bytes = 0
    for ver in _snapshot_versions(dest_root):
        snap = dest_root / "snapshots" / f"v{ver}"
        if snap.is_dir():
            meta_bytes += measure_dir_bytes(snap)
        names.extend(list_snapshot_checkpoint_names(snap))
    # Legacy flat LATEST / inline layout fallback.
    latest = resolve_durable_latest_snapshot(full_state_dir, experiment_id, kind=kind)
    if latest is not None and latest.parent.name != "snapshots":
        meta_bytes += measure_dir_bytes(latest)
        names.extend(list_snapshot_checkpoint_names(latest))
    referenced = collect_referenced_checkpoint_names(
        planned_names=(),
        rollback_names=names,
    )
    if not store.is_dir():
        return int(meta_bytes)
    report = measure_unique_checkpoint_bytes(
        store,
        [n for n in referenced if (store / n).is_dir()],
    )
    return int(report["total_bytes"] + meta_bytes)

def assert_durable_checkpoint_budget(
    *,
    protected_bytes: int,
    upload_bytes: int,
    budget_bytes: int,
    label: str = "durable checkpoints",
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Budget gate for durable storage using explicit accounting.

    ``budget_bytes`` must be supplied by the caller (resolved from
    ``BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES``); there is no numeric default.
    Never resolved by deleting the last-known-good.
    """
    if budget_bytes is None or int(budget_bytes) <= 0:
        raise RuntimeError(
            f"Durable checkpoint budget missing/invalid for {label}: {budget_bytes!r}"
        )
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
            f"Durable checkpoint budget exceeded for {label}: "
            f"protected={protected / 1e9:.1f}GB + upload={upload / 1e9:.1f}GB "
            f"= {peak / 1e9:.1f}GB > budget={int(budget_bytes) / 1e9:.1f}GB. "
            f"Refusing to drop the last-known-good to make room; raise "
            f"{ENV_DURABLE_CHECKPOINT_BUDGET_BYTES} or lower save_total_limit. "
            f"detail={report['detail']}"
        )
    return report


def plan_local_checkpoint_disk_peak(
    *,
    existing_checkpoint_bytes: int,
    new_checkpoint_bytes: int,
    hydrated_wav_bytes: int = 0,
) -> Dict[str, Any]:
    """
    Peak local bytes before a durable upload finishes.

    Existing checkpoints stay on disk while a new checkpoint is written, so the
    peak is existing + temporary new — not max(existing, new). Hydrated WAV is
    counted once by the caller (do not pass both a union estimate and a duplicate).
    """
    existing = max(0, int(existing_checkpoint_bytes))
    new = max(0, int(new_checkpoint_bytes))
    wav = max(0, int(hydrated_wav_bytes))
    return {
        "existing_checkpoint_bytes": existing,
        "new_checkpoint_bytes": new,
        "hydrated_wav_bytes": wav,
        "checkpoint_peak_bytes": existing + new,
        "total_peak_bytes": existing + new + wav,
    }
REQUIRED_CHECKPOINT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
MODEL_FILE_CANDIDATES = ("model.safetensors", "pytorch_model.bin")


def _verify_checkpoint_dir_complete(path: Path) -> None:
    """A checkpoint counts as uploaded only when every resume artifact is there."""
    missing = [n for n in REQUIRED_CHECKPOINT_FILES if not (path / n).is_file()]
    if not any((path / n).is_file() for n in MODEL_FILE_CANDIDATES):
        missing.append("|".join(MODEL_FILE_CANDIDATES))
    if missing:
        raise RuntimeError(f"Checkpoint incomplete after copy (missing {missing}): {path}")


def checkpoint_content_digest(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Content identity for one checkpoint directory.

    Digest covers every regular file's relative path, size, and SHA-256 so a
    same-named durable directory with different model bytes cannot be reused.
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: List[Dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        files.append(
            {
                "path": rel,
                "size": int(path.stat().st_size),
                "sha256": hasher.hexdigest(),
            }
        )
    digest = hashlib.sha256(_stable_json({"files": files}).encode("utf-8")).hexdigest()
    return {"digest": digest, "files": files, "n_files": len(files)}


def list_snapshot_checkpoint_names(snapshot_dir: Union[str, Path]) -> List[str]:
    """Names declared by a snapshot manifest (or inline legacy layout). No I/O verify."""
    snap = Path(snapshot_dir)
    manifest = read_snapshot_manifest(snap)
    if manifest is None:
        return [p.name for p in list_step_checkpoints(snap)]
    return [str(name) for name in (manifest.get("checkpoints") or [])]


def known_snapshot_checkpoint_digests(
    dest_root: Union[str, Path],
) -> Dict[str, str]:
    """Latest known digest per checkpoint name across retained snapshots."""
    root = Path(dest_root)
    out: Dict[str, str] = {}
    for ver in _snapshot_versions(root):
        snap = root / "snapshots" / f"v{ver}"
        manifest = read_snapshot_manifest(snap) or {}
        digests = manifest.get("checkpoint_digests") or {}
        for name, value in digests.items():
            digest = value.get("digest") if isinstance(value, dict) else value
            if digest:
                out[str(name)] = str(digest)
    return out


def _checkpoint_dir_is_complete(path: Path) -> bool:
    try:
        _verify_checkpoint_dir_complete(path)
        return True
    except RuntimeError:
        return False


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


def sync_experiment_checkpoints_to_durable(
    local_experiment_dir: Union[str, Path],
    full_state_dir: Union[str, Path],
    *,
    experiment_id: str,
    kind: str,
    require_durable: bool = True,
    best_checkpoint_name: Optional[str] = None,
    save_total_limit: int = 2,
    budget_bytes: Optional[int] = None,
) -> Optional[Path]:
    """
    Incremental versioned sync: upload only checkpoints durable storage does not have yet,
    verify them, then commit a new ``snapshots/vN`` manifest and flip ``LATEST``.

    Copying the whole experiment tree on every save re-uploads gigabytes of
    already-synced checkpoints, so each ``checkpoint-N`` is written once into an
    immutable ``ckpts/`` store and snapshots merely reference it by name. The
    previous LATEST snapshot and every checkpoint it references stay untouched,
    so an interrupted sync always leaves a usable last-known-good.
    """
    local = Path(local_experiment_dir)
    if not local.is_dir():
        if require_durable:
            raise RuntimeError(f"Local experiment checkpoint dir missing for durable sync: {local}")
        return None
    durable_root = Path(full_state_dir)
    if not durable_root.exists():
        if require_durable:
            raise RuntimeError(
                f"Durable FULL_STATE_DIR missing for checkpoint sync (fail-closed): {durable_root}"
            )
        return None
    fp_name = "full_experiment_fingerprint.json"
    if not (local / fp_name).is_file():
        raise RuntimeError(f"Refusing durable sync without local fingerprint: {local / fp_name}")

    dest_root = durable_experiment_dir(durable_root, experiment_id, kind=kind)
    store = dest_root / CHECKPOINT_STORE_DIRNAME
    store.mkdir(parents=True, exist_ok=True)

    # The snapshot about to be replaced is the last-known-good until LATEST flips.
    lkg_snapshot = resolve_durable_latest_snapshot(durable_root, experiment_id, kind=kind)
    if lkg_snapshot is not None:
        # Fail-closed: LATEST must resolve fully (all declared checkpoints present + digests).
        lkg_resolved = resolve_snapshot_checkpoints(
            lkg_snapshot, require_complete=True, verify_digests=True,
        )
        lkg_names = {p.name for p in lkg_resolved}
        lkg_step = max((int(n.split("-")[-1]) for n in lkg_names), default=-1)
    else:
        lkg_names = set()
        lkg_step = None

    local_steps = list_step_checkpoints(local)
    # Uploads can only come from local; the store is the destination, and a
    # partial copy there must never be mistaken for a usable source.
    sources: Dict[str, Path] = {p.name: p for p in local_steps}
    # Only snapshot-referenced store dirs participate in retention — orphan
    # complete debris must not be kept alive by name alone.
    snap_ref_names: set = set()
    for ver in _snapshot_versions(dest_root):
        snap_ref_names.update(
            list_snapshot_checkpoint_names(dest_root / "snapshots" / f"v{ver}")
        )
    prior_digests = known_snapshot_checkpoint_digests(dest_root)
    digest_cache: Dict[str, str] = {}

    def _digest_of(path: Path) -> str:
        key = str(path.resolve())
        if key not in digest_cache:
            digest_cache[key] = str(checkpoint_content_digest(path)["digest"])
        return digest_cache[key]

    def _step_of(name: str) -> int:
        return int(str(name).split("-")[-1])

    candidate_names = set(sources) | {
        n for n in snap_ref_names if (store / n).is_dir()
    }
    retention = plan_checkpoint_retention(
        candidate_steps=[_step_of(n) for n in candidate_names],
        best_step=_step_of(best_checkpoint_name) if best_checkpoint_name else None,
        previous_latest_step=lkg_step if lkg_step and lkg_step >= 0 else None,
        save_total_limit=save_total_limit,
    )
    keep_names = [f"checkpoint-{s}" for s in retention["keep"]]

    # Budget only at sync: resolve explicit config (no 15GiB default).
    resolved_budget = resolve_durable_checkpoint_budget_bytes(budget_bytes, required=True)

    # Protected = unique bytes of everything still on disk that prune will keep
    # until after LATEST flips: every existing snapshot (LATEST + rollback) plus
    # the planned keep set. Omitting rollback-only names undercounts peak usage.
    existing_snap_names: List[str] = list(snap_ref_names)
    referenced = collect_referenced_checkpoint_names(
        planned_names=keep_names,
        lkg_names=sorted(lkg_names),
        rollback_names=existing_snap_names,
    )

    def _store_reusable(name: str, source: Optional[Path]) -> bool:
        target = store / name
        if not (target.is_dir() and _checkpoint_dir_is_complete(target)):
            return False
        store_digest = _digest_of(target)
        if source is not None and Path(source).is_dir():
            return store_digest == _digest_of(Path(source))
        known = prior_digests.get(name)
        if known is not None:
            return store_digest == known
        # Legacy store entry with no prior digest: refuse silent trust.
        return False

    pending: List[str] = []
    for name in keep_names:
        source = sources.get(name)
        if _store_reusable(name, source):
            continue
        # Snapshot-referenced store entries are immutable content: never overwrite.
        if name in snap_ref_names:
            raise RuntimeError(
                f"Refusing to mutate snapshot-referenced durable checkpoint {name} "
                f"under {store}: local digest differs or store entry is incomplete. "
                f"LATEST and rollback left unchanged."
            )
        if source is None or not Path(source).is_dir():
            raise RuntimeError(
                f"Cannot publish durable checkpoint {name}: no local source under {local}"
            )
        pending.append(name)
    protected_names = [n for n in referenced if (store / n).is_dir()]
    for name in pending:
        if (store / name).is_dir() and name not in protected_names:
            protected_names.append(name)
    protected_report = measure_unique_checkpoint_bytes(store, protected_names)
    upload_report = measure_unique_checkpoint_bytes(
        store,
        pending,
        local_sources=sources,
        prefer_local=True,
    )
    budget_report = assert_durable_checkpoint_budget(
        protected_bytes=protected_report["total_bytes"],
        upload_bytes=upload_report["total_bytes"],
        budget_bytes=int(resolved_budget),
        label=f"{kind}/{experiment_id}",
        detail={
            "keep": keep_names,
            "referenced": referenced,
            "uploading": pending,
            "lkg_step": lkg_step,
            "protected_names": protected_report["names"],
            "upload_names": upload_report["names"],
        },
    )

    def _recover_store_swap(target: Path) -> None:
        tmp = store / f"{target.name}.tmp_copy"
        bak = store / f"{target.name}.bak_replace"
        if bak.exists() and not target.exists():
            os.replace(str(bak), str(target))
        elif bak.exists() and target.exists():
            shutil.rmtree(bak, ignore_errors=True)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    uploaded: List[str] = []
    for name in pending:
        source = Path(sources[name])
        target = store / name
        _recover_store_swap(target)
        tmp = store / f"{name}.tmp_copy"
        bak = store / f"{name}.bak_replace"
        if tmp.exists():
            shutil.rmtree(tmp)
        if bak.exists():
            shutil.rmtree(bak, ignore_errors=True)
        # Orphan/new only: stage fully before touching the live target.
        shutil.copytree(source, tmp)
        _verify_checkpoint_dir_complete(tmp)
        staged_digest = checkpoint_content_digest(tmp)["digest"]
        digest_cache[str(tmp.resolve())] = str(staged_digest)
        if target.exists():
            os.replace(str(target), str(bak))
        os.replace(str(tmp), str(target))
        if bak.exists():
            shutil.rmtree(bak, ignore_errors=True)
        digest_cache[str(target.resolve())] = str(staged_digest)
        uploaded.append(name)

    names = sorted(
        (n for n in keep_names if (store / n).is_dir() and _checkpoint_dir_is_complete(store / n)),
        key=_step_of,
    )
    if not names:
        raise RuntimeError(
            f"No complete checkpoint available to snapshot under {store} "
            f"(retention wanted {keep_names})"
        )
    for name in names:
        _verify_checkpoint_dir_complete(store / name)

    checkpoint_digests = {name: _digest_of(store / name) for name in names}
    if set(checkpoint_digests) != set(names):
        raise RuntimeError(
            f"Refusing snapshot with incomplete checkpoint_digests: "
            f"digests={sorted(checkpoint_digests)} checkpoints={names}"
        )

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
        "checkpoint_digests": checkpoint_digests,
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
    resolved = resolve_snapshot_checkpoints(
        snap_dir, require_complete=True, verify_digests=True,
    )
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
    # go. Anything still referenced by the committed snapshot OR the retained
    # previous snapshot (rollback) is untouchable.
    referenced_keep: set = set(names)
    for old in _snapshot_versions(dest_root):
        if old < version - 1:
            continue
        referenced_keep.update(
            list_snapshot_checkpoint_names(dest_root / "snapshots" / f"v{old}")
        )
    for entry in sorted(store.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.endswith(".tmp_copy") or entry.name.endswith(".bak_replace"):
            shutil.rmtree(entry, ignore_errors=True)
            continue
        if entry.name.startswith("checkpoint-") and entry.name not in referenced_keep:
            shutil.rmtree(entry, ignore_errors=True)
    for old in _snapshot_versions(dest_root):
        if old >= version - 1:
            continue  # keep the committed snapshot and one rollback pointer
        shutil.rmtree(dest_root / "snapshots" / f"v{old}", ignore_errors=True)
    return snap_dir


def resolve_snapshot_checkpoints(
    snapshot_dir: Union[str, Path],
    *,
    require_complete: bool = True,
    verify_digests: bool = True,
) -> List[Path]:
    """
    Checkpoint directories a snapshot points at (manifest-based or inline).

    When ``require_complete`` is true (default), every name declared in the
    manifest must exist under the store — missing entries fail closed instead of
    returning a silently truncated list. When digests are recorded, on-disk
    content must match.
    """
    snap = Path(snapshot_dir)
    manifest = read_snapshot_manifest(snap)
    if manifest is None:
        return list_step_checkpoints(snap)  # legacy inline snapshot
    store = snap.parent.parent / str(manifest.get("store_dir") or CHECKPOINT_STORE_DIRNAME)
    declared = [str(name) for name in (manifest.get("checkpoints") or [])]
    digests = manifest.get("checkpoint_digests") or {}
    if require_complete and verify_digests:
        digest_keys = {str(k) for k in digests}
        declared_set = set(declared)
        if digest_keys != declared_set:
            raise RuntimeError(
                f"Snapshot {snap} checkpoint_digests keys {sorted(digest_keys)} "
                f"!= checkpoints {sorted(declared_set)}"
            )
    missing: List[str] = []
    mismatched: List[str] = []
    out: List[Path] = []
    for name in declared:
        p = store / name
        if not p.is_dir():
            missing.append(name)
            continue
        if verify_digests:
            expected = digests[name]
            if isinstance(expected, dict):
                expected = expected.get("digest")
            actual = checkpoint_content_digest(p)["digest"]
            if not expected or actual != expected:
                mismatched.append(name)
                continue
        out.append(p)
    if require_complete and missing:
        raise RuntimeError(
            f"Snapshot {snap} references missing checkpoints {missing} under {store}"
        )
    if require_complete and mismatched:
        raise RuntimeError(
            f"Snapshot {snap} digest mismatch for checkpoints {mismatched} under {store}"
        )
    return sorted(out, key=lambda p: int(p.name.split("-")[-1]))


def resolve_durable_latest_snapshot(
    full_state_dir: Union[str, Path],
    experiment_id: str,
    *,
    kind: str,
) -> Optional[Path]:
    """
    Resolve the snapshot named by ``LATEST``.

    Returns ``None`` only when the durable experiment tree is completely empty
    (no LATEST, no snapshots/, no ckpts/). Any half-initialized state fails closed.
    Never auto-selects the highest snapshot version.
    """
    dest_root = durable_experiment_dir(full_state_dir, experiment_id, kind=kind)
    latest = dest_root / "LATEST"
    store = dest_root / CHECKPOINT_STORE_DIRNAME
    snap_root = dest_root / "snapshots"

    def _nonempty_dir(path: Path) -> bool:
        return path.is_dir() and any(path.iterdir())

    def _has_store_checkpoints() -> bool:
        if not store.is_dir():
            return False
        return any(
            p.is_dir() and p.name.startswith("checkpoint-")
            for p in store.iterdir()
        )

    residue = (
        _nonempty_dir(snap_root)
        or _has_store_checkpoints()
        or bool(list_step_checkpoints(dest_root))
    )

    if latest.is_file():
        version = latest.read_text(encoding="utf-8").strip()
        if not version:
            raise RuntimeError(f"Durable LATEST pointer is empty under {dest_root}")
        snap = dest_root / "snapshots" / version
        if not snap.is_dir():
            raise RuntimeError(
                f"Durable LATEST points to missing snapshot {snap} under {dest_root}"
            )
        return snap

    if residue:
        raise RuntimeError(
            f"Durable experiment has snapshots/ckpts residue but no usable LATEST "
            f"under {dest_root}; refusing to auto-select a snapshot"
        )
    return None


def snapshot_max_step(snapshot_dir: Union[str, Path]) -> int:
    steps = [int(p.name.split("-")[-1]) for p in resolve_snapshot_checkpoints(snapshot_dir)]
    return max(steps) if steps else -1


def make_durable_checkpoint_sync_callback(
    *,
    local_experiment_dir: Union[str, Path],
    full_state_dir: Union[str, Path],
    experiment_id: str,
    kind: str = FULL_TRAIN_MARKER,
    training_contract: Optional[Dict[str, Any]] = None,
    save_total_limit: int = 2,
    budget_bytes: Optional[int] = None,
    best_checkpoint_name: Optional[str] = None,
):
    """HF TrainerCallback: versioned durable sync on every save."""
    resolved_budget = resolve_durable_checkpoint_budget_bytes(budget_bytes, required=True)
    try:
        from transformers import TrainerCallback
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"transformers TrainerCallback unavailable: {exc}") from exc

    class DurableCheckpointSyncCallback(TrainerCallback):
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
                # Enforce experiment fingerprint/contract on the local root before sync.
                ensure_experiment_fingerprint(
                    local_experiment_dir,
                    experiment_id=experiment_id,
                    kind=kind,
                    expected_contract=training_contract,
                    allow_create_if_empty=False,
                )
                best_name = best_checkpoint_name
                best = getattr(state, "best_model_checkpoint", None)
                if best:
                    best_name = Path(str(best)).name
                    if kind == FULL_TRAIN_MARKER and training_contract is not None:
                        assert_checkpoint_allowed_for_full_train(
                            best,
                            experiment_id=experiment_id,
                            expected_contract=training_contract,
                            experiment_root=local_experiment_dir,
                            require_complete=False,
                        )
                limit = getattr(args, "save_total_limit", None)
                if limit is None:
                    limit = save_total_limit
                sync_experiment_checkpoints_to_durable(
                    local_experiment_dir,
                    full_state_dir,
                    experiment_id=experiment_id,
                    kind=kind,
                    require_durable=True,
                    best_checkpoint_name=best_name,
                    save_total_limit=int(limit) if limit is not None else int(save_total_limit),
                    budget_bytes=resolved_budget,
                )
            except Exception as sync_exc:
                raise RuntimeError(
                    f"Durable checkpoint sync failed at step={getattr(state, 'global_step', None)}: "
                    f"{type(sync_exc).__name__}: {sync_exc}"
                ) from sync_exc
            return control

    return DurableCheckpointSyncCallback()


def restore_experiment_checkpoints_from_durable(
    local_experiment_dir: Union[str, Path],
    full_state_dir: Union[str, Path],
    *,
    experiment_id: str,
    kind: str,
    expected_contract: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Merge the LATEST durable snapshot into the local experiment dir.

    Durable is authoritative for every checkpoint name the snapshot references:
    same-named local directories are replaced from durable (never kept by default).

    Fail-closed before any mutation: a local tree that already has step
    checkpoints but no fingerprint is refused (never stamped with a durable
    fingerprint that would legitimize orphan weights). Contract / ahead
    conflicts are also checked before any copy.
    """
    local = Path(local_experiment_dir)
    snap = resolve_durable_latest_snapshot(full_state_dir, experiment_id, kind=kind)
    local_steps = list_step_checkpoints(local) if local.is_dir() else []
    fp_name = "full_experiment_fingerprint.json"
    local_fp = read_checkpoint_fingerprint(local) if local.is_dir() and (local / fp_name).is_file() else None

    # C1: never copy a durable fingerprint onto a local tree that already has
    # checkpoints but no identity document.
    if local_steps and local_fp is None:
        raise RuntimeError(
            f"Experiment dir has checkpoints but missing fingerprint (fail-closed); "
            f"refusing durable restore that would stamp a new identity over orphan "
            f"weights: {local}"
        )

    if snap is None:
        if local_steps:
            ensure_experiment_fingerprint(
                local,
                experiment_id=experiment_id,
                kind=kind,
                expected_contract=expected_contract,
                allow_create_if_empty=False,
            )
            return local
        return None

    fp = read_checkpoint_fingerprint(snap)
    if not fp or fp.get("kind") != kind or str(fp.get("experiment_id")) != str(experiment_id):
        raise RuntimeError(
            f"Durable checkpoint fingerprint mismatch for restore: {snap} fp={fp}"
        )
    if expected_contract is not None:
        assert_training_contract(
            extract_training_contract(fp),
            expected_contract,
            label=f"durable restore {snap}",
            require_hparams=bool(expected_contract.get("hparams")),
        )

    if local_fp is not None:
        if local_fp.get("kind") != kind or str(local_fp.get("experiment_id")) != str(experiment_id):
            raise RuntimeError(
                f"Local experiment fingerprint mismatch before restore: {local} fp={local_fp}"
            )

    # Verify fingerprint, checkpoint list, and digests before any local mutation.
    snap_cks = list(
        resolve_snapshot_checkpoints(snap, require_complete=True, verify_digests=True)
    )
    snap_names = {p.name for p in snap_cks}
    snap_max = max((int(p.name.split("-")[-1]) for p in snap_cks), default=-1)
    local_max = max((int(p.name.split("-")[-1]) for p in local_steps), default=-1)

    durable_contract = extract_training_contract(fp) or {}
    local_contract = extract_training_contract(local_fp) if local_fp else None
    contracts_differ = False
    if local_fp is not None and durable_contract:
        contracts_differ = not training_contract_matches(
            local_contract or {},
            durable_contract,
            require_hparams=bool(durable_contract.get("hparams")),
        )
    elif local_fp is not None and not durable_contract:
        contracts_differ = bool(extract_training_contract(local_fp))

    # Validate ahead/contract conflicts before mutating local bytes.
    if contracts_differ and local_max > snap_max:
        raise RuntimeError(
            f"Local tree is ahead of durable while training_contract differs; "
            f"refusing fingerprint upgrade that would mix identities: {local}"
        )
    if expected_contract is not None and local_fp is not None:
        local_matches = training_contract_matches(
            local_contract or {},
            expected_contract,
            require_hparams=bool(expected_contract.get("hparams")),
        )
        if not local_matches and local_max > snap_max:
            raise RuntimeError(
                f"Local training_contract mismatches expected_contract while local "
                f"is ahead of durable: {local}"
            )

    upgrade_fp = (
        not (local.is_dir() and (local / fp_name).is_file())
        or local_max <= snap_max
        or contracts_differ
    )

    local.mkdir(parents=True, exist_ok=True)

    # Materialise snapshot checkpoints from durable (replace same names).
    for ck in snap_cks:
        atomic_restore_checkpoint_dir(ck, local / ck.name)

    if upgrade_fp:
        for p in list_step_checkpoints(local):
            step = int(p.name.split("-")[-1])
            if p.name not in snap_names and step <= snap_max:
                shutil.rmtree(p, ignore_errors=True)
        src_fp = snap / fp_name
        if src_fp.is_file():
            shutil.copy2(src_fp, local / fp_name)
    elif local_steps or (local / fp_name).is_file():
        ensure_experiment_fingerprint(
            local,
            experiment_id=experiment_id,
            kind=kind,
            expected_contract=expected_contract,
            allow_create_if_empty=False,
        )
    return local


def resolve_best_checkpoint_from_durable(
    full_state_dir: Union[str, Path],
    *,
    experiment_id: str,
    train_summary: Dict[str, Any],
    local_experiment_dir: Union[str, Path],
    expected_contract: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Restore LATEST snapshot into ``local_experiment_dir`` then resolve best.

    ``local_experiment_dir`` is required: durable store paths live under
    ``.../ckpts/checkpoint-N`` and cannot satisfy the direct-parent experiment
    root guard used for full-train validation.
    """
    preferred = train_summary.get("best_checkpoint")
    if not preferred:
        raise RuntimeError("train summary missing best_checkpoint — refuse latest fallback")
    name = Path(str(preferred)).name
    if not name.startswith("checkpoint-"):
        raise RuntimeError(f"train summary best_checkpoint is not a step checkpoint: {preferred!r}")

    restored = restore_experiment_checkpoints_from_durable(
        local_experiment_dir, full_state_dir,
        experiment_id=experiment_id, kind=FULL_TRAIN_MARKER,
        expected_contract=expected_contract,
    )
    if restored is None:
        raise RuntimeError("No durable LATEST snapshot available for evaluate")
    root = Path(local_experiment_dir)
    candidate = root / name
    if not candidate.exists():
        raise RuntimeError(
            f"Best checkpoint {name} not found under restored tree {root} — "
            "refusing to substitute the latest checkpoint"
        )
    assert_checkpoint_allowed_for_full_train(
        candidate,
        experiment_id=experiment_id,
        expected_contract=expected_contract,
        experiment_root=local_experiment_dir,
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
    weights, so the resume test would prove nothing about a Pod/session reset.
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

    Loading 1.2GB of weights twice inside one session to compare them is
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


PRIVATE_BATCH_UID_KEY = "_bahnar_record_uids"


def make_uid_tracking_collator(inner_collator: Any, sink: Dict[str, Any]) -> Any:
    """
    Wrap a data collator so each batch carries its record UIDs privately.

    UIDs travel on the batch dict under ``PRIVATE_BATCH_UID_KEY``. Prefetching
    extra microbatches cannot confuse the proof: ``training_step`` reads UIDs
    from the exact ``inputs`` it receives, not from a global last-collated sink.
    """

    def collate(features):
        uids = [
            str(f.get("record_uid"))
            for f in features
            if isinstance(f, dict) and f.get("record_uid") is not None
        ]
        sink.setdefault("collated_batches", []).append(uids)
        clean = [
            {k: v for k, v in f.items() if k != "record_uid"} if isinstance(f, dict) else f
            for f in features
        ]
        batch = inner_collator(clean)
        if not isinstance(batch, dict):
            raise TypeError(
                "UID-tracking collator requires the inner collator to return a dict batch"
            )
        batch[PRIVATE_BATCH_UID_KEY] = list(uids)
        return batch

    return collate


def make_uid_tracking_trainer_cls(base_cls: Any, sink: Dict[str, Any]) -> Any:
    """
    Trainer subclass that records the first microbatch ``training_step`` consumes.

    Reads UIDs from the private field on ``inputs`` (attached by the tracking
    collator), then strips that field before the model forward. Skipped resume
    batches never reach ``training_step``, so prefetch cannot steal the proof.
    """

    class UidTrackingTrainer(base_cls):  # type: ignore[misc,valid-type]
        def training_step(self, model, inputs, *args, **kwargs):
            uids: List[str] = []
            if isinstance(inputs, dict) and PRIVATE_BATCH_UID_KEY in inputs:
                raw = inputs.pop(PRIVATE_BATCH_UID_KEY)
                uids = [str(u) for u in (raw or [])]
            if "first_consumed_uids" not in sink:
                sink["first_consumed_uids"] = list(uids)
                sink["first_consumed_from_inputs"] = True
            return super().training_step(model, inputs, *args, **kwargs)

    return UidTrackingTrainer


def evaluate_data_position(
    sink: Dict[str, Any],
    *,
    expected_uid: Optional[str],
) -> Dict[str, Any]:
    """
    Compare the first UID actually trained on after resume with Phase A's plan.

    Proof is UID-only (plus restore/RNG gates elsewhere). Observed collate/skip
    batch indexes are never compared to an expected offset — HF Trainer's
    ``skip_first_batches`` accounting is not a reliable proof surface.
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
    return {
        "data_position_ok": bool(ok),
        "data_position_detail": detail,
        "consumed_first_uid": str(actual),
        "expected_first_uid": str(expected_uid),
        "consumed_batch_uids": consumed,
        "first_consumed_from_inputs": bool(sink.get("first_consumed_from_inputs")),
    }


def plan_expected_resume_position(
    ordered_uids: Sequence[str],
    *,
    resume_step: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int = 1,
    dataloader_length: Optional[int] = None,
    drop_last: bool = False,
) -> Dict[str, Any]:
    """
    Phase A's record of where Phase B must pick up.

    Matches HuggingFace Trainer resume semantics: within the current epoch it
    skips ``(global_step % num_update_steps_per_epoch) * gradient_accumulation_steps``
    dataloader batches, then the first batch that reaches ``training_step`` is the
    proof UID. Absolute ``global_step * gradient_accumulation`` is incorrect once
    training wraps past one epoch.
    """
    uids = [str(u) for u in ordered_uids]
    if not uids:
        raise ValueError("ordered_uids must be non-empty for resume position planning")
    per_batch = max(1, int(per_device_train_batch_size))
    accum = max(1, int(gradient_accumulation_steps))
    n = len(uids)
    if dataloader_length is None:
        if drop_last:
            dl_len = max(1, n // per_batch) if n >= per_batch else 1
        else:
            dl_len = max(1, (n + per_batch - 1) // per_batch)
    else:
        dl_len = max(1, int(dataloader_length))
    num_update_steps_per_epoch = max(1, dl_len // accum)
    step = int(resume_step)
    steps_in_epoch = step % num_update_steps_per_epoch
    batches_skipped = steps_in_epoch * accum
    sample_index = batches_skipped * per_batch
    if sample_index >= n:
        sample_index = sample_index % n
    return {
        "ordered_uids": uids,
        "resume_step": step,
        "dataloader_length": int(dl_len),
        "num_update_steps_per_epoch": int(num_update_steps_per_epoch),
        "steps_trained_in_current_epoch": int(steps_in_epoch),
        "expected_batch_offset": int(batches_skipped),
        "expected_sample_index": int(sample_index),
        "expected_sample_index_in_epoch": int(sample_index),
        "expected_first_uid": uids[sample_index],
        "wrapped_epoch": step >= num_update_steps_per_epoch,
        "n_epochs_wrapped": int(step // num_update_steps_per_epoch),
    }


def summarize_resume_proof(
    sink: Dict[str, Any],
    *,
    expected_first_uid: Optional[str] = None,
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
    position = evaluate_data_position(sink, expected_uid=expected_first_uid)
    return {
        **restore,
        **position,
        "rng_restored": bool(rng.get("rng_restored")),
        "rng_detail": rng.get("rng_detail"),
        "rng_streams": rng.get("rng_streams"),
        "first_step_global_step": sink.get("first_step_global_step"),
    }


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
    best_checkpoint_from_durable_valid: bool,
    metrics_finite: bool,
    frozen_test_accessed: bool,
    contract_matches: bool,
) -> Dict[str, Any]:
    """Evaluate succeeds only when every upstream gate holds (fail-closed)."""
    checks = {
        "full_train_success": bool(full_train_success),
        "best_checkpoint_from_durable_valid": bool(best_checkpoint_from_durable_valid),
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
    for s in splits:
        if is_frozen_split_label(s):
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
    """Resolve WAV under local / NB02 cache roots (never durable bulk WAV)."""
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
    existence alone is not enough after a session reset.

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


def probe_gpu_memory_for_duration(
    *,
    model,
    processor,
    duration_seconds: float = 40.0,
    target_sr: int = 16000,
    device=None,
    vocab_size: int = 140,
):
    """
    Forward+backward one synthetic utterance of ``duration_seconds``.

    Manual preflight only — do not call from Run-all stages.
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = max(1, int(float(duration_seconds) * int(target_sr)))
    wav = torch.zeros(n_samples, dtype=torch.float32)
    feats = processor(wav.numpy(), sampling_rate=int(target_sr), return_tensors="pt")
    input_values = feats["input_values"].to(device)
    labels = torch.randint(0, max(2, int(vocab_size) - 1), (1, 8), device=device)
    model = model.to(device)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    out = model(input_values=input_values, labels=labels)
    loss = out.loss
    loss.backward()
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else after
    model.zero_grad(set_to_none=True)
    return {
        "ok": True,
        "duration_seconds": float(duration_seconds),
        "n_samples": int(n_samples),
        "device": str(device),
        "allocated_before_bytes": int(before),
        "allocated_after_bytes": int(after),
        "peak_allocated_bytes": int(peak),
        "loss": float(loss.detach().cpu()),
    }
