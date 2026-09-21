"""Locked scientific/data contracts for Notebook 05 Direct S2TT D0."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

STATUS_DIRECT_PREPARE = "SUCCESS_DIRECT_PREPARE"
STATUS_DIRECT_PILOT = "SUCCESS_DIRECT_PILOT"
STATUS_DIRECT_RESUME_TEST = "SUCCESS_DIRECT_RESUME_TEST"
STATUS_DIRECT_TRAINING = "SUCCESS_DIRECT_TRAINING"
STATUS_DIRECT_EVALUATE = "SUCCESS_DIRECT_EVALUATE"
STATUS_FAILED = "FAILED"

# RQ1 locked architecture: XLS-R encoder + mBART-50 VI decoder.
LOCKED_ENCODER_ID = "facebook/wav2vec2-xls-r-300m"
LOCKED_ENCODER_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
LOCKED_DECODER_ID = "facebook/mbart-large-50-many-to-many-mmt"
LOCKED_DECODER_REVISION = "1fc5c3d1fc340141fe0daf7b9898d85c4e60b436"
LOCKED_TARGET_LANG = "vi_VN"
LOCKED_EXPERIMENT_ID = "direct_xlsr300m_mbart50_vi_v1"
LOCKED_ASR_PREPARE_CONTRACT_HASH = "b5172ee0cc391e4f02f3834975c62f6fb3b56dc53ebc79717e8d5ded69bc880c"
LOCKED_NB03_TRAIN_COUNT = 102486
LOCKED_NB03_VALIDATION_COUNT = 11112
LOCKED_TORCH_VERSION = "2.8.0"
LOCKED_TRANSFORMERS_VERSION = "4.57.6"
LOCKED_ACCELERATE_VERSION = "1.10.1"
LOCKED_METRIC_FOR_BEST_MODEL = "eval_sacrebleu"
LOCKED_GREATER_IS_BETTER = True
LOCKED_DIRECT_MONITOR_SIZE = 256
LOCKED_HARD_MAX_UNK_RATE = 0.05
LOCKED_HARD_MAX_TRUNCATION_RATE = 0.05
LOCKED_SAMPLE_RATE = 16000
LOCKED_MAX_AUDIO_DURATION = 40.0
LOCKED_MAX_TARGET_LENGTH = 256
TARGET_NORMALIZATION_VERSION = "mt_bahnar_vi_nfc_ws_v1"
MATCHED_INPUT_POLICY = "exact_nb03_asr_eligible_uids_join_locked_rq1_text_vi_v1"
DIRECT_CONFIG_RELPATH = "configs/direct.yaml"

_FORBIDDEN_REVISIONS = {"", "main", "master", "latest", "head"}
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$", re.I)

SOURCE_FINGERPRINT_NOTEBOOK = "notebooks/05_train_direct_s2tt.ipynb"
SOURCE_FINGERPRINT_RELPATHS = (
    SOURCE_FINGERPRINT_NOTEBOOK,
    "configs/direct.yaml",
    "src/direct_contract.py",
    "src/direct_data.py",
    "src/direct_dataset.py",
    "src/direct_export.py",
    "src/direct_full_train.py",
    "src/direct_model.py",
    "src/direct_runtime_paths.py",
    "src/asr_full_train.py",
    "src/asr_full_shards.py",
    "src/asr_full_pcm.py",
    "src/asr_full_data.py",
    "src/asr_utils.py",
    "src/asr_runtime_paths.py",
    "src/audio_utils.py",
    "src/data_utils.py",
    "src/metrics.py",
    "src/seed.py",
    "src/mt_normalize.py",
    "src/mt_runtime_paths.py",
    "src/mt_tokenize.py",
)
_RUNTIME_CONTROL_ASSIGN_RE = re.compile(
    r"^(\s*)(FULL_STAGE|ALLOW_FULL_TRAINING|RUN_ID)\s*=\s*.+$"
)


def _stable_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: Mapping[str, Any], keys: Sequence[str]) -> str:
    subset = {k: payload.get(k) for k in keys}
    return hashlib.sha256(_stable_json(subset).encode("utf-8")).hexdigest()


def assert_revision_pinned(model_id: str, revision: str) -> None:
    rev = str(revision or "").strip()
    if not str(model_id or "").strip():
        raise RuntimeError("model id must be non-empty")
    if rev.lower() in _FORBIDDEN_REVISIONS or not _COMMIT_RE.fullmatch(rev):
        raise RuntimeError(f"revision must be an explicit commit pin, got {revision!r}")


def assert_locked_direct_baseline(
    encoder_id: str,
    encoder_revision: str,
    decoder_id: str,
    decoder_revision: str,
    target_lang: str,
) -> None:
    assert_revision_pinned(encoder_id, encoder_revision)
    assert_revision_pinned(decoder_id, decoder_revision)
    expected = (
        LOCKED_ENCODER_ID,
        LOCKED_ENCODER_REVISION,
        LOCKED_DECODER_ID,
        LOCKED_DECODER_REVISION,
        LOCKED_TARGET_LANG,
    )
    actual = (encoder_id, encoder_revision, decoder_id, decoder_revision, target_lang)
    if actual != expected:
        raise RuntimeError(f"Direct baseline pin mismatch. expected={expected!r} actual={actual!r}")


DIRECT_DATA_CONTRACT_KEYS = (
    "dataset_id",
    "dataset_revision",
    "parquet_revision",
    "asr_prepare_contract_hash",
    "matched_input_policy",
    "train_uid_set_hash",
    "validation_uid_set_hash",
    "train_ordered_uid_hash",
    "validation_ordered_uid_hash",
    "train_pair_hash",
    "validation_pair_hash",
    "train_ordered_row_hash",
    "validation_ordered_row_hash",
    "train_file_sha256",
    "validation_file_sha256",
    "locked_train_manifest_sha256",
    "locked_validation_manifest_sha256",
    "asr_train_eligible_uid_set_hash",
    "asr_validation_eligible_uid_set_hash",
    "asr_train_eligible_file_sha256",
    "asr_validation_eligible_file_sha256",
    "train_count",
    "validation_count",
    "encoder_id",
    "encoder_revision",
    "decoder_id",
    "decoder_revision",
    "target_lang",
    "tokenizer_fingerprint",
    "target_normalization_version",
    "sample_rate",
    "max_audio_duration",
    "max_target_length",
    "stage_version",
)


def build_direct_data_contract(**kwargs: Any) -> Dict[str, Any]:
    assert_locked_direct_baseline(
        kwargs["encoder_id"], kwargs["encoder_revision"],
        kwargs["decoder_id"], kwargs["decoder_revision"], kwargs["target_lang"],
    )
    payload = dict(kwargs)
    payload.setdefault("matched_input_policy", MATCHED_INPUT_POLICY)
    payload.setdefault("target_normalization_version", TARGET_NORMALIZATION_VERSION)
    payload.setdefault("stage_version", "direct_prepare_v3")
    payload["train_count"] = int(payload["train_count"])
    payload["validation_count"] = int(payload["validation_count"])
    payload["sample_rate"] = int(payload["sample_rate"])
    payload["max_audio_duration"] = float(payload["max_audio_duration"])
    payload["max_target_length"] = int(payload["max_target_length"])
    required = (
        "train_ordered_uid_hash",
        "validation_ordered_uid_hash",
        "train_pair_hash",
        "validation_pair_hash",
        "train_ordered_row_hash",
        "validation_ordered_row_hash",
        "train_file_sha256",
        "validation_file_sha256",
        "locked_train_manifest_sha256",
        "locked_validation_manifest_sha256",
        "asr_train_eligible_uid_set_hash",
        "asr_validation_eligible_uid_set_hash",
        "asr_train_eligible_file_sha256",
        "asr_validation_eligible_file_sha256",
    )
    missing = [k for k in required if not str(payload.get(k) or "").strip()]
    if missing:
        raise RuntimeError(f"Direct data contract missing content locks: {missing}")
    payload["contract_hash"] = _hash_payload(payload, DIRECT_DATA_CONTRACT_KEYS)
    return payload


def assert_direct_data_contract_self_consistent(payload: Mapping[str, Any]) -> None:
    """Recompute outer contract_hash from locked child fields; refuse tampered JSON."""
    want = _hash_payload(payload, DIRECT_DATA_CONTRACT_KEYS)
    got = str(payload.get("contract_hash") or "")
    if got != want:
        raise RuntimeError(f"Direct data contract hash mismatch: recorded={got} recomputed={want}")


DIRECT_TRAINING_CONTRACT_KEYS = (
    "direct_data_contract_hash",
    "experiment_id",
    "encoder_id",
    "encoder_revision",
    "decoder_id",
    "decoder_revision",
    "target_lang",
    "tokenizer_fingerprint",
    "train_uid_set_hash",
    "validation_uid_set_hash",
    "monitor_uid_set_hash",
    "monitor_pair_hash",
    "monitor_ordered_row_hash",
    "monitor_file_sha256",
    "monitor_size",
    "source_fingerprint_sha256",
    "torch_version",
    "transformers_version",
    "accelerate_version",
    "seed",
    "learning_rate",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "warmup_ratio",
    "weight_decay",
    "fp16",
    "bf16",
    "gradient_checkpointing",
    "save_steps",
    "eval_steps",
    "save_total_limit",
    "max_target_length",
    "generation_max_length",
    "num_beams",
    "metric_for_best_model",
    "greater_is_better",
    "freeze_feature_encoder",
    "stage_version",
)


def build_direct_training_contract(**kwargs: Any) -> Dict[str, Any]:
    payload = dict(kwargs)
    payload.setdefault("stage_version", "direct_train_v2")
    payload["direct_training_contract_hash"] = _hash_payload(payload, DIRECT_TRAINING_CONTRACT_KEYS)
    return payload


def assert_direct_training_contract_self_consistent(payload: Mapping[str, Any]) -> None:
    want = _hash_payload(payload, DIRECT_TRAINING_CONTRACT_KEYS)
    got = str(payload.get("direct_training_contract_hash") or "")
    if got != want:
        raise RuntimeError(f"Direct training contract hash mismatch: recorded={got} recomputed={want}")


def build_direct_generation_config(**kwargs: Any) -> Dict[str, Any]:
    return {
        "target_lang": kwargs["target_lang"],
        "forced_bos_token_id": int(kwargs["forced_bos_token_id"]),
        "decoder_start_token_id": int(kwargs["decoder_start_token_id"]),
        "pad_token_id": int(kwargs["pad_token_id"]),
        "eos_token_id": int(kwargs["eos_token_id"]),
        "generation_max_length": int(kwargs["generation_max_length"]),
        "num_beams": int(kwargs["num_beams"]),
        "metric_for_best_model": str(kwargs.get("metric_for_best_model", LOCKED_METRIC_FOR_BEST_MODEL)),
        "greater_is_better": bool(kwargs.get("greater_is_better", LOCKED_GREATER_IS_BETTER)),
    }


def assert_direct_training_contracts_match(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    assert_direct_training_contract_self_consistent(recorded)
    assert_direct_training_contract_self_consistent(current)
    diffs = []
    for k in DIRECT_TRAINING_CONTRACT_KEYS:
        if recorded.get(k) != current.get(k):
            diffs.append(f"{k}: recorded={recorded.get(k)!r} current={current.get(k)!r}")
    if diffs:
        raise RuntimeError("Direct training contract mismatch: " + "; ".join(diffs))


def load_direct_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Single operational config source for Notebook 05 knobs (pins still fail-closed)."""
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"Direct config missing: {p}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load configs/direct.yaml") from exc
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Direct config must be a mapping: {p}")
    enc = dict(data.get("encoder") or {})
    dec = dict(data.get("decoder") or {})
    train = dict(data.get("training") or {})
    gen = dict(data.get("generation") or {})
    cfg = {
        "experiment_id": str(data.get("experiment_id") or ""),
        "encoder_id": str(enc.get("model_id") or ""),
        "encoder_revision": str(enc.get("revision") or ""),
        "decoder_id": str(dec.get("model_id") or ""),
        "decoder_revision": str(dec.get("revision") or ""),
        "target_lang": str(dec.get("target_lang") or ""),
        "per_device_train_batch_size": int(train.get("per_device_train_batch_size")),
        "per_device_eval_batch_size": int(train.get("per_device_eval_batch_size")),
        "gradient_accumulation_steps": int(train.get("gradient_accumulation_steps")),
        "learning_rate": float(train.get("learning_rate")),
        "warmup_ratio": float(train.get("warmup_ratio")),
        "weight_decay": float(train.get("weight_decay")),
        "num_train_epochs": float(train.get("num_train_epochs")),
        "save_steps": int(train.get("save_steps")),
        "eval_steps": int(train.get("eval_steps")),
        "save_total_limit": int(train.get("save_total_limit")),
        "fp16": bool(train.get("fp16")),
        "bf16": bool(train.get("bf16")),
        "gradient_checkpointing": bool(train.get("gradient_checkpointing")),
        "freeze_feature_encoder": bool(train.get("freeze_feature_encoder")),
        "generation_max_length": int(gen.get("max_length")),
        "num_beams": int(gen.get("num_beams")),
        "monitor_size": int(data.get("monitor_size")),
        "config_path": str(p.resolve()),
    }
    assert_locked_direct_baseline(
        cfg["encoder_id"], cfg["encoder_revision"],
        cfg["decoder_id"], cfg["decoder_revision"], cfg["target_lang"],
    )
    if cfg["experiment_id"] != LOCKED_EXPERIMENT_ID:
        raise RuntimeError(
            f"YAML experiment_id {cfg['experiment_id']!r} != locked {LOCKED_EXPERIMENT_ID!r}"
        )
    if int(cfg["monitor_size"]) != int(LOCKED_DIRECT_MONITOR_SIZE):
        raise RuntimeError(
            f"YAML monitor_size {cfg['monitor_size']!r} != locked {LOCKED_DIRECT_MONITOR_SIZE}"
        )
    nb03 = dict(data.get("nb03") or {})
    cfg["nb03_train_count"] = int(nb03.get("train_count") or LOCKED_NB03_TRAIN_COUNT)
    cfg["nb03_validation_count"] = int(nb03.get("validation_count") or LOCKED_NB03_VALIDATION_COUNT)
    cfg["nb03_train_eligible_uid_set_hash"] = str(nb03.get("train_eligible_uid_set_hash") or "").strip()
    cfg["nb03_validation_eligible_uid_set_hash"] = str(nb03.get("validation_eligible_uid_set_hash") or "").strip()
    cfg["nb03_train_eligible_file_sha256"] = str(nb03.get("train_eligible_file_sha256") or "").strip()
    cfg["nb03_validation_eligible_file_sha256"] = str(nb03.get("validation_eligible_file_sha256") or "").strip()
    if cfg["nb03_train_count"] != LOCKED_NB03_TRAIN_COUNT or cfg["nb03_validation_count"] != LOCKED_NB03_VALIDATION_COUNT:
        raise RuntimeError(
            f"YAML NB03 eligible counts {cfg['nb03_train_count']}/{cfg['nb03_validation_count']} "
            f"!= locked {LOCKED_NB03_TRAIN_COUNT}/{LOCKED_NB03_VALIDATION_COUNT}"
        )
    return cfg


def _normalize_pkg_version(value: Any) -> str:
    return str(value or "").split("+", 1)[0].strip()


def direct_runtime_environment() -> Dict[str, str]:
    """Installed torch/transformers/accelerate versions (local import)."""
    try:
        import torch
        import transformers
        import accelerate
    except ImportError as exc:
        raise RuntimeError(f"Direct runtime packages missing: {exc}") from exc
    return {
        "torch_version": _normalize_pkg_version(torch.__version__),
        "transformers_version": _normalize_pkg_version(transformers.__version__),
        "accelerate_version": _normalize_pkg_version(accelerate.__version__),
    }


def assert_locked_direct_runtime(payload: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    got = dict(payload) if payload is not None else direct_runtime_environment()
    expected = {
        "torch_version": LOCKED_TORCH_VERSION,
        "transformers_version": LOCKED_TRANSFORMERS_VERSION,
        "accelerate_version": LOCKED_ACCELERATE_VERSION,
    }
    diffs = []
    for key, want in expected.items():
        actual = _normalize_pkg_version(got.get(key))
        if actual != want:
            diffs.append(f"{key}: expected={want!r} actual={actual!r}")
    if diffs:
        raise RuntimeError("Direct runtime version pin mismatch: " + "; ".join(diffs))
    return {k: _normalize_pkg_version(got.get(k)) for k in expected}


def git_head_commit(project_root: Union[str, Path]) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def canonicalize_notebook_code(notebook: Union[str, Path, Mapping[str, Any]]) -> str:
    """
    Production logic of Notebook 05, ignoring runtime/autosave noise.

    Drops outputs and execution counts. Masks ``FULL_STAGE``,
    ``ALLOW_FULL_TRAINING`` and ``RUN_ID`` assignments so stage switches do
    not invalidate a recorded prepare fingerprint.
    """
    if isinstance(notebook, Mapping):
        nb = notebook
    else:
        p = Path(notebook)
        if not p.is_file():
            raise RuntimeError(f"Source fingerprint missing file: {p}")
        nb = json.loads(p.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in nb.get("cells") or []:
        if cell.get("cell_type") != "code":
            continue
        raw = cell.get("source") or []
        text = "".join(raw) if isinstance(raw, list) else str(raw)
        masked: list[str] = []
        for line in text.splitlines(keepends=True):
            ending = ""
            body = line
            if body.endswith("\r\n"):
                ending = "\r\n"
                body = body[:-2]
            elif body.endswith("\n"):
                ending = "\n"
                body = body[:-1]
            match = _RUNTIME_CONTROL_ASSIGN_RE.match(body)
            if match:
                masked.append(f"{match.group(1)}{match.group(2)} = <RUNTIME_CONTROL>{ending}")
            else:
                masked.append(line)
        parts.append("".join(masked))
    return "\n".join(parts)


def _fingerprint_file(rel: str, path: Path) -> Dict[str, Any]:
    from src.data_utils import sha256_file

    if rel.endswith(".ipynb"):
        canonical = canonicalize_notebook_code(path)
        sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return {
            "path": rel,
            "sha256": sha,
            "size_bytes": len(canonical.encode("utf-8")),
            "canonical": True,
        }
    sha = sha256_file(path)
    return {"path": rel, "sha256": sha, "size_bytes": int(path.stat().st_size), "canonical": False}


def compute_source_fingerprint(project_root: Union[str, Path]) -> Dict[str, Any]:
    """Content fingerprint of Notebook 05 production sources + optional git HEAD."""
    root = Path(project_root).resolve()
    files = []
    digest = hashlib.sha256()
    for rel in SOURCE_FINGERPRINT_RELPATHS:
        p = root / rel
        if not p.is_file():
            raise RuntimeError(f"Source fingerprint missing file: {p}")
        info = _fingerprint_file(rel, p)
        files.append(info)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(info["sha256"].encode("utf-8"))
        digest.update(b"\n")
    return {
        "git_head": git_head_commit(root),
        "aggregate_sha256": digest.hexdigest(),
        "files": files,
    }


def assert_source_fingerprint_matches(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Fail closed when Notebook 05 sources changed between stages."""
    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping):
        raise RuntimeError("Source fingerprint missing; refuse to continue with unbound sources")
    rec = str(recorded.get("aggregate_sha256") or "")
    cur = str(current.get("aggregate_sha256") or "")
    if not rec or not cur:
        raise RuntimeError("Source fingerprint aggregate_sha256 missing")
    if rec != cur:
        raise RuntimeError(
            f"Direct source fingerprint changed since prepare: recorded={rec} current={cur}"
        )
