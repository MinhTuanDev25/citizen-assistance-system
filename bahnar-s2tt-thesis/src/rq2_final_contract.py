"""RQ2 Notebook 14 contracts: upstream freeze, arm isolation, G_test firewall.

D0 is the already-frozen RQ1 Direct checkpoint. It is bound, not retrained.
D-Random and D-Quality CONTINUE FROM that same frozen D0 model state, using
the same supervised G_train plus the frozen NB13 selection of that arm.

G_test stays closed until a human sets ALLOW_G_TEST_EVALUATION after every
scientific decision is frozen. Default notebook flags start no training and
do not open G_test.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.direct_contract import (
    LOCKED_ACCELERATE_VERSION,
    LOCKED_DECODER_ID,
    LOCKED_DECODER_REVISION,
    LOCKED_ENCODER_ID,
    LOCKED_ENCODER_REVISION,
    LOCKED_EXPERIMENT_ID,
    LOCKED_GREATER_IS_BETTER,
    LOCKED_HARD_MAX_TRUNCATION_RATE,
    LOCKED_MAX_AUDIO_DURATION,
    LOCKED_MAX_TARGET_LENGTH,
    LOCKED_METRIC_FOR_BEST_MODEL,
    LOCKED_SAMPLE_RATE,
    LOCKED_TARGET_LANG,
    LOCKED_TORCH_VERSION,
    LOCKED_TRANSFORMERS_VERSION,
    TARGET_NORMALIZATION_VERSION,
    canonicalize_notebook_code,
    git_head_commit,
)
from src.direct_contract import STATUS_DIRECT_EVALUATE as DIRECT_EVALUATE_STATUS
from src.rq1_contract import (
    BOOTSTRAP_METHOD_PAIRED_CLUSTER,
    BOOTSTRAP_UNIT_GROUP_ID,
    sha256_file,
    sha256_json,
)
from src.rq2_pseudo_contract import (
    NB11_RELATIVE_DIR,
    NB11_SUCCESS_STATUS,
    PSEUDO_RELATIVE_DIR,
    assert_not_g_test_path,
    resolve_nb11_input,
)
from src.rq2_selection_contract import (
    ARM_QUALITY,
    ARM_RANDOM,
    SELECTION_RELATIVE_DIR,
    STATUS_SELECTION_FROZEN,
    verify_selection_contract,
)
from src.rq2_selection import verify_published_selection

FINAL_RELATIVE_DIR = "artifacts/rq2/final"
FINAL_SCHEMA_VERSION = "rq2-final-1.0"
FINAL_CONTRACT_VERSION = "rq2_final_contract_v1"
FINAL_CODE_VERSION = "rq2_final_train_evaluate_v1"
STATUS_SUCCESS = "SUCCESS_RQ2_FINAL"
STATUS_FAIL = "FAIL_RQ2_FINAL"
STATUS_PREFLIGHT = "SUCCESS_RQ2_FINAL_PREFLIGHT"
STATUS_ARMS_FROZEN = "SUCCESS_RQ2_FINAL_ARMS_FROZEN"
STATUS_UNLOCK = "SUCCESS_RQ2_FINAL_G_TEST_UNLOCK"

ARM_D0 = "d0"
ARMS = (ARM_D0, ARM_RANDOM, ARM_QUALITY)
TRAINABLE_ARMS = (ARM_RANDOM, ARM_QUALITY)

D0_POLICY = "reuse_frozen_rq1_direct_checkpoint"
AUGMENTATION_INIT_POLICY = "continue_from_frozen_rq1_d0_checkpoint"
REJECTED_PUBLIC_PRETRAINED_INIT_POLICY = "public_pretrained_xlsr_mbart50_same_as_rq1_d0"
SUPERVISED_BASE_POLICY = "rq1_d0_matched_g_train_plus_frozen_nb13_selection"
VALIDATION_SPLIT = "g_validation"
FROZEN_TEST_SPLIT = "g_test"
BEST_CHECKPOINT_METRIC = LOCKED_METRIC_FOR_BEST_MODEL
BEST_CHECKPOINT_TIE_BREAK = ("global_step descending", "checkpoint_name ascending")

GOLD_PSEUDO_MIX_POLICY_UNSET = "UNSET_REQUIRE_EXPLICIT_CONFIG"
GOLD_PSEUDO_MIX_POLICY_SLOTTED = "configured_gold_pseudo_slot_ratio"
SEED_POLICY_UNSET = "UNSET_REQUIRE_EXPLICIT_CONFIG"
SEED_POLICY_SINGLE = "compute_constrained_single_seed"
SEED_POLICY_MULTI = "multi_seed"

RQ1_CONFIG_RELPATH = "configs/rq1.yaml"
DIRECT_CONFIG_RELPATH = "configs/direct.yaml"
SOURCE_FINGERPRINT_NOTEBOOK = "notebooks/14_RQ2_Final_Train_Evaluate.ipynb"
SOURCE_FINGERPRINT_RELPATHS = (
    SOURCE_FINGERPRINT_NOTEBOOK,
    "src/rq2_final_contract.py",
    "src/rq2_final_data.py",
    "src/rq2_final_train.py",
    "src/rq2_final_evaluate.py",
    "configs/direct.yaml",
    "configs/rq1.yaml",
    "src/direct_contract.py",
    "src/direct_dataset.py",
    "src/direct_full_train.py",
    "src/direct_model.py",
    "src/metrics.py",
    "src/rq1_contract.py",
    "src/rq1_evaluation.py",
    "src/rq1_inference.py",
    "src/mt_normalize.py",
    "src/seed.py",
)

_RUNTIME_CONTROL_ASSIGN_RE = re.compile(
    r"^(\s*)(RUN_REAL_TRAINING|ALLOW_G_TEST_EVALUATION|RQ2_FINAL_FROZEN|DIRECT_STATE_DIR|RUN_ID)\s*=\s*.+$"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PATH_MARKERS = ("/Users/", "/home/", "/workspace/", "\\")

SCIENTIFIC_TRAINING_KEYS = (
    "arm",
    "d0_policy",
    "init_policy",
    "d0_init_model_state_sha256",
    "d0_checkpoint_fingerprint_sha256",
    "d0_best_checkpoint_name",
    "gold_pseudo_mix_policy",
    "gold_slots",
    "pseudo_slots",
    "seed_policy",
    "seed_policy_seeds",
    "supervised_base_policy",
    "data_manifest_sha256",
    "ordered_training_uid_hash",
    "training_pair_hash",
    "supervised_ordered_uid_hash",
    "supervised_pair_hash",
    "supervised_uid_set_hash",
    "supervised_audio_identity_hash",
    "pseudo_ordered_uid_hash",
    "pseudo_pair_hash",
    "validation_ordered_uid_hash",
    "validation_pair_hash",
    "validation_uid_set_hash",
    "validation_audio_identity_hash",
    "nb11_input_contract_sha256",
    "nb12_contract_sha256",
    "nb13_selection_contract_sha256",
    "selection_budget_hours",
    "selection_budget_seconds",
    "realized_pseudo_duration_seconds",
    "encoder_id",
    "encoder_revision",
    "decoder_id",
    "decoder_revision",
    "target_lang",
    "target_normalization_version",
    "sample_rate",
    "max_audio_duration",
    "max_target_length",
    "truncation_policy",
    "hard_max_truncation_rate",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "num_train_epochs",
    "fp16",
    "bf16",
    "gradient_checkpointing",
    "freeze_feature_encoder",
    "optimizer",
    "lr_scheduler_type",
    "max_grad_norm",
    "generation_max_length",
    "num_beams",
    "metric_for_best_model",
    "greater_is_better",
    "seed",
    "dataloader_seed",
    "torch_version",
    "transformers_version",
    "accelerate_version",
    "source_fingerprint_sha256",
    "rq1_direct_training_contract_hash",
    "stage_version",
)

RUNTIME_ONLY_KEYS = (
    "dataloader_num_workers",
    "logging_steps",
    "report_to",
    "disable_tqdm",
    "progress_refresh_seconds",
)

FAIRNESS_KEYS = (
    "init_policy",
    "d0_init_model_state_sha256",
    "d0_checkpoint_fingerprint_sha256",
    "gold_pseudo_mix_policy",
    "gold_slots",
    "pseudo_slots",
    "seed_policy",
    "seed_policy_seeds",
    "supervised_base_policy",
    "supervised_ordered_uid_hash",
    "supervised_pair_hash",
    "supervised_uid_set_hash",
    "supervised_audio_identity_hash",
    "validation_ordered_uid_hash",
    "validation_pair_hash",
    "validation_uid_set_hash",
    "validation_audio_identity_hash",
    "nb11_input_contract_sha256",
    "nb12_contract_sha256",
    "nb13_selection_contract_sha256",
    "selection_budget_hours",
    "selection_budget_seconds",
    "encoder_id",
    "encoder_revision",
    "decoder_id",
    "decoder_revision",
    "target_lang",
    "target_normalization_version",
    "sample_rate",
    "max_audio_duration",
    "max_target_length",
    "truncation_policy",
    "hard_max_truncation_rate",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "num_train_epochs",
    "fp16",
    "bf16",
    "gradient_checkpointing",
    "freeze_feature_encoder",
    "optimizer",
    "lr_scheduler_type",
    "max_grad_norm",
    "generation_max_length",
    "num_beams",
    "metric_for_best_model",
    "greater_is_better",
    "seed",
    "dataloader_seed",
    "torch_version",
    "transformers_version",
    "accelerate_version",
    "source_fingerprint_sha256",
    "rq1_direct_training_contract_hash",
)

UNLOCK_REQUIRED = (
    "upstream_frozen",
    "d0_bound",
    "d_random_complete",
    "d_quality_complete",
    "best_checkpoints_frozen",
    "validation_selection_complete",
    "training_contracts_verified",
    "evaluation_protocol_frozen",
    "bootstrap_protocol_frozen",
    "no_pending_scientific_decisions",
    "rq2_final_frozen",
)

FINAL_ARTIFACT_FILES = (
    "final_contract.json",
    "summary.json",
    "arm_metrics.json",
    "arm_comparison.json",
    "bootstrap_results.json",
    "best_checkpoints.json",
    "fairness_proof.json",
    "d_random_training_complete.json",
    "d_quality_training_complete.json",
    "artifact_hashes.json",
    "predictions/d0.parquet",
    "predictions/d_random.parquet",
    "predictions/d_quality.parquet",
)

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_RESUME_AVAILABLE = "RESUME_AVAILABLE"
STATUS_TRAINING_RUNNING = "TRAINING_RUNNING"
STATUS_TRAINING_COMPLETE = "TRAINING_COMPLETE"
STATUS_FAILED_TRAINING = "FAILED"


class Rq2FinalError(RuntimeError):
    """NB14 fail-closed error."""


class GTestFirewallError(Rq2FinalError):
    """G_test was touched before an explicit unlock."""


class UpstreamGateError(Rq2FinalError):
    """An upstream RQ2/RQ1 artifact is missing, stale, or unfrozen."""


class ArmIsolationError(Rq2FinalError):
    """A checkpoint or manifest was used under the wrong arm."""


class TrainingContractError(Rq2FinalError):
    """A training contract is incomplete, drifted, or non-scientific."""


class EvaluationError(Rq2FinalError):
    """Paired evaluation, references, or bootstrap cannot proceed."""


@dataclass(frozen=True)
class Nb14Flags:
    run_real_training: bool = False
    allow_g_test_evaluation: bool = False
    rq2_final_frozen: bool = False
    direct_state_dir: str = ""
    durable_root: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "RUN_REAL_TRAINING": bool(self.run_real_training),
            "ALLOW_G_TEST_EVALUATION": bool(self.allow_g_test_evaluation),
            "RQ2_FINAL_FROZEN": bool(self.rq2_final_frozen),
            "DIRECT_STATE_DIR": str(self.direct_state_dir or ""),
            "DURABLE_ROOT": str(self.durable_root or ""),
        }


def default_nb14_flags() -> Nb14Flags:
    return Nb14Flags()


def resolve_nb14_layout(
    project_root: Union[str, Path],
    *,
    durable_root: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Path]:
    """Durable NB11/NB12/NB13/NB14 roots. The code checkout is not a fallback."""
    from src.rq1_runtime_paths import resolve_rq1_runtime_paths

    root = Path(project_root).resolve()
    runtime = resolve_rq1_runtime_paths(project_root=root, durable_root=durable_root, env=env)
    durable = Path(runtime.durable_root).resolve()
    rq2 = durable / "artifacts" / "rq2"
    return {
        "project_root": root,
        "durable_root": durable,
        "u_clean_dir": rq2 / "u_clean",
        "pseudo_dir": rq2 / "pseudo_labels",
        "selection_dir": rq2 / "selection",
        "final_dir": rq2 / "final",
    }


def assert_nb14_output_dir(
    out_dir: Union[str, Path],
    project_root: Union[str, Path],
    *,
    durable_root: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    layout = resolve_nb14_layout(project_root, durable_root=durable_root, env=env)
    root = layout["project_root"]
    out_input = Path(out_dir).absolute()
    out = Path(out_dir).resolve()
    expected = layout["final_dir"]
    if out != expected:
        raise Rq2FinalError(f"NB14 output dir must be {expected}, got {out}")

    project_final_lexical = Path(project_root).absolute() / FINAL_RELATIVE_DIR
    if layout["durable_root"] != root and out_input == project_final_lexical:
        raise Rq2FinalError("NB14 final output under PROJECT_ROOT is rejected")
    for protected in (
        root / NB11_RELATIVE_DIR,
        root / PSEUDO_RELATIVE_DIR,
        root / SELECTION_RELATIVE_DIR,
        root / "artifacts" / "rq1",
        root / "data" / "manifests",
    ):
        try:
            out.relative_to(protected.resolve())
        except ValueError:
            continue
        raise Rq2FinalError(f"NB14 output dir is inside a protected tree: {protected}")
    assert_g_test_blocked(out, allow=False)
    return out


def assert_g_test_blocked(path: Union[str, Path], *, allow: bool) -> Path:
    """Refuse frozen-test paths unless the human unlock flag is on."""
    p = Path(path)
    if allow:
        return p
    try:
        return assert_not_g_test_path(p)
    except Exception as exc:
        raise GTestFirewallError(f"NB14 G_test firewall blocked {p.name}") from exc


def assert_scientific_value_portable(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_scientific_value_portable(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_scientific_value_portable(item, label=f"{label}[{index}]")
        return
    if not isinstance(value, str):
        return
    text = value
    if Path(text).is_absolute() or any(marker in text for marker in _FORBIDDEN_PATH_MARKERS):
        raise Rq2FinalError(f"{label} contains a non-portable path")


def is_sha256(value: object) -> bool:
    return bool(_HEX64.match(str(value or "").strip().lower()))


def normalize_gold_pseudo_mix_policy(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind an explicit gold:pseudo slot policy. Does not invent a scientific ratio."""
    name = str(fields.get("gold_pseudo_mix_policy") or GOLD_PSEUDO_MIX_POLICY_UNSET).strip()
    if name in {"", GOLD_PSEUDO_MIX_POLICY_UNSET, "UNSET"}:
        return {
            "gold_pseudo_mix_policy": GOLD_PSEUDO_MIX_POLICY_UNSET,
            "gold_slots": None,
            "pseudo_slots": None,
            "configured": False,
        }
    try:
        gold_slots = int(fields.get("gold_slots"))
        pseudo_slots = int(fields.get("pseudo_slots"))
    except (TypeError, ValueError) as exc:
        raise TrainingContractError("gold:pseudo mix slots must be explicit positive integers") from exc
    if gold_slots <= 0 or pseudo_slots <= 0:
        raise TrainingContractError("gold:pseudo mix slots must be positive")
    return {
        "gold_pseudo_mix_policy": name,
        "gold_slots": gold_slots,
        "pseudo_slots": pseudo_slots,
        "configured": True,
        "configured_ratio": f"{gold_slots}:{pseudo_slots}",
    }


def require_configured_mix_policy(fields: Mapping[str, Any], *, for_real_training: bool) -> Dict[str, Any]:
    policy = normalize_gold_pseudo_mix_policy(fields)
    if for_real_training and not policy["configured"]:
        raise TrainingContractError(
            "GOLD_PSEUDO_MIX_POLICY is unset; real training requires an explicit configured gold:pseudo ratio"
        )
    return policy


def normalize_seed_policy(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind an explicit seed policy. Does not invent the production seed set."""
    mode = str(fields.get("seed_policy") or SEED_POLICY_UNSET).strip()
    raw_seeds = fields.get("seed_policy_seeds")
    if raw_seeds is None and fields.get("seeds") is not None:
        raw_seeds = fields.get("seeds")
    if mode in {"", SEED_POLICY_UNSET, "UNSET"}:
        return {
            "seed_policy": SEED_POLICY_UNSET,
            "seed_policy_seeds": [],
            "configured": False,
            "report_mean_std": False,
        }
    if mode not in {SEED_POLICY_SINGLE, SEED_POLICY_MULTI}:
        raise TrainingContractError(f"unknown seed_policy {mode!r}")
    try:
        seeds = [int(x) for x in list(raw_seeds or [])]
    except (TypeError, ValueError) as exc:
        raise TrainingContractError("seed_policy_seeds must be an explicit list of integers") from exc
    if mode == SEED_POLICY_SINGLE and len(seeds) != 1:
        raise TrainingContractError("compute-constrained single-seed protocol requires exactly one declared seed")
    if mode == SEED_POLICY_MULTI and len(seeds) < 2:
        raise TrainingContractError("multi-seed protocol requires at least two declared seeds")
    result = {
        "seed_policy": mode,
        "seed_policy_seeds": seeds,
        "configured": True,
        "report_mean_std": mode == SEED_POLICY_MULTI and len(seeds) >= 2,
        "seed_runs": [{"seed": seed, "dataloader_seed": seed} for seed in seeds],
        "seed_policy_record": {
            "mode": "single" if mode == SEED_POLICY_SINGLE else "multi",
            "seeds": list(seeds),
        },
    }
    if mode == SEED_POLICY_SINGLE:
        only_seed, = seeds
        result["seed"] = only_seed
        result["active_seed"] = only_seed
        result["dataloader_seed"] = only_seed
    return result


def require_configured_seed_policy(fields: Mapping[str, Any], *, for_real_training: bool) -> Dict[str, Any]:
    policy = normalize_seed_policy(fields)
    if for_real_training and not policy["configured"]:
        raise TrainingContractError(
            "RQ2 seed_policy is unset; real training requires an explicit seed list or declared single-seed mode"
        )
    return policy


def summarize_seed_runs(runs: Sequence[Mapping[str, Any]], *, seed_policy: Mapping[str, Any]) -> Dict[str, Any]:
    valid = [dict(run) for run in runs if run]
    mode = str(seed_policy.get("seed_policy") or seed_policy.get("mode") or SEED_POLICY_UNSET)
    declared = list(seed_policy.get("seed_policy_seeds") or seed_policy.get("seeds") or [])
    if mode in {SEED_POLICY_MULTI, "multi"} and not valid:
        raise TrainingContractError("multi-seed summary cannot be computed from an empty run list")
    report_mean = bool(seed_policy.get("report_mean_std")) and len(valid) >= 2
    payload = {
        "seed_policy": mode,
        "declared_seeds": declared,
        "n_valid_runs": len(valid),
        "mean_std_reported": report_mean,
        "protocol": (
            "multi-seed protocol"
            if mode in {SEED_POLICY_MULTI, "multi"}
            else "explicitly declared compute-constrained single-seed protocol"
        ),
    }
    if report_mean:
        import statistics

        for metric in ("sacrebleu", "chrfpp"):
            values = [float(run[metric]) for run in valid if metric in run]
            if len(values) >= 2:
                payload[f"{metric}_mean"] = float(statistics.fmean(values))
                payload[f"{metric}_std"] = float(statistics.pstdev(values))
    return payload


def aggregate_paired_seed_metrics(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    seed_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Mean and population std from real per-seed Random/Quality metrics.

    ``std_convention`` is ``population_pstdev`` for every metric. One declared
    seed reports ``std=None``; two or more use ``statistics.pstdev``. An empty
    multi-seed list is rejected. Missing seeds are not dropped.
    """
    import statistics

    policy = normalize_seed_policy(seed_policy)
    declared = [int(seed) for seed in policy.get("seed_policy_seeds") or []]
    if policy.get("seed_policy") == SEED_POLICY_MULTI and not per_seed:
        raise TrainingContractError("multi-seed summary cannot be computed from an empty run list")
    by_seed: Dict[int, Mapping[str, Any]] = {}
    for row in per_seed:
        seed = int(row["active_seed"])
        if seed in by_seed:
            raise TrainingContractError(f"duplicate per-seed result for seed {seed}")
        by_seed[seed] = row
    if set(by_seed) != set(declared):
        missing = sorted(set(declared) - set(by_seed))
        extra = sorted(set(by_seed) - set(declared))
        raise TrainingContractError(
            f"per-seed results do not match declared seeds; missing={missing} extra={extra}"
        )
    ordered = [by_seed[seed] for seed in declared]

    def _mean_std(values: Sequence[float]) -> Dict[str, Any]:
        if len(values) == 1:
            return {"n_valid_runs": 1, "mean": float(values[0]), "std": None}
        return {
            "n_valid_runs": len(values),
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)),
        }

    def _arm_metric(arm_key: str, metric: str) -> List[float]:
        return [float(row[arm_key][metric]) for row in ordered]

    per_seed_rows = []
    bleu_deltas = []
    chrf_deltas = []
    for row in ordered:
        bleu_delta = float(row[ARM_QUALITY]["sacrebleu"]) - float(row[ARM_RANDOM]["sacrebleu"])
        chrf_delta = float(row[ARM_QUALITY]["chrfpp"]) - float(row[ARM_RANDOM]["chrfpp"])
        bleu_deltas.append(bleu_delta)
        chrf_deltas.append(chrf_delta)
        per_seed_rows.append({
            "active_seed": int(row["active_seed"]),
            "pairing": f"{ARM_RANDOM}:{int(row['active_seed'])}|{ARM_QUALITY}:{int(row['active_seed'])}",
            ARM_RANDOM: dict(row[ARM_RANDOM]),
            ARM_QUALITY: dict(row[ARM_QUALITY]),
            "delta_sacrebleu": bleu_delta,
            "delta_chrfpp": chrf_delta,
        })
    multi = policy.get("seed_policy") == SEED_POLICY_MULTI
    return {
        "seed_policy": policy["seed_policy_record"],
        "declared_seeds": declared,
        "n_valid_runs": len(ordered),
        "mean_std_reported": bool(multi and len(ordered) >= 2),
        "std_convention": "population_pstdev",
        "per_seed": per_seed_rows,
        ARM_RANDOM: {
            "sacrebleu": _mean_std(_arm_metric(ARM_RANDOM, "sacrebleu")),
            "chrfpp": _mean_std(_arm_metric(ARM_RANDOM, "chrfpp")),
        },
        ARM_QUALITY: {
            "sacrebleu": _mean_std(_arm_metric(ARM_QUALITY, "sacrebleu")),
            "chrfpp": _mean_std(_arm_metric(ARM_QUALITY, "chrfpp")),
        },
        "treatment": {
            "per_seed_delta_sacrebleu": bleu_deltas,
            "per_seed_delta_chrfpp": chrf_deltas,
            "sacrebleu": _mean_std(bleu_deltas),
            "chrfpp": _mean_std(chrf_deltas),
        },
    }


def _is_sha256(value: object) -> bool:
    return is_sha256(value)


def flatten_readiness(readiness: Mapping[str, Any]) -> Dict[str, bool]:
    """Accept either a boolean map or a derived {gate: {ok: bool, ...}} report."""
    if "flat" in readiness and isinstance(readiness.get("flat"), Mapping):
        source = readiness["flat"]
        return {str(k): bool(v) for k, v in source.items()}
    if "gates" in readiness and isinstance(readiness.get("gates"), Mapping):
        source = readiness["gates"]
    else:
        source = readiness
    out: Dict[str, bool] = {}
    for key, value in source.items():
        if key in {"flat", "gates", "all_ok", "evidence"}:
            continue
        if isinstance(value, Mapping):
            out[str(key)] = value.get("ok") is True
        else:
            out[str(key)] = value is True
    return out


def _require_sha256(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if not _is_sha256(text):
        raise UpstreamGateError(f"{label} is not a sha256")
    return text


def _without(payload: Mapping[str, Any], key: str) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k != key}


def verify_self_hash(payload: Mapping[str, Any], key: str, label: str) -> None:
    if sha256_json(_without(payload, key)) != str(payload.get(key) or ""):
        raise TrainingContractError(f"{label} hash does not recompute from its body")


def load_rq1_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise UpstreamGateError(f"RQ1 config missing: {p}")
    import yaml

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise UpstreamGateError("RQ1 config must be a mapping")
    dataset = dict(data.get("dataset") or {})
    runtime = dict(data.get("runtime") or {})
    return {
        "seed": int(data.get("seed")),
        "bootstrap_samples": int(data.get("bootstrap_samples")),
        "confidence": float(data.get("confidence")),
        "expected_test_count": int(data.get("expected_test_count")),
        "dataset_id": str(dataset.get("id") or ""),
        "dataset_revision": str(dataset.get("revision") or ""),
        "parquet_revision": str(dataset.get("parquet_revision") or ""),
        "torch_version": str(runtime.get("torch") or LOCKED_TORCH_VERSION),
        "transformers_version": str(runtime.get("transformers") or LOCKED_TRANSFORMERS_VERSION),
        "accelerate_version": str(runtime.get("accelerate") or LOCKED_ACCELERATE_VERSION),
    }


def _fingerprint_file(rel: str, path: Path) -> Dict[str, Any]:
    if rel.endswith(".ipynb"):
        text = canonicalize_notebook_code(path)
        masked = []
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
        canonical = "".join(masked)
        sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return {"path": rel, "sha256": sha, "size_bytes": len(canonical.encode("utf-8")), "canonical": True}
    sha = sha256_file(path)
    return {"path": rel, "sha256": sha, "size_bytes": int(path.stat().st_size), "canonical": False}


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_nb14_source_fingerprint(project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    root = Path(project_root or code_root()).resolve()
    files = []
    digest = hashlib.sha256()
    for rel in SOURCE_FINGERPRINT_RELPATHS:
        path = root / rel
        if not path.is_file():
            raise TrainingContractError(f"NB14 source fingerprint missing file: {rel}")
        info = _fingerprint_file(rel, path)
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


def resolve_d0_direct_state(flags: Nb14Flags) -> Path:
    text = str(flags.direct_state_dir or "").strip()
    if not text:
        raise UpstreamGateError("DIRECT_STATE_DIR is empty; NB14 cannot bind frozen RQ1 D0")
    path = Path(text)
    if not path.is_dir():
        raise UpstreamGateError(f"DIRECT_STATE_DIR is not a directory: {path}")
    return path


def _audio_pair_hash_from_frame(frame: Any) -> str:
    import pandas as pd

    work = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    audio_col = None
    for candidate in ("pcm16_sha256", "source_sha256", "sha256_pcm"):
        if candidate in work.columns:
            audio_col = candidate
            break
    if audio_col is None:
        raise UpstreamGateError(
            "supervised audio identity is missing; pcm16_sha256/sha256_pcm is required "
            "(do not hash an empty list)"
        )
    pairs = []
    seen = {}
    for uid, sha in zip(work["record_uid"].astype(str), work[audio_col].astype(str)):
        digest = str(sha or "").strip().lower()
        if not is_sha256(digest):
            raise UpstreamGateError(f"supervised audio identity for {uid} is not a sha256")
        if uid in seen and seen[uid] != digest:
            raise UpstreamGateError(f"duplicate conflicting audio identity for {uid}")
        seen[uid] = digest
        pairs.append({"uid": uid, "audio": digest})
    return sha256_json(pairs)


def _derive_split_identity(frame: Any, *, path: Path) -> Dict[str, str]:
    from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash
    from src.direct_data import compute_ordered_row_hash, compute_pair_hash

    return {
        "uid_set_hash": compute_uid_set_hash(frame),
        "ordered_uid_hash": compute_ordered_uid_hash(frame),
        "pair_hash": compute_pair_hash(frame),
        "ordered_row_hash": compute_ordered_row_hash(frame),
        "file_sha256": sha256_file(path),
        "audio_pair_hash": _audio_pair_hash_from_frame(frame),
    }


def bind_frozen_d0_identity(
    direct_state_dir: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Reuse the frozen RQ1 Direct checkpoint. D0 is not retrained in NB14.

    Strong train/validation identity is always derived from immutable Direct
    prepared CSVs and, when ``project_root`` is given, the locked RQ1 split
    manifests. ``direct_supervised_identity.json`` is an optional cached proof
    and cannot be required for scientific correctness.
    """
    state = Path(direct_state_dir)
    evaluate = _read_json(state / "direct_evaluate_summary.json")
    train = _read_json(state / "direct_train_summary.json")
    contract = _read_json(state / "direct_training_contract.json")
    if evaluate.get("status") != DIRECT_EVALUATE_STATUS:
        raise UpstreamGateError(f"RQ1 D0 evaluate status is {evaluate.get('status')!r}")
    if evaluate.get("frozen_test_accessed") is not False:
        raise UpstreamGateError("RQ1 D0 evaluate summary is not frozen-test clean")
    from src.direct_contract import assert_direct_training_contract_self_consistent
    from src.direct_data import DIRECT_TRAIN_CSV, DIRECT_VAL_CSV, load_direct_prepared_frames

    assert_direct_training_contract_self_consistent(contract)
    hash_value = str(contract.get("direct_training_contract_hash") or "")
    train_hash = str(
        train.get("direct_training_contract_hash")
        or (train.get("training_contract") or {}).get("direct_training_contract_hash")
        or hash_value
    )
    eval_hash = str(
        evaluate.get("direct_training_contract_hash")
        or (evaluate.get("training_contract") or {}).get("direct_training_contract_hash")
        or hash_value
    )
    if hash_value != train_hash or hash_value != eval_hash:
        raise UpstreamGateError("RQ1 D0 training-contract hash mismatch across train/evaluate")
    best = str(train.get("best_checkpoint_name") or train.get("best_checkpoint") or "")
    eval_best = str(evaluate.get("best_checkpoint_name") or evaluate.get("best_checkpoint") or best)
    if Path(best).name != Path(eval_best).name:
        raise UpstreamGateError("RQ1 D0 best-checkpoint name mismatch")
    best_name = Path(best).name
    if not best_name:
        raise UpstreamGateError("RQ1 D0 best-checkpoint name is empty")
    experiment_id = str(contract.get("experiment_id") or LOCKED_EXPERIMENT_ID)
    checkpoint = _resolve_d0_checkpoint_dir(state, experiment_id=experiment_id, best_name=best_name)
    from src.asr_full_train import assert_checkpoint_complete_for_resume
    from src.direct_full_train import assert_direct_checkpoint_fingerprint

    try:
        assert_checkpoint_complete_for_resume(checkpoint)
        fingerprint = assert_direct_checkpoint_fingerprint(
            checkpoint,
            experiment_id=experiment_id,
            expected_training_contract_hash=hash_value,
            experiment_root=checkpoint.parent,
        )
    except Exception as exc:
        raise UpstreamGateError(f"RQ1 D0 checkpoint identity failed: {exc}") from exc
    model_state_sha256 = _model_state_sha256(checkpoint)
    fingerprint_sha256 = sha256_json(fingerprint)

    train_csv = state / DIRECT_TRAIN_CSV
    val_csv = state / DIRECT_VAL_CSV
    if not train_csv.is_file() or not val_csv.is_file():
        raise UpstreamGateError("frozen D0 Direct prepared train/validation CSVs are missing")

    # The canonical supervised gold identity for NB14 is the exact Direct
    # prepared eligible data that trained D0, not the larger raw RQ1 manifests.
    prepared_train, prepared_val = load_direct_prepared_frames(state)
    derived_train = _derive_split_identity(prepared_train, path=train_csv)
    derived_val = _derive_split_identity(prepared_val, path=val_csv)

    # D0 training contract locks the eligible UID sets used by Direct.
    if derived_train["uid_set_hash"] != str(contract.get("train_uid_set_hash") or ""):
        raise UpstreamGateError("D0 prepared train UID-set hash does not match the Direct training contract")
    if derived_val["uid_set_hash"] != str(contract.get("validation_uid_set_hash") or ""):
        raise UpstreamGateError("D0 prepared validation UID-set hash does not match the Direct training contract")

    # Notebook 05 prepare summary is the frozen bridge from raw RQ1
    # manifests -> Direct prepared eligible data.
    prepare_summary = _read_json(state / "direct_prepare_summary.json")
    prepare_contract = prepare_summary.get("data_contract") or {}
    if not isinstance(prepare_contract, dict) or not prepare_contract:
        raise UpstreamGateError("RQ1 D0 direct_prepare_summary.json is missing data_contract")

    prepare_checks = (
        ("asr_train_eligible_uid_set_hash", derived_train["uid_set_hash"], "train UID-set"),
        ("asr_validation_eligible_uid_set_hash", derived_val["uid_set_hash"], "validation UID-set"),
        ("train_ordered_uid_hash", derived_train["ordered_uid_hash"], "train ordered UID"),
        ("validation_ordered_uid_hash", derived_val["ordered_uid_hash"], "validation ordered UID"),
        ("train_pair_hash", derived_train["pair_hash"], "train UID->text"),
        ("validation_pair_hash", derived_val["pair_hash"], "validation UID->text"),
        ("train_file_sha256", derived_train["file_sha256"], "prepared train file"),
        ("validation_file_sha256", derived_val["file_sha256"], "prepared validation file"),
    )

    for field, actual, label in prepare_checks:
        expected = str(prepare_contract.get(field) or "").strip().lower()
        actual = str(actual or "").strip().lower()
        if not expected:
            raise UpstreamGateError(f"D0 prepare data contract is missing {field}")
        if actual != expected:
            raise UpstreamGateError(
                f"D0 prepared {label} identity does not match direct_prepare_summary: "
                f"{actual} != {expected}"
            )

    # These remain the prepared-file hashes. Do NOT replace them with the
    # raw RQ1 manifest hashes: the prepared files are what actually trained D0.
    train_file_sha = derived_train["file_sha256"]
    val_file_sha = derived_val["file_sha256"]

    if project_root is not None:
        root = Path(project_root)
        g_train_path = root / "data" / "manifests" / "rq1_train.csv"
        g_val_path = root / "data" / "manifests" / "rq1_validation.csv"
        if not g_train_path.is_file() or not g_val_path.is_file():
            raise UpstreamGateError("locked RQ1 train/validation manifests are missing")

        # Raw RQ1 manifests establish provenance only. They are intentionally
        # not row-identical to the Direct eligible data after filtering,
        # normalization and PCM preparation.
        locked_train_sha = str(
            prepare_contract.get("locked_train_manifest_sha256") or ""
        ).strip().lower()
        locked_val_sha = str(
            prepare_contract.get("locked_validation_manifest_sha256") or ""
        ).strip().lower()

        if not is_sha256(locked_train_sha):
            raise UpstreamGateError("D0 prepare data contract has invalid locked_train_manifest_sha256")
        if not is_sha256(locked_val_sha):
            raise UpstreamGateError("D0 prepare data contract has invalid locked_validation_manifest_sha256")

        actual_train_manifest_sha = sha256_file(g_train_path)
        actual_val_manifest_sha = sha256_file(g_val_path)

        if actual_train_manifest_sha != locked_train_sha:
            raise UpstreamGateError("locked RQ1 train manifest SHA256 does not match D0 prepare provenance")
        if actual_val_manifest_sha != locked_val_sha:
            raise UpstreamGateError("locked RQ1 validation manifest SHA256 does not match D0 prepare provenance")


    identity = {
        "arm": ARM_D0,
        "d0_policy": D0_POLICY,
        "experiment_id": experiment_id,
        "direct_training_contract_hash": hash_value,
        "direct_data_contract_hash": str(contract.get("direct_data_contract_hash") or ""),
        "best_checkpoint_name": best_name,
        "checkpoint_fingerprint_sha256": fingerprint_sha256,
        "model_state_sha256": model_state_sha256,
        "tokenizer_fingerprint": str(contract.get("tokenizer_fingerprint") or ""),
        "train_uid_set_hash": derived_train["uid_set_hash"],
        "validation_uid_set_hash": derived_val["uid_set_hash"],
        "train_ordered_uid_hash": derived_train["ordered_uid_hash"],
        "validation_ordered_uid_hash": derived_val["ordered_uid_hash"],
        "train_pair_hash": derived_train["pair_hash"],
        "validation_pair_hash": derived_val["pair_hash"],
        "train_file_sha256": train_file_sha,
        "validation_file_sha256": val_file_sha,
        "train_audio_pair_hash": derived_train["audio_pair_hash"],
        "validation_audio_pair_hash": derived_val["audio_pair_hash"],
        "encoder_id": str(contract.get("encoder_id") or LOCKED_ENCODER_ID),
        "encoder_revision": str(contract.get("encoder_revision") or LOCKED_ENCODER_REVISION),
        "decoder_id": str(contract.get("decoder_id") or LOCKED_DECODER_ID),
        "decoder_revision": str(contract.get("decoder_revision") or LOCKED_DECODER_REVISION),
        "target_lang": str(contract.get("target_lang") or LOCKED_TARGET_LANG),
        "seed": int(contract.get("seed") or 42),
        "frozen_test_accessed": False,
    }
    extra_path = state / "direct_supervised_identity.json"
    if extra_path.is_file():
        extra = _read_json(extra_path)
        for key in (
            "train_ordered_uid_hash",
            "validation_ordered_uid_hash",
            "train_pair_hash",
            "validation_pair_hash",
            "train_file_sha256",
            "validation_file_sha256",
            "train_audio_pair_hash",
            "validation_audio_pair_hash",
            "train_uid_set_hash",
            "validation_uid_set_hash",
        ):
            if extra.get(key) and str(extra[key]) != str(identity.get(key) or ""):
                raise UpstreamGateError(
                    f"optional direct_supervised_identity.json disagrees on {key}"
                )
    assert_scientific_value_portable(
        {k: v for k, v in identity.items() if k != "best_checkpoint_name"},
        label="d0_identity",
    )
    return identity


def resolve_d0_best_checkpoint_dir(direct_state_dir: Union[str, Path], identity: Mapping[str, Any]) -> Path:
    return _resolve_d0_checkpoint_dir(
        Path(direct_state_dir),
        experiment_id=str(identity.get("experiment_id") or LOCKED_EXPERIMENT_ID),
        best_name=str(identity.get("best_checkpoint_name") or ""),
    )


def resolve_frozen_d0_init(
    flags: Nb14Flags,
    *,
    project_root: Optional[Union[str, Path]] = None,
    identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the immutable D0 checkpoint both RQ2 arms must continue from."""
    state = resolve_d0_direct_state(flags)
    live_identity = bind_frozen_d0_identity(state, project_root=project_root)
    if identity:
        provided_hash = str(identity.get("direct_training_contract_hash") or "")
        if provided_hash and provided_hash != str(live_identity.get("direct_training_contract_hash") or ""):
            raise UpstreamGateError("stale RQ1 Direct training contract does not match frozen D0")
        provided_fp = str(identity.get("model_state_sha256") or identity.get("checkpoint_fingerprint_sha256") or "")
        live_fp = str(live_identity.get("model_state_sha256") or "")
        if provided_fp and is_sha256(provided_fp) and provided_fp != live_fp and provided_fp != str(live_identity.get("checkpoint_fingerprint_sha256") or ""):
            raise UpstreamGateError("provided D0 identity does not match the frozen RQ1 Direct checkpoint")
    bound = dict(live_identity)
    checkpoint = resolve_d0_best_checkpoint_dir(state, bound)
    model_state = str(bound.get("model_state_sha256") or "")
    fingerprint = str(bound.get("checkpoint_fingerprint_sha256") or "")
    contract_hash = str(bound.get("direct_training_contract_hash") or "")
    if not checkpoint.is_dir():
        raise UpstreamGateError("frozen D0 checkpoint directory is missing")
    if not is_sha256(model_state):
        raise UpstreamGateError("frozen D0 model-state fingerprint is empty or invalid")
    if not is_sha256(fingerprint):
        raise UpstreamGateError("frozen D0 checkpoint fingerprint is empty or invalid")
    if not is_sha256(contract_hash):
        raise UpstreamGateError("frozen D0 Direct training contract hash is empty or invalid")
    live = _model_state_sha256(checkpoint)
    if live != model_state:
        raise UpstreamGateError("frozen D0 model-state fingerprint does not match checkpoint bytes")
    ckpt_fp = str(bound.get("checkpoint_fingerprint_sha256") or "")
    if not is_sha256(ckpt_fp):
        raise UpstreamGateError("frozen D0 checkpoint ownership fingerprint is empty")
    return {
        "identity": bound,
        "checkpoint_dir": checkpoint,
        "best_checkpoint_name": str(bound.get("best_checkpoint_name") or checkpoint.name),
        "model_state_sha256": model_state,
        "checkpoint_fingerprint_sha256": fingerprint,
        "direct_training_contract_hash": contract_hash,
        "init_policy": AUGMENTATION_INIT_POLICY,
    }


def verify_upstream_rq2(
    project_root: Union[str, Path],
    *,
    flags: Nb14Flags,
    artifact_root: Optional[Union[str, Path]] = None,
    expected_nb13_generation_id: Optional[str] = None,
    expected_budget_hours: Optional[float] = None,
    expected_random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Hash-verify NB11, NB12, NB13 and bind frozen RQ1 D0. Does not open G_test.

    ``artifact_root`` is the durable tree that holds ``artifacts/rq2``. When it
    is omitted the runtime durable root is used. The code checkout is not a
    fallback, and ``CURRENT`` does not replace an explicit NB13 pin.
    """
    root = Path(project_root)
    if flags.allow_g_test_evaluation:
        raise GTestFirewallError("preflight must run with ALLOW_G_TEST_EVALUATION=False")
    layout = resolve_nb14_layout(root, durable_root=artifact_root)
    artifact = layout["durable_root"]
    try:
        nb11 = resolve_nb11_input(root, u_clean_dir=layout["u_clean_dir"])
        summary = _read_json(nb11.generation_dir / "summary.json")
        if summary.get("status") != NB11_SUCCESS_STATUS:
            raise UpstreamGateError(f"NB11 status is {summary.get('status')!r}, not {NB11_SUCCESS_STATUS}")
        from src.rq2_pseudo_contract import read_current_generation_id
        from src.rq2_selection_contract import resolve_frozen_u_prime

        frozen = resolve_frozen_u_prime(
            root,
            durable_root=artifact,
            pseudo_dir=layout["pseudo_dir"],
            u_clean_dir=layout["u_clean_dir"],
        )
        nb13_dir = layout["selection_dir"]
        current_id = ""
        if (nb13_dir / "CURRENT").is_file():
            try:
                current_id = read_current_generation_id(nb13_dir, UpstreamGateError)
            except UpstreamGateError:
                current_id = ""
        expected = str(expected_nb13_generation_id or "").strip()
        nb13_generation_id = expected or current_id
        if not nb13_generation_id:
            raise UpstreamGateError("NB13 generation_id is required; CURRENT is missing")
        selection = verify_published_selection(
            nb13_dir,
            project_root=root,
            generation_id=nb13_generation_id,
            durable_root=artifact,
            pseudo_dir=layout["pseudo_dir"],
            u_clean_dir=layout["u_clean_dir"],
        )
    except GTestFirewallError:
        raise
    except Exception as exc:
        raise UpstreamGateError(str(exc)) from exc
    resolved_nb13 = str(selection["summary"].get("generation_id") or "")
    if expected and resolved_nb13 != expected:
        raise RuntimeError(f"pinned NB13 generation {expected} does not match resolved generation {resolved_nb13}")
    if selection["summary"].get("status") != STATUS_SELECTION_FROZEN:
        raise UpstreamGateError("NB13 is not frozen")
    verify_selection_contract(selection["contract"])
    if expected_budget_hours is not None and float(selection["contract"]["selection_budget_hours"]) != float(expected_budget_hours):
        raise RuntimeError(
            f"NB13 selection_budget_hours is {selection['contract']['selection_budget_hours']}, not {expected_budget_hours}"
        )
    if expected_random_seed is not None and int(selection["contract"]["random_seed"]) != int(expected_random_seed):
        raise RuntimeError(
            f"NB13 random_seed is {selection['contract']['random_seed']}, not {expected_random_seed}"
        )
    d0 = bind_frozen_d0_identity(resolve_d0_direct_state(flags), project_root=root)
    rq1_cfg = load_rq1_yaml_config(code_root() / RQ1_CONFIG_RELPATH)
    source = compute_nb14_source_fingerprint(code_root())
    payload = {
        "nb11_generation_id": nb11.generation_id,
        "nb11_input_contract_sha256": nb11.contract_sha256,
        "nb12_generation_id": frozen.generation_id,
        "nb12_contract_sha256": frozen.identity["nb12_contract_sha256"],
        "u_prime_manifest_sha256": frozen.identity["u_prime_manifest_sha256"],
        "u_prime_ordered_uid_sha256": frozen.identity["u_prime_ordered_uid_sha256"],
        "nb13_generation_id": resolved_nb13,
        "expected_nb13_generation_id": expected or resolved_nb13,
        "resolved_nb13_generation_id": resolved_nb13,
        "nb13_current_generation_id": current_id,
        "nb13_selection_contract_sha256": selection["contract"]["selection_contract_sha256"],
        "d_random_manifest_sha256": selection["summary"]["d_random_manifest_sha256"],
        "d_quality_manifest_sha256": selection["summary"]["d_quality_manifest_sha256"],
        "selection_budget_hours": selection["contract"]["selection_budget_hours"],
        "selection_budget_seconds": selection["contract"]["selection_budget_seconds"],
        "d0": d0,
        "rq1_eval": {
            "seed": rq1_cfg["seed"],
            "bootstrap_samples": rq1_cfg["bootstrap_samples"],
            "confidence": rq1_cfg["confidence"],
            "bootstrap_method": BOOTSTRAP_METHOD_PAIRED_CLUSTER,
            "bootstrap_unit": BOOTSTRAP_UNIT_GROUP_ID,
            "cluster_col": BOOTSTRAP_UNIT_GROUP_ID,
            "expected_test_count": rq1_cfg["expected_test_count"],
            "dataset_id": rq1_cfg["dataset_id"],
            "dataset_revision": rq1_cfg["dataset_revision"],
            "parquet_revision": rq1_cfg["parquet_revision"],
        },
        "source_fingerprint_sha256": source["aggregate_sha256"],
        "status": STATUS_PREFLIGHT,
    }
    payload["upstream_preflight_sha256"] = sha256_json(_without(payload, "upstream_preflight_sha256"))
    return payload


def locked_architecture() -> Dict[str, Any]:
    return {
        "encoder_id": LOCKED_ENCODER_ID,
        "encoder_revision": LOCKED_ENCODER_REVISION,
        "decoder_id": LOCKED_DECODER_ID,
        "decoder_revision": LOCKED_DECODER_REVISION,
        "target_lang": LOCKED_TARGET_LANG,
        "target_normalization_version": TARGET_NORMALIZATION_VERSION,
        "sample_rate": LOCKED_SAMPLE_RATE,
        "max_audio_duration": LOCKED_MAX_AUDIO_DURATION,
        "max_target_length": LOCKED_MAX_TARGET_LENGTH,
        "truncation_policy": "train_labels_may_truncate_eval_uses_canonical_text_vi_norm",
        "hard_max_truncation_rate": LOCKED_HARD_MAX_TRUNCATION_RATE,
        "metric_for_best_model": BEST_CHECKPOINT_METRIC,
        "greater_is_better": LOCKED_GREATER_IS_BETTER,
        "torch_version": LOCKED_TORCH_VERSION,
        "transformers_version": LOCKED_TRANSFORMERS_VERSION,
        "accelerate_version": LOCKED_ACCELERATE_VERSION,
    }


def build_arm_training_contract(fields: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {key: fields.get(key) for key in SCIENTIFIC_TRAINING_KEYS}
    payload["contract_version"] = FINAL_CONTRACT_VERSION
    payload["schema_version"] = FINAL_SCHEMA_VERSION
    payload["code_version"] = FINAL_CODE_VERSION
    payload.setdefault("stage_version", "rq2_final_train_v1")
    extra = {key: fields[key] for key in fields if key not in payload and key not in RUNTIME_ONLY_KEYS}
    payload.update(extra)
    if payload.get("arm") not in ARMS:
        raise TrainingContractError(f"unknown arm {payload.get('arm')!r}")
    if payload["arm"] == ARM_D0 and payload.get("d0_policy") != D0_POLICY:
        raise TrainingContractError("D0 contract must reuse the frozen RQ1 Direct checkpoint")
    if str(payload.get("init_policy") or "") == REJECTED_PUBLIC_PRETRAINED_INIT_POLICY:
        raise TrainingContractError("public pretrained XLS-R+mBART reinitialization is rejected")
    if payload["arm"] in TRAINABLE_ARMS:
        if payload.get("init_policy") != AUGMENTATION_INIT_POLICY:
            raise TrainingContractError("augmentation arms must continue from the frozen RQ1 D0 checkpoint")
        if not is_sha256(payload.get("d0_init_model_state_sha256")):
            raise TrainingContractError("trainable arm is missing frozen D0 model-state fingerprint")
        if not is_sha256(payload.get("d0_checkpoint_fingerprint_sha256")):
            raise TrainingContractError("trainable arm is missing frozen D0 checkpoint fingerprint")
        mix = normalize_gold_pseudo_mix_policy(payload)
        payload["gold_pseudo_mix_policy"] = mix["gold_pseudo_mix_policy"]
        payload["gold_slots"] = mix["gold_slots"]
        payload["pseudo_slots"] = mix["pseudo_slots"]
        if not mix["configured"]:
            raise TrainingContractError(
                "GOLD_PSEUDO_MIX_POLICY is unset; trainable-arm training contracts require an explicit gold:pseudo ratio"
            )
        seed_policy = require_configured_seed_policy(payload, for_real_training=True)
        payload["seed_policy"] = seed_policy["seed_policy"]
        payload["seed_policy_seeds"] = list(seed_policy["seed_policy_seeds"])
        payload["seed_runs"] = list(seed_policy["seed_runs"])
        payload["seed_policy_record"] = dict(seed_policy["seed_policy_record"])
        declared = [int(seed) for seed in seed_policy["seed_policy_seeds"]]
        if payload.get("active_seed") is None:
            raise TrainingContractError(
                "seed-specific training contract is missing active_seed; it does not default to seeds[0]"
            )
        active = int(payload["active_seed"])
        if active not in declared:
            raise TrainingContractError(f"active_seed {active} is not in the declared seed list")
        if seed_policy["seed_policy"] == SEED_POLICY_SINGLE and declared != [active]:
            raise TrainingContractError("single-seed active_seed must be the only declared seed")
        payload["active_seed"] = active
        payload["seed"] = active
        payload["dataloader_seed"] = active
    if payload["arm"] == ARM_D0:
        payload["init_policy"] = D0_POLICY
        payload.setdefault("gold_pseudo_mix_policy", GOLD_PSEUDO_MIX_POLICY_UNSET)
        payload.setdefault("seed_policy", payload.get("seed_policy") or SEED_POLICY_UNSET)
    for key in ("encoder_id", "encoder_revision", "decoder_id", "decoder_revision", "target_lang"):
        if payload.get(key) != locked_architecture()[key]:
            raise TrainingContractError(f"training contract {key} drifted from the locked Direct architecture")
    for banned in RUNTIME_ONLY_KEYS:
        if banned in payload:
            raise TrainingContractError(f"{banned} is a runtime control and cannot enter the scientific hash")
    assert_scientific_value_portable(
        {k: v for k, v in payload.items() if k not in ("direct_state_dir",)},
        label="training_contract",
    )
    payload["arm_training_contract_sha256"] = sha256_json(_without(payload, "arm_training_contract_sha256"))
    return payload


def materialize_seed_bound_contracts(fields: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build one contract per declared seed.

    Single-seed mode returns exactly one contract. Multi-seed mode binds
    ``active_seed`` for every declared seed and never keeps only ``seeds[0]``.
    """
    if fields.get("arm") == ARM_D0:
        return [build_arm_training_contract(fields)]
    policy = normalize_seed_policy(fields)
    if not policy.get("configured"):
        return [build_arm_training_contract(dict(fields))]
    if policy.get("seed_policy") == SEED_POLICY_MULTI:
        built = []
        for run in policy["seed_runs"]:
            payload = dict(fields)
            payload["active_seed"] = int(run["seed"])
            built.append(build_arm_training_contract(payload))
        declared = [int(seed) for seed in policy["seed_policy_seeds"]]
        produced = [int(contract["active_seed"]) for contract in built]
        if produced != declared:
            raise TrainingContractError("multi-seed materialization did not cover every declared seed")
        return built
    only_seed = int(policy["active_seed"])
    payload = dict(fields)
    payload["active_seed"] = only_seed
    built = [build_arm_training_contract(payload)]
    if policy.get("configured") and policy.get("seed_policy") == SEED_POLICY_SINGLE and len(built) != 1:
        raise TrainingContractError("single-seed materialization must produce exactly one contract")
    return built


def index_seed_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Index seed runs by active_seed. Duplicate or missing seeds fail closed."""
    indexed: Dict[int, Dict[str, Any]] = {}
    for run in runs:
        contract = run.get("contract") or run
        if contract.get("active_seed") is None:
            raise TrainingContractError("seed run is missing active_seed")
        seed = int(contract["active_seed"])
        if seed in indexed:
            raise TrainingContractError(f"duplicate active_seed {seed}")
        indexed[seed] = dict(run)
    return indexed


def iter_declared_seed_runs(seed_runs: Mapping[str, Any]) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Yield ``(arm, active_seed, run)`` from an arm to seed-map.

    ``seed_runs[arm]`` is always a mapping of active seed to run. Iterating the
    mapping itself yields seeds, not run objects.
    """
    found: List[Tuple[str, Any, Dict[str, Any]]] = []
    for arm, runs in seed_runs.items():
        if not isinstance(runs, Mapping):
            raise TrainingContractError(f"{arm} seed runs must be a mapping of active_seed to run")
        for active_seed, run in runs.items():
            if not isinstance(run, Mapping) or "layout" not in run:
                raise TrainingContractError(f"{arm} seed {active_seed} run is missing layout")
            found.append((str(arm), active_seed, dict(run)))
    return found


def verify_arm_training_contract(contract: Mapping[str, Any]) -> None:
    verify_self_hash(contract, "arm_training_contract_sha256", "arm training contract")
    rebuilt = build_arm_training_contract(_without(dict(contract), "arm_training_contract_sha256"))
    if rebuilt["arm_training_contract_sha256"] != contract["arm_training_contract_sha256"]:
        raise TrainingContractError("arm training contract hash drifted")


def assert_augmentation_fairness(d_random: Mapping[str, Any], d_quality: Mapping[str, Any]) -> None:
    verify_arm_training_contract(d_random)
    verify_arm_training_contract(d_quality)
    if d_random.get("arm") != ARM_RANDOM or d_quality.get("arm") != ARM_QUALITY:
        raise TrainingContractError("fairness check requires d_random and d_quality contracts")
    diffs = []
    for key in FAIRNESS_KEYS:
        if d_random.get(key) != d_quality.get(key):
            diffs.append(key)
    if diffs:
        raise TrainingContractError("D-Random/D-Quality fairness mismatch: " + ", ".join(diffs))
    return build_fairness_proof(d_random, d_quality)


def build_fairness_proof(d_random: Mapping[str, Any], d_quality: Mapping[str, Any]) -> Dict[str, Any]:
    proof = {
        "supervised_ordered_uid_hash": str(d_random.get("supervised_ordered_uid_hash") or ""),
        "supervised_uid_set_hash": str(d_random.get("supervised_uid_set_hash") or ""),
        "supervised_pair_hash": str(d_random.get("supervised_pair_hash") or ""),
        "supervised_audio_identity_hash": str(d_random.get("supervised_audio_identity_hash") or ""),
        "validation_ordered_uid_hash": str(d_random.get("validation_ordered_uid_hash") or ""),
        "validation_uid_set_hash": str(d_random.get("validation_uid_set_hash") or ""),
        "validation_pair_hash": str(d_random.get("validation_pair_hash") or ""),
        "validation_audio_identity_hash": str(d_random.get("validation_audio_identity_hash") or ""),
        "d_random_pseudo_ordered_uid_hash": str(d_random.get("pseudo_ordered_uid_hash") or ""),
        "d_quality_pseudo_ordered_uid_hash": str(d_quality.get("pseudo_ordered_uid_hash") or ""),
        "d_random_pseudo_pair_hash": str(d_random.get("pseudo_pair_hash") or ""),
        "d_quality_pseudo_pair_hash": str(d_quality.get("pseudo_pair_hash") or ""),
        "only_pseudo_labeled_component_differs": True,
    }
    for key in (
        "supervised_ordered_uid_hash",
        "supervised_uid_set_hash",
        "supervised_pair_hash",
        "supervised_audio_identity_hash",
        "validation_ordered_uid_hash",
        "validation_uid_set_hash",
        "validation_pair_hash",
        "validation_audio_identity_hash",
    ):
        if not is_sha256(proof[key]):
            raise TrainingContractError(f"fairness proof missing {key}")
        if str(d_random.get(key) or "") != str(d_quality.get(key) or ""):
            raise TrainingContractError(f"fairness proof {key} is not shared by both arms")
    proof["fairness_proof_sha256"] = sha256_json(_without(proof, "fairness_proof_sha256"))
    return proof


def experiment_root_for_arm(
    *,
    durable_root: Union[str, Path],
    contract_hash: str,
    arm: str,
) -> Path:
    if arm not in ARMS:
        raise ArmIsolationError(f"unknown arm {arm!r}")
    h = _require_sha256(contract_hash, "final contract hash")
    return Path(durable_root) / "bahnar_s2tt" / "rq2_final_state" / f"contract_{h[:16]}" / arm


def assert_checkpoint_arm_isolation(
    checkpoint_path: Union[str, Path],
    *,
    arm: str,
    expected_contract_hash: str,
    expected_data_hash: str,
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise ArmIsolationError(f"checkpoint missing: {path}")
    fingerprint = _read_json(path / "full_experiment_fingerprint.json") if (path / "full_experiment_fingerprint.json").is_file() else _read_json(path.parent / "full_experiment_fingerprint.json")
    if str(fingerprint.get("arm") or "") != arm:
        raise ArmIsolationError(f"checkpoint arm {fingerprint.get('arm')!r} is not {arm}")
    if str(fingerprint.get("arm_training_contract_sha256") or "") != expected_contract_hash:
        raise ArmIsolationError("checkpoint training contract does not match the arm")
    if str(fingerprint.get("data_manifest_sha256") or "") != expected_data_hash:
        raise ArmIsolationError("checkpoint data contract does not match the arm")
    return fingerprint


def evaluation_protocol(rq1_eval: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = {
        "reference_field": "text_vi_norm",
        "hypothesis_field": "hypothesis_vi",
        "metrics": ["sacrebleu", "chrfpp"],
        "sacrebleu_source": "src.metrics.mt_corpus_metrics",
        "chrf_word_order": 2,
        "bootstrap_method": BOOTSTRAP_METHOD_PAIRED_CLUSTER,
        "bootstrap_unit": BOOTSTRAP_UNIT_GROUP_ID,
        "cluster_col": BOOTSTRAP_UNIT_GROUP_ID,
        "bootstrap_samples": int(rq1_eval["bootstrap_samples"]),
        "seed": int(rq1_eval["seed"]),
        "confidence": float(rq1_eval["confidence"]),
        "comparisons": [
            [ARM_RANDOM, ARM_D0],
            [ARM_QUALITY, ARM_D0],
            [ARM_QUALITY, ARM_RANDOM],
        ],
        "eval_uses_untruncated_references": True,
    }
    protocol["evaluation_protocol_sha256"] = sha256_json(_without(protocol, "evaluation_protocol_sha256"))
    return protocol


def unlock_g_test(
    flags: Nb14Flags,
    readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    """Human-gated unlock. Nothing here opens G_test; it only checks freeze state."""
    if flags.allow_g_test_evaluation is not True:
        raise GTestFirewallError("ALLOW_G_TEST_EVALUATION is false")
    if flags.rq2_final_frozen is not True:
        raise GTestFirewallError("RQ2_FINAL_FROZEN is false; unlock refused")
    flat = flatten_readiness(readiness)
    missing = [name for name in UNLOCK_REQUIRED if flat.get(name) is not True]
    if missing:
        raise GTestFirewallError("G_test unlock refused; pending: " + ", ".join(missing))
    return {"status": STATUS_UNLOCK, "allow_g_test_evaluation": True}


def derive_final_status(checks: Mapping[str, bool]) -> Dict[str, Any]:
    required = (
        "upstream_contracts_verified",
        "selections_frozen",
        "training_contracts_verified",
        "required_arms_complete",
        "checkpoint_isolation_proven",
        "best_checkpoints_frozen",
        "validation_model_selection_complete",
        "g_test_explicitly_unlocked",
        "identical_paired_uid_order",
        "identical_references",
        "metrics_finite",
        "bootstrap_complete",
        "artifacts_hashed",
        "final_contract_hashed",
        "final_artifacts_reverified",
        "current_committed",
    )
    failed = [name for name in required if checks.get(name) is not True]
    status = STATUS_SUCCESS if not failed else STATUS_FAIL
    return {"status": status, "failed_checks": failed, "checks": {name: bool(checks.get(name)) for name in required}}


PRE_PUBLISH_REQUIRED = (
    "upstream_contracts_verified",
    "selections_frozen",
    "training_contracts_verified",
    "required_arms_complete",
    "checkpoint_isolation_proven",
    "best_checkpoints_frozen",
    "validation_model_selection_complete",
    "g_test_explicitly_unlocked",
    "identical_paired_uid_order",
    "identical_references",
    "metrics_finite",
    "bootstrap_complete",
    "artifacts_hashed",
    "final_contract_hashed",
    "final_artifacts_reverified",
)


def build_final_contract(fields: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(fields)
    payload["contract_version"] = FINAL_CONTRACT_VERSION
    payload["schema_version"] = FINAL_SCHEMA_VERSION
    payload["code_version"] = FINAL_CODE_VERSION
    payload.pop("created_at_utc", None)
    payload.pop("sealed_at_utc", None)
    payload["rq2_final_contract_sha256"] = sha256_json(_without(payload, "rq2_final_contract_sha256"))
    verify_self_hash(payload, "rq2_final_contract_sha256", "RQ2 final contract")
    return payload


def _resolve_d0_checkpoint_dir(state: Path, *, experiment_id: str, best_name: str) -> Path:
    candidates = [
        state / "checkpoints" / "full_train" / experiment_id / "ckpts" / best_name,
        state / "checkpoints" / "full_train" / experiment_id / best_name,
        state / best_name,
        state / "full_train" / experiment_id / best_name,
        state / experiment_id / best_name,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise UpstreamGateError(f"RQ1 D0 checkpoint directory missing: {best_name}")


def _model_state_sha256(checkpoint: Path) -> str:
    for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin"):
        path = checkpoint / name
        if path.is_file():
            return sha256_file(path)
    raise UpstreamGateError(f"RQ1 D0 checkpoint has no model-state file: {checkpoint.name}")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise UpstreamGateError(f"missing required JSON: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UpstreamGateError(f"JSON object required: {path.name}")
    return payload
