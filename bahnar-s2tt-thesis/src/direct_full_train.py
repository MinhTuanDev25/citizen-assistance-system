"""Training/evaluation helpers for Notebook 05 Direct S2TT."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.asr_full_train import (
    ENV_DURABLE_CHECKPOINT_BUDGET_BYTES,
    FULL_TRAIN_MARKER,
    assert_checkpoint_complete_for_resume,
    estimate_checkpoint_bytes,
    plan_expected_resume_position,
    read_checkpoint_fingerprint,
    resolve_best_checkpoint_from_durable,
    resolve_durable_checkpoint_budget_bytes,
)
from src.data_utils import compute_uid_set_hash, sha256_file
from src.direct_contract import STATUS_DIRECT_TRAINING, STATUS_FAILED
from src.direct_data import DirectAccessTracker
from src.metrics import mt_corpus_metrics

RESUME_TEST_PHASE_A_STEPS = 20
RESUME_TEST_PHASE_B_STEPS = 40
RESUME_TEST_SUBSET_SIZE = 128
CHECKPOINT_PEAK_COPIES = 4  # latest + best + rollback/LKG + temporary upload


def build_direct_training_hparams(
    *,
    n_train: int,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    num_train_epochs: float = 3.0,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    save_total_limit: int = 2,
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = True,
) -> Dict[str, Any]:
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    steps_per_epoch = math.ceil(n_train / max(1, per_device_train_batch_size * gradient_accumulation_steps))
    return {
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "per_device_eval_batch_size": int(per_device_eval_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "learning_rate": float(learning_rate),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "num_train_epochs": float(num_train_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "estimated_total_steps": int(math.ceil(steps_per_epoch * float(num_train_epochs))),
        "save_steps": int(save_steps),
        "eval_steps": int(eval_steps),
        "save_total_limit": int(save_total_limit),
        "fp16": bool(fp16),
        "bf16": bool(bf16),
        "gradient_checkpointing": bool(gradient_checkpointing),
    }


def fixed_subset(df: pd.DataFrame, n: int, *, seed: int = 42) -> pd.DataFrame:
    if len(df) <= int(n):
        return df.sort_values("record_uid").reset_index(drop=True)
    return df.sample(n=int(n), random_state=int(seed)).sort_values("record_uid").reset_index(drop=True)


def monitor_manifest(df: pd.DataFrame, *, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Semantic content lock for the validation monitor used as train-time eval_dataset.

    Byte integrity is a separate check against the persisted CSV
    (``assert_monitor_file_sha256``). Do not compare ``file_sha256`` to a
    re-serialized DataFrame: pandas float/NaN round-trips change bytes.
    """
    from src.direct_data import compute_ordered_row_hash, compute_pair_hash

    if "record_uid" not in df.columns or "text_vi_norm" not in df.columns:
        raise RuntimeError("monitor frame must include record_uid and text_vi_norm")
    uids = df["record_uid"].astype(str).tolist()
    payload = {
        "n": len(uids),
        "uid_set_hash": compute_uid_set_hash(pd.DataFrame({"record_uid": uids})),
        "pair_hash": compute_pair_hash(df),
        "ordered_row_hash": compute_ordered_row_hash(df),
        "record_uids": uids,
    }
    if path is not None:
        payload["file_sha256"] = sha256_file(path)
    return payload


def assert_monitor_file_sha256(
    monitor_path: Union[str, Path],
    expected_sha256: str,
) -> str:
    """Byte lock: SHA-256 of the on-disk monitor CSV vs training contract."""
    actual = sha256_file(monitor_path)
    want = str(expected_sha256 or "").strip().lower()
    if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
        raise RuntimeError(f"monitor_file_sha256 must be a 64-hex SHA-256, got {expected_sha256!r}")
    if actual != want:
        raise RuntimeError(
            f"Direct monitor file SHA-256 mismatch: file={actual} contract={want}"
        )
    return actual


def assert_monitor_matches_validation(
    monitor_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    size: int,
    seed: int = 42,
    monitor_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Semantic authority for the train-time monitor is the locked validation frame.

    Persisted monitor rows must equal ``fixed_subset(validation, size, seed)``
    on n / uid_set_hash / pair_hash / ordered_row_hash. Byte integrity of the
    CSV is ``assert_monitor_file_sha256`` vs ``training_contract['monitor_file_sha256']``.
    """
    expected = fixed_subset(validation_df, int(size), seed=int(seed))
    got = monitor_manifest(monitor_df)
    want = monitor_manifest(expected)
    diffs = []
    for key in ("n", "uid_set_hash", "pair_hash", "ordered_row_hash"):
        if got.get(key) != want.get(key):
            diffs.append(f"{key}: persisted={got.get(key)!r} expected={want.get(key)!r}")
    if diffs:
        raise RuntimeError("Direct monitor drifted from locked validation subset: " + "; ".join(diffs))
    if int(got["n"]) != int(size) and len(validation_df) >= int(size):
        raise RuntimeError(f"Direct monitor size {got['n']} != {size}")
    if monitor_path is not None:
        got["file_sha256"] = sha256_file(monitor_path)
    return got


def compute_direct_seq2seq_metrics(eval_preds: Any, *, tokenizer: Any) -> Dict[str, float]:
    """Monitor-set metrics. References here may be truncated labels (train selection only)."""
    if hasattr(eval_preds, "predictions") and hasattr(eval_preds, "label_ids"):
        preds, labels = eval_preds.predictions, eval_preds.label_ids
    elif isinstance(eval_preds, (tuple, list)) and len(eval_preds) >= 2:
        preds, labels = eval_preds[0], eval_preds[1]
    else:
        raise RuntimeError(f"Unexpected eval_preds shape/type: {type(eval_preds)!r}")
    if isinstance(preds, tuple):
        preds = preds[0]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer.pad_token_id is required")
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    preds = np.where(preds != -100, preds, int(pad_id))
    labels = np.where(labels != -100, labels, int(pad_id))
    pred_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    ref_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    m = mt_corpus_metrics(pred_text, ref_text)
    return {"sacrebleu": float(m["sacrebleu"]), "chrfpp": float(m["chrfpp"]), "monitor_n": int(m["n"])}


def _reference_from_frame(frame: pd.DataFrame, uid: str) -> str:
    hit = frame.loc[frame["record_uid"].astype(str) == str(uid)]
    if hit.empty:
        raise RuntimeError(f"Evaluation reference missing for record_uid={uid}")
    if "text_vi_norm" not in hit.columns:
        raise RuntimeError("Evaluation frame lacks text_vi_norm")
    ref = str(hit.iloc[0]["text_vi_norm"])
    if not ref.strip():
        raise RuntimeError(f"Empty text_vi_norm for record_uid={uid}")
    return ref


def generate_direct_predictions(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    collator: Any,
    batch_size: int,
    generation_max_length: int,
    num_beams: int,
    forced_bos_token_id: int,
) -> list[Dict[str, Any]]:
    """
    Generate Vietnamese hypotheses.

    References are always the full ``text_vi_norm`` from the dataset frame keyed
    by ``record_uid``. Truncated training labels are never decoded as references.
    """
    import torch
    from torch.utils.data import DataLoader

    frame = dataset.frame
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collator)
    device = next(model.parameters()).device
    model.eval()
    rows = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch = dict(batch)
            batch.pop("labels", None)
            inputs = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
            generated = model.generate(
                **inputs,
                max_length=int(generation_max_length),
                num_beams=int(num_beams),
                forced_bos_token_id=int(forced_bos_token_id),
            )
            preds = tokenizer.batch_decode(generated.detach().cpu().numpy(), skip_special_tokens=True)
            for p in preds:
                uid = str(frame.iloc[offset]["record_uid"])
                rows.append({
                    "record_uid": uid,
                    "prediction_vi": p,
                    "reference_vi": _reference_from_frame(frame, uid),
                })
                offset += 1
    return rows


def evaluate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    preds = [str(r["prediction_vi"]) for r in rows]
    refs = [str(r["reference_vi"]) for r in rows]
    return mt_corpus_metrics(preds, refs)


def write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def load_stage_success(
    path: Union[str, Path],
    *,
    status: str,
    expected_contract_hash: str,
    expected_experiment_id: str,
    expected_training_contract_hash: str,
) -> Dict[str, Any]:
    """Fail-closed load of a prior Direct stage summary (stale/wrong config cannot pass)."""
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"Required success artifact missing: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("status") != status:
        raise RuntimeError(f"Unexpected status in {p.name}: {data.get('status')!r}")
    if str(data.get("contract_hash")) != str(expected_contract_hash):
        raise RuntimeError(f"Data contract mismatch in {p.name}")
    if str(data.get("experiment_id")) != str(expected_experiment_id):
        raise RuntimeError(f"experiment_id mismatch in {p.name}")
    got_tc = str(
        data.get("direct_training_contract_hash")
        or data.get("training_contract_hash")
        or ""
    )
    if got_tc != str(expected_training_contract_hash):
        raise RuntimeError(f"direct_training_contract_hash mismatch in {p.name}")
    return data


def load_success(
    path: Union[str, Path],
    *,
    status: str,
    expected_contract_hash: Optional[str] = None,
    expected_experiment_id: Optional[str] = None,
    expected_training_contract_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper; full-stage gates should call load_stage_success."""
    if expected_experiment_id is not None or expected_training_contract_hash is not None:
        if expected_contract_hash is None or expected_experiment_id is None or expected_training_contract_hash is None:
            raise RuntimeError("Stage success load requires contract_hash, experiment_id and training contract hash")
        return load_stage_success(
            path,
            status=status,
            expected_contract_hash=expected_contract_hash,
            expected_experiment_id=expected_experiment_id,
            expected_training_contract_hash=expected_training_contract_hash,
        )
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"Required success artifact missing: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("status") != status:
        raise RuntimeError(f"Unexpected status in {p.name}: {data.get('status')!r}")
    if expected_contract_hash is not None and str(data.get("contract_hash")) != str(expected_contract_hash):
        raise RuntimeError(f"Contract mismatch in {p.name}")
    return data


def derive_direct_handoff(status: str, *, frozen_test_accessed: bool) -> Dict[str, bool]:
    accessed = bool(frozen_test_accessed)
    return {
        "ready_for_rq1_evaluation": status == "SUCCESS_DIRECT_EVALUATE" and not accessed,
        "ready_for_rq1_final": False,
        "frozen_test_accessed": accessed,
    }


def frozen_flag_from_tracker(tracker: Optional[DirectAccessTracker]) -> bool:
    if tracker is None:
        raise RuntimeError("Direct frozen-test flag requires an access tracker (not a hardcoded False)")
    return bool(tracker.frozen_test_accessed())


def seedable_random_epoch_indices(n: int, *, seed: int, epoch: int) -> list[int]:
    """
    Index order of Accelerate ``SeedableRandomSampler`` at ``epoch``.

    HuggingFace Trainer (transformers 4.57 + accelerate 1.10 in this repo)
    wraps the default ``RandomSampler`` with ``SeedableRandomSampler`` and
    calls ``set_epoch(epoch)``. The permutation is ``randperm`` with
    ``generator.manual_seed(seed + epoch)``, not a one-shot epoch-0
    ``torch.manual_seed`` + ``RandomSampler``.
    """
    import torch

    if n <= 0:
        raise ValueError("dataset length must be positive")

    class _DS:
        def __len__(self):
            return int(n)

    try:
        from accelerate.data_loader import SeedableRandomSampler

        sampler = SeedableRandomSampler(_DS(), replacement=False, data_seed=int(seed))
        sampler.set_epoch(int(epoch))
        return list(sampler)
    except ImportError:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + int(epoch))
        return torch.randperm(int(n), generator=generator).tolist()


def random_sampler_epoch_indices(n: int, *, seed: int, epoch: int = 0) -> list[int]:
    """Epoch-aware train sampler indices. Prefer ``seedable_random_epoch_indices``."""
    return seedable_random_epoch_indices(n, seed=int(seed), epoch=int(epoch))


def random_sampler_uid_order(uids: Sequence[str], *, seed: int, epoch: int = 0) -> list[str]:
    names = [str(u) for u in uids]
    order = seedable_random_epoch_indices(len(names), seed=int(seed), epoch=int(epoch))
    return [names[i] for i in order]


def plan_direct_resume_position(
    uids: Sequence[str],
    *,
    seed: int,
    resume_step: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> Dict[str, Any]:
    """
    Expected first UID after resume, using the permutation of the resume epoch.

    ``plan_expected_resume_position`` only indexes into a provided order. This
    helper builds that order from Accelerate's seedable sampler at
    ``resume_step // num_update_steps_per_epoch``.
    """
    names = [str(u) for u in uids]
    if not names:
        raise ValueError("uids must be non-empty for resume position planning")
    skeleton = plan_expected_resume_position(
        names,
        resume_step=int(resume_step),
        per_device_train_batch_size=int(per_device_train_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
    epoch = int(skeleton["n_epochs_wrapped"])
    ordered = random_sampler_uid_order(names, seed=int(seed), epoch=epoch)
    plan = plan_expected_resume_position(
        ordered,
        resume_step=int(resume_step),
        per_device_train_batch_size=int(per_device_train_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
    plan["epoch"] = epoch
    plan["seed"] = int(seed)
    plan["sampler"] = "accelerate.SeedableRandomSampler"
    plan["epoch0_first_uid"] = random_sampler_uid_order(names, seed=int(seed), epoch=0)[
        int(plan["expected_sample_index"])
    ]
    return plan


def skip_first_batches_preserving_epoch(skip_fn: Any, dataloader: Any, num_batches: int = 0) -> Any:
    """
    Compat for accelerate 1.10.1 ``skip_first_batches``.

    Hugging Face Trainer calls ``set_epoch(epochs_trained)`` then
    ``skip_first_batches(...)``. The pinned Accelerate rebuilds a
    ``DataLoaderShard`` with ``iteration=0``, so ``__iter__`` replays
    epoch-0 shuffle. Preserve the pre-skip iteration (upstream semantics:
    ``epoch_dataloader.iteration = epochs_trained`` after skip).
    """
    resumed_epoch = int(getattr(dataloader, "iteration", 0) or 0)
    skipped = skip_fn(dataloader, num_batches)
    if hasattr(skipped, "iteration"):
        skipped.iteration = resumed_epoch
    return skipped


def make_resume_safe_seq2seq_trainer_cls(base_cls: Any = None) -> Any:
    """
    Seq2SeqTrainer subclass for transformers==4.57.6 + accelerate==1.10.1.

    Patches ``transformers.trainer.skip_first_batches`` only for the duration
    of ``_inner_training_loop`` so resume keeps the SeedableRandomSampler
    epoch after data skip. Fresh training (no resume skip) is unchanged.
    """
    from transformers import Seq2SeqTrainer

    Base = Seq2SeqTrainer if base_cls is None else base_cls

    class ResumeSafeSeq2SeqTrainer(Base):  # type: ignore[misc,valid-type]
        def _inner_training_loop(self, *args, **kwargs):
            import transformers.trainer as trainer_mod

            original = trainer_mod.skip_first_batches

            def _skip(dataloader, num_batches=0):
                return skip_first_batches_preserving_epoch(original, dataloader, num_batches)

            trainer_mod.skip_first_batches = _skip
            try:
                return super()._inner_training_loop(*args, **kwargs)
            finally:
                trainer_mod.skip_first_batches = original

    ResumeSafeSeq2SeqTrainer.__name__ = "ResumeSafeSeq2SeqTrainer"
    ResumeSafeSeq2SeqTrainer.__qualname__ = "ResumeSafeSeq2SeqTrainer"
    return ResumeSafeSeq2SeqTrainer


def _finite_metric_values(metrics: Mapping[str, Any]) -> list[bool]:
    flags = []
    for key in ("eval_sacrebleu", "eval_chrfpp", "sacrebleu", "chrfpp", "train_loss"):
        if key not in metrics:
            continue
        try:
            flags.append(math.isfinite(float(metrics[key])))
        except (TypeError, ValueError):
            flags.append(False)
    return flags


def derive_direct_training_status(
    *,
    global_step: int,
    max_steps: int,
    metrics: Mapping[str, Any],
    latest_checkpoint: Union[str, Path],
    best_checkpoint: Union[str, Path],
    durable_best_resolved: bool,
    frozen_test_accessed: bool,
) -> Dict[str, Any]:
    """Fail-closed SUCCESS_DIRECT_TRAINING gate. Status is never assigned first."""
    checks: Dict[str, bool] = {}
    checks["max_steps_positive"] = int(max_steps) > 0
    checks["reached_max_steps"] = int(global_step) == int(max_steps) and int(max_steps) > 0
    finite = _finite_metric_values(metrics)
    checks["metrics_present"] = bool(finite)
    checks["metrics_finite"] = bool(finite) and all(finite)
    try:
        assert_checkpoint_complete_for_resume(latest_checkpoint)
        checks["latest_checkpoint_complete"] = True
    except Exception:
        checks["latest_checkpoint_complete"] = False
    best = Path(best_checkpoint)
    checks["best_checkpoint_exists"] = best.is_dir()
    try:
        assert_checkpoint_complete_for_resume(best)
        checks["best_checkpoint_complete"] = True
    except Exception:
        checks["best_checkpoint_complete"] = False
    checks["durable_best_resolved"] = bool(durable_best_resolved)
    checks["frozen_test_not_accessed"] = frozen_test_accessed is False
    failed = sorted(k for k, ok in checks.items() if not ok)
    return {
        "status": STATUS_DIRECT_TRAINING if not failed else STATUS_FAILED,
        "checks": checks,
        "failed_checks": failed,
    }


def assert_direct_checkpoint_budget(
    *,
    n_parameters: int,
    env: Optional[Mapping[str, str]] = None,
    copies: int = CHECKPOINT_PEAK_COPIES,
) -> Dict[str, Any]:
    """
    Require env budget at checkpoint-write stages.

    Peak = latest + best + rollback/LKG + one temporary upload. Never delete
    last-known-good to create headroom.
    """
    budget = resolve_durable_checkpoint_budget_bytes(None, env=env, required=True)
    est = estimate_checkpoint_bytes(n_parameters=int(n_parameters))
    peak = int(est["checkpoint_bytes"]) * int(copies)
    report = {
        **est,
        "peak_copies": int(copies),
        "peak_bytes": peak,
        "budget_bytes": int(budget),
        "env_name": ENV_DURABLE_CHECKPOINT_BUDGET_BYTES,
        "protect_last_known_good": True,
    }
    if peak > int(budget):
        raise RuntimeError(
            f"Insufficient durable checkpoint budget: peak(latest+best+rollback+upload)="
            f"{peak} bytes > {ENV_DURABLE_CHECKPOINT_BUDGET_BYTES}={budget}. "
            "Refusing to delete last-known-good to make space."
        )
    return report


def validate_direct_best_checkpoint(
    *,
    experiment_dir: Union[str, Path],
    experiment_id: str,
    best_checkpoint: Union[str, Path],
    expected_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Fail closed before durable sync: best must be a complete step checkpoint
    under ``full_train/<experiment_id>`` matching the training contract.
    """
    root = Path(experiment_dir).resolve()
    if root.name != str(experiment_id) or root.parent.name != FULL_TRAIN_MARKER:
        raise RuntimeError(
            f"Best checkpoint must live under {FULL_TRAIN_MARKER}/{experiment_id}, got {root}"
        )
    name = Path(str(best_checkpoint)).name
    if not name.startswith("checkpoint-"):
        raise RuntimeError(f"best_checkpoint_name is not a step checkpoint: {best_checkpoint!r}")
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Best checkpoint {path} is not under {root}") from exc
    if not path.is_dir():
        raise RuntimeError(f"Best checkpoint missing under experiment dir: {path}")
    assert_checkpoint_complete_for_resume(path)
    assert_direct_checkpoint_fingerprint(
        path,
        experiment_id=experiment_id,
        expected_training_contract_hash=str(expected_contract.get("direct_training_contract_hash") or ""),
        experiment_root=root,
    )
    return {"best_checkpoint_name": name, "best_checkpoint_path": str(path)}


def restore_direct_best_checkpoint(
    *,
    state_dir: Union[str, Path],
    experiment_id: str,
    train_summary: Mapping[str, Any],
    local_experiment_dir: Union[str, Path],
    expected_contract: Mapping[str, Any],
) -> str:
    """Restore durable LATEST snapshot and resolve the verified best checkpoint."""
    preferred = train_summary.get("best_checkpoint") or train_summary.get("best_checkpoint_name")
    summary = dict(train_summary)
    if preferred:
        summary["best_checkpoint"] = Path(str(preferred)).name
    path = resolve_best_checkpoint_from_durable(
        state_dir,
        experiment_id=experiment_id,
        train_summary=summary,
        local_experiment_dir=local_experiment_dir,
        expected_contract=dict(expected_contract),
    )
    assert_direct_checkpoint_fingerprint(
        path,
        experiment_id=experiment_id,
        expected_training_contract_hash=str(expected_contract.get("direct_training_contract_hash") or ""),
        experiment_root=local_experiment_dir,
    )
    return path


def assert_direct_checkpoint_fingerprint(
    checkpoint_path: Union[str, Path],
    *,
    experiment_id: str,
    expected_training_contract_hash: str,
    experiment_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Verify completeness is handled by ASR; this binds Direct training identity."""
    cp = Path(checkpoint_path)
    if experiment_root is not None:
        root = Path(experiment_root).resolve()
        try:
            cp.resolve().relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Best checkpoint {cp} is not under experiment root {root}") from exc
    fp = read_checkpoint_fingerprint(cp) or read_checkpoint_fingerprint(cp.parent)
    if not fp:
        raise RuntimeError(f"Direct checkpoint fingerprint missing: {cp}")
    if str(fp.get("experiment_id")) != str(experiment_id):
        raise RuntimeError(f"Direct checkpoint experiment_id mismatch: {fp.get('experiment_id')!r}")
    got = str(fp.get("direct_training_contract_hash") or "")
    if got != str(expected_training_contract_hash):
        raise RuntimeError("Direct checkpoint training-contract hash mismatch")
    return fp
