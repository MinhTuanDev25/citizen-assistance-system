"""RQ2 Notebook 14 evaluation: firewall, untruncated references, paired bootstrap.

Inference and G_test reads are refused until unlock_g_test succeeds.
Metrics reuse ``src.metrics.mt_corpus_metrics`` (SacreBLEU + chrF++ word_order=2).
Bootstrap reuses the RQ1 paired cluster protocol on ``group_id``.
G_test identity reuses ``build_rq1_test_contract`` / ``assert_rq1_test_contract``.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.metrics import mt_corpus_metrics
from src.mt_normalize import normalize_mt_text_v1
from src.rq1_contract import (
    BOOTSTRAP_METHOD_PAIRED_CLUSTER,
    BOOTSTRAP_UNIT_GROUP_ID,
    assert_group_id_complete,
    assert_locked_manifest_bundle_matches_contract,
    assert_rq1_test_contract,
    build_rq1_test_contract,
    ordered_uid_hash,
    reference_content_hash,
    sha256_file,
    sha256_json,
    uid_set_hash,
)
from src.rq2_final_contract import (
    ARM_D0,
    ARM_QUALITY,
    ARM_RANDOM,
    ARMS,
    EvaluationError,
    FINAL_ARTIFACT_FILES,
    FINAL_RELATIVE_DIR,
    GTestFirewallError,
    Nb14Flags,
    STATUS_FAIL,
    STATUS_SUCCESS,
    STATUS_TRAINING_COMPLETE,
    UNLOCK_REQUIRED,
    UpstreamGateError,
    assert_g_test_blocked,
    assert_nb14_output_dir,
    build_final_contract,
    compute_nb14_source_fingerprint,
    derive_final_status,
    evaluation_protocol,
    flatten_readiness,
    is_sha256,
    unlock_g_test,
)
from src.rq2_pseudo_contract import (
    RQ1_FINAL_CONTRACT_FILENAME,
    RQ1_FINAL_STATUS,
    RQ1_FINAL_SUMMARY_FILENAME,
    atomic_write_text,
    finalize_generation,
    project_rq1_final_summary,
    read_current_generation_id,
    stage_generation,
    write_json,
)

RQ1_TEST_CONTRACT_FILENAME = "rq1_test_contract.json"
RQ1_LATEST_UNLOCKED = "LATEST_UNLOCKED.json"


PREDICTION_COLUMNS = (
    "record_uid",
    "group_id",
    "hypothesis_vi",
    "reference_vi",
    "arm",
    "checkpoint_fingerprint",
)


def canonical_references(frame: pd.DataFrame) -> List[str]:
    if "text_vi_norm" in frame.columns:
        return [normalize_mt_text_v1(x) for x in frame["text_vi_norm"].tolist()]
    if "text_vi" in frame.columns:
        return [normalize_mt_text_v1(x) for x in frame["text_vi"].tolist()]
    raise EvaluationError("evaluation frame has no canonical Vietnamese reference")


def audit_target_truncation(token_lengths: Sequence[int], *, max_target_length: int) -> Dict[str, Any]:
    lengths = [int(x) for x in token_lengths]
    n = len(lengths)
    truncated = sum(1 for x in lengths if x > int(max_target_length))
    ordered = sorted(lengths)
    def pct(p: float) -> int:
        if not ordered:
            return 0
        index = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return int(ordered[index])
    return {
        "n_target_truncated": int(truncated),
        "truncation_rate": float(truncated) / float(n) if n else 0.0,
        "max_target_tokens": int(max(ordered) if ordered else 0),
        "p50_target_tokens": pct(50),
        "p90_target_tokens": pct(90),
        "p99_target_tokens": pct(99),
        "max_target_length": int(max_target_length),
    }


def prediction_row(
    *,
    record_uid: str,
    group_id: str,
    hypothesis_vi: str,
    reference_vi: str,
    arm: str,
    checkpoint_fingerprint: str,
) -> Dict[str, Any]:
    ref = normalize_mt_text_v1(reference_vi)
    hyp = normalize_mt_text_v1(hypothesis_vi)
    if not record_uid or not ref:
        raise EvaluationError(f"invalid prediction row for {record_uid!r}")
    if hypothesis_vi is None or not isinstance(hypothesis_vi, str):
        raise EvaluationError(f"missing hypothesis for {record_uid}")
    hyp = normalize_mt_text_v1(hypothesis_vi)
    if hyp is None or not str(hyp).strip():
        raise EvaluationError(f"non-finite hypothesis for {record_uid}")
    if arm not in ARMS:
        raise EvaluationError(f"invalid arm label {arm!r}")
    fingerprint = str(checkpoint_fingerprint or "")
    if not is_sha256(fingerprint):
        raise EvaluationError(f"invalid checkpoint fingerprint for {record_uid}")
    return {
        "record_uid": str(record_uid),
        "group_id": str(group_id),
        "hypothesis_vi": hyp,
        "reference_vi": ref,
        "arm": arm,
        "checkpoint_fingerprint": fingerprint,
    }


def write_predictions_parquet(path: Union[str, Path], rows: Sequence[Mapping[str, Any]]) -> str:
    frame = pd.DataFrame(list(rows), columns=list(PREDICTION_COLUMNS))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return sha256_file(path)


def assert_paired_alignment(frames: Mapping[str, pd.DataFrame]) -> List[str]:
    if set(frames) != set(ARMS):
        raise EvaluationError("paired evaluation requires d0, d_random, and d_quality")
    for arm, frame in frames.items():
        if "record_uid" not in frame.columns:
            raise EvaluationError(f"{arm} missing record_uid")
        if "reference_vi" not in frame.columns:
            raise EvaluationError(f"{arm} missing reference_vi")
        if "group_id" not in frame.columns:
            raise EvaluationError(f"{arm} missing group_id")
        if "checkpoint_fingerprint" not in frame.columns:
            raise EvaluationError(f"{arm} missing checkpoint_fingerprint")
        if "arm" in frame.columns and {str(x) for x in frame["arm"].tolist()} != {arm}:
            raise EvaluationError(f"{arm} prediction arm column is not {arm}")
        fps = frame["checkpoint_fingerprint"].astype(str).tolist()
        if any(not is_sha256(fp) for fp in fps):
            raise EvaluationError(f"{arm} has an invalid checkpoint fingerprint")
        if frame_missing(frame):
            raise EvaluationError(f"{arm} has missing predictions")
        hyps = frame["hypothesis_vi"].tolist()
        if any(not isinstance(h, str) for h in hyps):
            raise EvaluationError(f"{arm} has a non-string hypothesis")
    orders = {arm: frame["record_uid"].astype(str).tolist() for arm, frame in frames.items()}
    refs = {arm: frame["reference_vi"].astype(str).tolist() for arm, frame in frames.items()}
    groups = {arm: frame["group_id"].astype(str).tolist() for arm, frame in frames.items()}
    first = ARM_D0
    expected = orders[first]
    if len(expected) != len(set(expected)):
        raise EvaluationError("duplicate record_uid in paired predictions")
    for arm, uids in orders.items():
        if uids != expected:
            raise EvaluationError(f"{arm} UID order does not match D0")
        if refs[arm] != refs[first]:
            raise EvaluationError(f"{arm} references do not match D0")
        if groups[arm] != groups[first]:
            raise EvaluationError(f"{arm} group_id does not match D0")
    return expected


def frame_missing(frame: pd.DataFrame) -> bool:
    if frame["record_uid"].isna().any():
        return True
    if frame["hypothesis_vi"].isna().any():
        return True
    return bool(frame["hypothesis_vi"].astype(str).map(str.strip).eq("").any())


def arm_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    refs = frame["reference_vi"].astype(str).tolist()
    hyps = frame["hypothesis_vi"].astype(str).tolist()
    payload = mt_corpus_metrics(hyps, refs)
    if not payload.get("finite"):
        raise EvaluationError("non-finite SacreBLEU/chrF++")
    return payload


def comparison_key(left: str, right: str) -> str:
    return f"{left}-{right}"


def paired_cluster_bootstrap_arms(
    frames: Mapping[str, pd.DataFrame],
    *,
    n_samples: int,
    seed: int,
    confidence: float = 0.95,
    cluster_col: str = BOOTSTRAP_UNIT_GROUP_ID,
    comparisons: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Any]:
    if cluster_col != BOOTSTRAP_UNIT_GROUP_ID:
        raise EvaluationError(f"RQ2 bootstrap cluster_col must be {BOOTSTRAP_UNIT_GROUP_ID}")
    uids = assert_paired_alignment(frames)
    work = {arm: frame.reset_index(drop=True) for arm, frame in frames.items()}
    n_clusters = assert_group_id_complete(work[ARM_D0], cluster_col=cluster_col)
    groups = work[ARM_D0][cluster_col].astype(str).map(lambda x: x.strip())
    for arm in (ARM_RANDOM, ARM_QUALITY):
        other = work[arm][cluster_col].astype(str).map(lambda x: x.strip())
        if other.tolist() != groups.tolist():
            raise EvaluationError(f"{arm} cluster identity does not match D0")
    unique_groups = sorted(groups.unique().tolist())
    if len(unique_groups) != n_clusters:
        raise EvaluationError("cluster accounting mismatch")
    cluster_indices = {g: np.flatnonzero(groups.to_numpy() == g) for g in unique_groups}
    pairs = list(comparisons or ((ARM_RANDOM, ARM_D0), (ARM_QUALITY, ARM_D0), (ARM_QUALITY, ARM_RANDOM)))
    observed = {}
    for left, right in pairs:
        lm = arm_metrics(work[left])
        rm = arm_metrics(work[right])
        observed[comparison_key(left, right)] = {
            "sacrebleu": float(lm["sacrebleu"] - rm["sacrebleu"]),
            "chrfpp": float(lm["chrfpp"] - rm["chrfpp"]),
            "left": left,
            "right": right,
        }
    rng = np.random.default_rng(int(seed))
    samples = {key: {"sacrebleu": [], "chrfpp": []} for key in observed}
    for _ in range(int(n_samples)):
        chosen = rng.choice(unique_groups, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_indices[g] for g in chosen])
        for left, right in pairs:
            key = comparison_key(left, right)
            left_m = mt_corpus_metrics(
                work[left].iloc[idx]["hypothesis_vi"].astype(str).tolist(),
                work[left].iloc[idx]["reference_vi"].astype(str).tolist(),
            )
            right_m = mt_corpus_metrics(
                work[right].iloc[idx]["hypothesis_vi"].astype(str).tolist(),
                work[right].iloc[idx]["reference_vi"].astype(str).tolist(),
            )
            samples[key]["sacrebleu"].append(float(left_m["sacrebleu"] - right_m["sacrebleu"]))
            samples[key]["chrfpp"].append(float(left_m["chrfpp"] - right_m["chrfpp"]))
    alpha = (1.0 - float(confidence)) / 2.0
    results = {}
    for key, obs in observed.items():
        entry = {
            "left": obs["left"],
            "right": obs["right"],
            "observed_sacrebleu_delta": obs["sacrebleu"],
            "observed_chrfpp_delta": obs["chrfpp"],
        }
        for metric in ("sacrebleu", "chrfpp"):
            arr = np.asarray(samples[key][metric], dtype=np.float64)
            lo, hi = np.quantile(arr, [alpha, 1.0 - alpha])
            entry[f"{metric}_ci_lower"] = float(lo)
            entry[f"{metric}_ci_upper"] = float(hi)
            if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
                raise EvaluationError(f"non-finite bootstrap CI for {key} {metric}")
            if not math.isfinite(float(obs["sacrebleu" if metric == "sacrebleu" else "chrfpp"])):
                raise EvaluationError(f"non-finite observed delta for {key} {metric}")
        results[key] = entry
    return {
        "bootstrap_method": BOOTSTRAP_METHOD_PAIRED_CLUSTER,
        "bootstrap_unit": cluster_col,
        "cluster_col": cluster_col,
        "n_clusters": int(n_clusters),
        "n_rows": int(len(uids)),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "confidence": float(confidence),
        "comparisons": results,
    }


def stream_progress(*, records: int, batches: int, started: float) -> Dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - started)
    rate = float(records) / elapsed if elapsed > 0 else 0.0
    return {
        "records_processed": int(records),
        "batches_processed": int(batches),
        "elapsed_seconds": elapsed,
        "records_per_second": rate,
    }


def resolve_frozen_rq1_final_state(
    project_root: Union[str, Path],
    *,
    durable_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Resolve the durable RQ1 final + test contracts. Does not load G_test audio."""
    from src.rq1_runtime_paths import resolve_rq1_runtime_paths
    from src.rq2_u_clean import verify_rq1_contract_hash

    runtime = resolve_rq1_runtime_paths(
        project_root=project_root,
        durable_root=durable_root,
    )
    latest_path = runtime.durable_state_root / RQ1_LATEST_UNLOCKED
    if not latest_path.is_file():
        raise EvaluationError("RQ1 LATEST_UNLOCKED.json is missing; cannot resolve the durable final contract")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    contract_hash = str(latest.get("final_contract_hash") or "").strip().lower()
    if not is_sha256(contract_hash):
        raise EvaluationError("LATEST_UNLOCKED.final_contract_hash is missing or invalid")
    state_dir = runtime.state_dir(contract_hash)
    final_path = state_dir / RQ1_FINAL_CONTRACT_FILENAME
    test_path = state_dir / RQ1_TEST_CONTRACT_FILENAME
    summary_path = state_dir / RQ1_FINAL_SUMMARY_FILENAME
    if not final_path.is_file():
        raise EvaluationError(f"RQ1 final contract is missing: {final_path.name}")
    if not test_path.is_file():
        raise EvaluationError("RQ1 test contract is missing from the durable final state")
    final_contract = json.loads(final_path.read_text(encoding="utf-8"))
    test_contract = json.loads(test_path.read_text(encoding="utf-8"))
    if not isinstance(final_contract, dict) or not isinstance(test_contract, dict):
        raise EvaluationError("RQ1 durable contracts must be JSON objects")
    try:
        verify_rq1_contract_hash(final_contract, "rq1_final_contract_hash", contract_hash)
        test_hash = str(final_contract.get("rq1_test_contract_hash") or "").strip().lower()
        verify_rq1_contract_hash(test_contract, "rq1_test_contract_hash", test_hash)
    except Exception as exc:
        raise EvaluationError(f"RQ1 durable contract hash verification failed: {exc}") from exc
    if state_dir.name != f"contract_{contract_hash[:16]}":
        raise EvaluationError("RQ1 durable state dir does not match the final contract hash")
    if not summary_path.is_file():
        raise EvaluationError("RQ1 final summary is missing from the durable final state")
    summary = project_rq1_final_summary(json.loads(summary_path.read_text(encoding="utf-8")))
    if summary["status"] != RQ1_FINAL_STATUS or not summary["ready_rq1_final"]:
        raise EvaluationError(f"RQ1 durable status is {summary['status']!r}, not {RQ1_FINAL_STATUS}")
    if summary["rq1_final_contract_hash"] != contract_hash:
        raise EvaluationError("RQ1 final summary hash does not match the durable final contract")
    bound_test = str(summary.get("rq1_test_contract_hash") or test_hash)
    if bound_test and bound_test != test_hash:
        raise EvaluationError("RQ1 final summary test-contract hash does not match the durable pair")
    return {
        "runtime": runtime,
        "state_dir": state_dir,
        "final_contract": final_contract,
        "test_contract": test_contract,
        "summary": summary,
        "rq1_final_contract_hash": contract_hash,
        "rq1_test_contract_hash": test_hash,
        "latest": latest,
    }


def _best_checkpoint_records(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Flatten either one checkpoint record or an arm×seed map."""
    if not isinstance(payload, Mapping) or not payload:
        return []
    if "checkpoint_fingerprint" in payload or "checkpoint_fingerprint_sha256" in payload or "g_test_used" in payload:
        return [payload]
    records = []
    for value in payload.values():
        if isinstance(value, Mapping) and (
            "checkpoint_fingerprint" in value or "checkpoint_fingerprint_sha256" in value
        ):
            records.append(value)
    return records


_BOOTSTRAP_COMPARE_FIELDS = (
    "observed_sacrebleu_delta",
    "sacrebleu_ci_lower",
    "sacrebleu_ci_upper",
    "observed_chrfpp_delta",
    "chrfpp_ci_lower",
    "chrfpp_ci_upper",
)


def _best_record_ready(record: Mapping[str, Any], *, expected_arm: Optional[str] = None) -> bool:
    """Arm-aware best-checkpoint readiness.

    D0 is the frozen RQ1 checkpoint. NB14 does not select it with an NB14
    validation metric, so ``validation_metric=None`` and ``global_step=None``
    are valid for D0. Random and Quality still require a completed validation
    selection.
    """
    if not isinstance(record, Mapping) or not record:
        return False
    labeled = str(record.get("arm") or "")
    if expected_arm is not None and labeled not in ("", expected_arm):
        return False
    arm = expected_arm or labeled
    if labeled and expected_arm and labeled != expected_arm:
        return False
    fingerprint = str(record.get("checkpoint_fingerprint") or record.get("checkpoint_fingerprint_sha256") or "")
    if record.get("g_test_used") is not False or not is_sha256(fingerprint):
        return False
    if arm == ARM_D0:
        if labeled not in ("", ARM_D0):
            return False
        if not is_sha256(record.get("model_state_sha256")):
            return False
        contract_hash = str(record.get("direct_training_contract_hash") or record.get("arm_training_contract_sha256") or "")
        return is_sha256(contract_hash)
    if arm in (ARM_RANDOM, ARM_QUALITY):
        if labeled not in ("", arm):
            return False
        if record.get("active_seed") is None:
            return False
        try:
            int(record["active_seed"])
            int(record["global_step"])
            metric = float(record["validation_metric"])
        except (TypeError, ValueError, KeyError):
            return False
        if not math.isfinite(metric):
            return False
        manifest = str(record.get("selected_manifest_sha256") or record.get("data_manifest_sha256") or "")
        return bool(
            is_sha256(record.get("arm_training_contract_sha256"))
            and is_sha256(manifest)
            and str(record.get("nb13_generation_id") or "")
            and is_sha256(record.get("d0_init_model_state_sha256"))
        )
    if "validation_metric" in record and record.get("validation_metric") is None:
        return False
    return True


def assert_recorded_best_fingerprint(contract: Mapping[str, Any], arm: str, fingerprint: str) -> None:
    """Fail closed when a published best-checkpoint fingerprint is missing or wrong."""
    expected = str(contract.get(f"{arm}_best_checkpoint_fingerprint") or "")
    if not is_sha256(expected):
        raise EvaluationError(f"final contract missing {arm} best checkpoint fingerprint")
    if expected != str(fingerprint):
        raise EvaluationError(f"{arm} best checkpoint fingerprint does not match the final contract")


def prove_trainable_arm_ready_for_g_test(
    *,
    arm: str,
    layout: Mapping[str, Path],
    contract: Mapping[str, Any],
    best: Mapping[str, Any],
    d0_identity: Mapping[str, Any],
    nb13_selection_contract_sha256: str = "",
) -> Dict[str, Any]:
    """Independently re-prove one trainable arm. Missing ownership fails closed."""
    from src.rq2_final_train import (
        checkpoint_model_state_sha256,
        read_training_complete_proof,
    )

    if arm not in (ARM_RANDOM, ARM_QUALITY):
        raise EvaluationError(f"G_test proof does not apply to arm {arm!r}")
    complete_path = layout.get("training_complete")
    if not complete_path or not Path(complete_path).is_file():
        raise GTestFirewallError(f"{arm} training_complete proof is missing")
    expected_contract = str(contract.get("arm_training_contract_sha256") or "")
    if not is_sha256(expected_contract):
        raise GTestFirewallError(f"{arm} training contract hash is missing")
    if contract.get("active_seed") is not None or best.get("active_seed") is not None:
        try:
            if int(contract.get("active_seed")) != int(best.get("active_seed")):
                raise GTestFirewallError(f"{arm} active_seed does not match the training contract")
        except GTestFirewallError:
            raise
        except (TypeError, ValueError) as exc:
            raise GTestFirewallError(f"{arm} active_seed is invalid") from exc
    complete = read_training_complete_proof(
        layout, arm=arm, expected_contract_hash=expected_contract,
    )
    if str(complete.get("arm") or "") != arm:
        raise GTestFirewallError(f"{arm} training_complete arm identity is wrong")
    if str(complete.get("training_contract_sha256") or "") != expected_contract:
        raise GTestFirewallError(f"{arm} training contract hash does not match the frozen contract")
    data_hash = str(contract.get("data_manifest_sha256") or contract.get("nb13_selection_contract_sha256") or "")
    if nb13_selection_contract_sha256 and str(contract.get("nb13_selection_contract_sha256") or "") != nb13_selection_contract_sha256:
        raise GTestFirewallError(f"{arm} data/selection contract does not match frozen NB13")
    d0_fp = str(d0_identity.get("model_state_sha256") or "")
    if not is_sha256(d0_fp):
        raise GTestFirewallError("frozen D0 initialization fingerprint is empty")
    bound_d0 = str(contract.get("d0_init_model_state_sha256") or "")
    if not is_sha256(bound_d0):
        raise GTestFirewallError(f"{arm} D0 initialization fingerprint is empty")
    if bound_d0 != d0_fp:
        raise GTestFirewallError(f"{arm} D0 initialization fingerprint does not match frozen D0")
    ckpt_fp = str(d0_identity.get("checkpoint_fingerprint_sha256") or "")
    bound_ckpt = str(contract.get("d0_checkpoint_fingerprint_sha256") or ckpt_fp)
    if bound_ckpt and is_sha256(ckpt_fp) and bound_ckpt != ckpt_fp:
        raise GTestFirewallError(f"{arm} D0 checkpoint fingerprint does not match frozen D0")
    best_name = str(best.get("checkpoint_name") or complete.get("best_checkpoint") or "")
    if not best_name:
        raise GTestFirewallError(f"{arm} best checkpoint name is missing")
    ckpt = Path(layout["checkpoints"]) / best_name
    if not ckpt.is_dir():
        raise GTestFirewallError(f"{arm} best checkpoint does not exist")
    from src.rq2_final_contract import assert_checkpoint_arm_isolation

    ownership = assert_checkpoint_arm_isolation(
        ckpt,
        arm=arm,
        expected_contract_hash=expected_contract,
        expected_data_hash=str(contract.get("data_manifest_sha256") or data_hash),
    )
    for key in ("arm", "arm_training_contract_sha256", "data_manifest_sha256", "experiment_fingerprint_sha256"):
        value = ownership.get(key)
        if value in (None, ""):
            raise GTestFirewallError(f"{arm} checkpoint ownership field {key} is missing")
        if key.endswith("sha256") and not is_sha256(value):
            raise GTestFirewallError(f"{arm} checkpoint ownership field {key} is empty or invalid")
    live_fp = checkpoint_model_state_sha256(ckpt)
    recorded_fp = str(best.get("checkpoint_fingerprint") or best.get("checkpoint_fingerprint_sha256") or "")
    if not recorded_fp:
        raise GTestFirewallError(f"{arm} recorded checkpoint fingerprint is empty")
    if live_fp != recorded_fp:
        raise GTestFirewallError(f"{arm} model-state fingerprint does not match the recorded fingerprint")
    if best.get("g_test_used") is not False:
        raise GTestFirewallError(f"{arm} best checkpoint does not prove G_test was unused")
    metric = best.get("validation_metric")
    if metric is None:
        raise GTestFirewallError(f"{arm} required validation metric is missing")
    try:
        numeric = float(metric)
    except (TypeError, ValueError) as exc:
        raise GTestFirewallError(f"{arm} validation metric is not numeric") from exc
    if not math.isfinite(numeric):
        raise GTestFirewallError(f"{arm} validation metric is not finite")
    return {
        "arm": arm,
        "training_complete": complete,
        "checkpoint": str(ckpt),
        "checkpoint_fingerprint": live_fp,
        "ownership": ownership,
    }


def prove_trainable_arms_ready_for_g_test(
    *,
    flags: Nb14Flags,
    layouts: Mapping[str, Mapping[str, Path]],
    contracts: Mapping[str, Mapping[str, Any]],
    best: Mapping[str, Mapping[str, Any]],
    d0_identity: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> Dict[str, Any]:
    if not layouts or not contracts or not best:
        raise GTestFirewallError("G_test unlock requires independent D-Random and D-Quality proofs")
    proofs = {}
    nb13 = str(upstream.get("nb13_selection_contract_sha256") or "")
    for arm in (ARM_RANDOM, ARM_QUALITY):
        layout_entry = layouts.get(arm) or {}
        contract_entry = contracts.get(arm) or {}
        payload = best.get(arm) or {}
        records = _best_checkpoint_records(payload)
        seed_map = "checkpoint_fingerprint" not in payload and "checkpoint_fingerprint_sha256" not in payload and "g_test_used" not in payload and bool(records)
        if not seed_map:
            contract = contract_entry.get("contract") or contract_entry
            proofs[arm] = prove_trainable_arm_ready_for_g_test(
                arm=arm,
                layout=layout_entry,
                contract=contract,
                best=payload if "checkpoint_fingerprint" in payload else (records[0] if records else payload),
                d0_identity=d0_identity,
                nb13_selection_contract_sha256=nb13,
            )
            continue
        arm_proofs = {}
        for record in records:
            seed = int(record["active_seed"])
            layout = layout_entry.get(seed) or layout_entry.get(str(seed)) or {}
            contract = (
                (contract_entry.get("by_seed") or {}).get(seed)
                or (contract_entry.get("by_seed") or {}).get(str(seed))
                or contract_entry.get("contract")
                or contract_entry
            )
            if isinstance(contract, Mapping) and "contract" in contract and "arm_training_contract_sha256" not in contract:
                contract = contract["contract"]
            arm_proofs[seed] = prove_trainable_arm_ready_for_g_test(
                arm=arm,
                layout=layout,
                contract=contract,
                best=record,
                d0_identity=d0_identity,
                nb13_selection_contract_sha256=nb13,
            )
        proofs[arm] = arm_proofs
    return proofs


def assert_declared_seed_checkpoints_ready(
    best: Mapping[str, Any],
    *,
    seed_policy: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Require every declared seed to have a frozen Random and Quality checkpoint.

    Partial multi-seed completion stays locked. This does not rank seeds by G_test.
    """
    from src.rq2_final_contract import normalize_seed_policy

    policy = normalize_seed_policy(seed_policy)
    if not policy.get("configured"):
        raise GTestFirewallError("G_test seed gate requires an explicit seed policy")
    declared = [int(seed) for seed in policy["seed_policy_seeds"]]
    pairs = []
    by_arm: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for arm in (ARM_RANDOM, ARM_QUALITY):
        records = _best_checkpoint_records(best.get(arm) or {})
        found: Dict[int, Mapping[str, Any]] = {}
        for record in records:
            if record.get("active_seed") is None:
                raise GTestFirewallError(f"{arm} best checkpoint is missing active_seed")
            seed = int(record["active_seed"])
            if seed in found:
                raise GTestFirewallError(f"{arm} has a duplicate checkpoint for seed {seed}")
            fingerprint = str(record.get("checkpoint_fingerprint") or "")
            if not is_sha256(fingerprint):
                raise GTestFirewallError(f"{arm} seed {seed} checkpoint fingerprint is missing")
            if record.get("g_test_used") is not False:
                raise GTestFirewallError(f"{arm} seed {seed} checkpoint does not prove G_test was unused")
            if str(record.get("arm") or arm) != arm:
                raise GTestFirewallError(f"{arm} seed {seed} checkpoint belongs to another arm")
            found[seed] = record
        missing = [seed for seed in declared if seed not in found]
        extra = sorted(set(found) - set(declared))
        if missing or extra:
            raise GTestFirewallError(
                f"{arm} checkpoints do not match declared seeds; missing={missing} extra={extra}"
            )
        by_arm[arm] = found
    for seed in declared:
        random_row = by_arm[ARM_RANDOM][seed]
        quality_row = by_arm[ARM_QUALITY][seed]
        pairs.append({
            "active_seed": seed,
            "pairing": f"{ARM_RANDOM}:{seed}|{ARM_QUALITY}:{seed}",
            ARM_RANDOM: {
                "active_seed": seed,
                "checkpoint_fingerprint": random_row["checkpoint_fingerprint"],
                "arm_training_contract_sha256": random_row.get("arm_training_contract_sha256"),
            },
            ARM_QUALITY: {
                "active_seed": seed,
                "checkpoint_fingerprint": quality_row["checkpoint_fingerprint"],
                "arm_training_contract_sha256": quality_row.get("arm_training_contract_sha256"),
            },
        })
    return pairs


def plan_paired_seed_evaluations(
    best: Mapping[str, Any],
    *,
    seed_policy: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Evaluation jobs paired on the same seed. Does not choose a best seed."""
    return assert_declared_seed_checkpoints_ready(best, seed_policy=seed_policy)


def require_g_test_unlocked(
    flags: Nb14Flags,
    readiness: Mapping[str, Any],
    *,
    layouts: Optional[Mapping[str, Mapping[str, Path]]] = None,
    contracts: Optional[Mapping[str, Mapping[str, Any]]] = None,
    best: Optional[Mapping[str, Mapping[str, Any]]] = None,
    d0_identity: Optional[Mapping[str, Any]] = None,
    upstream: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    random_best = (best or {}).get(ARM_RANDOM) or {}
    if isinstance(random_best, Mapping) and "checkpoint_fingerprint" not in random_best and "g_test_used" not in random_best:
        seed_policy = ((contracts or {}).get(ARM_RANDOM) or {}).get("seed_policy") or (upstream or {}).get("seed_policy") or {}
        if not seed_policy:
            raise GTestFirewallError("multi-seed G_test unlock requires the declared seed policy")
        assert_declared_seed_checkpoints_ready(best or {}, seed_policy=seed_policy)
    proofs = prove_trainable_arms_ready_for_g_test(
        flags=flags,
        layouts=layouts or {},
        contracts=contracts or {},
        best=best or {},
        d0_identity=d0_identity or (upstream or {}).get("d0") or {},
        upstream=upstream or {},
    )
    unlocked = unlock_g_test(flags, readiness)
    unlocked["arm_proofs"] = proofs
    return unlocked


def load_frozen_test_frame(
    *,
    project_root: Union[str, Path],
    flags: Nb14Flags,
    readiness: Mapping[str, Any],
    test_contract: Optional[Mapping[str, Any]] = None,
    layouts: Optional[Mapping[str, Mapping[str, Path]]] = None,
    contracts: Optional[Mapping[str, Mapping[str, Any]]] = None,
    best: Optional[Mapping[str, Mapping[str, Any]]] = None,
    d0_identity: Optional[Mapping[str, Any]] = None,
    upstream: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    require_g_test_unlocked(
        flags,
        readiness,
        layouts=layouts,
        contracts=contracts,
        best=best,
        d0_identity=d0_identity,
        upstream=upstream,
    )
    root = Path(project_root)
    durable = resolve_frozen_rq1_final_state(root, durable_root=flags.durable_root or None)
    path = root / "data" / "manifests" / "rq1_test.csv"
    assert_g_test_blocked(path, allow=True)
    contract = dict(test_contract or durable["test_contract"])
    if str(contract.get("rq1_test_contract_hash") or "") != durable["rq1_test_contract_hash"]:
        raise EvaluationError("caller test contract does not match the durable RQ1 test contract")
    frame = pd.read_csv(path)
    try:
        assert_rq1_test_contract(frame, contract, manifest_path=path)
        assert_locked_manifest_bundle_matches_contract(project_root=root, test_contract=contract)
    except Exception as exc:
        raise EvaluationError(f"G_test identity failed: {exc}") from exc
    frame["text_vi_norm"] = frame["text_vi"].map(normalize_mt_text_v1)
    return frame


def run_g_test_inference(
    *,
    arm: str,
    test_frame: pd.DataFrame,
    checkpoint_fingerprint: str,
    generate_fn: Callable[[pd.DataFrame], Sequence[Mapping[str, Any]]],
) -> pd.DataFrame:
    if arm not in ARMS:
        raise EvaluationError(f"cannot infer unknown arm {arm!r}")
    if not is_sha256(checkpoint_fingerprint):
        raise EvaluationError("inference requires a sha256 checkpoint fingerprint")
    raw_rows = list(generate_fn(test_frame))
    expected = test_frame["record_uid"].astype(str).tolist()
    if len(expected) != len(set(expected)):
        raise EvaluationError("frozen G_test contains duplicate record_uid values")
    refs = canonical_references(test_frame)
    groups = test_frame["group_id"].astype(str).tolist()
    seen: Dict[str, Mapping[str, Any]] = {}
    extras = []
    for row in raw_rows:
        uid = str(row.get("record_uid") or "")
        if not uid:
            raise EvaluationError(f"{arm} inference returned a row without record_uid")
        if uid in seen:
            raise EvaluationError(f"{arm} inference duplicate UID: {uid}")
        if uid not in set(expected):
            extras.append(uid)
        seen[uid] = row
    if extras:
        raise EvaluationError(f"{arm} inference extra UIDs: {extras[:5]}")
    missing = [uid for uid in expected if uid not in seen]
    if missing:
        raise EvaluationError(f"{arm} inference missing UIDs: {missing[:5]}")
    if len(seen) != len(expected):
        raise EvaluationError(f"{arm} inference UID count drifted from frozen G_test")
    rows = []
    for uid, ref, group in zip(expected, refs, groups):
        item = seen[uid]
        hyp = item.get("hypothesis_vi")
        if hyp is None and "prediction_vi" in item:
            hyp = item.get("prediction_vi")
        if hyp is None and "d0_pred_vi" in item:
            hyp = item.get("d0_pred_vi")
        if hyp is None:
            raise EvaluationError(f"{arm} inference missing hypothesis for {uid}")
        if not isinstance(hyp, str):
            raise EvaluationError(f"{arm} inference hypothesis for {uid} is not a string")
        supplied_group = item.get("group_id")
        if supplied_group is not None and str(supplied_group) != str(group):
            raise EvaluationError(f"{arm} inference group_id for {uid} does not match frozen G_test")
        supplied_ref = item.get("reference_vi")
        if supplied_ref is not None and normalize_mt_text_v1(supplied_ref) != ref:
            raise EvaluationError(f"{arm} inference reference_vi for {uid} does not match frozen G_test")
        supplied_arm = item.get("arm")
        if supplied_arm is not None and str(supplied_arm) != arm:
            raise EvaluationError(f"{arm} inference arm label does not match the caller")
        rows.append(prediction_row(
            record_uid=uid,
            group_id=str(group),
            hypothesis_vi=hyp,
            reference_vi=ref,
            arm=arm,
            checkpoint_fingerprint=checkpoint_fingerprint,
        ))
    frame = pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))
    if frame["record_uid"].astype(str).tolist() != expected:
        raise EvaluationError(f"{arm} inference UID order drifted from frozen G_test")
    if frame["reference_vi"].astype(str).tolist() != refs:
        raise EvaluationError(f"{arm} inference references drifted from frozen G_test")
    if frame["group_id"].astype(str).tolist() != groups:
        raise EvaluationError(f"{arm} inference group_id drifted from frozen G_test")
    return frame


def generate_from_direct_checkpoint(
    *,
    checkpoint_dir: Union[str, Path],
    training_contract: Mapping[str, Any],
    audio_cache_roots: Sequence[Union[str, Path]] = (),
    processors: Optional[tuple] = None,
    model: Any = None,
) -> Callable[[pd.DataFrame], List[Dict[str, Any]]]:
    """RQ1 Direct inference bound to a frozen best checkpoint. Does not train."""

    def _generate(test_frame: pd.DataFrame) -> List[Dict[str, Any]]:
        from transformers import SpeechEncoderDecoderModel

        from src.direct_dataset import DirectDataCollator, DirectSpeechTranslationDataset
        from src.direct_full_train import generate_direct_predictions
        from src.direct_model import load_direct_processors

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
        loaded = model if model is not None else SpeechEncoderDecoderModel.from_pretrained(str(checkpoint_dir))
        dataset = DirectSpeechTranslationDataset(
            test_frame,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            audio_cache_roots=list(audio_cache_roots),
            sample_rate=int(training_contract.get("sample_rate") or 16000),
            max_target_length=int(training_contract.get("max_target_length") or 256),
        )
        collator = DirectDataCollator(feature_extractor, tokenizer)
        forced_bos = int(tokenizer.lang_code_to_id[str(training_contract["target_lang"])])
        return generate_direct_predictions(
            model=loaded,
            tokenizer=tokenizer,
            dataset=dataset,
            collator=collator,
            batch_size=int(training_contract.get("per_device_eval_batch_size") or 1),
            generation_max_length=int(training_contract.get("generation_max_length") or 256),
            num_beams=int(training_contract.get("num_beams") or 4),
            forced_bos_token_id=forced_bos,
        )

    return _generate


def derive_final_readiness(
    *,
    flags: Nb14Flags,
    upstream: Optional[Mapping[str, Any]] = None,
    layouts: Optional[Mapping[str, Mapping[str, Path]]] = None,
    contracts: Optional[Mapping[str, Mapping[str, Any]]] = None,
    best: Optional[Mapping[str, Mapping[str, Any]]] = None,
    protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    layouts = layouts or {}
    contracts = contracts or {}
    best = best or {}
    gates: Dict[str, Dict[str, Any]] = {}

    def gate(name: str, ok: bool, **evidence: Any) -> None:
        gates[name] = {"ok": bool(ok), **evidence}

    gate("upstream_frozen", bool(upstream and upstream.get("nb13_generation_id") and upstream.get("nb13_selection_contract_sha256")),
         nb13_generation_id=str((upstream or {}).get("nb13_generation_id") or ""))
    d0 = (upstream or {}).get("d0") or {}
    gate("d0_bound", bool(d0.get("checkpoint_fingerprint_sha256") and d0.get("direct_training_contract_hash")),
         checkpoint_fingerprint=str(d0.get("checkpoint_fingerprint_sha256") or ""))
    def _training_complete(layout: Mapping[str, Any], contract: Mapping[str, Any], *, expected_seed: Any = None) -> bool:
        complete_path = layout.get("training_complete") if isinstance(layout, Mapping) else None
        payload: Dict[str, Any] = {}
        if complete_path and Path(complete_path).is_file():
            payload = json.loads(Path(complete_path).read_text(encoding="utf-8"))
        ready = (
            payload.get("status") == STATUS_TRAINING_COMPLETE
            and str(payload.get("arm") or "") == arm
            and str(payload.get("training_contract_sha256") or "") == str(contract.get("arm_training_contract_sha256") or "")
            and bool(contract.get("arm_training_contract_sha256"))
            and is_sha256(payload.get("terminal_checkpoint_fingerprint"))
        )
        if not ready or expected_seed is None:
            return ready
        try:
            seed_ok = int(payload.get("active_seed")) == int(expected_seed)
        except (TypeError, ValueError):
            return False
        return (
            seed_ok
            and payload.get("g_test_used") is False
            and is_sha256(payload.get("best_checkpoint_fingerprint"))
            and payload.get("validation_metric") is not None
            and payload.get("global_step") is not None
            and bool(payload.get("nb13_generation_id"))
            and is_sha256(payload.get("d0_init_model_state_sha256"))
            and is_sha256(payload.get("selected_manifest_sha256"))
        )

    def _seed_contract_identity(
        contract: Mapping[str, Any],
        fingerprint: Mapping[str, Any],
        *,
        arm_name: str,
        seed: Any,
        upstream_payload: Optional[Mapping[str, Any]],
    ) -> bool:
        if str(contract.get("arm") or "") != arm_name:
            return False
        try:
            if int(contract.get("active_seed")) != int(seed):
                return False
        except (TypeError, ValueError):
            return False
        if not is_sha256(contract.get("arm_training_contract_sha256")):
            return False
        if not upstream_payload:
            return True
        expected_nb13 = str(upstream_payload.get("resolved_nb13_generation_id") or upstream_payload.get("nb13_generation_id") or "")
        nb13 = str(contract.get("nb13_generation_id") or fingerprint.get("nb13_generation_id") or "")
        if not expected_nb13 or nb13 != expected_nb13:
            return False
        d0_fp = str((upstream_payload.get("d0") or {}).get("model_state_sha256") or "")
        bound = str(contract.get("d0_init_model_state_sha256") or fingerprint.get("d0_init_model_state_sha256") or "")
        if not is_sha256(d0_fp) or bound != d0_fp:
            return False
        manifest_key = "d_random_manifest_sha256" if arm_name == ARM_RANDOM else "d_quality_manifest_sha256"
        expected_manifest = str(upstream_payload.get(manifest_key) or "")
        selected = str(fingerprint.get("selected_manifest_sha256") or contract.get("selected_manifest_sha256") or "")
        return bool(is_sha256(expected_manifest) and selected == expected_manifest)

    for arm, key in ((ARM_RANDOM, "d_random_complete"), (ARM_QUALITY, "d_quality_complete")):
        layout = layouts.get(arm) or {}
        contract_entry = contracts.get(arm) or {}
        contract = contract_entry.get("contract") or contract_entry
        seed_layouts = (
            isinstance(layout, Mapping)
            and layout
            and "training_complete" not in layout
            and "checkpoints" not in layout
        )
        if seed_layouts:
            by_seed = contract_entry.get("by_seed") or {}
            ok = bool(by_seed) and set(by_seed) == set(layout)
            for seed, seed_layout in layout.items():
                seed_run = by_seed.get(seed) or by_seed.get(str(seed)) or {}
                seed_contract = seed_run.get("contract") or seed_run
                if isinstance(seed_contract, Mapping) and "contract" in seed_contract and "arm_training_contract_sha256" not in seed_contract:
                    seed_contract = seed_contract["contract"]
                fingerprint = seed_run.get("fingerprint") if isinstance(seed_run, Mapping) else {}
                ok = ok and _training_complete(seed_layout, seed_contract if isinstance(seed_contract, Mapping) else {}, expected_seed=seed)
                ok = ok and _seed_contract_identity(
                    seed_contract if isinstance(seed_contract, Mapping) else {},
                    fingerprint if isinstance(fingerprint, Mapping) else {},
                    arm_name=arm,
                    seed=seed,
                    upstream_payload=upstream,
                )
        else:
            ok = _training_complete(layout, contract if isinstance(contract, Mapping) else {})
        gate(
            key,
            ok,
            contract_sha256=str(contract.get("arm_training_contract_sha256") or "") if isinstance(contract, Mapping) else "",
            n_seed_runs=len(layout) if seed_layouts else 1,
        )
    frozen_best = True
    for arm in ARMS:
        payload = best.get(arm) or {}
        layout = layouts.get(arm) or {}
        path = layout.get("best_checkpoint")
        disk = {}
        if path and Path(path).is_file():
            disk = json.loads(Path(path).read_text(encoding="utf-8"))
            payload = payload or disk
        records = _best_checkpoint_records(payload)
        ok = bool(records) and all(_best_record_ready(record, expected_arm=arm) for record in records)
        if arm != ARM_D0:
            frozen_best = frozen_best and ok
        gate(
            f"{arm}_best_checkpoint_frozen",
            ok,
            checkpoint_fingerprint=str((records[0].get("checkpoint_fingerprint") if records else "") or ""),
            n_seed_runs=len(records),
        )
    gate("best_checkpoints_frozen", frozen_best and gates.get("d0_best_checkpoint_frozen", {}).get("ok") is True)
    gate("validation_selection_complete", gates["best_checkpoints_frozen"]["ok"])
    train_ok = True
    for arm in (ARM_RANDOM, ARM_QUALITY):
        entry = contracts.get(arm) or {}
        by_seed = entry.get("by_seed") or {}
        if by_seed:
            found_contracts = []
            for run in by_seed.values():
                seed_contract = run.get("contract") if isinstance(run, Mapping) and "arm_training_contract_sha256" not in run else run
                found_contracts.append(seed_contract if isinstance(seed_contract, Mapping) else {})
            train_ok = train_ok and bool(found_contracts) and all(
                is_sha256(item.get("arm_training_contract_sha256")) for item in found_contracts
            )
        else:
            contract = entry.get("contract") or entry
            train_ok = train_ok and bool(isinstance(contract, Mapping) and is_sha256(contract.get("arm_training_contract_sha256")))
    gate("training_contracts_verified", train_ok)
    gate("evaluation_protocol_frozen", bool(protocol and protocol.get("evaluation_protocol_sha256")))
    gate("bootstrap_protocol_frozen", bool(protocol and protocol.get("bootstrap_method") == BOOTSTRAP_METHOD_PAIRED_CLUSTER))
    gate("no_pending_scientific_decisions", flags.rq2_final_frozen is True)
    gate("rq2_final_frozen", flags.rq2_final_frozen is True)
    flat = {name: gates.get(name, {}).get("ok") is True for name in UNLOCK_REQUIRED}
    return {"gates": gates, "flat": flat, "all_ok": all(flat.values())}


# Scientific hash exclusions. These are not part of the NB14 result that
# artifact_hashes.json locks:
# - artifact_hashes.json: the manifest must not hash itself
# - COMPLETE.json: written by finalize_generation after the scientific hash
# - artifact_manifest.json: lifecycle seal written by finalize_generation
# - CURRENT: publication pointer, stored beside generations, not inside one
# - *.partial path components: staging directories
# - filenames ending in .tmp or .temp: temporary files
SCIENTIFIC_HASH_EXCLUDED_BASENAMES = frozenset({
    "artifact_hashes.json",
    "COMPLETE.json",
    "artifact_manifest.json",
    "CURRENT",
})


def iter_scientific_artifact_files(directory: Union[str, Path]) -> List[str]:
    """Relative scientific artifact paths, sorted, excluding the hash manifest and lifecycle files."""
    root = Path(directory)
    if not root.is_dir():
        return []
    found: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        if any(part in SCIENTIFIC_HASH_EXCLUDED_BASENAMES or part.endswith(".partial") for part in parts):
            continue
        if path.name.endswith(".tmp") or path.name.endswith(".temp"):
            continue
        found.append(relative.as_posix())
    return sorted(found)


def hash_artifact_files(directory: Union[str, Path]) -> Dict[str, str]:
    """SHA-256 every scientific file under a generation, including dynamic per-seed artifacts."""
    root = Path(directory)
    return {rel: sha256_file(root / rel) for rel in iter_scientific_artifact_files(root)}


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvaluationError(f"JSON object required: {path.name}")
    return payload


def _recorded_sha(entry: Any) -> str:
    if isinstance(entry, Mapping):
        return str(entry.get("sha256") or "")
    return str(entry or "")


def _is_seed_map(payload: Mapping[str, Any]) -> bool:
    if not isinstance(payload, Mapping) or not payload:
        return False
    if "checkpoint_fingerprint" in payload or "checkpoint_fingerprint_sha256" in payload or "g_test_used" in payload:
        return False
    return bool(_best_checkpoint_records(payload))


def _best_record_for_seed(payload: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    if not _is_seed_map(payload):
        if payload.get("active_seed") is not None and int(payload.get("active_seed")) != int(seed):
            raise EvaluationError(f"best checkpoint active_seed does not match evaluation seed {seed}")
        return payload
    for key in (seed, str(seed)):
        record = payload.get(key)
        if isinstance(record, Mapping) and (
            "checkpoint_fingerprint" in record or "checkpoint_fingerprint_sha256" in record
        ):
            return record
    for record in _best_checkpoint_records(payload):
        if int(record.get("active_seed")) == int(seed):
            return record
    raise EvaluationError(f"best checkpoint for seed {seed} is missing")


def _seed_runs_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        fingerprint = str(row.get("best_checkpoint_fingerprint") or row.get("checkpoint_fingerprint") or "")
        if row.get("active_seed") is None or not is_sha256(fingerprint):
            return False
        if row.get("g_test_used") is not False:
            return False
    return True


def _seed_run_record(arm: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    fingerprint = str(record.get("checkpoint_fingerprint") or record.get("checkpoint_fingerprint_sha256") or "")
    return {
        "active_seed": int(record["active_seed"]),
        "arm": arm,
        "g_test_used": False,
        "checkpoint_name": record.get("checkpoint_name"),
        "checkpoint_fingerprint": fingerprint,
        "best_checkpoint": record.get("checkpoint_name"),
        "best_checkpoint_fingerprint": fingerprint,
        "arm_training_contract_sha256": record.get("arm_training_contract_sha256"),
        "training_contract_sha256": record.get("arm_training_contract_sha256"),
        "nb13_generation_id": record.get("nb13_generation_id"),
        "d0_init_model_state_sha256": record.get("d0_init_model_state_sha256"),
        "selected_manifest_sha256": record.get("selected_manifest_sha256") or record.get("data_manifest_sha256"),
        "validation_metric": record.get("validation_metric"),
        "validation_metric_value": record.get("validation_metric"),
        "global_step": record.get("global_step"),
    }


def _prediction_identity(frame: pd.DataFrame) -> Dict[str, Any]:
    uids = frame["record_uid"].astype(str).tolist()
    refs = frame["reference_vi"].astype(str).tolist()
    if len(uids) != len(set(uids)):
        raise EvaluationError("prediction rows contain a duplicate record_uid")
    return {
        "record_uids": uids,
        "references": refs,
        "record_uid_sha256": sha256_json(uids),
        "reference_sha256": sha256_json(refs),
        "n_rows": len(uids),
    }


def _persist_per_seed_evaluation(
    root: Path,
    per_seed_frames: Mapping[Any, Mapping[str, pd.DataFrame]],
    *,
    best: Mapping[str, Any],
    n_samples: int,
    seed: int,
    confidence: float,
    seed_policy: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """Write one paired evaluation per declared seed. Does not pool seeds."""
    rows = []
    base_uids = None
    base_refs = None
    extra: Dict[str, Path] = {}
    for active_seed, frames in per_seed_frames.items():
        active_seed = int(active_seed)
        if set(frames) != set(ARMS):
            raise EvaluationError(f"seed {active_seed} evaluation is missing an arm")
        assert_paired_alignment(frames)
        identity = _prediction_identity(frames[ARM_RANDOM])
        quality_identity = _prediction_identity(frames[ARM_QUALITY])
        if identity["record_uids"] != quality_identity["record_uids"]:
            raise EvaluationError(f"seed {active_seed} Random and Quality UID order differ")
        if identity["references"] != quality_identity["references"]:
            raise EvaluationError(f"seed {active_seed} Random and Quality references differ")
        if base_uids is None:
            base_uids = identity["record_uids"]
            base_refs = identity["references"]
        elif identity["record_uids"] != base_uids or identity["references"] != base_refs:
            raise EvaluationError("G_test UID/reference base differs across seeds")
        seed_dir = root / "evaluation" / f"seed-{active_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics_row: Dict[str, Any] = {"active_seed": active_seed}
        for arm in (ARM_RANDOM, ARM_QUALITY):
            rel = f"evaluation/seed-{active_seed}/{arm}.parquet"
            write_predictions_parquet(root / rel, frames[arm].to_dict("records"))
            extra[rel] = root / rel
            owned = _best_record_for_seed(best.get(arm) or {}, active_seed)
            fingerprints = set(frames[arm]["checkpoint_fingerprint"].astype(str).tolist())
            if fingerprints != {str(owned.get("checkpoint_fingerprint") or "")}:
                raise EvaluationError(f"{arm} seed {active_seed} prediction fingerprint does not match the best checkpoint")
            if {str(value) for value in frames[arm]["arm"].tolist()} != {arm}:
                raise EvaluationError(f"{arm} seed {active_seed} prediction arm does not match")
            metric = arm_metrics(frames[arm])
            metrics_row[arm] = {
                "sacrebleu": metric["sacrebleu"],
                "chrfpp": metric["chrfpp"],
                "arm": arm,
                "active_seed": active_seed,
                "checkpoint_name": owned.get("checkpoint_name"),
                "checkpoint_fingerprint": str(owned.get("checkpoint_fingerprint") or ""),
            }
        metrics_row["delta_sacrebleu"] = float(metrics_row[ARM_QUALITY]["sacrebleu"]) - float(metrics_row[ARM_RANDOM]["sacrebleu"])
        metrics_row["delta_chrfpp"] = float(metrics_row[ARM_QUALITY]["chrfpp"]) - float(metrics_row[ARM_RANDOM]["chrfpp"])
        metrics_row["record_uid_sha256"] = identity["record_uid_sha256"]
        metrics_row["reference_sha256"] = identity["reference_sha256"]
        metrics_row["n_rows"] = identity["n_rows"]
        metrics_path = seed_dir / "metrics.json"
        write_json(metrics_path, metrics_row)
        extra[f"evaluation/seed-{active_seed}/metrics.json"] = metrics_path
        bootstrap = paired_cluster_bootstrap_arms(
            frames,
            n_samples=int(n_samples),
            seed=int(seed),
            confidence=float(confidence),
        )
        bootstrap_path = seed_dir / "bootstrap.json"
        write_json(bootstrap_path, bootstrap)
        extra[f"evaluation/seed-{active_seed}/bootstrap.json"] = bootstrap_path
        rows.append(metrics_row)
    from src.rq2_final_contract import aggregate_paired_seed_metrics

    summary = aggregate_paired_seed_metrics(rows, seed_policy=seed_policy)
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = aggregate_dir / "metrics.json"
    write_json(aggregate_path, summary)
    extra["aggregate/metrics.json"] = aggregate_path
    return summary, extra


def persist_final_artifacts(
    directory: Union[str, Path],
    *,
    frames: Mapping[str, pd.DataFrame],
    contract_fields: Mapping[str, Any],
    best: Mapping[str, Any],
    n_samples: int,
    seed: int,
    confidence: float = 0.95,
    per_seed_frames: Optional[Mapping[Any, Mapping[str, pd.DataFrame]]] = None,
) -> Dict[str, Path]:
    """Write a self-consistent NB14 artifact set. Does not publish CURRENT."""
    root = Path(directory)
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    assert_paired_alignment(frames)
    pred_hashes: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    for arm in ARMS:
        pred_hashes[arm] = write_predictions_parquet(
            root / f"predictions/{arm}.parquet",
            frames[arm].to_dict("records"),
        )
        metrics[arm] = arm_metrics(frames[arm])
        if not metrics[arm].get("finite"):
            raise EvaluationError(f"{arm} metrics are not finite")
    bootstrap = paired_cluster_bootstrap_arms(
        frames,
        n_samples=int(n_samples),
        seed=int(seed),
        confidence=float(confidence),
    )
    write_json(root / "arm_metrics.json", metrics)
    write_json(root / "arm_comparison.json", bootstrap["comparisons"])
    write_json(root / "bootstrap_results.json", bootstrap)
    write_json(root / "best_checkpoints.json", dict(best))
    fairness = dict(contract_fields.get("fairness_proof") or {})
    if not fairness:
        raise EvaluationError("fairness_proof is required before publication")
    write_json(root / "fairness_proof.json", fairness)
    extra_paths: Dict[str, Path] = {}
    for arm, rel in ((ARM_RANDOM, "d_random_training_complete.json"), (ARM_QUALITY, "d_quality_training_complete.json")):
        payload = dict(contract_fields.get(f"{arm}_training_complete") or {})
        if payload.get("status") != STATUS_TRAINING_COMPLETE:
            raise EvaluationError(f"{arm} training_complete proof is missing")
        records = _best_checkpoint_records(best.get(arm) or {})
        top = best.get(arm) or {}
        if _is_seed_map(top):
            built = [_seed_run_record(arm, record) for record in records]
            existing = list(payload.get("seed_runs") or [])
            if not _seed_runs_ready(existing):
                payload["seed_runs"] = built
            arm_dir = root / "training" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            for row in payload["seed_runs"]:
                rel_seed = f"training/{arm}/seed-{int(row['active_seed'])}.json"
                write_json(root / rel_seed, row)
                extra_paths[rel_seed] = root / rel_seed
        write_json(root / rel, payload)
    write_json(root / "summary.json", {
        "status": STATUS_SUCCESS,
        "arms": list(ARMS),
        "n_test_rows": int(len(frames[ARM_D0])),
        "rq2_final_contract_sha256": "",
    })
    fields = dict(contract_fields)
    fields["prediction_artifact_sha256"] = pred_hashes
    fields["metrics_artifact_sha256"] = sha256_file(root / "arm_metrics.json")
    fields["comparison_artifact_sha256"] = sha256_file(root / "arm_comparison.json")
    fields["bootstrap_artifact_sha256"] = sha256_file(root / "bootstrap_results.json")
    fields["best_checkpoints_artifact_sha256"] = sha256_file(root / "best_checkpoints.json")
    fields["fairness_proof_sha256"] = sha256_file(root / "fairness_proof.json")
    fields["d_random_training_complete_sha256"] = sha256_file(root / "d_random_training_complete.json")
    fields["d_quality_training_complete_sha256"] = sha256_file(root / "d_quality_training_complete.json")
    if contract_fields.get("gold_pseudo_mix") or contract_fields.get("gold_pseudo_mix_policy"):
        mix = dict(contract_fields.get("gold_pseudo_mix") or {})
        fields["gold_pseudo_mix"] = {
            "policy": mix.get("policy") or contract_fields.get("gold_pseudo_mix_policy"),
            "gold_slots": mix.get("gold_slots", contract_fields.get("gold_slots")),
            "pseudo_slots": mix.get("pseudo_slots", contract_fields.get("pseudo_slots")),
        }
        fields["gold_pseudo_mix_policy"] = fields["gold_pseudo_mix"]["policy"]
        fields["gold_slots"] = fields["gold_pseudo_mix"]["gold_slots"]
        fields["pseudo_slots"] = fields["gold_pseudo_mix"]["pseudo_slots"]
    if per_seed_frames is not None or contract_fields.get("per_seed_results") is not None or contract_fields.get("seed_policy_record"):
        from src.rq2_final_contract import aggregate_paired_seed_metrics

        seed_policy = dict(contract_fields.get("seed_policy_record") or {})
        if contract_fields.get("seed_policy") and "seed_policy" not in seed_policy and "mode" not in seed_policy:
            seed_policy = {
                "seed_policy": contract_fields.get("seed_policy"),
                "seed_policy_seeds": list(contract_fields.get("seed_policy_seeds") or []),
            }
        elif seed_policy.get("mode"):
            seed_policy = {
                "seed_policy": "multi_seed" if seed_policy.get("mode") == "multi" else "compute_constrained_single_seed",
                "seed_policy_seeds": list(seed_policy.get("seeds") or []),
            }
        if per_seed_frames is not None:
            summary, seeded_paths = _persist_per_seed_evaluation(
                root,
                per_seed_frames,
                best=best,
                n_samples=int(n_samples),
                seed=int(seed),
                confidence=float(confidence),
                seed_policy=seed_policy,
            )
            extra_paths.update(seeded_paths)
            fields["seed_run_summary"] = summary
            fields["seed_policy_record"] = summary["seed_policy"]
            per_seed = list(summary["per_seed"])
        else:
            per_seed = list(contract_fields.get("per_seed_results") or [])
            if str(seed_policy.get("seed_policy") or "") == "multi_seed" and not per_seed:
                raise EvaluationError("multi-seed publication requires real per-seed results")
            if per_seed:
                summary = aggregate_paired_seed_metrics(per_seed, seed_policy=seed_policy)
                fields["seed_run_summary"] = summary
                fields["seed_policy_record"] = summary["seed_policy"]
        if per_seed:
            (root / "evaluation").mkdir(parents=True, exist_ok=True)
            (root / "evaluation" / "per_seed").mkdir(parents=True, exist_ok=True)
            summary = fields["seed_run_summary"]
            for row in summary["per_seed"]:
                rel_seed = f"evaluation/per_seed/seed-{row['active_seed']}.json"
                write_json(root / rel_seed, row)
                extra_paths[rel_seed] = root / rel_seed
            rel_agg = "evaluation/aggregate.json"
            write_json(root / rel_agg, summary)
            extra_paths[rel_agg] = root / rel_agg
        fields["seed_policy"] = seed_policy.get("seed_policy")
        fields["seed_policy_seeds"] = list(seed_policy.get("seed_policy_seeds") or [])
    fields["metric_package_versions"] = {
        arm: {
            "sacrebleu_version": metrics[arm].get("sacrebleu_version"),
            "chrf_word_order": 2,
        }
        for arm in ARMS
    }
    contract = build_final_contract(fields)
    write_json(root / "final_contract.json", contract)
    write_json(root / "summary.json", {
        "status": STATUS_SUCCESS,
        "arms": list(ARMS),
        "n_test_rows": int(len(frames[ARM_D0])),
        "rq2_final_contract_sha256": contract["rq2_final_contract_sha256"],
        "seed_policy": contract.get("seed_policy_record"),
        "gold_pseudo_mix": contract.get("gold_pseudo_mix"),
        "seed_run_summary": contract.get("seed_run_summary"),
    })
    write_json(root / "artifact_hashes.json", {"files": hash_artifact_files(root)})
    published = {rel: root / rel for rel in FINAL_ARTIFACT_FILES}
    published.update(extra_paths)
    return published


def verify_published_seed_provenance(contract: Mapping[str, Any], best: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute the published seed aggregate and reject missing, duplicate, or extra seeds."""
    from src.rq2_final_contract import SEED_POLICY_MULTI, SEED_POLICY_SINGLE, aggregate_paired_seed_metrics

    record = dict(contract.get("seed_policy_record") or {})
    mode = str(record.get("mode") or "")
    if mode not in {"single", "multi"}:
        raise EvaluationError("final contract seed policy mode is missing")
    try:
        declared = [int(seed) for seed in list(record.get("seeds") or [])]
    except (TypeError, ValueError) as exc:
        raise EvaluationError("final contract declared seeds are invalid") from exc
    if mode == "single" and len(declared) != 1:
        raise EvaluationError("single-seed publication must declare exactly one seed")
    if mode == "multi" and len(declared) < 2:
        raise EvaluationError("multi-seed publication must declare at least two seeds")
    if len(set(declared)) != len(declared):
        raise EvaluationError("final contract declares a duplicate seed")
    mix = dict(contract.get("gold_pseudo_mix") or {})
    if not str(mix.get("policy") or "").strip() or mix.get("gold_slots") is None or mix.get("pseudo_slots") is None:
        raise EvaluationError("final contract is missing the gold:pseudo mix policy")
    seed_policy = {
        "seed_policy": SEED_POLICY_MULTI if mode == "multi" else SEED_POLICY_SINGLE,
        "seed_policy_seeds": declared,
    }
    try:
        pairs = assert_declared_seed_checkpoints_ready(best, seed_policy=seed_policy)
    except GTestFirewallError as exc:
        raise EvaluationError(str(exc)) from exc
    summary = dict(contract.get("seed_run_summary") or {})
    per_seed = list(summary.get("per_seed") or [])
    if int(summary.get("n_valid_runs") or -1) != len(declared):
        raise EvaluationError("seed summary n_valid_runs does not match declared seeds")
    got = [int(row.get("active_seed")) for row in per_seed]
    if got != declared:
        raise EvaluationError("per-seed results do not match the declared seed order")
    rebuilt = aggregate_paired_seed_metrics(
        [
            {
                "active_seed": int(row["active_seed"]),
                ARM_RANDOM: row[ARM_RANDOM],
                ARM_QUALITY: row[ARM_QUALITY],
            }
            for row in per_seed
        ],
        seed_policy=seed_policy,
    )
    for arm in (ARM_RANDOM, ARM_QUALITY):
        for metric in ("sacrebleu", "chrfpp"):
            left = (summary.get(arm) or {}).get(metric) or {}
            right = rebuilt[arm][metric]
            if int(left.get("n_valid_runs") or -1) != int(right["n_valid_runs"]):
                raise EvaluationError(f"{arm} {metric} n_valid_runs does not recompute")
            if abs(float(left.get("mean")) - float(right["mean"])) > 1e-9:
                raise EvaluationError(f"{arm} {metric} mean does not recompute from per-seed results")
            if left.get("std") is None or right.get("std") is None:
                if left.get("std") != right.get("std"):
                    raise EvaluationError(f"{arm} {metric} std does not recompute")
            elif abs(float(left["std"]) - float(right["std"])) > 1e-9:
                raise EvaluationError(f"{arm} {metric} std does not recompute")
    if mode == "single":
        for arm, key in (
            (ARM_RANDOM, "d_random_best_checkpoint_fingerprint"),
            (ARM_QUALITY, "d_quality_best_checkpoint_fingerprint"),
        ):
            if str(contract.get(key) or "") != str(pairs[0][arm]["checkpoint_fingerprint"]):
                raise EvaluationError(f"{key} does not match the single declared seed")
    return rebuilt


def _assert_recomputed_bootstrap(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Reject a bootstrap JSON that does not recompute, including every CI bound."""
    if recorded.get("bootstrap_method") != recomputed.get("bootstrap_method"):
        raise EvaluationError(f"{label} bootstrap method does not recompute")
    if str(recorded.get("cluster_col") or "") != str(recomputed.get("cluster_col") or ""):
        raise EvaluationError(f"{label} bootstrap cluster_col does not recompute")
    if str(recorded.get("bootstrap_unit") or "") != str(recomputed.get("bootstrap_unit") or ""):
        raise EvaluationError(f"{label} bootstrap unit does not recompute")
    for key in ("n_samples", "seed", "n_clusters", "n_rows"):
        if key not in recorded or recorded.get(key) is None or key not in recomputed or recomputed.get(key) is None:
            raise EvaluationError(f"{label} bootstrap {key} is missing")
        try:
            if int(recorded[key]) != int(recomputed[key]):
                raise EvaluationError(f"{label} bootstrap {key} does not recompute")
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"{label} bootstrap {key} is not an integer") from exc
    try:
        if abs(float(recorded.get("confidence")) - float(recomputed.get("confidence"))) > 1e-9:
            raise EvaluationError(f"{label} bootstrap confidence does not recompute")
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{label} bootstrap confidence is missing") from exc
    persisted = recorded.get("comparisons")
    expected = recomputed.get("comparisons")
    if not isinstance(persisted, Mapping) or not persisted:
        raise EvaluationError(f"{label} bootstrap comparisons are missing")
    if not isinstance(expected, Mapping) or not expected:
        raise EvaluationError(f"{label} recomputed bootstrap comparisons are missing")
    required = comparison_key(ARM_QUALITY, ARM_RANDOM)
    if required not in persisted:
        raise EvaluationError(f"{label} Quality-Random bootstrap comparison is missing")
    if set(persisted) != set(expected):
        raise EvaluationError(f"{label} bootstrap comparisons do not match the recomputed set")
    for key, entry in persisted.items():
        other = expected.get(key) or {}
        if not isinstance(entry, Mapping):
            raise EvaluationError(f"{label} bootstrap {key} comparison is malformed")
        for field in _BOOTSTRAP_COMPARE_FIELDS:
            if field not in entry or field not in other:
                raise EvaluationError(f"{label} bootstrap {key} {field} is missing")
            try:
                left = float(entry[field])
                right = float(other[field])
            except (TypeError, ValueError) as exc:
                raise EvaluationError(f"{label} bootstrap {key} {field} is not finite") from exc
            if not math.isfinite(left) or abs(left - right) > 1e-9:
                raise EvaluationError(f"{label} bootstrap {key} {field} does not recompute")


def _compare_delta_list(stored: Sequence[Any], rebuilt: Sequence[Any], label: str) -> None:
    try:
        left = [float(value) for value in stored]
        right = [float(value) for value in rebuilt]
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{label} is not numeric") from exc
    if len(left) != len(right) or any(abs(item - other) > 1e-9 for item, other in zip(left, right)):
        raise EvaluationError(f"{label} does not recompute")


def _assert_seed_summary_matches(stored: Mapping[str, Any], rebuilt: Mapping[str, Any], *, label: str) -> None:
    if int(stored.get("n_valid_runs") or -1) != int(rebuilt.get("n_valid_runs") or -2):
        raise EvaluationError(f"{label} n_valid_runs does not recompute")
    if "std_convention" in stored and stored.get("std_convention") != rebuilt.get("std_convention"):
        raise EvaluationError(f"{label} std convention does not recompute")
    for arm in (ARM_RANDOM, ARM_QUALITY, "treatment"):
        persisted_arm = stored.get(arm) or {}
        for metric in ("sacrebleu", "chrfpp"):
            if metric in persisted_arm:
                _compare_mean_std(persisted_arm[metric] or {}, rebuilt[arm][metric], f"{label} {arm} {metric}")
    treatment = stored.get("treatment") or {}
    for key in ("per_seed_delta_sacrebleu", "per_seed_delta_chrfpp"):
        if key not in treatment:
            raise EvaluationError(f"{label} {key} is missing")
        _compare_delta_list(treatment[key], rebuilt["treatment"][key], f"{label} {key}")


def _compare_mean_std(stored: Mapping[str, Any], rebuilt: Mapping[str, Any], label: str) -> None:
    if int(stored.get("n_valid_runs") or -1) != int(rebuilt.get("n_valid_runs") or -2):
        raise EvaluationError(f"{label} n_valid_runs does not recompute")
    if abs(float(stored.get("mean")) - float(rebuilt.get("mean"))) > 1e-9:
        raise EvaluationError(f"{label} mean does not recompute from per-seed metrics")
    if stored.get("std") is None or rebuilt.get("std") is None:
        if stored.get("std") != rebuilt.get("std"):
            raise EvaluationError(f"{label} std does not recompute")
    elif abs(float(stored["std"]) - float(rebuilt["std"])) > 1e-9:
        raise EvaluationError(f"{label} std does not recompute")


def _verify_persisted_seed_training(
    gen_dir: Path,
    contract: Mapping[str, Any],
    best: Mapping[str, Any],
    arm: str,
    runs: Sequence[Mapping[str, Any]],
) -> None:
    declared = [int(seed) for seed in (contract.get("seed_policy_record") or {}).get("seeds") or []]
    got = []
    for row in runs:
        if row.get("active_seed") is None:
            raise EvaluationError(f"{arm} training_complete run is missing active_seed")
        seed = int(row["active_seed"])
        if seed in got:
            raise EvaluationError(f"{arm} training_complete duplicates seed {seed}")
        got.append(seed)
        if str(row.get("arm") or "") != arm:
            raise EvaluationError(f"{arm} seed {seed} training_complete belongs to another arm")
        if row.get("g_test_used") is not False:
            raise EvaluationError(f"{arm} seed {seed} training_complete does not prove G_test was unused")
        fingerprint = str(row.get("best_checkpoint_fingerprint") or row.get("checkpoint_fingerprint") or "")
        if not is_sha256(fingerprint):
            raise EvaluationError(f"{arm} seed {seed} training_complete fingerprint is invalid")
        owned = _best_record_for_seed(best.get(arm) or {}, seed)
        if str(owned.get("checkpoint_fingerprint") or "") != fingerprint:
            raise EvaluationError(f"{arm} seed {seed} training fingerprint does not match the best checkpoint")
        if int(owned.get("active_seed")) != seed:
            raise EvaluationError(f"{arm} seed {seed} best checkpoint belongs to another seed")
        contract_hash = str(row.get("training_contract_sha256") or row.get("arm_training_contract_sha256") or "")
        if contract_hash != str(owned.get("arm_training_contract_sha256") or ""):
            raise EvaluationError(f"{arm} seed {seed} training contract hash does not match the best checkpoint")
        expected_nb13 = str(contract.get("resolved_nb13_generation_id") or contract.get("nb13_generation_id") or "")
        if str(row.get("nb13_generation_id") or "") != expected_nb13:
            raise EvaluationError(f"{arm} seed {seed} NB13 generation does not match the final contract")
        if str(row.get("d0_init_model_state_sha256") or "") != str(contract.get("d0_init_model_state_sha256") or ""):
            raise EvaluationError(f"{arm} seed {seed} D0 identity does not match the final contract")
        manifest_key = "d_random_manifest_sha256" if arm == ARM_RANDOM else "d_quality_manifest_sha256"
        if str(row.get("selected_manifest_sha256") or "") != str(contract.get(manifest_key) or ""):
            raise EvaluationError(f"{arm} seed {seed} manifest hash does not match the arm")
        disk = gen_dir / "training" / arm / f"seed-{seed}.json"
        if not disk.is_file():
            raise EvaluationError(f"{arm} seed {seed} training artifact is missing")
        reopened = _load_json(disk)
        reopened_fp = str(reopened.get("best_checkpoint_fingerprint") or reopened.get("checkpoint_fingerprint") or "")
        if reopened_fp != fingerprint or int(reopened.get("active_seed")) != seed or str(reopened.get("arm") or "") != arm:
            raise EvaluationError(f"{arm} seed {seed} reopened training artifact does not match")
    if declared and set(got) != set(declared):
        raise EvaluationError(f"{arm} training_complete seeds do not match the declaration")


def _verify_seeded_evaluation_files(
    gen_dir: Path,
    contract: Mapping[str, Any],
    best: Mapping[str, Any],
    *,
    n_samples: int,
    seed: int,
    confidence: float,
) -> None:
    """Recompute per-seed metrics, alignment, ownership, and the aggregate."""
    from src.rq2_final_contract import SEED_POLICY_MULTI, SEED_POLICY_SINGLE, aggregate_paired_seed_metrics

    policy_record = dict(contract.get("seed_policy_record") or {})
    declared = [int(item) for item in policy_record.get("seeds") or []]
    if not declared:
        raise EvaluationError("seeded publication is missing declared seeds")
    d0 = pd.read_parquet(gen_dir / "predictions" / f"{ARM_D0}.parquet")
    d0_uids = d0["record_uid"].astype(str).tolist()
    d0_refs = d0["reference_vi"].astype(str).tolist()
    if len(d0_uids) != len(set(d0_uids)):
        raise EvaluationError("D0 predictions contain a duplicate record_uid")
    base = None
    rows = []
    for active_seed in declared:
        seed_dir = gen_dir / "evaluation" / f"seed-{active_seed}"
        metrics_path = seed_dir / "metrics.json"
        bootstrap_path = seed_dir / "bootstrap.json"
        if not metrics_path.is_file() or not bootstrap_path.is_file():
            raise EvaluationError(f"seed {active_seed} evaluation artifacts are missing")
        recorded = _load_json(metrics_path)
        frames = {ARM_D0: d0}
        for arm in (ARM_RANDOM, ARM_QUALITY):
            path = seed_dir / f"{arm}.parquet"
            if not path.is_file():
                raise EvaluationError(f"seed {active_seed} {arm} predictions are missing")
            frame = pd.read_parquet(path)
            frames[arm] = frame
            if {str(value) for value in frame["arm"].tolist()} != {arm}:
                raise EvaluationError(f"seed {active_seed} {arm} prediction arm does not match")
            owned = _best_record_for_seed(best.get(arm) or {}, active_seed)
            if str(owned.get("arm") or arm) != arm:
                raise EvaluationError(f"seed {active_seed} {arm} checkpoint belongs to another arm")
            if int(owned.get("active_seed")) != int(active_seed):
                raise EvaluationError(f"seed {active_seed} {arm} checkpoint belongs to another seed")
            expected_fp = str(owned.get("checkpoint_fingerprint") or "")
            fingerprints = set(frame["checkpoint_fingerprint"].astype(str).tolist())
            if fingerprints != {expected_fp}:
                raise EvaluationError(f"seed {active_seed} {arm} evaluation fingerprint does not match the best checkpoint")
            frames[arm] = frame
        random_uids = frames[ARM_RANDOM]["record_uid"].astype(str).tolist()
        quality_uids = frames[ARM_QUALITY]["record_uid"].astype(str).tolist()
        random_refs = frames[ARM_RANDOM]["reference_vi"].astype(str).tolist()
        quality_refs = frames[ARM_QUALITY]["reference_vi"].astype(str).tolist()
        if random_uids != quality_uids:
            raise EvaluationError(f"seed {active_seed} Random and Quality UID order differ")
        if random_refs != quality_refs:
            raise EvaluationError(f"seed {active_seed} Random and Quality references differ")
        if len(random_uids) != len(set(random_uids)):
            raise EvaluationError(f"seed {active_seed} predictions contain a duplicate record_uid")
        if random_uids != d0_uids or random_refs != d0_refs:
            raise EvaluationError(f"seed {active_seed} G_test UID/reference base does not match")
        for arm in (ARM_RANDOM, ARM_QUALITY):
            frame = frames[arm]
            owned = _best_record_for_seed(best.get(arm) or {}, active_seed)
            expected_fp = str(owned.get("checkpoint_fingerprint") or "")
            recorded_arm = recorded.get(arm) or {}
            if str(recorded_arm.get("checkpoint_fingerprint") or "") != expected_fp:
                raise EvaluationError(f"seed {active_seed} {arm} metrics fingerprint does not match the best checkpoint")
            if str(recorded_arm.get("arm") or "") != arm or int(recorded_arm.get("active_seed")) != int(active_seed):
                raise EvaluationError(f"seed {active_seed} {arm} metrics ownership does not match")
            recomputed = arm_metrics(frame)
            for key in ("sacrebleu", "chrfpp"):
                if abs(float(recomputed[key]) - float(recorded_arm.get(key))) > 1e-9:
                    raise EvaluationError(f"seed {active_seed} {arm} {key} does not recompute from predictions")
        identity = (sha256_json(random_uids), sha256_json(random_refs), len(random_uids))
        if base is None:
            base = identity
        elif identity != base:
            raise EvaluationError("G_test UID/reference base differs across seeds")
        if str(recorded.get("record_uid_sha256") or "") != identity[0] or str(recorded.get("reference_sha256") or "") != identity[1]:
            raise EvaluationError(f"seed {active_seed} stored alignment hash does not recompute")
        recorded_bootstrap = _load_json(bootstrap_path)
        recomputed_bootstrap = paired_cluster_bootstrap_arms(
            frames,
            n_samples=int(recorded_bootstrap.get("n_samples") or n_samples),
            seed=int(recorded_bootstrap.get("seed") or seed),
            confidence=float(recorded_bootstrap.get("confidence") or confidence),
            cluster_col=str(recorded_bootstrap.get("cluster_col") or BOOTSTRAP_UNIT_GROUP_ID),
        )
        _assert_recomputed_bootstrap(
            recorded_bootstrap,
            recomputed_bootstrap,
            label=f"seed {active_seed}",
        )
        rows.append({
            "active_seed": active_seed,
            ARM_RANDOM: recorded[ARM_RANDOM],
            ARM_QUALITY: recorded[ARM_QUALITY],
        })
    seed_policy = {
        "seed_policy": SEED_POLICY_MULTI if policy_record.get("mode") == "multi" else SEED_POLICY_SINGLE,
        "seed_policy_seeds": declared,
    }
    rebuilt = aggregate_paired_seed_metrics(rows, seed_policy=seed_policy)
    aggregate_path = gen_dir / "aggregate" / "metrics.json"
    if not aggregate_path.is_file():
        raise EvaluationError("aggregate metrics are missing")
    stored = _load_json(aggregate_path)
    if int(stored.get("n_valid_runs") or -1) != len(declared):
        raise EvaluationError("aggregate n_valid_runs does not match declared seeds")
    if stored.get("std_convention") != "population_pstdev" or rebuilt.get("std_convention") != "population_pstdev":
        raise EvaluationError("aggregate std convention is not population_pstdev")
    _assert_seed_summary_matches(stored, rebuilt, label="aggregate")
    contract_summary = contract.get("seed_run_summary") or {}
    _assert_seed_summary_matches(contract_summary, rebuilt, label="contract seed summary")


def _verify_staged_final_generation(
    gen_dir: Path,
    *,
    project_root: Union[str, Path],
    durable_root: Optional[Union[str, Path]] = None,
    artifact_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    missing = [rel for rel in FINAL_ARTIFACT_FILES if not (gen_dir / rel).is_file()]
    if missing:
        raise EvaluationError("final artifacts missing: " + ", ".join(missing))
    hashes = hash_artifact_files(gen_dir)
    recorded = _load_json(gen_dir / "artifact_hashes.json")
    if "files" not in recorded or not isinstance(recorded.get("files"), Mapping):
        raise EvaluationError("artifact_hashes.json files must be an object")
    files = recorded["files"]
    actual_paths = set(hashes)
    recorded_paths = {str(rel) for rel in files}
    missing_coverage = sorted(actual_paths - recorded_paths)
    listed_missing = sorted(recorded_paths - actual_paths)
    if missing_coverage or listed_missing:
        parts = []
        if missing_coverage:
            parts.append(
                "artifact_hashes.json missing hash coverage for un-hashed scientific artifact: "
                + ", ".join(missing_coverage)
            )
        if listed_missing:
            parts.append("artifact_hashes.json lists a missing artifact: " + ", ".join(listed_missing))
        raise EvaluationError("; ".join(parts))
    for rel in sorted(hashes):
        if _recorded_sha(files.get(rel)) != hashes[rel]:
            raise EvaluationError(f"artifact hash mismatch: {rel}")
    contract = _load_json(gen_dir / "final_contract.json")
    rebuilt = build_final_contract({k: v for k, v in contract.items() if k != "rq2_final_contract_sha256"})
    if rebuilt["rq2_final_contract_sha256"] != contract.get("rq2_final_contract_sha256"):
        raise EvaluationError("final contract hash does not recompute")
    seed_mode = str((contract.get("seed_policy_record") or {}).get("mode") or "")
    for key in (
        "nb11_input_contract_sha256",
        "nb12_contract_sha256",
        "nb13_selection_contract_sha256",
        "d_random_manifest_sha256",
        "d_quality_manifest_sha256",
        "d0_checkpoint_fingerprint",
        "d0_init_model_state_sha256",
        "source_fingerprint_sha256",
        "g_test_manifest_sha256",
        "g_test_ordered_uid_hash",
        "g_test_reference_hash",
        "comparison_artifact_sha256",
        "best_checkpoints_artifact_sha256",
        "fairness_proof_sha256",
        "d_random_training_complete_sha256",
        "d_quality_training_complete_sha256",
    ):
        if not is_sha256(contract.get(key)):
            raise EvaluationError(f"final contract missing locked identity: {key}")
    if seed_mode != "multi":
        for key in (
            "d_random_best_checkpoint_fingerprint",
            "d_quality_best_checkpoint_fingerprint",
            "d_random_training_contract_sha256",
            "d_quality_training_contract_sha256",
        ):
            if not is_sha256(contract.get(key)):
                raise EvaluationError(f"final contract missing locked identity: {key}")
    if str(contract.get("comparison_artifact_sha256") or "") != hashes.get("arm_comparison.json"):
        raise EvaluationError("comparison artifact SHA256 does not match contract")
    if str(contract.get("best_checkpoints_artifact_sha256") or "") != hashes.get("best_checkpoints.json"):
        raise EvaluationError("best_checkpoints artifact SHA256 does not match contract")
    if str(contract.get("fairness_proof_sha256") or "") != hashes.get("fairness_proof.json"):
        raise EvaluationError("fairness_proof SHA256 does not match contract")
    if str(contract.get("d_random_training_complete_sha256") or "") != hashes.get("d_random_training_complete.json"):
        raise EvaluationError("d_random training_complete SHA256 does not match contract")
    if str(contract.get("d_quality_training_complete_sha256") or "") != hashes.get("d_quality_training_complete.json"):
        raise EvaluationError("d_quality training_complete SHA256 does not match contract")

    source = compute_nb14_source_fingerprint()
    if source["aggregate_sha256"] != str(contract.get("source_fingerprint_sha256") or ""):
        raise EvaluationError("source fingerprint drifted from the repository helper")

    root = Path(project_root)
    from src.rq2_pseudo_contract import resolve_nb11_generation
    from src.rq2_selection_contract import resolve_frozen_u_prime
    from src.rq2_final_data import verify_pinned_nb13_selection

    nb11_id = str(contract.get("nb11_generation_id") or "")
    nb12_id = str(contract.get("nb12_generation_id") or "")
    nb13_id = str(contract.get("nb13_generation_id") or "")
    if not nb11_id or not nb12_id or not nb13_id:
        raise EvaluationError("final contract is missing pinned upstream generation IDs")
    from src.rq2_final_contract import resolve_nb14_layout

    artifact = Path(artifact_root) if artifact_root is not None else root
    layout = resolve_nb14_layout(root, durable_root=artifact)
    nb11 = resolve_nb11_generation(
        root, nb11_id, u_clean_dir=layout["u_clean_dir"], require_audio_files=False,
    )
    if nb11.contract_sha256 != str(contract["nb11_input_contract_sha256"]):
        raise EvaluationError("NB11 contract hash does not match the pinned generation")
    frozen = resolve_frozen_u_prime(
        root,
        generation_id=nb12_id,
        durable_root=layout["durable_root"],
        pseudo_dir=layout["pseudo_dir"],
        u_clean_dir=layout["u_clean_dir"],
    )
    if str(frozen.identity["nb12_contract_sha256"]) != str(contract["nb12_contract_sha256"]):
        raise EvaluationError("NB12 contract hash does not match the pinned generation")
    selection = verify_pinned_nb13_selection(root, generation_id=nb13_id, artifact_root=layout["durable_root"])
    if str(selection["contract"]["selection_contract_sha256"]) != str(contract["nb13_selection_contract_sha256"]):
        raise EvaluationError("NB13 selection contract hash does not match the pinned generation")
    if str(selection["summary"]["d_random_manifest_sha256"]) != str(contract["d_random_manifest_sha256"]):
        raise EvaluationError("NB13 D-Random manifest hash does not match the pinned generation")
    if str(selection["summary"]["d_quality_manifest_sha256"]) != str(contract["d_quality_manifest_sha256"]):
        raise EvaluationError("NB13 D-Quality manifest hash does not match the pinned generation")

    rq1 = resolve_frozen_rq1_final_state(root, durable_root=durable_root)
    test_contract = rq1["test_contract"]
    if str(test_contract.get("manifest_sha256") or "") != str(contract["g_test_manifest_sha256"]):
        raise EvaluationError("G_test manifest hash does not match the durable RQ1 test contract")
    if str(test_contract.get("ordered_uid_hash") or "") != str(contract["g_test_ordered_uid_hash"]):
        raise EvaluationError("G_test ordered UID hash does not match the durable RQ1 test contract")
    if str(test_contract.get("reference_content_hash") or "") != str(contract["g_test_reference_hash"]):
        raise EvaluationError("G_test reference hash does not match the durable RQ1 test contract")
    if rq1["rq1_test_contract_hash"] != str(test_contract.get("rq1_test_contract_hash") or ""):
        raise EvaluationError("durable RQ1 test contract hash drifted")

    frames = {arm: pd.read_parquet(gen_dir / f"predictions/{arm}.parquet") for arm in ARMS}
    assert_paired_alignment(frames)
    pred_map = contract.get("prediction_artifact_sha256") or {}
    for arm in ARMS:
        rel = f"predictions/{arm}.parquet"
        if str(pred_map.get(arm) or "") != hashes.get(rel):
            raise EvaluationError(f"{arm} prediction SHA256 does not match contract")
    metrics = _load_json(gen_dir / "arm_metrics.json")
    if str(contract.get("metrics_artifact_sha256") or "") != hashes.get("arm_metrics.json"):
        raise EvaluationError("metrics artifact SHA256 does not match contract")
    for arm, frame in frames.items():
        recomputed = arm_metrics(frame)
        recorded_m = metrics.get(arm) or {}
        if not recomputed.get("finite"):
            raise EvaluationError(f"{arm} metrics are not finite")
        for key in ("sacrebleu", "chrfpp"):
            if abs(float(recomputed[key]) - float(recorded_m[key])) > 1e-9:
                raise EvaluationError(f"{arm} {key} does not match prediction bytes")
    bootstrap = _load_json(gen_dir / "bootstrap_results.json")
    if str(contract.get("bootstrap_artifact_sha256") or "") != hashes.get("bootstrap_results.json"):
        raise EvaluationError("bootstrap artifact SHA256 does not match contract")
    if bootstrap.get("bootstrap_method") != BOOTSTRAP_METHOD_PAIRED_CLUSTER:
        raise EvaluationError("bootstrap protocol is not paired cluster")
    if bootstrap.get("cluster_col") != BOOTSTRAP_UNIT_GROUP_ID:
        raise EvaluationError("bootstrap cluster_col drifted")
    if not bootstrap.get("comparisons"):
        raise EvaluationError("bootstrap comparisons are missing")
    recomputed_bootstrap = paired_cluster_bootstrap_arms(
        frames,
        n_samples=int(bootstrap.get("n_samples") or (contract.get("bootstrap_config") or {}).get("bootstrap_samples") or 0),
        seed=int(bootstrap.get("seed") or (contract.get("bootstrap_config") or {}).get("seed") or 0),
        confidence=float(bootstrap.get("confidence") or (contract.get("bootstrap_config") or {}).get("confidence") or 0.95),
        cluster_col=str(bootstrap.get("cluster_col") or BOOTSTRAP_UNIT_GROUP_ID),
    )
    _assert_recomputed_bootstrap(bootstrap, recomputed_bootstrap, label="legacy")
    best = _load_json(gen_dir / "best_checkpoints.json")
    from src.rq2_final_contract import assert_checkpoint_arm_isolation, experiment_root_for_arm
    from src.rq2_final_train import arm_state_layout, checkpoint_model_state_sha256

    for arm in ARMS:
        payload = best.get(arm) or {}
        records = _best_checkpoint_records(payload) if _is_seed_map(payload) else [payload]
        if not records:
            raise EvaluationError(f"{arm} best checkpoint is missing")
        for record in records:
            labeled = str(record.get("arm") or arm)
            if labeled != arm:
                raise EvaluationError(f"best checkpoint arm mismatch for {arm}")
            fp = str(record.get("checkpoint_fingerprint") or record.get("checkpoint_fingerprint_sha256") or "")
            if not fp:
                raise EvaluationError(f"{arm} best checkpoint fingerprint is missing")
            if not is_sha256(fp):
                raise EvaluationError(f"{arm} best checkpoint fingerprint is invalid")
            if seed_mode and arm != ARM_D0:
                if record.get("active_seed") is None:
                    raise EvaluationError(f"{arm} best checkpoint is missing active_seed")
                if record.get("g_test_used") is not False:
                    raise EvaluationError(f"{arm} best checkpoint does not prove G_test was unused")
            if arm == ARM_D0 and fp != str(contract.get("d0_checkpoint_fingerprint") or ""):
                raise EvaluationError("D0 checkpoint fingerprint does not match final contract")
            owned_contract = str(record.get("arm_training_contract_sha256") or record.get("direct_training_contract_hash") or "")
            if not owned_contract:
                raise EvaluationError(f"{arm} best checkpoint training-contract ownership is missing")
            if arm != ARM_D0:
                scalar = str(contract.get(f"{arm}_training_contract_sha256") or "")
                expected = str(record.get("arm_training_contract_sha256") or scalar)
                if not is_sha256(expected) or owned_contract != expected:
                    raise EvaluationError(f"{arm} best checkpoint training-contract ownership drifted")
                if scalar and seed_mode != "multi" and owned_contract != scalar:
                    raise EvaluationError(f"{arm} best checkpoint training-contract ownership drifted")
                data_hash = str(record.get("data_contract_sha256") or record.get("data_manifest_sha256") or "")
                if not data_hash:
                    raise EvaluationError(f"{arm} best checkpoint data/selection ownership is missing")
                experiment_fp = str(record.get("experiment_fingerprint_sha256") or "")
                if not experiment_fp:
                    raise EvaluationError(f"{arm} best checkpoint experiment fingerprint is missing")
                if durable_root:
                    root = experiment_root_for_arm(
                        durable_root=durable_root,
                        contract_hash=expected,
                        arm=arm,
                    )
                    layout = arm_state_layout(root)
                    ckpt = Path(layout["checkpoints"]) / str(record.get("checkpoint_name") or "")
                    if not ckpt.is_dir():
                        raise EvaluationError(f"{arm} best checkpoint cannot be re-opened")
                    assert_checkpoint_arm_isolation(
                        ckpt,
                        arm=arm,
                        expected_contract_hash=expected,
                        expected_data_hash=data_hash,
                    )
                    live_fp = checkpoint_model_state_sha256(ckpt)
                    if live_fp != fp:
                        raise EvaluationError(f"{arm} best checkpoint model-state fingerprint does not match")
                if seed_mode != "multi":
                    assert_recorded_best_fingerprint(contract, arm, fp)
    if str(contract.get("d0_init_model_state_sha256") or "") and not is_sha256(contract.get("d0_init_model_state_sha256")):
        raise EvaluationError("final contract D0 init fingerprint is invalid")
    fairness = _load_json(gen_dir / "fairness_proof.json")
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
        if not is_sha256(fairness.get(key)):
            raise EvaluationError(f"fairness proof missing {key}")
    for arm, rel in ((ARM_RANDOM, "d_random_training_complete.json"), (ARM_QUALITY, "d_quality_training_complete.json")):
        complete = _load_json(gen_dir / rel)
        if complete.get("status") != STATUS_TRAINING_COMPLETE:
            raise EvaluationError(f"{arm} training_complete is not TRAINING_COMPLETE")
        if str(complete.get("arm") or "") != arm:
            raise EvaluationError(f"{arm} training_complete arm mismatch")
        runs = list(complete.get("seed_runs") or [])
        if seed_mode == "multi":
            if not runs:
                raise EvaluationError(f"{arm} multi-seed training_complete runs are missing")
            _verify_persisted_seed_training(gen_dir, contract, best, arm, runs)
        elif runs:
            _verify_persisted_seed_training(gen_dir, contract, best, arm, runs)
            if str(complete.get("training_contract_sha256") or "") != str(contract.get(f"{arm}_training_contract_sha256") or ""):
                raise EvaluationError(f"{arm} training_complete contract hash mismatch")
            if not is_sha256(complete.get("terminal_checkpoint_fingerprint")):
                raise EvaluationError(f"{arm} training_complete fingerprint is invalid")
        else:
            if str(complete.get("arm") or "") != arm:
                raise EvaluationError(f"{arm} training_complete arm mismatch")
            if str(complete.get("training_contract_sha256") or "") != str(contract.get(f"{arm}_training_contract_sha256") or ""):
                raise EvaluationError(f"{arm} training_complete contract hash mismatch")
            if not is_sha256(complete.get("terminal_checkpoint_fingerprint")):
                raise EvaluationError(f"{arm} training_complete fingerprint is invalid")
    summary = _load_json(gen_dir / "summary.json")
    if summary.get("status") != STATUS_SUCCESS:
        raise EvaluationError("summary status is not SUCCESS_RQ2_FINAL")
    if str(summary.get("rq2_final_contract_sha256") or "") != str(contract.get("rq2_final_contract_sha256") or ""):
        raise EvaluationError("summary contract hash does not match the actual final contract")
    if seed_mode:
        verify_published_seed_provenance(contract, best)
        _verify_seeded_evaluation_files(
            gen_dir,
            contract,
            best,
            n_samples=int(bootstrap.get("n_samples") or 0),
            seed=int(bootstrap.get("seed") or 0),
            confidence=float(bootstrap.get("confidence") or 0.95),
        )
    return {
        "hashes": hashes,
        "contract_sha256": contract["rq2_final_contract_sha256"],
        "summary_status": summary.get("status"),
    }


def publish_final_generation(
    project_root: Union[str, Path],
    *,
    artifacts: Mapping[str, Any],
    checks: Optional[Mapping[str, bool]] = None,
    durable_root: Optional[Union[str, Path]] = None,
    artifact_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Copy artifacts, independently re-verify them, then move CURRENT last.

    Caller ``checks`` are diagnostics only and cannot force SUCCESS.
    CURRENT is written only after the immutable generation verifies.
    """
    del checks  # not authoritative
    root = Path(project_root)
    from src.rq2_final_contract import resolve_nb14_layout

    publication_root = Path(durable_root) if durable_root else root
    layout = resolve_nb14_layout(root, durable_root=publication_root)
    out = layout["final_dir"]
    out.mkdir(parents=True, exist_ok=True)
    out = assert_nb14_output_dir(out, root, durable_root=publication_root)
    upstream_root = Path(artifact_root) if artifact_root is not None else root
    staged = stage_generation(out)
    gen_dir: Path = staged["staging_dir"]
    (gen_dir / "predictions").mkdir(parents=True, exist_ok=True)
    try:
        copied = list(FINAL_ARTIFACT_FILES)
        for rel in FINAL_ARTIFACT_FILES:
            if rel not in artifacts:
                raise EvaluationError(f"missing artifact {rel}")
            dest = gen_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(Path(artifacts[rel]), dest)
        for rel, src in artifacts.items():
            if rel in FINAL_ARTIFACT_FILES:
                continue
            dest = gen_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(Path(src), dest)
            copied.append(rel)
        _verify_staged_final_generation(
            gen_dir, project_root=root, durable_root=durable_root, artifact_root=upstream_root,
        )
    except Exception as exc:
        return {
            "status": STATUS_FAIL,
            "wrote_current": False,
            "error": str(exc),
            "verdict": derive_final_status({}),
        }
    verified = finalize_generation(out, staged, copied, move_current=False)
    published = Path(staged["final_dir"])
    try:
        second = _verify_staged_final_generation(
            published, project_root=root, durable_root=durable_root, artifact_root=upstream_root,
        )
    except Exception as exc:
        return {
            "status": STATUS_FAIL,
            "wrote_current": False,
            "generation": verified,
            "error": str(exc),
            "verdict": derive_final_status({}),
        }
    from src.rq2_pseudo_contract import atomic_write_text as _atomic_current

    _atomic_current(out / "CURRENT", str(staged["generation_id"]) + "\n")
    if read_current_generation_id(out) != str(staged["generation_id"]):
        return {
            "status": STATUS_FAIL,
            "wrote_current": False,
            "generation": verified,
            "error": "CURRENT did not move to the verified generation",
            "verdict": derive_final_status({}),
        }
    return {
        "status": STATUS_SUCCESS,
        "generation": verified,
        "wrote_current": True,
        "reverified": second,
        "verdict": derive_final_status({name: True for name in derive_final_status({}).get("checks")}),
    }


def _copy_file(src: Union[str, Path], dest: Path) -> None:
    data = Path(src).read_bytes()
    dest.write_bytes(data)
