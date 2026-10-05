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
    summarize_seed_runs,
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
        layout = layouts.get(arm) or {}
        contract = (contracts.get(arm) or {}).get("contract") or contracts.get(arm) or {}
        payload = best.get(arm) or {}
        proofs[arm] = prove_trainable_arm_ready_for_g_test(
            arm=arm,
            layout=layout,
            contract=contract,
            best=payload,
            d0_identity=d0_identity,
            nb13_selection_contract_sha256=nb13,
        )
    return proofs


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
    for arm, key in ((ARM_RANDOM, "d_random_complete"), (ARM_QUALITY, "d_quality_complete")):
        layout = layouts.get(arm) or {}
        contract = (contracts.get(arm) or {}).get("contract") or contracts.get(arm) or {}
        complete_path = layout.get("training_complete")
        payload: Dict[str, Any] = {}
        if complete_path and Path(complete_path).is_file():
            payload = json.loads(Path(complete_path).read_text(encoding="utf-8"))
        ok = (
            payload.get("status") == STATUS_TRAINING_COMPLETE
            and str(payload.get("arm") or "") == arm
            and str(payload.get("training_contract_sha256") or "") == str(contract.get("arm_training_contract_sha256") or "")
            and bool(contract.get("arm_training_contract_sha256"))
            and is_sha256(payload.get("terminal_checkpoint_fingerprint"))
        )
        gate(
            key,
            ok,
            contract_sha256=str(contract.get("arm_training_contract_sha256") or ""),
            status=str(payload.get("status") or ""),
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
        fp = str(payload.get("checkpoint_fingerprint") or payload.get("checkpoint_fingerprint_sha256") or "")
        ok = bool(payload) and payload.get("g_test_used") is False and (not fp or is_sha256(fp))
        if arm != ARM_D0:
            frozen_best = frozen_best and ok
        gate(f"{arm}_best_checkpoint_frozen", ok, checkpoint_fingerprint=fp)
    gate("best_checkpoints_frozen", frozen_best and gates.get("d0_best_checkpoint_frozen", {}).get("ok") is True)
    gate("validation_selection_complete", gates["best_checkpoints_frozen"]["ok"])
    train_ok = True
    for arm in (ARM_RANDOM, ARM_QUALITY):
        contract = (contracts.get(arm) or {}).get("contract") or contracts.get(arm) or {}
        train_ok = train_ok and bool(contract.get("arm_training_contract_sha256"))
    gate("training_contracts_verified", train_ok)
    gate("evaluation_protocol_frozen", bool(protocol and protocol.get("evaluation_protocol_sha256")))
    gate("bootstrap_protocol_frozen", bool(protocol and protocol.get("bootstrap_method") == BOOTSTRAP_METHOD_PAIRED_CLUSTER))
    gate("no_pending_scientific_decisions", flags.rq2_final_frozen is True)
    gate("rq2_final_frozen", flags.rq2_final_frozen is True)
    flat = {name: gates.get(name, {}).get("ok") is True for name in UNLOCK_REQUIRED}
    return {"gates": gates, "flat": flat, "all_ok": all(flat.values())}


def hash_artifact_files(directory: Union[str, Path]) -> Dict[str, str]:
    root = Path(directory)
    return {
        rel: sha256_file(root / rel)
        for rel in FINAL_ARTIFACT_FILES
        if rel != "artifact_hashes.json" and (root / rel).is_file()
    }


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvaluationError(f"JSON object required: {path.name}")
    return payload


def _recorded_sha(entry: Any) -> str:
    if isinstance(entry, Mapping):
        return str(entry.get("sha256") or "")
    return str(entry or "")


def persist_final_artifacts(
    directory: Union[str, Path],
    *,
    frames: Mapping[str, pd.DataFrame],
    contract_fields: Mapping[str, Any],
    best: Mapping[str, Any],
    n_samples: int,
    seed: int,
    confidence: float = 0.95,
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
    for arm, rel in ((ARM_RANDOM, "d_random_training_complete.json"), (ARM_QUALITY, "d_quality_training_complete.json")):
        payload = dict(contract_fields.get(f"{arm}_training_complete") or {})
        if payload.get("status") != STATUS_TRAINING_COMPLETE:
            raise EvaluationError(f"{arm} training_complete proof is missing")
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
    if contract_fields.get("seed_policy"):
        seed_policy = {
            "seed_policy": contract_fields.get("seed_policy"),
            "seed_policy_seeds": list(contract_fields.get("seed_policy_seeds") or []),
            "report_mean_std": str(contract_fields.get("seed_policy") or "") == "multi_seed",
        }
        fields["seed_policy"] = seed_policy["seed_policy"]
        fields["seed_policy_seeds"] = seed_policy["seed_policy_seeds"]
        fields["seed_run_summary"] = summarize_seed_runs([], seed_policy=seed_policy)
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
    })
    write_json(root / "artifact_hashes.json", {"files": hash_artifact_files(root)})
    return {rel: root / rel for rel in FINAL_ARTIFACT_FILES}


def _verify_staged_final_generation(
    gen_dir: Path,
    *,
    project_root: Union[str, Path],
    durable_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    missing = [rel for rel in FINAL_ARTIFACT_FILES if not (gen_dir / rel).is_file()]
    if missing:
        raise EvaluationError("final artifacts missing: " + ", ".join(missing))
    hashes = hash_artifact_files(gen_dir)
    recorded = _load_json(gen_dir / "artifact_hashes.json")
    files = recorded.get("files") or recorded
    missing_hashes = [rel for rel in hashes if rel not in files]
    if missing_hashes:
        raise EvaluationError("artifact_hashes.json missing files: " + ", ".join(missing_hashes))
    for rel, sha in hashes.items():
        if _recorded_sha(files.get(rel)) != sha:
            raise EvaluationError(f"artifact hash mismatch: {rel}")
    contract = _load_json(gen_dir / "final_contract.json")
    rebuilt = build_final_contract({k: v for k, v in contract.items() if k != "rq2_final_contract_sha256"})
    if rebuilt["rq2_final_contract_sha256"] != contract.get("rq2_final_contract_sha256"):
        raise EvaluationError("final contract hash does not recompute")
    for key in (
        "nb11_input_contract_sha256",
        "nb12_contract_sha256",
        "nb13_selection_contract_sha256",
        "d_random_manifest_sha256",
        "d_quality_manifest_sha256",
        "d0_checkpoint_fingerprint",
        "d0_init_model_state_sha256",
        "d_random_best_checkpoint_fingerprint",
        "d_quality_best_checkpoint_fingerprint",
        "d_random_training_contract_sha256",
        "d_quality_training_contract_sha256",
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
    nb11 = resolve_nb11_generation(root, nb11_id, require_audio_files=False)
    if nb11.contract_sha256 != str(contract["nb11_input_contract_sha256"]):
        raise EvaluationError("NB11 contract hash does not match the pinned generation")
    frozen = resolve_frozen_u_prime(root, generation_id=nb12_id)
    if str(frozen.identity["nb12_contract_sha256"]) != str(contract["nb12_contract_sha256"]):
        raise EvaluationError("NB12 contract hash does not match the pinned generation")
    selection = verify_pinned_nb13_selection(root, generation_id=nb13_id)
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
    for key, entry in (bootstrap.get("comparisons") or {}).items():
        other = (recomputed_bootstrap.get("comparisons") or {}).get(key) or {}
        for field in (
            "observed_sacrebleu_delta",
            "observed_chrfpp_delta",
            "sacrebleu_ci_lower",
            "sacrebleu_ci_upper",
            "chrfpp_ci_lower",
            "chrfpp_ci_upper",
        ):
            if not math.isfinite(float(entry.get(field))):
                raise EvaluationError(f"bootstrap {key} {field} is not finite")
            if abs(float(entry.get(field)) - float(other.get(field))) > 1e-9:
                raise EvaluationError(f"bootstrap {key} {field} does not recompute from prediction bytes")
    best = _load_json(gen_dir / "best_checkpoints.json")
    from src.rq2_final_contract import assert_checkpoint_arm_isolation, experiment_root_for_arm
    from src.rq2_final_train import arm_state_layout, checkpoint_model_state_sha256

    for arm in ARMS:
        payload = best.get(arm) or {}
        labeled = str(payload.get("arm") or arm)
        if labeled != arm:
            raise EvaluationError(f"best checkpoint arm mismatch for {arm}")
        fp = str(payload.get("checkpoint_fingerprint") or payload.get("checkpoint_fingerprint_sha256") or "")
        if not fp:
            raise EvaluationError(f"{arm} best checkpoint fingerprint is missing")
        if not is_sha256(fp):
            raise EvaluationError(f"{arm} best checkpoint fingerprint is invalid")
        if arm == ARM_D0 and fp != str(contract.get("d0_checkpoint_fingerprint") or ""):
            raise EvaluationError("D0 checkpoint fingerprint does not match final contract")
        owned_contract = str(payload.get("arm_training_contract_sha256") or payload.get("direct_training_contract_hash") or "")
        if not owned_contract:
            raise EvaluationError(f"{arm} best checkpoint training-contract ownership is missing")
        if arm != ARM_D0:
            expected = str(contract.get(f"{arm}_training_contract_sha256") or "")
            if owned_contract != expected:
                raise EvaluationError(f"{arm} best checkpoint training-contract ownership drifted")
            data_hash = str(payload.get("data_contract_sha256") or payload.get("data_manifest_sha256") or "")
            if not data_hash:
                raise EvaluationError(f"{arm} best checkpoint data/selection ownership is missing")
            experiment_fp = str(payload.get("experiment_fingerprint_sha256") or "")
            if not experiment_fp:
                raise EvaluationError(f"{arm} best checkpoint experiment fingerprint is missing")
            if durable_root:
                root = experiment_root_for_arm(
                    durable_root=durable_root,
                    contract_hash=expected,
                    arm=arm,
                )
                layout = arm_state_layout(root)
                ckpt = Path(layout["checkpoints"]) / str(payload.get("checkpoint_name") or "")
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
            expected_best_fp = str(contract.get(f"{arm}_best_checkpoint_fingerprint") or "")
            if expected_best_fp and expected_best_fp != fp:
                raise EvaluationError(f"{arm} best checkpoint fingerprint does not match the final contract")
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
        if str(complete.get("training_contract_sha256") or "") != str(contract.get(f"{arm}_training_contract_sha256") or ""):
            raise EvaluationError(f"{arm} training_complete contract hash mismatch")
        if not is_sha256(complete.get("terminal_checkpoint_fingerprint")):
            raise EvaluationError(f"{arm} training_complete fingerprint is invalid")
    summary = _load_json(gen_dir / "summary.json")
    if summary.get("status") != STATUS_SUCCESS:
        raise EvaluationError("summary status is not SUCCESS_RQ2_FINAL")
    if str(summary.get("rq2_final_contract_sha256") or "") != str(contract.get("rq2_final_contract_sha256") or ""):
        raise EvaluationError("summary contract hash does not match the actual final contract")
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
) -> Dict[str, Any]:
    """Copy artifacts, independently re-verify them, then move CURRENT last.

    Caller ``checks`` are diagnostics only and cannot force SUCCESS.
    CURRENT is written only after the immutable generation verifies.
    """
    del checks  # not authoritative
    root = Path(project_root)
    out = Path(project_root) / FINAL_RELATIVE_DIR
    out.mkdir(parents=True, exist_ok=True)
    out = assert_nb14_output_dir(out, root)
    staged = stage_generation(out)
    gen_dir: Path = staged["staging_dir"]
    (gen_dir / "predictions").mkdir(parents=True, exist_ok=True)
    try:
        for rel in FINAL_ARTIFACT_FILES:
            if rel not in artifacts:
                raise EvaluationError(f"missing artifact {rel}")
            dest = gen_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(Path(artifacts[rel]), dest)
        _verify_staged_final_generation(gen_dir, project_root=root, durable_root=durable_root)
    except Exception as exc:
        return {
            "status": STATUS_FAIL,
            "wrote_current": False,
            "error": str(exc),
            "verdict": derive_final_status({}),
        }
    verified = finalize_generation(out, staged, FINAL_ARTIFACT_FILES, move_current=False)
    published = Path(staged["final_dir"])
    try:
        second = _verify_staged_final_generation(published, project_root=root, durable_root=durable_root)
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
