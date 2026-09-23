"""Notebook 06 helpers: frozen-test unlock, paired C0/D0 evaluation and analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data_utils import normalize_bahnar_ctc_v1
from src.metrics import mt_corpus_metrics
from src.mt_normalize import normalize_mt_text_v1
from src.rq1_contract import (
    STATUS_RQ1_C0,
    STATUS_RQ1_D0,
    STATUS_RQ1_UNLOCK,
    STATUS_RQ1_VERIFY,
    BOOTSTRAP_METHOD_PAIRED_CLUSTER,
    BOOTSTRAP_UNIT_GROUP_ID,
    assert_group_id_complete,
    assert_no_split_key_overlap,
    assert_prediction_frame,
    build_rq1_test_contract,
    resolve_parquet_revision_pin,
    sha256_file,
    sha256_json,
    validate_frozen_test_frame,
)


def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"Missing required artifact: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _best_name(*payloads: Mapping[str, Any]) -> str:
    for payload in payloads:
        for key in ("best_checkpoint_name", "best_checkpoint"):
            val = payload.get(key)
            if val is None or val == "":
                continue
            return Path(str(val)).name
    raise RuntimeError("best checkpoint name missing from upstream artifacts")


def _require_same(label: str, **fields: Any) -> None:
    values = list(fields.values())
    if len({json.dumps(v, sort_keys=True, default=str) for v in values}) != 1:
        raise RuntimeError(f"{label} mismatch across upstream artifacts: {fields}")


def _assert_asr_identity(train: Mapping[str, Any], evaluate: Mapping[str, Any]) -> Dict[str, Any]:
    from src.asr_full_train import extract_training_contract

    if evaluate.get("status") != "SUCCESS_FULL_EVALUATE":
        raise RuntimeError("NB03 is not SUCCESS_FULL_EVALUATE")
    if evaluate.get("frozen_test_accessed") is not False:
        raise RuntimeError("NB03 evaluate summary is not frozen-test clean")
    contract = extract_training_contract(dict(train))
    if not contract:
        raise RuntimeError("NB03 train summary missing training contract")
    eval_contract = extract_training_contract(dict(evaluate)) or {}
    exp = str(train.get("experiment_id") or contract.get("experiment_id") or "")
    if not exp:
        raise RuntimeError("NB03 experiment_id missing")
    _require_same(
        "NB03 experiment_id",
        train=exp,
        evaluate=str(evaluate.get("experiment_id") or exp),
        contract=str(contract.get("experiment_id") or exp),
    )
    train_hash = str(
        contract.get("contract_hash")
        or contract.get("train_contract_hash")
        or contract.get("data_contract_hash")
        or ""
    )
    if not train_hash:
        raise RuntimeError("NB03 training/data contract hash missing")
    if eval_contract:
        eval_hash = str(
            eval_contract.get("contract_hash")
            or eval_contract.get("train_contract_hash")
            or eval_contract.get("data_contract_hash")
            or train_hash
        )
        if eval_hash != train_hash:
            raise RuntimeError("NB03 train/evaluate contract hash mismatch")
    best = _best_name(train, evaluate)
    eval_best = evaluate.get("best_checkpoint") or evaluate.get("best_checkpoint_name")
    if eval_best is not None and Path(str(eval_best)).name != best:
        raise RuntimeError("NB03 train/evaluate best-checkpoint name mismatch")
    return {
        "experiment_id": exp,
        "contract_hash": train_hash,
        "best_checkpoint_name": best,
        "training_contract": contract,
        "model_id": contract.get("model_id") or contract.get("pretrained_model_name_or_path"),
        "model_revision": contract.get("model_revision") or contract.get("revision"),
    }


def _assert_mt_identity(
    train: Mapping[str, Any],
    evaluate: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    from src.mt_contract import (
        GENERATION_CONFIG_KEYS,
        assert_generation_config_matches,
        assert_mt_training_contract_self_consistent,
    )

    if evaluate.get("status") != "SUCCESS_MT_EVALUATE":
        raise RuntimeError("NB04 is not SUCCESS_MT_EVALUATE")
    if evaluate.get("frozen_test_accessed") is not False:
        raise RuntimeError("NB04 evaluate summary is not frozen-test clean")
    assert_mt_training_contract_self_consistent(contract)
    exp = str(contract.get("experiment_id") or train.get("experiment_id") or "")
    if not exp:
        raise RuntimeError("NB04 experiment_id missing")
    _require_same(
        "NB04 experiment_id",
        contract=exp,
        train=str(train.get("experiment_id") or exp),
        evaluate=str(evaluate.get("experiment_id") or exp),
    )
    contract_hash = str(contract.get("mt_training_contract_hash") or "")
    train_hash = str(
        train.get("mt_training_contract_hash")
        or (train.get("training_contract") or {}).get("mt_training_contract_hash")
        or contract_hash
    )
    eval_hash = str(
        evaluate.get("mt_training_contract_hash")
        or (evaluate.get("training_contract") or {}).get("mt_training_contract_hash")
        or contract_hash
    )
    _require_same("NB04 mt_training_contract_hash", contract=contract_hash, train=train_hash, evaluate=eval_hash)
    best = _best_name(train, evaluate)
    eval_best = evaluate.get("best_checkpoint") or evaluate.get("best_checkpoint_name")
    if eval_best is not None and Path(str(eval_best)).name != best:
        raise RuntimeError("NB04 train/evaluate best-checkpoint name mismatch")
    expected_gen = {k: contract.get(k) for k in GENERATION_CONFIG_KEYS if k in contract}
    if expected_gen:
        actual_gen = evaluate.get("generation_config") or {
            k: evaluate.get(k) for k in GENERATION_CONFIG_KEYS if k in evaluate
        }
        if actual_gen:
            # Fill missing keys from contract so helper compares locked surface.
            merged = dict(expected_gen)
            merged.update({k: v for k, v in actual_gen.items() if v is not None})
            assert_generation_config_matches(expected_gen, {k: merged.get(k) for k in expected_gen})
    return {
        "experiment_id": exp,
        "contract_hash": contract_hash,
        "best_checkpoint_name": best,
        "training_contract": dict(contract),
        "model_id": contract.get("model_id"),
        "model_revision": contract.get("model_revision"),
        "generation_config": expected_gen,
    }


def _assert_direct_identity(
    train: Mapping[str, Any],
    evaluate: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    from src.direct_contract import assert_direct_training_contract_self_consistent

    if evaluate.get("status") != "SUCCESS_DIRECT_EVALUATE":
        raise RuntimeError("NB05 is not SUCCESS_DIRECT_EVALUATE")
    if evaluate.get("frozen_test_accessed") is not False:
        raise RuntimeError("NB05 evaluate summary is not frozen-test clean")
    assert_direct_training_contract_self_consistent(contract)
    exp = str(contract.get("experiment_id") or train.get("experiment_id") or "")
    if not exp:
        raise RuntimeError("NB05 experiment_id missing")
    _require_same(
        "NB05 experiment_id",
        contract=exp,
        train=str(train.get("experiment_id") or exp),
        evaluate=str(evaluate.get("experiment_id") or exp),
    )
    contract_hash = str(contract.get("direct_training_contract_hash") or "")
    train_hash = str(
        train.get("direct_training_contract_hash")
        or train.get("training_contract_hash")
        or contract_hash
    )
    eval_hash = str(
        evaluate.get("direct_training_contract_hash")
        or evaluate.get("training_contract_hash")
        or contract_hash
    )
    _require_same(
        "NB05 direct_training_contract_hash",
        contract=contract_hash,
        train=train_hash,
        evaluate=eval_hash,
    )
    best = _best_name(train, evaluate)
    eval_best = evaluate.get("best_checkpoint") or evaluate.get("best_checkpoint_name")
    if eval_best is not None and Path(str(eval_best)).name != best:
        raise RuntimeError("NB05 train/evaluate best-checkpoint name mismatch")
    for key in (
        "encoder_id",
        "encoder_revision",
        "decoder_id",
        "decoder_revision",
        "target_lang",
        "generation_max_length",
        "num_beams",
    ):
        if key in train and train.get(key) is not None and train.get(key) != contract.get(key):
            raise RuntimeError(f"NB05 train/contract mismatch on {key}")
        if key in evaluate and evaluate.get(key) is not None and evaluate.get(key) != contract.get(key):
            raise RuntimeError(f"NB05 evaluate/contract mismatch on {key}")
    return {
        "experiment_id": exp,
        "contract_hash": contract_hash,
        "best_checkpoint_name": best,
        "training_contract": dict(contract),
        "encoder_id": contract.get("encoder_id"),
        "encoder_revision": contract.get("encoder_revision"),
        "decoder_id": contract.get("decoder_id"),
        "decoder_revision": contract.get("decoder_revision"),
        "target_lang": contract.get("target_lang"),
        "generation_max_length": contract.get("generation_max_length"),
        "num_beams": contract.get("num_beams"),
    }


def verify_upstream_handoffs(
    *,
    asr_state_dir: Union[str, Path],
    mt_state_dir: Union[str, Path],
    direct_state_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Verify only summary/contract metadata; MUST NOT open frozen test."""
    asr = Path(asr_state_dir)
    mt = Path(mt_state_dir)
    direct = Path(direct_state_dir)
    asr_eval = _read_json(asr / "full_evaluate_summary.json")
    asr_train = _read_json(asr / "full_train_summary.json")
    mt_eval = _read_json(mt / "mt_evaluate_summary.json")
    mt_train = _read_json(mt / "mt_train_summary.json")
    mt_contract = _read_json(mt / "mt_training_contract.json")
    direct_eval = _read_json(direct / "direct_evaluate_summary.json")
    direct_train = _read_json(direct / "direct_train_summary.json")
    direct_contract = _read_json(direct / "direct_training_contract.json")

    asr_id = _assert_asr_identity(asr_train, asr_eval)
    mt_id = _assert_mt_identity(mt_train, mt_eval, mt_contract)
    direct_id = _assert_direct_identity(direct_train, direct_eval, direct_contract)

    checks = {
        "asr_evaluate_success": True,
        "mt_evaluate_success": True,
        "direct_evaluate_success": True,
        "asr_identity_ok": True,
        "mt_identity_ok": True,
        "direct_identity_ok": True,
        "asr_no_frozen_test": True,
        "mt_no_frozen_test": True,
        "direct_no_frozen_test": True,
    }
    handoff = {
        "status": STATUS_RQ1_VERIFY,
        "checks": checks,
        "asr": {
            "state_dir": str(asr),
            "evaluate": asr_eval,
            "train": asr_train,
            "identity": asr_id,
        },
        "mt": {
            "state_dir": str(mt),
            "evaluate": mt_eval,
            "train": mt_train,
            "training_contract": mt_contract,
            "identity": mt_id,
        },
        "direct": {
            "state_dir": str(direct),
            "evaluate": direct_eval,
            "train": direct_train,
            "training_contract": direct_contract,
            "identity": direct_id,
        },
    }
    handoff["asr_handoff_hash"] = sha256_json(handoff["asr"])
    handoff["mt_handoff_hash"] = sha256_json(handoff["mt"])
    handoff["direct_handoff_hash"] = sha256_json(handoff["direct"])
    return handoff


def unlock_frozen_test_manifest(
    *,
    project_root: Union[str, Path],
    upstream_handoff: Mapping[str, Any],
    allow_frozen_test_access: bool,
    dataset_id: str,
    dataset_revision: str,
    parquet_revision: str,
    expected_test_count: Optional[int] = 215,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """First legitimate full-content read of rq1_test.csv. Fails closed."""
    if upstream_handoff.get("status") != STATUS_RQ1_VERIFY:
        raise RuntimeError("Frozen test cannot be unlocked before SUCCESS_RQ1_VERIFY")
    if not allow_frozen_test_access:
        raise RuntimeError("Set ALLOW_FROZEN_TEST_ACCESS=True explicitly for stage unlock_test")
    root = Path(project_root)
    man = root / "data" / "manifests"
    summary_path = man / "split_summary.json"
    test_path = man / "rq1_test.csv"
    train_path = man / "rq1_train.csv"
    val_path = man / "rq1_validation.csv"
    summary = _read_json(summary_path)
    recorded_dataset = str(summary.get("dataset_id") or summary.get("dataset_name") or "").strip()
    if not recorded_dataset:
        raise RuntimeError("split_summary.json missing dataset_id/dataset_name")
    if recorded_dataset != str(dataset_id):
        raise RuntimeError(f"Frozen-test dataset id mismatch: {recorded_dataset} != {dataset_id}")
    recorded_revision = str(summary.get("dataset_commit_sha") or summary.get("dataset_commit_sha_after") or "").strip()
    if not recorded_revision:
        raise RuntimeError("split_summary.json missing dataset_commit_sha")
    if recorded_revision != str(dataset_revision):
        raise RuntimeError(f"Frozen-test dataset revision mismatch: {recorded_revision} != {dataset_revision}")
    parquet_pin = resolve_parquet_revision_pin(summary, locked_parquet_revision=str(parquet_revision))

    manifest_pins = summary.get("manifest_sha256") or {}
    for name, path in (
        ("rq1_test.csv", test_path),
        ("rq1_train.csv", train_path),
        ("rq1_validation.csv", val_path),
    ):
        expected_sha = str(manifest_pins.get(name) or "")
        if len(expected_sha) != 64:
            raise RuntimeError(f"split_summary.json does not pin {name} SHA256")
        if not path.is_file():
            raise RuntimeError(f"Missing locked manifest: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{name} SHA256 mismatch: {actual_sha} != {expected_sha}")

    frame = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    validate_frozen_test_frame(frame)
    if expected_test_count is not None and len(frame) != int(expected_test_count):
        raise RuntimeError(f"Frozen-test count mismatch: {len(frame)} != {expected_test_count}")
    overlap = assert_no_split_key_overlap(test_df=frame, train_df=train_df, validation_df=val_df)
    contract = build_rq1_test_contract(
        frame,
        manifest_path=test_path,
        split_summary_sha256=sha256_file(summary_path),
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        parquet_revision=parquet_pin["parquet_revision"],
        parquet_revision_source=parquet_pin["parquet_revision_source"],
        train_manifest_sha256=str(manifest_pins["rq1_train.csv"]),
        validation_manifest_sha256=str(manifest_pins["rq1_validation.csv"]),
        split_integrity=overlap,
    )
    return frame, {
        "status": STATUS_RQ1_UNLOCK,
        "contract": contract,
        "manifest_path": str(test_path),
        "overlap": overlap,
        "parquet_revision": parquet_pin["parquet_revision"],
        "parquet_revision_source": parquet_pin["parquet_revision_source"],
    }


def join_paired_predictions(test_frame: pd.DataFrame, c0: pd.DataFrame, d0: pd.DataFrame) -> pd.DataFrame:
    assert_prediction_frame(c0, test_frame, system="C0", pred_col="c0_pred_vi")
    assert_prediction_frame(d0, test_frame, system="D0", pred_col="d0_pred_vi")
    assert_group_id_complete(test_frame)
    base = test_frame.copy().reset_index(drop=True)
    base["reference_bahnar"] = base["text_bahnar"].map(normalize_bahnar_ctc_v1)
    base["reference_vi"] = base["text_vi"].map(normalize_mt_text_v1)
    keep = [
        c
        for c in [
            "record_uid",
            "source_split",
            "duration_seconds",
            "group_id",
            "reference_bahnar",
            "reference_vi",
        ]
        if c in base.columns
    ]
    out = base[keep].copy()
    c0_idx = c0.set_index("record_uid", drop=False)
    d0_idx = d0.set_index("record_uid", drop=False)
    out["asr_pred_bahnar"] = [str(c0_idx.loc[u, "asr_pred_bahnar"]) for u in out["record_uid"]]
    out["c0_pred_vi"] = [normalize_mt_text_v1(c0_idx.loc[u, "c0_pred_vi"]) for u in out["record_uid"]]
    out["d0_pred_vi"] = [normalize_mt_text_v1(d0_idx.loc[u, "d0_pred_vi"]) for u in out["record_uid"]]
    return out


def system_metrics(paired: pd.DataFrame) -> Dict[str, Any]:
    refs = paired["reference_vi"].astype(str).tolist()
    c0 = mt_corpus_metrics(paired["c0_pred_vi"].astype(str).tolist(), refs)
    d0 = mt_corpus_metrics(paired["d0_pred_vi"].astype(str).tolist(), refs)
    return {
        "c0": c0,
        "d0": d0,
        "delta": {
            "sacrebleu": float(d0["sacrebleu"] - c0["sacrebleu"]),
            "chrfpp": float(d0["chrfpp"] - c0["chrfpp"]),
        },
    }


def paired_bootstrap(
    paired: pd.DataFrame,
    *,
    n_samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
    cluster_col: str = BOOTSTRAP_UNIT_GROUP_ID,
) -> Dict[str, Any]:
    """
    Deterministic paired *cluster* bootstrap by ``group_id``.

    Samples unique clusters with replacement, includes all rows of each selected
    cluster, and keeps C0/D0/reference pairing. Never silently falls back to
    row bootstrap when ``group_id`` exists.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    n = len(paired)
    if n == 0:
        raise ValueError("paired frame is empty")
    if cluster_col != BOOTSTRAP_UNIT_GROUP_ID:
        raise RuntimeError(f"RQ1 bootstrap cluster_col must be {BOOTSTRAP_UNIT_GROUP_ID}")
    n_clusters = assert_group_id_complete(paired, cluster_col=cluster_col)
    work = paired.reset_index(drop=True)
    groups = work[cluster_col].astype(str).map(lambda x: x.strip())
    unique_groups = sorted(groups.unique().tolist())
    if len(unique_groups) != n_clusters:
        raise RuntimeError("cluster accounting mismatch")
    # Pre-index rows belonging to each cluster (stable row order within cluster).
    cluster_indices: Dict[str, np.ndarray] = {
        g: work.index[groups.to_numpy() == g].to_numpy(dtype=int) for g in unique_groups
    }
    rng = np.random.default_rng(int(seed))
    refs_all = work["reference_vi"].astype(str).to_numpy()
    c0_all = work["c0_pred_vi"].astype(str).to_numpy()
    d0_all = work["d0_pred_vi"].astype(str).to_numpy()
    db: List[float] = []
    dc: List[float] = []
    for _ in range(int(n_samples)):
        chosen = rng.choice(unique_groups, size=n_clusters, replace=True)
        idx_parts = [cluster_indices[g] for g in chosen]
        idx = np.concatenate(idx_parts) if idx_parts else np.asarray([], dtype=int)
        refs = refs_all[idx].tolist()
        c0 = c0_all[idx].tolist()
        d0 = d0_all[idx].tolist()
        mc0 = mt_corpus_metrics(c0, refs)
        md0 = mt_corpus_metrics(d0, refs)
        db.append(float(md0["sacrebleu"] - mc0["sacrebleu"]))
        dc.append(float(md0["chrfpp"] - mc0["chrfpp"]))
    alpha = (1.0 - float(confidence)) / 2.0

    def _ci(x: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(x, dtype=float)
        return {
            "mean": float(a.mean()),
            "lower": float(np.quantile(a, alpha)),
            "upper": float(np.quantile(a, 1 - alpha)),
        }

    return {
        "bootstrap_method": BOOTSTRAP_METHOD_PAIRED_CLUSTER,
        "bootstrap_unit": cluster_col,
        "cluster_col": cluster_col,
        "n_clusters": int(n_clusters),
        "n_rows": int(n),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "confidence": float(confidence),
        "delta_sacrebleu": _ci(db),
        "delta_chrfpp": _ci(dc),
    }


def slice_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    work = paired.copy()
    if "duration_seconds" in work.columns:
        work["duration_slice"] = pd.cut(
            work["duration_seconds"].astype(float),
            bins=[-np.inf, 10, 20, 30, np.inf],
            labels=["<=10s", "10-20s", "20-30s", ">30s"],
        )
    work["target_chars"] = work["reference_vi"].astype(str).map(len)
    work["target_length_slice"] = pd.qcut(
        work["target_chars"],
        q=min(3, max(1, work["target_chars"].nunique())),
        labels=False,
        duplicates="drop",
    )
    for dimension in [c for c in ("duration_slice", "target_length_slice", "source_split") if c in work.columns]:
        for value, grp in work.groupby(dimension, dropna=False, observed=False):
            if len(grp) == 0:
                continue
            m = system_metrics(grp)
            rows.append(
                {
                    "dimension": dimension,
                    "slice": str(value),
                    "n": int(len(grp)),
                    "c0_sacrebleu": m["c0"]["sacrebleu"],
                    "d0_sacrebleu": m["d0"]["sacrebleu"],
                    "delta_sacrebleu": m["delta"]["sacrebleu"],
                    "c0_chrfpp": m["c0"]["chrfpp"],
                    "d0_chrfpp": m["d0"]["chrfpp"],
                    "delta_chrfpp": m["delta"]["chrfpp"],
                }
            )
    return pd.DataFrame(rows)


def chr_f_sentence_score(hyp: str, ref: str) -> float:
    import sacrebleu

    return float(sacrebleu.sentence_chrf(str(hyp), [str(ref)], word_order=2).score)


def error_examples(paired: pd.DataFrame, *, top_k: int = 20) -> pd.DataFrame:
    out = paired.copy()
    out["c0_sent_chrfpp"] = [chr_f_sentence_score(h, r) for h, r in zip(out["c0_pred_vi"], out["reference_vi"])]
    out["d0_sent_chrfpp"] = [chr_f_sentence_score(h, r) for h, r in zip(out["d0_pred_vi"], out["reference_vi"])]
    out["delta_sent_chrfpp"] = out["d0_sent_chrfpp"] - out["c0_sent_chrfpp"]
    best_d0 = out.nlargest(int(top_k), "delta_sent_chrfpp").assign(example_group="d0_better")
    best_c0 = out.nsmallest(int(top_k), "delta_sent_chrfpp").assign(example_group="c0_better")
    return pd.concat([best_d0, best_c0], ignore_index=True)


def prediction_artifact_manifest(paths: Mapping[str, Union[str, Path]]) -> Dict[str, Any]:
    files = {}
    for name, p in paths.items():
        path = Path(p)
        if not path.is_file():
            raise RuntimeError(f"Missing final artifact {name}: {path}")
        files[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"files": files, "manifest_hash": sha256_json(files)}


def recompute_and_assert_finalize_integrity(
    *,
    integrity_path: Union[str, Path],
    c0_summary: Mapping[str, Any],
    d0_summary: Mapping[str, Any],
) -> str:
    """Rehash current integrity file and require equality with both stage summaries."""
    current = sha256_file(integrity_path)
    c0_hash = str(c0_summary.get("audio_integrity_sha256") or "")
    d0_hash = str(d0_summary.get("audio_integrity_sha256") or "")
    if not c0_hash or not d0_hash:
        raise RuntimeError("C0/D0 summaries missing audio_integrity_sha256")
    if c0_hash != d0_hash:
        raise RuntimeError("C0/D0 audio integrity hashes disagree")
    if current != c0_hash:
        raise RuntimeError(
            f"finalize integrity rehash mismatch vs C0/D0: current={current} staged={c0_hash}"
        )
    return current
