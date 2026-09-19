"""Notebook 04 prepare: locked-text QA, exclusions, eligible manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.asr_full_data import drop_train_pair_key_overlaps
from src.asr_full_train import assert_no_frozen_test_access
from src.asr_utils import is_forbidden_test_path
from src.data_utils import (
    build_clean_split,
    compute_ordered_uid_hash,
    compute_uid_set_hash,
    sha256_file,
)
from src.mt_contract import (
    STATUS_MT_PREPARE,
    STAGE_VERSION_PREPARE,
    assert_model_revision_pinned,
    build_mt_data_contract,
)
from src.mt_normalize import normalize_mt_text_v1, mt_normalization_version
from src.mt_runtime_paths import SOURCE_FIELD, TARGET_FIELD


def compute_mt_manifest_content_hash(df: pd.DataFrame) -> str:
    """Hash UID + source/target text (no audio columns required)."""
    cols = [c for c in ("record_uid", SOURCE_FIELD, TARGET_FIELD, "pair_key") if c in df.columns]
    if "record_uid" not in cols or SOURCE_FIELD not in cols or TARGET_FIELD not in cols:
        raise RuntimeError(f"MT manifest hash missing required columns; have {list(df.columns)}")
    work = df.loc[:, cols].copy()
    for c in cols:
        work[c] = work[c].astype(str)
    work = work.sort_values(cols, kind="mergesort")
    blob = json.dumps(
        {"columns": cols, "n_rows": int(len(work)), "rows": work.to_numpy().tolist()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_notebook02_locked_contract(
    *,
    contract_path: Union[str, Path],
    train_manifest: Union[str, Path],
    validation_manifest: Union[str, Path],
    train_exclusion: Union[str, Path],
    validation_exclusion: Union[str, Path],
    expected_dataset_revision: str,
    opened_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fail-closed verification of Notebook 02 locked hashes/counts (no G_test open)."""
    contract_path = Path(contract_path)
    train_manifest = Path(train_manifest)
    validation_manifest = Path(validation_manifest)
    train_exclusion = Path(train_exclusion)
    validation_exclusion = Path(validation_exclusion)
    for p in (contract_path, train_manifest, validation_manifest, train_exclusion, validation_exclusion):
        if is_forbidden_test_path(p):
            raise RuntimeError(f"Frozen-test path refused: {p}")
        if opened_paths is not None:
            opened_paths.append(str(p.resolve()))
        if not p.is_file():
            raise RuntimeError(f"Missing locked artifact: {p}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    errors: List[str] = []
    if contract.get("dataset_revision") != expected_dataset_revision:
        errors.append(
            f"dataset_revision mismatch: {contract.get('dataset_revision')} != {expected_dataset_revision}"
        )

    base = contract.get("base_manifest_sha256") or {}
    for name, path in (("rq1_train.csv", train_manifest), ("rq1_validation.csv", validation_manifest)):
        got = sha256_file(path)
        exp = base.get(name)
        if got != exp:
            errors.append(f"{name} SHA256 mismatch: got={got} expected={exp}")

    train_c = contract.get("train") or {}
    val_c = contract.get("validation") or {}
    if sha256_file(train_exclusion) != train_c.get("exclusion_csv_sha256"):
        errors.append("train exclusion CSV SHA256 mismatch")
    if sha256_file(validation_exclusion) != val_c.get("exclusion_csv_sha256"):
        errors.append("validation exclusion CSV SHA256 mismatch")

    train_raw = pd.read_csv(train_manifest)
    val_raw = pd.read_csv(validation_manifest)
    train_clean = build_clean_split(train_raw, pd.read_csv(train_exclusion))
    val_clean = build_clean_split(val_raw, pd.read_csv(validation_exclusion))

    if int(len(train_clean)) != int(train_c.get("clean_count", -1)):
        errors.append(
            f"train clean_count mismatch: {len(train_clean)} != {train_c.get('clean_count')}"
        )
    if int(len(val_clean)) != int(val_c.get("clean_count", -1)):
        errors.append(
            f"validation clean_count mismatch: {len(val_clean)} != {val_c.get('clean_count')}"
        )

    train_ordered = compute_ordered_uid_hash(train_clean)
    val_ordered = compute_ordered_uid_hash(val_clean)
    train_set = compute_uid_set_hash(train_clean)
    val_set = compute_uid_set_hash(val_clean)
    if train_ordered != train_c.get("ordered_uid_sha256"):
        errors.append("train ordered UID SHA256 mismatch")
    if val_ordered != val_c.get("ordered_uid_sha256"):
        errors.append("validation ordered UID SHA256 mismatch")
    if train_set != train_c.get("uid_set_sha256"):
        errors.append("train UID-set SHA256 mismatch")
    if val_set != val_c.get("uid_set_sha256"):
        errors.append("validation UID-set SHA256 mismatch")

    if errors:
        raise RuntimeError("Notebook02 locked-contract verification FAILED: " + "; ".join(errors))

    return {
        "passed": True,
        "dataset_revision": expected_dataset_revision,
        "train_clean_count": int(len(train_clean)),
        "validation_clean_count": int(len(val_clean)),
        "train_ordered_uid_sha256": train_ordered,
        "validation_ordered_uid_sha256": val_ordered,
        "train_uid_set_sha256": train_set,
        "validation_uid_set_sha256": val_set,
    }


REQUIRED_MT_COLUMNS = (
    "record_uid",
    SOURCE_FIELD,
    TARGET_FIELD,
)


def load_locked_mt_frames(
    *,
    train_manifest: Union[str, Path],
    validation_manifest: Union[str, Path],
    train_exclusion: Optional[Union[str, Path]] = None,
    validation_exclusion: Optional[Union[str, Path]] = None,
    opened_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load locked manifests, apply NB02 contamination exclusions if provided."""
    paths = [Path(train_manifest), Path(validation_manifest)]
    if train_exclusion:
        paths.append(Path(train_exclusion))
    if validation_exclusion:
        paths.append(Path(validation_exclusion))
    for p in paths:
        if is_forbidden_test_path(p):
            raise RuntimeError(f"Frozen-test path refused: {p}")
        if opened_paths is not None:
            opened_paths.append(str(p.resolve()))

    train_raw = pd.read_csv(train_manifest)
    val_raw = pd.read_csv(validation_manifest)
    for name, df in (("train", train_raw), ("validation", val_raw)):
        missing = [c for c in REQUIRED_MT_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{name} manifest missing columns: {missing}")

    if train_exclusion and Path(train_exclusion).is_file():
        train_df = build_clean_split(train_raw, pd.read_csv(train_exclusion))
    else:
        train_df = train_raw.copy()
    if validation_exclusion and Path(validation_exclusion).is_file():
        val_df = build_clean_split(val_raw, pd.read_csv(validation_exclusion))
    else:
        val_df = val_raw.copy()

    return {
        "train_raw": train_raw,
        "validation_raw": val_raw,
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "train_manifest_content_hash": compute_mt_manifest_content_hash(train_df),
        "validation_manifest_content_hash": compute_mt_manifest_content_hash(val_df),
    }


def _empty_row_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda x: normalize_mt_text_v1(x) == "")


def build_mt_exclusions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    source_col: str = SOURCE_FIELD,
    target_col: str = TARGET_FIELD,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Deterministic MT exclusions with row-accurate accounting:

        eligible_rows + excluded_rows == input_rows  (per split)
    """
    excl_rows: List[Dict[str, Any]] = []

    def _mark_exclude(df: pd.DataFrame, split: str, mask: pd.Series, reason: str) -> pd.Series:
        """Return boolean mask of newly excluded rows; append exclusion records."""
        new_mask = mask.fillna(False).astype(bool)
        for idx in df.index[new_mask]:
            r = df.loc[idx]
            excl_rows.append(
                {
                    "record_uid": str(r["record_uid"]),
                    "split": split,
                    "reason": reason,
                    "source_norm": normalize_mt_text_v1(r.get(source_col)),
                    "target_norm": normalize_mt_text_v1(r.get(target_col)),
                    "row_index": int(idx) if isinstance(idx, (int,)) else str(idx),
                }
            )
        return new_mask

    train_drop = pd.Series(False, index=train_df.index)
    val_drop = pd.Series(False, index=val_df.index)

    train_drop |= _mark_exclude(train_df, "train", _empty_row_mask(train_df[source_col]) & ~train_drop, "empty_source_after_norm")
    train_drop |= _mark_exclude(train_df, "train", _empty_row_mask(train_df[target_col]) & ~train_drop, "empty_target_after_norm")
    val_drop |= _mark_exclude(val_df, "validation", _empty_row_mask(val_df[source_col]) & ~val_drop, "empty_source_after_norm")
    val_drop |= _mark_exclude(val_df, "validation", _empty_row_mask(val_df[target_col]) & ~val_drop, "empty_target_after_norm")

    # Duplicate UIDs: keep first row, exclude subsequent rows (row-accurate).
    for split, df, drop in (("train", train_df, train_drop), ("validation", val_df, val_drop)):
        dup = df["record_uid"].astype(str).duplicated(keep="first") & ~drop
        if split == "train":
            train_drop |= _mark_exclude(df, split, dup, "duplicate_record_uid")
        else:
            val_drop |= _mark_exclude(df, split, dup, "duplicate_record_uid")

    train_elig = train_df.loc[~train_drop].copy()
    val_elig = val_df.loc[~val_drop].copy()

    # Pair-key policy: drop overlapping rows from train only
    train_elig, pair_excl, pair_rep = drop_train_pair_key_overlaps(train_elig, val_elig)
    if len(pair_excl):
        for _, r in pair_excl.iterrows():
            excl_rows.append(
                {
                    "record_uid": str(r["record_uid"]),
                    "split": "train",
                    "reason": str(r.get("reason") or "train_validation_pair_key_overlap"),
                    "source_norm": normalize_mt_text_v1(r.get(source_col)),
                    "target_norm": normalize_mt_text_v1(r.get(target_col)),
                    "row_index": "",
                }
            )

    train_uids = set(train_elig["record_uid"].astype(str))
    val_uids = set(val_elig["record_uid"].astype(str))
    uid_overlap = train_uids & val_uids
    if uid_overlap:
        raise RuntimeError(f"train/validation record_uid overlap after MT prepare: {len(uid_overlap)}")

    def _col_overlap(col: str) -> int:
        if col not in train_elig.columns or col not in val_elig.columns:
            return 0
        a = set(train_elig[col].dropna().astype(str)) - {""}
        b = set(val_elig[col].dropna().astype(str)) - {""}
        return len(a & b)

    for col in ("group_id", "recording_group_id"):
        n = _col_overlap(col)
        if n:
            raise RuntimeError(f"train/validation {col} overlap after MT prepare: {n}")

    for df in (train_elig, val_elig):
        df[f"{source_col}_norm"] = df[source_col].map(normalize_mt_text_v1)
        df[f"{target_col}_norm"] = df[target_col].map(normalize_mt_text_v1)

    train_elig = train_elig.reset_index(drop=True)
    val_elig = val_elig.reset_index(drop=True)

    excl = pd.DataFrame(excl_rows)
    if len(excl) == 0:
        excl = pd.DataFrame(columns=["record_uid", "split", "reason", "source_norm", "target_norm", "row_index"])

    train_excluded_rows = int(len(train_df) - len(train_elig))
    val_excluded_rows = int(len(val_df) - len(val_elig))
    report = {
        "normalization_version": mt_normalization_version(),
        "train_input": int(len(train_df)),
        "validation_input": int(len(val_df)),
        "train_eligible": int(len(train_elig)),
        "validation_eligible": int(len(val_elig)),
        "train_excluded_rows": train_excluded_rows,
        "validation_excluded_rows": val_excluded_rows,
        # Back-compat aliases (row counts, not unique UIDs)
        "train_excluded_uids": train_excluded_rows,
        "validation_excluded_uids": val_excluded_rows,
        "exclusions_rows": int(len(excl)),
        "pair_key_report": pair_rep,
        "accounting_train_ok": int(len(train_elig)) + train_excluded_rows == int(len(train_df)),
        "accounting_validation_ok": int(len(val_elig)) + val_excluded_rows == int(len(val_df)),
        "source_len_stats": _length_stats(
            train_elig[f"{source_col}_norm"].tolist() + val_elig[f"{source_col}_norm"].tolist()
        ),
        "target_len_stats": _length_stats(
            train_elig[f"{target_col}_norm"].tolist() + val_elig[f"{target_col}_norm"].tolist()
        ),
    }
    if not report["accounting_train_ok"] or not report["accounting_validation_ok"]:
        raise RuntimeError(f"MT exclusion accounting failed: {report}")
    return train_elig, val_elig, excl.reset_index(drop=True), report


def _length_stats(texts: Sequence[str]) -> Dict[str, float]:
    lengths = [len(t) for t in texts]
    if not lengths:
        return {"n": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "n": len(lengths),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "mean": float(sum(lengths) / len(lengths)),
    }


def length_ratio_outliers(
    df: pd.DataFrame,
    *,
    source_col: str = f"{SOURCE_FIELD}_norm",
    target_col: str = f"{TARGET_FIELD}_norm",
    min_ratio: float = 0.05,
    max_ratio: float = 20.0,
) -> pd.DataFrame:
    """Report-only abnormal length ratios (does not exclude)."""
    rows = []
    for _, r in df.iterrows():
        s = len(str(r.get(source_col) or ""))
        t = len(str(r.get(target_col) or ""))
        if s == 0:
            continue
        ratio = t / s
        if ratio < min_ratio or ratio > max_ratio:
            rows.append(
                {
                    "record_uid": str(r["record_uid"]),
                    "source_len": s,
                    "target_len": t,
                    "ratio": ratio,
                }
            )
    return pd.DataFrame(rows)


def persist_mt_prepare_artifacts(
    state_dir: Union[str, Path],
    *,
    train_eligible: pd.DataFrame,
    val_eligible: pd.DataFrame,
    exclusions: pd.DataFrame,
    summary: Dict[str, Any],
    contract: Dict[str, Any],
) -> Dict[str, Path]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mt_train_eligible": state_dir / "mt_train_eligible.csv",
        "mt_validation_eligible": state_dir / "mt_validation_eligible.csv",
        "mt_data_exclusions": state_dir / "mt_data_exclusions.csv",
        "mt_data_summary": state_dir / "mt_data_summary.json",
        "mt_contract": state_dir / "mt_contract.json",
        "mt_prepare_state": state_dir / "mt_prepare_state.json",
    }
    train_eligible.to_csv(paths["mt_train_eligible"], index=False)
    val_eligible.to_csv(paths["mt_validation_eligible"], index=False)
    exclusions.to_csv(paths["mt_data_exclusions"], index=False)
    paths["mt_data_summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["mt_contract"].write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    hashed_keys = (
        "mt_train_eligible",
        "mt_validation_eligible",
        "mt_data_exclusions",
        "mt_data_summary",
        "mt_contract",
    )
    artifact_meta = {}
    for k in hashed_keys:
        p = paths[k]
        artifact_meta[k] = {
            "path": p.name,
            "sha256": sha256_file(p),
            "size_bytes": int(p.stat().st_size),
        }
    state = {
        "status": STATUS_MT_PREPARE,
        "stage_version": STAGE_VERSION_PREPARE,
        "contract_hash": contract.get("contract_hash"),
        "train_eligible_count": int(len(train_eligible)),
        "validation_eligible_count": int(len(val_eligible)),
        "exclusions_count": int(len(exclusions)),
        "artifact_sha256": {k: artifact_meta[k]["sha256"] for k in hashed_keys},
        "artifact_meta": artifact_meta,
    }
    paths["mt_prepare_state"].write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def load_mt_prepare_success(
    state_dir: Union[str, Path],
    *,
    expected_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state_dir = Path(state_dir)
    path = state_dir / "mt_prepare_state.json"
    if not path.is_file():
        raise RuntimeError(f"Missing MT prepare state: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != STATUS_MT_PREPARE:
        raise RuntimeError(f"MT prepare not successful: {payload.get('status')}")

    required = {
        "mt_train_eligible": state_dir / "mt_train_eligible.csv",
        "mt_validation_eligible": state_dir / "mt_validation_eligible.csv",
        "mt_data_exclusions": state_dir / "mt_data_exclusions.csv",
        "mt_data_summary": state_dir / "mt_data_summary.json",
        "mt_contract": state_dir / "mt_contract.json",
    }
    expected_sha = dict(payload.get("artifact_sha256") or {})
    expected_meta = dict(payload.get("artifact_meta") or {})
    for key, fpath in required.items():
        if not fpath.is_file():
            raise RuntimeError(f"Prepared artifact missing: {fpath}")
        got = sha256_file(fpath)
        exp = expected_sha.get(key) or (expected_meta.get(key) or {}).get("sha256")
        if not exp:
            raise RuntimeError(f"Prepare state missing SHA256 for {key}")
        if got != exp:
            raise RuntimeError(f"Prepared artifact SHA256 mismatch for {key}: got={got} expected={exp}")
        meta = expected_meta.get(key) or {}
        if "size_bytes" in meta and int(fpath.stat().st_size) != int(meta["size_bytes"]):
            raise RuntimeError(
                f"Prepared artifact size mismatch for {key}: "
                f"{fpath.stat().st_size} != {meta['size_bytes']}"
            )

    contract = json.loads(required["mt_contract"].read_text(encoding="utf-8"))
    if expected_contract is not None:
        if contract.get("contract_hash") != expected_contract.get("contract_hash"):
            raise RuntimeError(
                "MT prepare contract_hash mismatch: "
                f"{contract.get('contract_hash')} != {expected_contract.get('contract_hash')}"
            )

    train_n = len(pd.read_csv(required["mt_train_eligible"]))
    val_n = len(pd.read_csv(required["mt_validation_eligible"]))
    if int(payload.get("train_eligible_count", -1)) != int(train_n):
        raise RuntimeError(
            f"train_eligible_count mismatch: state={payload.get('train_eligible_count')} file={train_n}"
        )
    if int(payload.get("validation_eligible_count", -1)) != int(val_n):
        raise RuntimeError(
            f"validation_eligible_count mismatch: state={payload.get('validation_eligible_count')} file={val_n}"
        )

    assert_no_frozen_test_access([required["mt_train_eligible"]], ["train", "validation"])
    return {"state": payload, "contract": contract}


def run_mt_prepare(
    *,
    state_dir: Union[str, Path],
    train_manifest: Union[str, Path],
    validation_manifest: Union[str, Path],
    train_exclusion: Optional[Union[str, Path]],
    validation_exclusion: Optional[Union[str, Path]],
    dataset_id: str,
    dataset_revision: str,
    model_id: str,
    model_revision: str,
    tokenizer_fingerprint: str,
    max_source_length: int,
    max_target_length: int,
    opened_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    assert_model_revision_pinned(model_id, model_revision)
    loaded = load_locked_mt_frames(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        train_exclusion=train_exclusion,
        validation_exclusion=validation_exclusion,
        opened_paths=opened_paths,
    )
    train_elig, val_elig, excl, report = build_mt_exclusions(loaded["train"], loaded["validation"])
    outliers = length_ratio_outliers(pd.concat([train_elig, val_elig], ignore_index=True))
    report["length_ratio_outlier_count"] = int(len(outliers))
    report["train_uid_set_hash"] = compute_uid_set_hash(train_elig)
    report["validation_uid_set_hash"] = compute_uid_set_hash(val_elig)
    report["train_ordered_uid_hash"] = compute_ordered_uid_hash(train_elig)
    report["validation_ordered_uid_hash"] = compute_ordered_uid_hash(val_elig)

    contract = build_mt_data_contract(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        train_manifest_content_hash=loaded["train_manifest_content_hash"],
        validation_manifest_content_hash=loaded["validation_manifest_content_hash"],
        train_uid_set_hash=report["train_uid_set_hash"],
        validation_uid_set_hash=report["validation_uid_set_hash"],
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_fingerprint=tokenizer_fingerprint,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    paths = persist_mt_prepare_artifacts(
        state_dir,
        train_eligible=train_elig,
        val_eligible=val_elig,
        exclusions=excl,
        summary=report,
        contract=contract,
    )
    return {
        "status": STATUS_MT_PREPARE,
        "contract": contract,
        "summary": report,
        "paths": {k: str(v) for k, v in paths.items()},
        "train_eligible": train_elig,
        "validation_eligible": val_elig,
        "exclusions": excl,
    }
