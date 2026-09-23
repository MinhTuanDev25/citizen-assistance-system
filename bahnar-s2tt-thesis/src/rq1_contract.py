"""Fail-closed contracts for Notebook 06 final frozen RQ1 evaluation."""
from __future__ import annotations

import hashlib
import importlib.metadata as im
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import pandas as pd

from src.data_utils import normalize_bahnar_ctc_v1
from src.mt_normalize import normalize_mt_text_v1

STATUS_RQ1_VERIFY = "SUCCESS_RQ1_VERIFY"
STATUS_RQ1_UNLOCK = "SUCCESS_RQ1_UNLOCK"
STATUS_RQ1_C0 = "SUCCESS_RQ1_C0"
STATUS_RQ1_D0 = "SUCCESS_RQ1_D0"
STATUS_RQ1_FINAL = "SUCCESS_RQ1_FINAL"
STATUS_FAILED = "FAILED"

RQ1_CONTRACT_VERSION = "rq1_final_contract_v1"
PARQUET_SOURCE_SPLIT_SUMMARY = "split_summary"
PARQUET_SOURCE_RQ1_CONFIG_LEGACY = "rq1_config_locked_legacy_summary"
BOOTSTRAP_METHOD_PAIRED_CLUSTER = "paired_cluster"
BOOTSTRAP_UNIT_GROUP_ID = "group_id"

REQUIRED_TEST_COLUMNS = (
    "record_uid", "record_id", "source_split", "parquet_file", "shard_row_index",
    "text_bahnar", "text_vi", "split", "group_id",
)
OVERLAP_KEYS = (
    "record_uid",
    "record_id",
    "audio_key",
    "recording_group_id",
    "group_id",
    "pair_key",
)


def _canon(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    """Write JSON via temp file + os.replace so crashes never leave a valid partial."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    payload = json.dumps(obj, ensure_ascii=False, indent=indent) + "\n"
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(str(tmp), str(out))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write CSV via temp file + os.replace."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(str(tmp), str(out))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def uid_set_hash(frame: pd.DataFrame) -> str:
    vals = sorted(frame["record_uid"].astype(str).tolist())
    return sha256_json(vals)


def ordered_uid_hash(frame: pd.DataFrame) -> str:
    return sha256_json(frame["record_uid"].astype(str).tolist())


def reference_content_hash(frame: pd.DataFrame) -> str:
    rows = []
    for _, r in frame.iterrows():
        rows.append({
            "record_uid": str(r["record_uid"]),
            "text_bahnar_norm": normalize_bahnar_ctc_v1(r["text_bahnar"]),
            "text_vi_norm": normalize_mt_text_v1(r["text_vi"]),
        })
    return sha256_json(rows)


def ordered_row_hash(frame: pd.DataFrame) -> str:
    cols = [c for c in (
        "record_uid", "record_id", "source_split", "parquet_file", "shard_row_index",
        "group_id", "recording_group_id", "pair_key", "text_bahnar", "text_vi", "split",
    ) if c in frame.columns]
    rows = []
    for _, r in frame.iterrows():
        row = {}
        for c in cols:
            v = r[c]
            if pd.isna(v):
                v = None
            elif c == "text_bahnar":
                v = normalize_bahnar_ctc_v1(v)
            elif c == "text_vi":
                v = normalize_mt_text_v1(v)
            elif hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            row[c] = v
        rows.append(row)
    return sha256_json(rows)


def assert_group_id_complete(frame: pd.DataFrame, *, cluster_col: str = BOOTSTRAP_UNIT_GROUP_ID) -> int:
    if cluster_col not in frame.columns:
        raise RuntimeError(f"Frozen RQ1 test missing required cluster column {cluster_col}")
    vals = frame[cluster_col]
    if vals.isna().any():
        raise RuntimeError(f"Frozen RQ1 test has null {cluster_col}")
    as_str = vals.astype(str).map(lambda x: x.strip())
    if as_str.eq("").any() or as_str.str.lower().isin({"nan", "none", "null"}).any():
        raise RuntimeError(f"Frozen RQ1 test has empty {cluster_col}")
    return int(as_str.nunique())


def validate_frozen_test_frame(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TEST_COLUMNS if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Frozen RQ1 test missing required columns: {missing}")
    if len(frame) == 0:
        raise RuntimeError("Frozen RQ1 test is empty")
    if frame["record_uid"].astype(str).duplicated().any():
        raise RuntimeError("Frozen RQ1 test has duplicate record_uid")
    split_values = {str(x).strip().lower() for x in frame["split"].tolist()}
    if split_values != {"test"}:
        raise RuntimeError(f"Frozen RQ1 split column must be exactly test, got {sorted(split_values)}")
    if frame["text_vi"].fillna("").astype(str).map(normalize_mt_text_v1).eq("").any():
        raise RuntimeError("Frozen RQ1 test contains empty Vietnamese reference")
    if frame["text_bahnar"].fillna("").astype(str).map(normalize_bahnar_ctc_v1).eq("").any():
        raise RuntimeError("Frozen RQ1 test contains empty Bahnar reference")
    assert_group_id_complete(frame)


def assert_no_split_key_overlap(
    *,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"checked_keys": [], "overlaps": {}}
    for key in OVERLAP_KEYS:
        if key not in test_df.columns:
            continue
        evidence["checked_keys"].append(key)
        test_vals = {str(x) for x in test_df[key].dropna().astype(str).tolist() if str(x)}
        for split_name, other in (("train", train_df), ("validation", validation_df)):
            if key not in other.columns:
                continue
            other_vals = {str(x) for x in other[key].dropna().astype(str).tolist() if str(x)}
            hit = sorted(test_vals & other_vals)
            if hit:
                raise RuntimeError(f"Frozen test overlaps {split_name} on {key}: {hit[:5]}")
            evidence["overlaps"][f"{key}:{split_name}"] = 0
    return evidence


def resolve_parquet_revision_pin(
    summary: Mapping[str, Any],
    *,
    locked_parquet_revision: str,
) -> Dict[str, str]:
    """
    Fail-closed parquet provenance.

    Prefer split_summary pin when present; otherwise allow the explicit locked
    rq1.yaml revision as a legacy provenance pin. Never invent a Hub HEAD revision.
    """
    locked = str(locked_parquet_revision or "").strip()
    if len(locked) < 7:
        raise RuntimeError("locked parquet_revision from rq1.yaml is missing/invalid")
    recorded = str(summary.get("parquet_revision") or summary.get("parquet_commit_sha") or "").strip()
    if recorded:
        if recorded != locked:
            raise RuntimeError(
                f"Frozen-test parquet revision mismatch: split_summary={recorded} locked={locked}"
            )
        return {"parquet_revision": recorded, "parquet_revision_source": PARQUET_SOURCE_SPLIT_SUMMARY}
    return {
        "parquet_revision": locked,
        "parquet_revision_source": PARQUET_SOURCE_RQ1_CONFIG_LEGACY,
    }


def build_rq1_test_contract(
    frame: pd.DataFrame,
    *,
    manifest_path: Union[str, Path],
    split_summary_sha256: str,
    dataset_id: str,
    dataset_revision: str,
    parquet_revision: str,
    parquet_revision_source: str,
    train_manifest_sha256: Optional[str] = None,
    validation_manifest_sha256: Optional[str] = None,
    split_integrity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    validate_frozen_test_frame(frame)
    n_clusters = assert_group_id_complete(frame)
    src = str(parquet_revision_source or "")
    if src not in {PARQUET_SOURCE_SPLIT_SUMMARY, PARQUET_SOURCE_RQ1_CONFIG_LEGACY}:
        raise RuntimeError(f"Invalid parquet_revision_source: {src!r}")
    p = Path(manifest_path)
    payload = {
        "contract_version": RQ1_CONTRACT_VERSION,
        "dataset_id": str(dataset_id),
        "dataset_revision": str(dataset_revision),
        "parquet_revision": str(parquet_revision),
        "parquet_revision_source": src,
        "test_count": int(len(frame)),
        "n_clusters": int(n_clusters),
        "cluster_col": BOOTSTRAP_UNIT_GROUP_ID,
        "uid_set_hash": uid_set_hash(frame),
        "ordered_uid_hash": ordered_uid_hash(frame),
        "reference_content_hash": reference_content_hash(frame),
        "ordered_row_hash": ordered_row_hash(frame),
        "manifest_sha256": sha256_file(p),
        "split_summary_sha256": str(split_summary_sha256),
        "train_manifest_sha256": str(train_manifest_sha256 or ""),
        "validation_manifest_sha256": str(validation_manifest_sha256 or ""),
        "split_integrity": dict(split_integrity or {}),
    }
    payload["rq1_test_contract_hash"] = sha256_json(payload)
    return payload


def assert_rq1_test_contract(frame: pd.DataFrame, contract: Mapping[str, Any], *, manifest_path: Union[str, Path]) -> None:
    expected = build_rq1_test_contract(
        frame,
        manifest_path=manifest_path,
        split_summary_sha256=str(contract["split_summary_sha256"]),
        dataset_id=str(contract["dataset_id"]),
        dataset_revision=str(contract["dataset_revision"]),
        parquet_revision=str(contract["parquet_revision"]),
        parquet_revision_source=str(contract.get("parquet_revision_source") or ""),
        train_manifest_sha256=str(contract.get("train_manifest_sha256") or ""),
        validation_manifest_sha256=str(contract.get("validation_manifest_sha256") or ""),
        split_integrity=contract.get("split_integrity") or {},
    )
    if dict(expected) != dict(contract):
        raise RuntimeError("Frozen RQ1 test contract mismatch — refuse evaluation")


def assert_locked_manifest_bundle_matches_contract(
    *,
    project_root: Union[str, Path],
    test_contract: Mapping[str, Any],
) -> Dict[str, str]:
    """Rehash the entire locked manifest bundle against the unlocked test contract."""
    man = Path(project_root) / "data" / "manifests"
    checks = {
        "split_summary.json": (man / "split_summary.json", str(test_contract.get("split_summary_sha256") or "")),
        "rq1_test.csv": (man / "rq1_test.csv", str(test_contract.get("manifest_sha256") or "")),
        "rq1_train.csv": (man / "rq1_train.csv", str(test_contract.get("train_manifest_sha256") or "")),
        "rq1_validation.csv": (man / "rq1_validation.csv", str(test_contract.get("validation_manifest_sha256") or "")),
    }
    actual: Dict[str, str] = {}
    for name, (path, expected) in checks.items():
        if len(expected) != 64:
            raise RuntimeError(f"rq1_test_contract missing SHA pin for {name}")
        if not path.is_file():
            raise RuntimeError(f"Missing locked manifest bundle file: {path}")
        got = sha256_file(path)
        actual[name] = got
        if got != expected:
            raise RuntimeError(f"Locked manifest bundle mutated after unlock: {name}")
    return actual


def build_final_contract(
    *,
    test_contract: Mapping[str, Any],
    asr_handoff_hash: str,
    mt_handoff_hash: str,
    direct_handoff_hash: str,
    source_fingerprint_sha256: str,
    runtime_versions: Mapping[str, str],
    seed: int,
    bootstrap_samples: int,
    confidence: float = 0.95,
    bootstrap_method: str = BOOTSTRAP_METHOD_PAIRED_CLUSTER,
    bootstrap_unit: str = BOOTSTRAP_UNIT_GROUP_ID,
    cluster_col: str = BOOTSTRAP_UNIT_GROUP_ID,
    checkpoint_proof: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    proof = dict(checkpoint_proof or {})
    if bootstrap_method != BOOTSTRAP_METHOD_PAIRED_CLUSTER:
        raise RuntimeError(f"Unsupported bootstrap_method: {bootstrap_method}")
    if bootstrap_unit != BOOTSTRAP_UNIT_GROUP_ID or cluster_col != BOOTSTRAP_UNIT_GROUP_ID:
        raise RuntimeError("RQ1 bootstrap must use group_id clusters")
    payload = {
        "contract_version": "rq1_final_experiment_v1",
        "rq1_test_contract_hash": str(test_contract["rq1_test_contract_hash"]),
        "asr_handoff_hash": str(asr_handoff_hash),
        "mt_handoff_hash": str(mt_handoff_hash),
        "direct_handoff_hash": str(direct_handoff_hash),
        "source_fingerprint_sha256": str(source_fingerprint_sha256),
        "runtime_versions": dict(runtime_versions),
        "seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
        "confidence": float(confidence),
        "bootstrap_method": str(bootstrap_method),
        "bootstrap_unit": str(bootstrap_unit),
        "cluster_col": str(cluster_col),
        "checkpoint_proof": {
            "asr": proof.get("asr") or {},
            "mt": proof.get("mt") or {},
            "direct": proof.get("direct") or {},
            "proof_hash": proof.get("proof_hash"),
        },
    }
    payload["rq1_final_contract_hash"] = sha256_json(payload)
    return payload


def assert_unlocked_final_contract_matches_current(
    final_contract: Mapping[str, Any],
    *,
    test_contract: Mapping[str, Any],
    asr_handoff_hash: str,
    mt_handoff_hash: str,
    direct_handoff_hash: str,
    source_fingerprint_sha256: str,
    runtime_versions: Mapping[str, str],
    seed: int,
    bootstrap_samples: int,
    confidence: float,
    checkpoint_proof: Mapping[str, Any],
    latest_meta: Optional[Mapping[str, Any]] = None,
    bootstrap_method: str = BOOTSTRAP_METHOD_PAIRED_CLUSTER,
    bootstrap_unit: str = BOOTSTRAP_UNIT_GROUP_ID,
    cluster_col: str = BOOTSTRAP_UNIT_GROUP_ID,
) -> Dict[str, Any]:
    expected = build_final_contract(
        test_contract=test_contract,
        asr_handoff_hash=asr_handoff_hash,
        mt_handoff_hash=mt_handoff_hash,
        direct_handoff_hash=direct_handoff_hash,
        source_fingerprint_sha256=source_fingerprint_sha256,
        runtime_versions=runtime_versions,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_method=bootstrap_method,
        bootstrap_unit=bootstrap_unit,
        cluster_col=cluster_col,
        checkpoint_proof=checkpoint_proof,
    )
    if dict(expected) != dict(final_contract):
        raise RuntimeError(
            "Unlocked rq1_final_contract.json does not match current upstream/"
            "checkpoint proof/runtime — refuse post-unlock inference"
        )
    if latest_meta is not None:
        got = str(latest_meta.get("final_contract_hash") or "")
        if got != expected["rq1_final_contract_hash"]:
            raise RuntimeError(
                "LATEST_UNLOCKED.final_contract_hash mismatch vs rebuilt final contract"
            )
        if str(latest_meta.get("source_fingerprint") or "") != str(source_fingerprint_sha256):
            raise RuntimeError("LATEST_UNLOCKED source fingerprint mismatch")
    return expected


def assert_prediction_frame(frame: pd.DataFrame, test_frame: pd.DataFrame, *, system: str, pred_col: str) -> None:
    req = {"record_uid", pred_col}
    if not req.issubset(frame.columns):
        raise RuntimeError(f"{system} predictions missing columns: {sorted(req - set(frame.columns))}")
    if frame["record_uid"].astype(str).duplicated().any():
        raise RuntimeError(f"{system} predictions contain duplicate UID")
    expected = test_frame["record_uid"].astype(str).tolist()
    got = frame["record_uid"].astype(str).tolist()
    if got != expected:
        raise RuntimeError(f"{system} predictions do not match exact frozen-test UID order")
    if frame[pred_col].isna().any():
        raise RuntimeError(f"{system} predictions contain null text")


def derive_final_status(
    *,
    test_contract_ok: bool,
    c0_ok: bool,
    d0_ok: bool,
    paired_uid_order_ok: bool,
    references_identical: bool,
    metrics_finite: bool,
    artifacts_hashed: bool,
    bootstrap_ok: bool,
) -> Dict[str, Any]:
    checks = {
        "test_contract_ok": bool(test_contract_ok),
        "c0_ok": bool(c0_ok),
        "d0_ok": bool(d0_ok),
        "paired_uid_order_ok": bool(paired_uid_order_ok),
        "references_identical": bool(references_identical),
        "metrics_finite": bool(metrics_finite),
        "artifacts_hashed": bool(artifacts_hashed),
        "bootstrap_ok": bool(bootstrap_ok),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    return {"status": STATUS_RQ1_FINAL if not failed else STATUS_FAILED, "checks": checks, "failed_checks": failed}


def assert_rq1_bootstrap_gate(
    bootstrap: Mapping[str, Any],
    *,
    final_contract: Mapping[str, Any],
    test_contract: Mapping[str, Any],
) -> None:
    """Fail-closed: cluster-bootstrap artifact must match unlocked scientific contract."""
    want_method = str(final_contract.get("bootstrap_method") or BOOTSTRAP_METHOD_PAIRED_CLUSTER)
    want_unit = str(final_contract.get("bootstrap_unit") or BOOTSTRAP_UNIT_GROUP_ID)
    want_cluster = str(final_contract.get("cluster_col") or BOOTSTRAP_UNIT_GROUP_ID)
    if str(bootstrap.get("bootstrap_method") or "") != want_method:
        raise RuntimeError(f"bootstrap_method mismatch: {bootstrap.get('bootstrap_method')!r} != {want_method!r}")
    if str(bootstrap.get("bootstrap_unit") or "") != want_unit:
        raise RuntimeError(f"bootstrap_unit mismatch: {bootstrap.get('bootstrap_unit')!r} != {want_unit!r}")
    if str(bootstrap.get("cluster_col") or "") != want_cluster:
        raise RuntimeError(f"bootstrap cluster_col mismatch: {bootstrap.get('cluster_col')!r} != {want_cluster!r}")
    if int(bootstrap.get("n_rows") or -1) != int(test_contract.get("test_count") or -2):
        raise RuntimeError(
            f"bootstrap n_rows mismatch: {bootstrap.get('n_rows')} != test_count {test_contract.get('test_count')}"
        )
    if int(bootstrap.get("n_clusters") or -1) != int(test_contract.get("n_clusters") or -2):
        raise RuntimeError(
            f"bootstrap n_clusters mismatch: {bootstrap.get('n_clusters')} != {test_contract.get('n_clusters')}"
        )
    if int(bootstrap.get("n_samples") or -1) != int(final_contract.get("bootstrap_samples") or -2):
        raise RuntimeError("bootstrap n_samples mismatch vs final contract")
    if int(bootstrap.get("seed") or -1) != int(final_contract.get("seed") or -2):
        raise RuntimeError("bootstrap seed mismatch vs final contract")
    if abs(float(bootstrap.get("confidence") or -1) - float(final_contract.get("confidence") or -2)) > 1e-12:
        raise RuntimeError("bootstrap confidence mismatch vs final contract")
    for key in ("delta_sacrebleu", "delta_chrfpp"):
        ci = bootstrap.get(key) or {}
        try:
            mean = float(ci["mean"])
            lower = float(ci["lower"])
            upper = float(ci["upper"])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"bootstrap {key} missing finite mean/lower/upper") from exc
        if not (math.isfinite(mean) and math.isfinite(lower) and math.isfinite(upper)):
            raise RuntimeError(f"bootstrap {key} contains non-finite CI values")
        if lower > upper:
            raise RuntimeError(f"bootstrap {key} invalid CI ordering: lower={lower} > upper={upper}")


def finite_metric_dict(obj: Mapping[str, Any], keys: Sequence[str]) -> bool:
    for k in keys:
        try:
            if not math.isfinite(float(obj[k])):
                return False
        except Exception:
            return False
    return True


def _pkg_version(name: str) -> str:
    try:
        return str(im.version(name))
    except Exception:
        return ""


def collect_runtime_provenance() -> Dict[str, str]:
    out = {
        "python": platform.python_version(),
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "accelerate": _pkg_version("accelerate"),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "pyarrow": _pkg_version("pyarrow"),
        "soundfile": _pkg_version("soundfile"),
        "tokenizers": _pkg_version("tokenizers"),
        "sacrebleu": _pkg_version("sacrebleu"),
        "libsndfile": "",
    }
    try:
        import soundfile as sf

        out["libsndfile"] = str(getattr(sf, "__libsndfile_version__", "") or getattr(sf, "LIBSNDFILE_VERSION", "") or "")
        if not out["libsndfile"]:
            out["libsndfile"] = str(getattr(getattr(sf, "_soundfile", None), "__version__", "") or "present")
    except Exception as exc:  # noqa: BLE001
        out["libsndfile"] = f"unavailable:{type(exc).__name__}"
    missing = [k for k, v in out.items() if k != "libsndfile" and not v]
    if missing:
        raise RuntimeError(f"RQ1 runtime provenance missing packages: {missing}")
    return out


RQ1_SOURCE_FILES = (
    "notebooks/06_rq1_evaluation.ipynb",
    "configs/rq1.yaml",
    "requirements.txt",
    "src/rq1_runtime_paths.py", "src/rq1_contract.py", "src/rq1_evaluation.py",
    "src/rq1_inference.py", "src/rq1_restore.py", "src/rq1_audio.py",
    "src/asr_full_data.py", "src/asr_full_pcm.py", "src/asr_full_shards.py",
    "src/asr_full_train.py", "src/asr_runtime_paths.py", "src/asr_utils.py",
    "src/audio_utils.py", "src/data_utils.py",
    "src/direct_contract.py", "src/direct_data.py", "src/direct_full_train.py",
    "src/direct_model.py",
    "src/metrics.py",
    "src/mt_contract.py", "src/mt_dataset.py", "src/mt_full_train.py",
    "src/mt_normalize.py", "src/mt_prepare.py", "src/mt_runtime_paths.py",
    "src/mt_tokenize.py",
)


def compute_rq1_source_fingerprint(project_root: Union[str, Path]) -> Dict[str, Any]:
    root = Path(project_root)
    files: Dict[str, str] = {}
    missing = []
    for rel in RQ1_SOURCE_FILES:
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif rel.endswith(".ipynb"):
            nb = json.loads(p.read_text(encoding="utf-8"))
            cells = []
            for cell in nb.get("cells", []):
                tags = set((cell.get("metadata") or {}).get("tags") or [])
                if "rq1-operator-controls" in tags:
                    continue
                clean = {
                    "cell_type": cell.get("cell_type"),
                    "metadata": {k: v for k, v in (cell.get("metadata") or {}).items() if k != "execution"},
                    "source": cell.get("source") or [],
                }
                cells.append(clean)
            files[rel] = sha256_json({"cells": cells})
        else:
            files[rel] = sha256_file(p)
    if missing:
        raise RuntimeError(f"RQ1 source fingerprint missing files: {missing}")
    return {"files": files, "aggregate_sha256": sha256_json(files)}



def assert_prediction_uid_order(frame: Any, expected_uids: Sequence[str], *, label: str) -> None:
    """Fail-closed exact ordered UID equality against the locked frozen test."""
    if "record_uid" not in getattr(frame, "columns", []):
        raise RuntimeError(f"{label} predictions missing record_uid")
    got = frame["record_uid"].astype(str).tolist()
    want = [str(u) for u in expected_uids]
    if got != want:
        raise RuntimeError(f"{label} UID order mismatch vs locked frozen test")


def assert_stage_summary_matches_final_contract(
    summary: Mapping[str, Any],
    *,
    final_contract: Mapping[str, Any],
    test_contract: Mapping[str, Any],
    expected_status: str,
    label: str,
) -> None:
    """Refuse stale C0/D0 summaries from another unlock/contract."""
    if summary.get("status") != expected_status:
        raise RuntimeError(f"{label} summary status is not {expected_status}")
    got_hash = str(summary.get("rq1_final_contract_hash") or "")
    want_hash = str(final_contract.get("rq1_final_contract_hash") or "")
    if not want_hash or got_hash != want_hash:
        raise RuntimeError(f"{label} summary rq1_final_contract_hash mismatch vs current FINAL_CONTRACT")
    got_n = int(summary.get("n") or -1)
    want_n = int(test_contract.get("test_count") or -2)
    if got_n != want_n:
        raise RuntimeError(f"{label} summary n={got_n} != test_count={want_n}")

def assert_locked_runtime(locked: Mapping[str, str], actual: Mapping[str, str]) -> None:
    def base(v: str) -> str:
        return str(v or "").split("+")[0]

    bad = {}
    for k in ("torch", "transformers", "accelerate"):
        if base(actual.get(k, "")) != base(locked.get(k, "")):
            bad[k] = {"actual": actual.get(k), "expected": locked.get(k)}
    if bad:
        raise RuntimeError(f"RQ1 runtime version mismatch: {bad}")
