"""RQ2 Notebook 14 training-data identity and equal-budget verification.

D0 uses the frozen RQ1 Direct supervised G_train only.
D-Random / D-Quality concatenate that same supervised base with the frozen
NB13 arm manifest. Selection rows are not re-scored, re-ranked, trimmed,
cut, shuffled, or rebalanced. Targets for pseudo-labeled rows are the
frozen ``pseudo_vi_norm`` strings from NB12/NB13.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from src.direct_data import compute_ordered_row_hash, compute_pair_hash
from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash
from src.mt_normalize import normalize_mt_text_v1
from src.rq1_contract import ordered_uid_hash, sha256_file, sha256_json, uid_set_hash
from src.rq2_final_contract import (
    ARM_D0,
    ARM_QUALITY,
    ARM_RANDOM,
    ArmIsolationError,
    EvaluationError,
    Rq2FinalError,
    SUPERVISED_BASE_POLICY,
    UpstreamGateError,
    assert_g_test_blocked,
    is_sha256,
    resolve_nb14_layout,
)
from src.rq2_selection import read_manifest_csv
from src.rq2_selection_contract import (
    ARM_QUALITY as SEL_QUALITY,
    ARM_RANDOM as SEL_RANDOM,
    DURATION_COLUMN,
    PSEUDO_LABEL_COLUMN,
    SELECTION_RELATIVE_DIR,
    UID_COLUMN,
)

EXAMPLE_KIND_SUPERVISED = "supervised_g_train"
EXAMPLE_KIND_PSEUDO = "pseudo_nb13"
COMPONENT_ORDER = (EXAMPLE_KIND_SUPERVISED, EXAMPLE_KIND_PSEUDO)


def _stable(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def ordered_example_uid_hash(uids: Sequence[str]) -> str:
    return sha256_json([str(uid) for uid in uids])


def pair_identity_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {"uid": str(row["example_uid"]), "text": str(row["target_text_norm"])}
        for row in rows
    ]
    return hashlib.sha256(_stable(payload).encode("utf-8")).hexdigest()


def uid_set_identity_hash(uids: Sequence[str]) -> str:
    return sha256_json(sorted(str(uid) for uid in uids))


def duration_sum(rows: Sequence[Mapping[str, Any]]) -> float:
    return round(sum(float(row.get("duration_seconds") or 0.0) for row in rows), 6)


def supervised_row(raw: Mapping[str, Any]) -> Dict[str, Any]:
    uid = str(raw.get("record_uid") or "")
    if not uid:
        raise Rq2FinalError("supervised row missing record_uid")
    text = normalize_mt_text_v1(raw.get("text_vi_norm") or raw.get("text_vi"))
    if not text:
        raise Rq2FinalError(f"supervised row {uid} has empty text_vi_norm")
    split = str(raw.get("split") or "train")
    if split.lower() in {"test", "g_test", "frozen_test"}:
        raise EvaluationError(f"supervised row {uid} belongs to the frozen test split")
    row = {
        "example_uid": uid,
        "kind": EXAMPLE_KIND_SUPERVISED,
        "split": "train",
        "source_split": str(raw.get("source_split") or "g_train"),
        "group_id": str(raw.get("group_id") or ""),
        "target_text_norm": text,
        "duration_seconds": float(raw.get("duration_seconds") or 0.0),
        "audio_sha256": str(raw.get("pcm16_sha256") or raw.get("sha256_pcm") or raw.get("source_sha256") or ""),
        "source_path": str(raw.get("wav_local_path") or raw.get("audio_locator") or raw.get("local_cache_relpath") or ""),
        "pseudo_vi_norm": "",
        "segment_uid": "",
        "selection_arm": "",
        "nb12_contract_sha256": "",
        "nb13_selection_contract_sha256": "",
    }
    return row


def pseudo_row(
    raw: Mapping[str, Any],
    *,
    arm: str,
    selection_contract_sha256: str,
    nb12_contract_sha256: str,
) -> Dict[str, Any]:
    uid = str(raw.get(UID_COLUMN) or "")
    if not uid:
        raise Rq2FinalError("pseudo-labeled row missing segment_uid")
    if str(raw.get("selection_arm") or "") != arm:
        raise ArmIsolationError(f"manifest arm {raw.get('selection_arm')!r} cannot be used as {arm}")
    text = str(raw.get(PSEUDO_LABEL_COLUMN) or "").strip()
    if not text:
        raise Rq2FinalError(f"pseudo-labeled row {uid} has empty pseudo_vi_norm")
    path = str(raw.get("segment_local_path") or "")
    if path:
        assert_g_test_blocked(path, allow=False)
    return {
        "example_uid": uid,
        "kind": EXAMPLE_KIND_PSEUDO,
        "split": "train",
        "source_split": "u_prime",
        "group_id": str(raw.get("source_group_id") or ""),
        "target_text_norm": text,
        "duration_seconds": float(raw.get(DURATION_COLUMN) or 0.0),
        "audio_sha256": str(raw.get("segment_pcm16_sha256") or ""),
        "source_path": path,
        "pseudo_vi_norm": text,
        "segment_uid": uid,
        "selection_arm": arm,
        "nb12_contract_sha256": str(nb12_contract_sha256),
        "quality_score_contract_sha256": str(raw.get("quality_score_contract_sha256") or ""),
        "nb13_selection_contract_sha256": str(selection_contract_sha256),
        "segment_wav_sha256": str(raw.get("segment_wav_sha256") or ""),
        "selection_rank": int(raw.get("selection_rank") or 0),
        "selection_key": str(raw.get("selection_key") or ""),
    }


def component_report(rows: Sequence[Mapping[str, Any]], *, kind: str) -> Dict[str, Any]:
    chosen = [row for row in rows if row["kind"] == kind]
    uids = [row["example_uid"] for row in chosen]
    audio_pairs = []
    for row in chosen:
        digest = str(row.get("audio_sha256") or "").strip().lower()
        if kind == EXAMPLE_KIND_SUPERVISED:
            if not is_sha256(digest):
                raise UpstreamGateError(
                    f"supervised audio identity for {row['example_uid']} is missing or invalid "
                    "(do not hash an empty list)"
                )
            audio_pairs.append({"uid": str(row["example_uid"]), "audio": digest})
        elif digest:
            audio_pairs.append({"uid": str(row["example_uid"]), "audio": digest})
    return {
        "kind": kind,
        "n_rows": len(chosen),
        "duration_seconds": duration_sum(chosen),
        "ordered_uid_hash": ordered_example_uid_hash(uids),
        "uid_set_hash": uid_set_identity_hash(uids),
        "pair_hash": pair_identity_hash(chosen),
        "audio_identity_hash": sha256_json(audio_pairs),
    }


def compose_arm_training_rows(
    *,
    arm: str,
    supervised_rows: Sequence[Mapping[str, Any]],
    pseudo_rows: Sequence[Mapping[str, Any]],
    selection_contract_sha256: str,
    nb12_contract_sha256: str,
) -> Dict[str, Any]:
    if arm not in (ARM_D0, ARM_RANDOM, ARM_QUALITY):
        raise Rq2FinalError(f"unknown arm {arm!r}")
    supervised = [supervised_row(row) for row in supervised_rows]
    if arm == ARM_D0:
        if pseudo_rows:
            raise Rq2FinalError("D0 training composition is supervised G_train only")
        mixed = list(supervised)
    else:
        wanted = ARM_RANDOM if arm == ARM_RANDOM else ARM_QUALITY
        mixed = list(supervised) + [
            pseudo_row(
                row,
                arm=wanted,
                selection_contract_sha256=selection_contract_sha256,
                nb12_contract_sha256=nb12_contract_sha256,
            )
            for row in pseudo_rows
        ]
    seen = set()
    for row in mixed:
        uid = row["example_uid"]
        if uid in seen:
            raise Rq2FinalError(f"duplicate example_uid inside {arm}: {uid}")
        seen.add(uid)
        if row["kind"] == EXAMPLE_KIND_PSEUDO:
            if str(row.get("nb13_selection_contract_sha256") or "") != selection_contract_sha256:
                raise ArmIsolationError(f"{uid} is not bound to the frozen NB13 contract")
            if str(row.get("nb12_contract_sha256") or "") != str(nb12_contract_sha256):
                raise ArmIsolationError(f"{uid} is not bound to the frozen NB12 contract")
    uids = [row["example_uid"] for row in mixed]
    report = {
        "arm": arm,
        "supervised_base_policy": SUPERVISED_BASE_POLICY,
        "n_rows": len(mixed),
        "duration_seconds": duration_sum(mixed),
        "ordered_training_uid_hash": ordered_example_uid_hash(uids),
        "training_pair_hash": pair_identity_hash(mixed),
        "uid_set_hash": uid_set_identity_hash(uids),
        "supervised": component_report(mixed, kind=EXAMPLE_KIND_SUPERVISED),
        "pseudo": component_report(mixed, kind=EXAMPLE_KIND_PSEUDO),
        "rows": mixed,
    }
    return report


def load_nb13_arm_manifest(
    project_root: Union[str, Path],
    *,
    arm: str,
    generation_id: str,
    expected_sha256: Optional[str] = None,
    artifact_root: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    if arm not in (ARM_RANDOM, ARM_QUALITY):
        raise ArmIsolationError(f"{arm} has no NB13 selection manifest")
    if not str(generation_id or "").strip():
        raise UpstreamGateError("NB13 generation_id is required; CURRENT is not re-read")
    root = Path(artifact_root or project_root)
    out = root / SELECTION_RELATIVE_DIR
    name = "d_random_manifest.csv" if arm == ARM_RANDOM else "d_quality_manifest.csv"
    path = assert_g_test_blocked(out / "generations" / str(generation_id) / name, allow=False)
    if not path.is_file():
        raise UpstreamGateError(f"pinned NB13 generation is missing {name}")
    got = sha256_file(path)
    if expected_sha256 and str(expected_sha256).lower() != got:
        raise UpstreamGateError(f"{name} SHA256 does not match the upstream-pinned hash")
    rows = read_manifest_csv(path)
    for row in rows:
        if str(row.get("selection_arm") or "") != arm:
            raise ArmIsolationError(f"{name} contains {row.get('selection_arm')!r}, not {arm}")
    return rows


def load_pinned_nb13_arm_manifest(
    project_root: Union[str, Path],
    *,
    arm: str,
    upstream: Mapping[str, Any],
    artifact_root: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    key = "d_random_manifest_sha256" if arm == ARM_RANDOM else "d_quality_manifest_sha256"
    return load_nb13_arm_manifest(
        project_root,
        arm=arm,
        generation_id=str(upstream["nb13_generation_id"]),
        expected_sha256=str(upstream[key]),
        artifact_root=artifact_root,
    )


def verify_pinned_nb13_selection(
    project_root: Union[str, Path],
    *,
    generation_id: str,
    artifact_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Re-verify one frozen NB13 generation. Never re-reads CURRENT."""
    from src.rq2_selection import verify_published_selection

    if not str(generation_id or "").strip():
        raise UpstreamGateError("NB13 generation_id is required; CURRENT is not re-read")
    root = Path(project_root)
    layout = resolve_nb14_layout(root, durable_root=artifact_root or project_root)
    return verify_published_selection(
        layout["selection_dir"],
        project_root=root,
        generation_id=str(generation_id),
        durable_root=layout["durable_root"],
        pseudo_dir=layout["pseudo_dir"],
        u_clean_dir=layout["u_clean_dir"],
    )


def verify_equal_budget_from_nb13(selection: Mapping[str, Any]) -> Dict[str, Any]:
    """Reuse NB13 target/realized budget semantics. No new tolerance is invented."""
    contract = selection.get("contract") or selection
    summary = selection.get("summary") or {}
    target = float(contract["selection_budget_seconds"])
    hours = float(contract["selection_budget_hours"])
    if abs(target - hours * 3600.0) > 0.0:
        raise UpstreamGateError("NB13 selection_budget_seconds does not match hours*3600")
    d_random = summary.get("d_random") or {}
    d_quality = summary.get("d_quality") or {}
    for arm_name, report in ((SEL_RANDOM, d_random), (SEL_QUALITY, d_quality)):
        if float(report["target_budget_seconds"]) != target:
            raise UpstreamGateError(f"{arm_name} target_budget_seconds drifted from the NB13 contract")
        if float(report["selected_duration_seconds"]) > target:
            raise UpstreamGateError(f"{arm_name} realized duration exceeds the NB13 target")
    return {
        "target_budget_hours": hours,
        "target_budget_seconds": target,
        "d_random_realized_seconds": float(d_random["selected_duration_seconds"]),
        "d_quality_realized_seconds": float(d_quality["selected_duration_seconds"]),
        "d_random_unused_seconds": float(d_random["unused_budget_seconds"]),
        "d_quality_unused_seconds": float(d_quality["unused_budget_seconds"]),
    }


def validation_identity(frame: pd.DataFrame) -> Dict[str, Any]:
    if "record_uid" not in frame.columns:
        raise Rq2FinalError("validation frame missing record_uid")
    work = frame.copy()
    if "text_vi_norm" not in work.columns:
        work["text_vi_norm"] = work["text_vi"].map(normalize_mt_text_v1)
    splits = {str(x).strip().lower() for x in work.get("split", pd.Series(["validation"] * len(work))).astype(str)}
    if splits & {"test", "g_test", "frozen_test"}:
        raise EvaluationError("validation frame contains frozen-test rows")
    uids = work["record_uid"].astype(str).tolist()
    audio_col = None
    for candidate in ("pcm16_sha256", "sha256_pcm", "source_sha256"):
        if candidate in work.columns:
            audio_col = candidate
            break
    if audio_col is None:
        raise UpstreamGateError(
            "validation audio identity is missing; pcm16_sha256/sha256_pcm is required "
            "(do not hash an empty list)"
        )
    audio_pairs = []
    seen = {}
    for uid, sha in zip(work["record_uid"].astype(str), work[audio_col].astype(str)):
        digest = str(sha or "").strip().lower()
        if not is_sha256(digest):
            raise UpstreamGateError(f"validation audio identity for {uid} is not a sha256")
        if uid in seen and seen[uid] != digest:
            raise UpstreamGateError(f"duplicate conflicting validation audio identity for {uid}")
        seen[uid] = digest
        audio_pairs.append({"uid": uid, "audio": digest})
    return {
        "n_rows": int(len(work)),
        "ordered_uid_hash": ordered_uid_hash(work),
        "uid_set_hash": uid_set_hash(work),
        "pair_hash": compute_pair_hash(work),
        "ordered_row_hash": compute_ordered_row_hash(work),
        "audio_pair_hash": sha256_json(audio_pairs),
    }


def assert_no_test_contamination(
    train_uids: Sequence[str],
    test_uids: Sequence[str],
) -> None:
    overlap = sorted(set(map(str, train_uids)) & set(map(str, test_uids)))
    if overlap:
        raise EvaluationError(f"train/test UID contamination: {overlap[:5]}")


def write_arm_manifest_csv(path: Union[str, Path], rows: Sequence[Mapping[str, Any]]) -> str:
    from src.rq2_pseudo_contract import atomic_write_text
    import csv
    import io

    columns = [
        "example_uid", "kind", "split", "source_split", "group_id",
        "target_text_norm", "duration_seconds", "audio_sha256", "source_path",
        "pseudo_vi_norm", "segment_uid", "selection_arm",
        "nb12_contract_sha256", "quality_score_contract_sha256", "nb13_selection_contract_sha256",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, buffer.getvalue())
    return sha256_file(path)


def data_contract_payload(
    composition: Mapping[str, Any],
    *,
    manifest_sha256: str,
    validation: Mapping[str, Any],
    upstream: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "arm": composition["arm"],
        "supervised_base_policy": SUPERVISED_BASE_POLICY,
        "data_manifest_sha256": manifest_sha256,
        "ordered_training_uid_hash": composition["ordered_training_uid_hash"],
        "training_pair_hash": composition["training_pair_hash"],
        "uid_set_hash": composition["uid_set_hash"],
        "supervised_ordered_uid_hash": composition["supervised"]["ordered_uid_hash"],
        "supervised_pair_hash": composition["supervised"]["pair_hash"],
        "supervised_uid_set_hash": composition["supervised"]["uid_set_hash"],
        "supervised_audio_identity_hash": composition["supervised"]["audio_identity_hash"],
        "supervised_n_rows": composition["supervised"]["n_rows"],
        "supervised_duration_seconds": composition["supervised"]["duration_seconds"],
        "pseudo_ordered_uid_hash": composition["pseudo"]["ordered_uid_hash"],
        "pseudo_pair_hash": composition["pseudo"]["pair_hash"],
        "pseudo_n_rows": composition["pseudo"]["n_rows"],
        "pseudo_duration_seconds": composition["pseudo"]["duration_seconds"],
        "total_n_rows": composition["n_rows"],
        "total_duration_seconds": composition["duration_seconds"],
        "validation_ordered_uid_hash": validation["ordered_uid_hash"],
        "validation_pair_hash": validation["pair_hash"],
        "validation_uid_set_hash": validation["uid_set_hash"],
        "validation_audio_identity_hash": validation.get("audio_pair_hash") or "",
        "nb11_input_contract_sha256": upstream["nb11_input_contract_sha256"],
        "nb12_contract_sha256": upstream["nb12_contract_sha256"],
        "nb13_selection_contract_sha256": upstream["nb13_selection_contract_sha256"],
        "selection_budget_hours": budget["target_budget_hours"],
        "selection_budget_seconds": budget["target_budget_seconds"],
        "realized_pseudo_duration_seconds": composition["pseudo"]["duration_seconds"],
    }
    payload["data_contract_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "data_contract_sha256"})
    return payload


RQ1_SPLIT_FILES = {
    "g_train": "rq1_train.csv",
    "g_validation": "rq1_validation.csv",
}


def load_rq1_split_csv(project_root: Union[str, Path], *, split: str) -> pd.DataFrame:
    """Load a frozen RQ1 train/validation CSV. G_test is refused here."""
    if split in {"g_test", "test", "frozen_test"}:
        raise EvaluationError("G_test cannot be loaded as training or validation data")
    name = RQ1_SPLIT_FILES.get(split)
    if not name:
        raise Rq2FinalError(f"unknown RQ1 split {split!r}")
    path = Path(project_root) / "data" / "manifests" / name
    path = assert_g_test_blocked(path, allow=False)
    if not path.is_file():
        raise UpstreamGateError(f"missing RQ1 split file {name}")
    frame = pd.read_csv(path)
    if "record_uid" not in frame.columns:
        raise Rq2FinalError(f"{name} missing record_uid")
    splits = {str(x).strip().lower() for x in frame.get("split", pd.Series(dtype=str)).astype(str)}
    if splits & {"test", "g_test", "frozen_test"}:
        raise EvaluationError(f"{name} contains frozen-test rows")
    return frame


def split_identity(frame: pd.DataFrame, *, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    work = frame.copy()
    if "text_vi_norm" not in work.columns:
        if "text_vi" not in work.columns:
            raise Rq2FinalError("supervised frame missing text_vi / text_vi_norm")
        work["text_vi_norm"] = work["text_vi"].map(normalize_mt_text_v1)
    audio_col = "pcm16_sha256" if "pcm16_sha256" in work.columns else ("source_sha256" if "source_sha256" in work.columns else None)
    audio_pairs = []
    if audio_col:
        audio_pairs = [
            {"uid": str(uid), "audio": str(sha)}
            for uid, sha in zip(work["record_uid"].astype(str), work[audio_col].astype(str))
        ]
    payload = {
        "n_rows": int(len(work)),
        "ordered_uid_hash": ordered_uid_hash(work),
        "uid_set_hash": uid_set_hash(work),
        "d0_uid_set_hash": compute_uid_set_hash(work),
        "d0_ordered_uid_hash": compute_ordered_uid_hash(work),
        "pair_hash": compute_pair_hash(work),
        "ordered_row_hash": compute_ordered_row_hash(work),
        "audio_pair_hash": sha256_json(audio_pairs) if audio_pairs else "",
    }
    if path is not None:
        payload["file_sha256"] = sha256_file(path)
    return payload


def assert_rq1_supervised_matches_d0(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    d0_identity: Mapping[str, Any],
    train_path: Optional[Union[str, Path]] = None,
    validation_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Prove G_train / G_validation are the frozen D0 supervised splits, not just same filenames."""
    train_id = split_identity(train, path=train_path)
    val_id = split_identity(validation, path=validation_path)
    expected_train = str(d0_identity.get("train_uid_set_hash") or "")
    expected_val = str(d0_identity.get("validation_uid_set_hash") or "")
    if not expected_train or not expected_val:
        raise UpstreamGateError("frozen D0 identity is missing train/validation UID-set hashes")
    if train_id["d0_uid_set_hash"] != expected_train:
        raise UpstreamGateError("G_train UID-set hash does not match frozen D0")
    if val_id["d0_uid_set_hash"] != expected_val:
        raise UpstreamGateError("G_validation UID-set hash does not match frozen D0")
    extra_train_audio = str(d0_identity.get("train_audio_pair_hash") or "")
    extra_val_audio = str(d0_identity.get("validation_audio_pair_hash") or "")
    extra_train_file = str(d0_identity.get("train_file_sha256") or "")
    extra_val_file = str(d0_identity.get("validation_file_sha256") or "")
    extra_train = str(d0_identity.get("train_ordered_uid_hash") or "")
    extra_val = str(d0_identity.get("validation_ordered_uid_hash") or "")
    extra_train_pair = str(d0_identity.get("train_pair_hash") or "")
    extra_val_pair = str(d0_identity.get("validation_pair_hash") or "")
    if extra_train and extra_train != train_id["d0_ordered_uid_hash"]:
        raise UpstreamGateError("G_train ordered UID hash does not match frozen D0")
    if extra_val and extra_val != val_id["d0_ordered_uid_hash"]:
        raise UpstreamGateError("G_validation ordered UID hash does not match frozen D0")
    if extra_train_pair and extra_train_pair != train_id["pair_hash"]:
        raise UpstreamGateError("G_train UID->text hash does not match frozen D0")
    if extra_val_pair and extra_val_pair != val_id["pair_hash"]:
        raise UpstreamGateError("G_validation UID->reference hash does not match frozen D0")
    if extra_train_audio and extra_train_audio != train_id["audio_pair_hash"]:
        raise UpstreamGateError("G_train UID->audio identity does not match frozen D0")
    if extra_val_audio and extra_val_audio != val_id["audio_pair_hash"]:
        raise UpstreamGateError("G_validation UID->audio identity does not match frozen D0")
    extra_train_file = str(d0_identity.get("train_file_sha256") or "")
    extra_val_file = str(d0_identity.get("validation_file_sha256") or "")
    if extra_train_file and train_path is not None and extra_train_file != train_id.get("file_sha256"):
        raise UpstreamGateError("G_train manifest SHA256 does not match frozen D0")
    if extra_val_file and validation_path is not None and extra_val_file != val_id.get("file_sha256"):
        raise UpstreamGateError("G_validation manifest SHA256 does not match frozen D0")
    return {"train": train_id, "validation": val_id}


def load_rq1_uid_audio_identity(
    durable_root: Union[str, Path],
    *,
    audio_index_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Read-only UID -> audio identity map from the durable RQ1 audio index."""
    from src.rq2_u_clean import _load_audio_index_jsonl, discover_full_audio_index_dir

    index_dir = Path(audio_index_dir) if audio_index_dir else discover_full_audio_index_dir(durable_root)
    try:
        loaded = _load_audio_index_jsonl(index_dir)
    except Exception as exc:
        raise UpstreamGateError(f"RQ1 audio identity map failed: {exc}") from exc
    out: Dict[str, Dict[str, Any]] = {}
    for uid, meta in loaded.items():
        pcm = str(meta.get("sha256_pcm") or "").strip().lower()
        if not is_sha256(pcm):
            raise UpstreamGateError(f"RQ1 audio identity for {uid} has invalid sha256_pcm")
        if uid in out and out[uid]["sha256_pcm"] != pcm:
            raise UpstreamGateError(f"duplicate conflicting RQ1 audio identity for {uid}")
        out[uid] = {
            "record_uid": uid,
            "split": str(meta.get("split") or ""),
            "shard_key": str(meta.get("shard_key") or ""),
            "shard_row_index": meta.get("shard_row_index"),
            "sha256_pcm": pcm,
            "sha256_source": str(meta.get("sha256_source") or ""),
            "local_cache_relpath": str(meta.get("local_cache_relpath") or ""),
            "sample_rate": int(meta.get("sample_rate") or 0),
            "n_samples": int(meta.get("n_samples") or 0),
        }
    return out


def bind_supervised_audio_identity(
    frame: pd.DataFrame,
    audio_map: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Attach durable RQ1 sha256_pcm to every supervised row. Fail closed on gaps."""
    work = frame.copy()
    uids = work["record_uid"].astype(str).tolist()
    if len(uids) != len(set(uids)):
        raise UpstreamGateError("supervised frame contains duplicate record_uid values")
    pcm = []
    relpaths = []
    for uid in uids:
        if uid not in audio_map:
            raise UpstreamGateError(f"supervised record_uid missing from RQ1 audio identity: {uid}")
        meta = audio_map[uid]
        digest = str(meta.get("sha256_pcm") or "").strip().lower()
        if not is_sha256(digest):
            raise UpstreamGateError(f"supervised audio identity for {uid} has invalid sha256_pcm")

        row_pos = len(pcm)
        for existing_col in ("pcm16_sha256", "sha256_pcm"):
            if existing_col not in work.columns:
                continue
            existing = str(work.iloc[row_pos][existing_col] or "").strip().lower()
            if existing and existing != "nan":
                if not is_sha256(existing):
                    raise UpstreamGateError(
                        f"supervised existing audio identity for {uid} is invalid"
                    )
                if existing != digest:
                    raise UpstreamGateError(
                        f"supervised existing audio identity conflicts with RQ1 audio index for {uid}"
                    )

        pcm.append(digest)
        relpaths.append(str(meta.get("local_cache_relpath") or ""))
    work["pcm16_sha256"] = pcm
    work["sha256_pcm"] = pcm
    if "local_cache_relpath" not in work.columns:
        work["local_cache_relpath"] = relpaths
    else:
        existing = work["local_cache_relpath"].astype(str).tolist()
        work["local_cache_relpath"] = [old if old and old != "nan" else new for old, new in zip(existing, relpaths)]
    return work


def load_frozen_supervised_splits(
    project_root: Union[str, Path],
    *,
    d0_identity: Mapping[str, Any],
    direct_state_dir: Optional[Union[str, Path]] = None,
    durable_root: Optional[Union[str, Path]] = None,
    audio_index_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    root = Path(project_root)

    if direct_state_dir:
        # Canonical NB14 gold data is the exact Direct prepared eligible
        # train/validation data that trained frozen D0.
        from src.direct_data import (
            DIRECT_TRAIN_CSV,
            DIRECT_VAL_CSV,
            load_direct_prepared_frames,
        )

        state = Path(direct_state_dir)
        train_path = state / DIRECT_TRAIN_CSV
        val_path = state / DIRECT_VAL_CSV

        if not train_path.is_file() or not val_path.is_file():
            raise UpstreamGateError(
                "frozen D0 Direct prepared train/validation CSVs are missing"
            )

        train, validation = load_direct_prepared_frames(state)

    else:
        # Compatibility fallback only. This is valid only when the raw RQ1
        # manifests themselves are exactly identical to D0 supervised data;
        # the D0 identity assertion below remains fail-closed.
        train_path = root / "data" / "manifests" / RQ1_SPLIT_FILES["g_train"]
        val_path = root / "data" / "manifests" / RQ1_SPLIT_FILES["g_validation"]
        train = load_rq1_split_csv(root, split="g_train")
        validation = load_rq1_split_csv(root, split="g_validation")

    audio_root = durable_root or d0_identity.get("durable_root")

    if audio_index_dir or audio_root:
        audio_map = load_rq1_uid_audio_identity(
            audio_root or ".",
            audio_index_dir=audio_index_dir,
        )
        train = bind_supervised_audio_identity(train, audio_map)
        validation = bind_supervised_audio_identity(validation, audio_map)

    elif (
        "pcm16_sha256" not in train.columns
        and "sha256_pcm" not in train.columns
    ):
        raise UpstreamGateError(
            "supervised G_train has no pcm16_sha256 and no durable RQ1 audio index"
        )

    identity = assert_rq1_supervised_matches_d0(
        train=train,
        validation=validation,
        d0_identity=d0_identity,
        train_path=train_path,
        validation_path=val_path,
    )

    return {
        "train": train,
        "validation": validation,
        "identity": identity,
        "train_path": train_path,
        "validation_path": val_path,
    }
