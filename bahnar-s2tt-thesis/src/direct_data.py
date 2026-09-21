"""Matched supervised-data loading for Notebook 05 Direct S2TT."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from src.asr_full_data import ELIGIBLE_TRAIN_CSV, ELIGIBLE_VAL_CSV, SUMMARY_JSON, STATUS_FULL_PREPARE
from src.asr_full_pcm import assert_local_disk_budget, cleanup_shard_download, wav_bytes_for_samples
from src.asr_full_shards import hydrate_union_audio, local_audio_ok, make_hf_parquet_stream_reader
from src.asr_full_train import assert_no_frozen_test_access, estimate_checkpoint_bytes, resolve_eligible_audio_path
from src.direct_contract import (
    LOCKED_DECODER_ID,
    LOCKED_DECODER_REVISION,
    LOCKED_ENCODER_ID,
    LOCKED_ENCODER_REVISION,
    LOCKED_NB03_TRAIN_COUNT,
    LOCKED_NB03_VALIDATION_COUNT,
    LOCKED_SAMPLE_RATE,
    assert_direct_data_contract_self_consistent,
    build_direct_training_contract,
)
from src.asr_utils import is_forbidden_test_path
from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash, sha256_file
from src.mt_normalize import normalize_mt_text_v1

DIRECT_TRAIN_CSV = "direct_train_eligible.csv"
DIRECT_VAL_CSV = "direct_validation_eligible.csv"
DIRECT_PREPARE_SUMMARY = "direct_prepare_summary.json"
DIRECT_TARGET_AUDIT = "direct_target_tokenizer_audit.json"
DIRECT_TRAINING_CONTRACT = "direct_training_contract.json"
DIRECT_MONITOR_CSV = "direct_validation_monitor.csv"
LATEST_PREPARE_POINTER = "LATEST_PREPARE.json"
DIRECT_PREPARE_FILES = (
    DIRECT_TRAIN_CSV,
    DIRECT_VAL_CSV,
    DIRECT_PREPARE_SUMMARY,
    DIRECT_TARGET_AUDIT,
    DIRECT_TRAINING_CONTRACT,
    DIRECT_MONITOR_CSV,
)
PAIR_HASH_COLS = ("record_uid", "text_vi_norm")
ORDERED_ROW_COLS = ("record_uid", "text_vi_norm", "split", "source_split")
HYDRATE_DISK_RESERVE_BYTES = 2 * 1024 ** 3
PARQUET_HEADROOM_BYTES = 4 * 1024 ** 3
CHECKPOINT_PEAK_COPIES = 4  # durable: latest + best + rollback/LKG + temporary upload
LOCAL_CHECKPOINT_PEAK_COPIES = 3  # save_total_limit=2 plus the in-progress save
PREPARE_REQUIRED_FILES = DIRECT_PREPARE_FILES
DEFAULT_HF_MODELS = (
    (LOCKED_ENCODER_ID, LOCKED_ENCODER_REVISION),
    (LOCKED_DECODER_ID, LOCKED_DECODER_REVISION),
)


NB03_ELIGIBLE_PIN_KEYS = (
    "train_eligible_uid_set_hash",
    "validation_eligible_uid_set_hash",
    "train_eligible_file_sha256",
    "validation_eligible_file_sha256",
)


def local_checkpoint_peak_copies(save_total_limit: int = 2) -> int:
    """HF writes the new checkpoint before rotating; peak = retained + 1."""
    return max(1, int(save_total_limit)) + 1


def _require_hex_hash(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise RuntimeError(f"{label} must be a 64-hex SHA-256, got {value!r}")
    return text


def _stable_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def compute_pair_hash(df: pd.DataFrame) -> str:
    """Order-independent hash of (record_uid, text_vi_norm)."""
    missing = [c for c in PAIR_HASH_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"pair hash missing columns: {missing}")
    work = df.loc[:, list(PAIR_HASH_COLS)].copy()
    for c in PAIR_HASH_COLS:
        work[c] = work[c].astype(str)
    work = work.sort_values(list(PAIR_HASH_COLS), kind="mergesort")
    payload = {"columns": list(PAIR_HASH_COLS), "n_rows": int(len(work)), "rows": work.to_numpy().tolist()}
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def compute_ordered_row_hash(df: pd.DataFrame) -> str:
    """Order-preserving hash of identity + target + split/source fields."""
    cols = [c for c in ORDERED_ROW_COLS if c in df.columns]
    if "record_uid" not in cols or "text_vi_norm" not in cols:
        raise RuntimeError(f"ordered-row hash missing required columns; have {list(df.columns)}")
    work = df.loc[:, cols].copy()
    for c in cols:
        work[c] = work[c].astype(str)
    payload = {"columns": cols, "n_rows": int(len(work)), "rows": work.to_numpy().tolist()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def frame_csv_sha256(df: pd.DataFrame) -> str:
    """SHA-256 of the exact CSV bytes persist_direct_prepare will write."""
    import tempfile

    fd, name = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    path = Path(name)
    try:
        df.to_csv(path, index=False)
        return sha256_file(path)
    finally:
        path.unlink(missing_ok=True)


@dataclass
class DirectAccessTracker:
    """Fail-closed record of files, splits, UIDs and Parquet refs actually touched."""

    files: List[str] = field(default_factory=list)
    splits: List[str] = field(default_factory=list)
    uids: List[str] = field(default_factory=list)
    parquet_refs: List[str] = field(default_factory=list)

    def record_file(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        self.files.append(key)
        if is_forbidden_test_path(key):
            raise RuntimeError(f"Frozen-test path refused by Direct pipeline: {key}")
        return p

    def record_split(self, split: Any) -> None:
        label = str(split or "").strip().lower()
        if label:
            self.splits.append(label)
        from src.asr_full_data import is_frozen_split_label

        if is_frozen_split_label(label):
            raise RuntimeError(f"Frozen-test split refused by Direct pipeline: {split!r}")

    def record_uids(self, uids: Iterable[Any], *, split: Optional[str] = None) -> None:
        if split is not None:
            self.record_split(split)
        for u in uids:
            s = str(u)
            self.uids.append(s)
            low = s.lower()
            if "rq1_test" in low or "frozen_test" in low:
                raise RuntimeError(f"Frozen-test UID refused by Direct pipeline: {s}")

    def record_parquet(self, ref: Any) -> None:
        if ref is None:
            return
        if hasattr(ref, "filename"):
            token = f"{getattr(ref, 'repo_id', '')}/{getattr(ref, 'filename', '')}@{getattr(ref, 'revision', '')}"
        else:
            token = str(ref)
        self.parquet_refs.append(token)
        if is_forbidden_test_path(token) or "rq1_test" in token.lower():
            raise RuntimeError(f"Frozen-test parquet refused by Direct pipeline: {token}")

    def frozen_test_accessed(self) -> bool:
        if any(is_forbidden_test_path(p) for p in self.files):
            return True
        from src.asr_full_data import is_frozen_split_label

        if any(is_frozen_split_label(s) for s in self.splits):
            return True
        if any("rq1_test" in str(u).lower() or "frozen_test" in str(u).lower() for u in self.uids):
            return True
        if any(is_forbidden_test_path(p) or "rq1_test" in str(p).lower() for p in self.parquet_refs):
            return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "files": list(self.files),
            "splits": list(self.splits),
            "uid_count": len(self.uids),
            "parquet_refs": list(self.parquet_refs),
            "frozen_test_accessed": self.frozen_test_accessed(),
        }


def _read_non_test(
    path: Union[str, Path],
    tracker: Optional[DirectAccessTracker] = None,
    opened_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    p = Path(path)
    if tracker is not None:
        p = tracker.record_file(p)
    else:
        if is_forbidden_test_path(p):
            raise RuntimeError(f"Frozen-test path refused by Direct pipeline: {p}")
        if opened_paths is not None:
            opened_paths.append(str(p.resolve()))
    return pd.read_csv(p)


def load_asr_matched_direct_frames(
    *,
    asr_state_dir: Union[str, Path],
    train_manifest: Union[str, Path],
    validation_manifest: Union[str, Path],
    expected_asr_contract_hash: str,
    opened_paths: Optional[List[str]] = None,
    expected_train_count: Optional[int] = None,
    expected_validation_count: Optional[int] = None,
    expected_asr_train_uid_set_hash: Optional[str] = None,
    expected_asr_validation_uid_set_hash: Optional[str] = None,
    expected_asr_train_file_sha256: Optional[str] = None,
    expected_asr_validation_file_sha256: Optional[str] = None,
    tracker: Optional[DirectAccessTracker] = None,
) -> Dict[str, Any]:
    """
    Primary RQ1 D0 uses the exact speech utterances that passed NB03 ASR prepare.

    Only Vietnamese target text is joined from locked RQ1 manifests. Bahnar text
    is never an input to D0. This makes C0/D0 supervision matched at utterance level.
    """
    state = Path(asr_state_dir)
    summary_path = state / SUMMARY_JSON
    if tracker is not None:
        tracker.record_file(summary_path)
    elif opened_paths is not None:
        opened_paths.append(str(summary_path.resolve()))
    if not summary_path.is_file():
        raise RuntimeError(f"NB03 full prepare summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != STATUS_FULL_PREPARE:
        raise RuntimeError(f"NB03 full prepare is not successful: {summary.get('status')!r}")
    recorded_hash = str(summary.get("contract_hash") or summary.get("data_contract_hash") or "")
    if recorded_hash != str(expected_asr_contract_hash):
        raise RuntimeError(
            f"NB03 prepare contract mismatch: recorded={recorded_hash!r} expected={expected_asr_contract_hash!r}"
        )

    asr_train_path = state / ELIGIBLE_TRAIN_CSV
    asr_val_path = state / ELIGIBLE_VAL_CSV
    asr_train = _read_non_test(asr_train_path, tracker, opened_paths)
    asr_val = _read_non_test(asr_val_path, tracker, opened_paths)
    locked_train = _read_non_test(train_manifest, tracker, opened_paths)
    locked_val = _read_non_test(validation_manifest, tracker, opened_paths)

    asr_train_uid_hash = compute_uid_set_hash(asr_train)
    asr_val_uid_hash = compute_uid_set_hash(asr_val)
    asr_train_file_sha = sha256_file(asr_train_path)
    asr_val_file_sha = sha256_file(asr_val_path)
    production_counts = (
        expected_train_count is not None
        and expected_validation_count is not None
        and int(expected_train_count) == LOCKED_NB03_TRAIN_COUNT
        and int(expected_validation_count) == LOCKED_NB03_VALIDATION_COUNT
    )
    provided_pins = {
        "train_eligible_uid_set_hash": str(expected_asr_train_uid_set_hash or "").strip(),
        "validation_eligible_uid_set_hash": str(expected_asr_validation_uid_set_hash or "").strip(),
        "train_eligible_file_sha256": str(expected_asr_train_file_sha256 or "").strip(),
        "validation_eligible_file_sha256": str(expected_asr_validation_file_sha256 or "").strip(),
    }
    missing_pins = [name for name in NB03_ELIGIBLE_PIN_KEYS if not provided_pins[name]]
    computed_pins = (
        "Computed hashes from current CSVs:\n"
        f"  train_eligible_uid_set_hash={asr_train_uid_hash}\n"
        f"  validation_eligible_uid_set_hash={asr_val_uid_hash}\n"
        f"  train_eligible_file_sha256={asr_train_file_sha}\n"
        f"  validation_eligible_file_sha256={asr_val_file_sha}\n"
        "Put these in configs/direct.yaml under nb03 before FULL_STAGE=prepare."
    )
    if production_counts and missing_pins:
        raise RuntimeError(
            "NB03 eligible provenance is not fully pinned. Missing: "
            + ", ".join(missing_pins)
            + "\n"
            + computed_pins
        )
    if provided_pins["train_eligible_uid_set_hash"]:
        want = _require_hex_hash(provided_pins["train_eligible_uid_set_hash"], label="nb03 train eligible uid hash")
        if asr_train_uid_hash != want:
            raise RuntimeError(
                f"NB03 train eligible UID set hash mismatch: csv={asr_train_uid_hash} expected={want}"
            )
    if provided_pins["validation_eligible_uid_set_hash"]:
        want = _require_hex_hash(
            provided_pins["validation_eligible_uid_set_hash"], label="nb03 validation eligible uid hash"
        )
        if asr_val_uid_hash != want:
            raise RuntimeError(
                f"NB03 validation eligible UID set hash mismatch: csv={asr_val_uid_hash} expected={want}"
            )
    if provided_pins["train_eligible_file_sha256"]:
        want = _require_hex_hash(provided_pins["train_eligible_file_sha256"], label="nb03 train eligible file sha256")
        if asr_train_file_sha != want:
            raise RuntimeError(
                f"NB03 train eligible file sha256 mismatch: csv={asr_train_file_sha} expected={want}"
            )
    if provided_pins["validation_eligible_file_sha256"]:
        want = _require_hex_hash(
            provided_pins["validation_eligible_file_sha256"], label="nb03 validation eligible file sha256"
        )
        if asr_val_file_sha != want:
            raise RuntimeError(
                f"NB03 validation eligible file sha256 mismatch: csv={asr_val_file_sha} expected={want}"
            )
    if expected_train_count is not None and len(asr_train) != int(expected_train_count):
        raise RuntimeError(f"Unexpected NB03 eligible train count: {len(asr_train)} != {expected_train_count}")
    if expected_validation_count is not None and len(asr_val) != int(expected_validation_count):
        raise RuntimeError(f"Unexpected NB03 eligible validation count: {len(asr_val)} != {expected_validation_count}")

    def _join(base: pd.DataFrame, locked: pd.DataFrame, split: str) -> pd.DataFrame:
        if base["record_uid"].astype(str).duplicated().any():
            raise RuntimeError(f"NB03 {split} eligible has duplicate record_uid")
        if locked["record_uid"].astype(str).duplicated().any():
            raise RuntimeError(f"Locked {split} manifest has duplicate record_uid")
        target = locked[["record_uid", "text_vi"]].copy()
        target["record_uid"] = target["record_uid"].astype(str)
        x = base.copy()
        x["record_uid"] = x["record_uid"].astype(str)
        out = x.merge(target, on="record_uid", how="left", validate="one_to_one")
        if len(out) != len(base):
            raise RuntimeError(f"{split} join count changed: {len(base)} -> {len(out)}")
        out["text_vi_norm"] = out["text_vi"].map(normalize_mt_text_v1)
        missing = out["text_vi_norm"].eq("")
        if missing.any():
            examples = out.loc[missing, "record_uid"].head(10).tolist()
            raise RuntimeError(
                f"{split} has {int(missing.sum())} empty Vietnamese targets after normalization; "
                f"matched RQ1 set must stay exact. examples={examples}"
            )
        if "split" in out.columns:
            from src.asr_full_data import is_frozen_split_label

            if bool(out["split"].map(is_frozen_split_label).any()):
                raise RuntimeError(f"{split} eligible contains frozen-test split labels")
        if tracker is not None:
            tracker.record_uids(out["record_uid"].astype(str).tolist(), split=split)
        return out.reset_index(drop=True)

    train = _join(asr_train, locked_train, "train")
    val = _join(asr_val, locked_val, "validation")

    uid_overlap = set(train["record_uid"].astype(str)) & set(val["record_uid"].astype(str))
    if uid_overlap:
        raise RuntimeError(f"Direct train/validation UID overlap: {len(uid_overlap)}")
    for col in ("group_id", "recording_group_id"):
        if col in train.columns and col in val.columns:
            a = set(train[col].dropna().astype(str)) - {""}
            b = set(val[col].dropna().astype(str)) - {""}
            overlap = a & b
            if overlap:
                raise RuntimeError(f"Direct train/validation {col} overlap: {len(overlap)}")

    opened = list(tracker.files) if tracker is not None else (opened_paths or [])
    splits = list(tracker.splits) if tracker is not None else ["train", "validation"]
    assert_no_frozen_test_access(opened, splits)
    return {
        "train": train,
        "validation": val,
        "asr_prepare_summary": summary,
        "train_uid_set_hash": compute_uid_set_hash(train),
        "validation_uid_set_hash": compute_uid_set_hash(val),
        "train_ordered_uid_hash": compute_ordered_uid_hash(train),
        "validation_ordered_uid_hash": compute_ordered_uid_hash(val),
        "train_pair_hash": compute_pair_hash(train),
        "validation_pair_hash": compute_pair_hash(val),
        "train_ordered_row_hash": compute_ordered_row_hash(train),
        "validation_ordered_row_hash": compute_ordered_row_hash(val),
        "locked_train_manifest_sha256": sha256_file(train_manifest),
        "locked_validation_manifest_sha256": sha256_file(validation_manifest),
        "asr_train_eligible_uid_set_hash": asr_train_uid_hash,
        "asr_validation_eligible_uid_set_hash": asr_val_uid_hash,
        "asr_train_eligible_file_sha256": asr_train_file_sha,
        "asr_validation_eligible_file_sha256": asr_val_file_sha,
    }


def _atomic_replace(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dest))


def _prepare_lkg_dir(state: Path) -> Path:
    return state / ".lkg_prepare"


def _prepare_staging_dir(state: Path) -> Path:
    return state / ".staging_prepare"


def _prepare_commit_marker(state: Path) -> Path:
    return state / ".prepare_commit_ok"


def _copy_prepare_files(src_dir: Path, dest_dir: Path, names: Sequence[str]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)


def _restore_prepare_lkg(state: Path, lkg: Path) -> None:
    if not lkg.is_dir():
        return
    for src in lkg.iterdir():
        if src.is_file():
            _atomic_replace(src, state / src.name)


def _recover_interrupted_prepare(state: Path) -> None:
    """If a previous persist died mid-commit, restore last-known-good."""
    lkg = _prepare_lkg_dir(state)
    marker = _prepare_commit_marker(state)
    staging = _prepare_staging_dir(state)
    if lkg.is_dir():
        if marker.is_file():
            shutil.rmtree(lkg, ignore_errors=True)
        else:
            _restore_prepare_lkg(state, lkg)
            shutil.rmtree(lkg, ignore_errors=True)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    if marker.is_file():
        marker.unlink()


def _lock_training_contract_to_monitor_file(
    training_contract: Mapping[str, Any],
    monitor_sha256: str,
) -> Dict[str, Any]:
    """Rebuild the training contract so ``monitor_file_sha256`` is the staged CSV SHA."""
    payload = dict(training_contract)
    payload.pop("direct_training_contract_hash", None)
    payload["monitor_file_sha256"] = str(monitor_sha256)
    locked = build_direct_training_contract(**payload)
    if str(locked.get("monitor_file_sha256") or "") != str(monitor_sha256):
        raise RuntimeError("Direct training contract failed to lock staged monitor SHA-256")
    return locked


def persist_direct_prepare(
    state_dir: Union[str, Path],
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    summary: Dict[str, Any],
    target_audit: Dict[str, Any],
    training_contract: Optional[Dict[str, Any]] = None,
    monitor: Optional[pd.DataFrame] = None,
    durable_state_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Stage the full prepare file set, then commit with last-known-good rollback.

    ``LATEST_PREPARE.json`` is published only after all six artifacts exist on
    disk. An interrupted commit restores LKG and does not move the pointer.

    Monitor byte lock: write the CSV first, hash the staged file, put that SHA
    into the training contract, verify after write, then commit.
    """
    if not isinstance(training_contract, dict):
        raise RuntimeError("Direct prepare requires training_contract before LATEST_PREPARE.json")
    if monitor is None or not isinstance(monitor, pd.DataFrame):
        raise RuntimeError("Direct prepare requires validation monitor before LATEST_PREPARE.json")

    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_prepare(state)

    staging = _prepare_staging_dir(state)
    staging.mkdir(parents=True, exist_ok=True)
    train.to_csv(staging / DIRECT_TRAIN_CSV, index=False)
    validation.to_csv(staging / DIRECT_VAL_CSV, index=False)
    monitor.to_csv(staging / DIRECT_MONITOR_CSV, index=False)
    monitor_sha = sha256_file(staging / DIRECT_MONITOR_CSV)
    locked_contract = _lock_training_contract_to_monitor_file(training_contract, monitor_sha)
    summary_out = dict(summary)
    summary_out["direct_training_contract_hash"] = locked_contract["direct_training_contract_hash"]
    (staging / DIRECT_PREPARE_SUMMARY).write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (staging / DIRECT_TARGET_AUDIT).write_text(
        json.dumps(target_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (staging / DIRECT_TRAINING_CONTRACT).write_text(
        json.dumps(dict(locked_contract), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    verify_sha = sha256_file(staging / DIRECT_MONITOR_CSV)
    if verify_sha != locked_contract["monitor_file_sha256"]:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            "Staged monitor SHA-256 does not match training contract "
            f"file={verify_sha} contract={locked_contract['monitor_file_sha256']}"
        )

    committed_names = [name for name in DIRECT_PREPARE_FILES if (staging / name).is_file()]
    missing_required = [name for name in PREPARE_REQUIRED_FILES if name not in committed_names]
    if missing_required:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"Direct prepare staging missing required files: {missing_required}")

    lkg = _prepare_lkg_dir(state)
    if lkg.exists():
        shutil.rmtree(lkg)
    live_names = [name for name in committed_names if (state / name).is_file()]
    if live_names:
        _copy_prepare_files(state, lkg, live_names)

    marker = _prepare_commit_marker(state)
    try:
        for name in committed_names:
            _atomic_replace(staging / name, state / name)
        marker.write_text("ok", encoding="utf-8")
    except Exception:
        if lkg.is_dir():
            _restore_prepare_lkg(state, lkg)
        if marker.is_file():
            marker.unlink()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)

    missing_live = [name for name in PREPARE_REQUIRED_FILES if not (state / name).is_file()]
    if missing_live:
        if lkg.is_dir():
            _restore_prepare_lkg(state, lkg)
        if marker.is_file():
            marker.unlink()
        raise RuntimeError(
            f"Direct prepare commit missing required files; pointer not published: {missing_live}"
        )

    pointer = {
        "contract_hash": summary_out.get("contract_hash"),
        "state_dir": str(state),
    }
    if durable_state_root is not None:
        root = Path(durable_state_root).resolve()
        try:
            state.resolve().relative_to(root)
        except ValueError as exc:
            if lkg.is_dir():
                _restore_prepare_lkg(state, lkg)
            if marker.is_file():
                marker.unlink()
            raise RuntimeError(
                f"Direct prepare pointer target {state} is not under durable state root {root}"
            ) from exc
        ptr_path = root / LATEST_PREPARE_POINTER
        ptr_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ptr_path.with_suffix(ptr_path.suffix + ".tmp")
        tmp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(str(tmp), str(ptr_path))
        pointer["pointer_path"] = str(ptr_path)
    if marker.is_file():
        marker.unlink()
    shutil.rmtree(lkg, ignore_errors=True)
    pointer["training_contract"] = locked_contract
    pointer["summary"] = summary_out
    return pointer


def load_direct_prepare_success(
    state_dir: Union[str, Path], *, expected_contract_hash: str
) -> Dict[str, Any]:
    state = Path(state_dir)
    p = state / DIRECT_PREPARE_SUMMARY
    if not p.is_file():
        raise RuntimeError(f"Direct prepare summary missing: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("status") != "SUCCESS_DIRECT_PREPARE":
        raise RuntimeError(f"Direct prepare not successful: {data.get('status')!r}")
    embedded = data.get("data_contract")
    if not isinstance(embedded, dict):
        raise RuntimeError("Direct prepare summary missing embedded data_contract")
    assert_direct_data_contract_self_consistent(embedded)
    if str(embedded.get("contract_hash")) != str(data.get("contract_hash")):
        raise RuntimeError("Direct prepare summary contract_hash does not match embedded data_contract")
    if str(data.get("contract_hash")) != str(expected_contract_hash):
        raise RuntimeError("Direct prepare contract hash mismatch")
    return data


def load_direct_prepared_frames(
    state_dir: Union[str, Path],
    *,
    tracker: Optional[DirectAccessTracker] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = Path(state_dir)
    train = _read_non_test(state / DIRECT_TRAIN_CSV, tracker)
    val = _read_non_test(state / DIRECT_VAL_CSV, tracker)
    if tracker is not None:
        tracker.record_uids(train["record_uid"].astype(str).tolist(), split="train")
        tracker.record_uids(val["record_uid"].astype(str).tolist(), split="validation")
    return train, val


def assert_direct_frames_match_contract(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    train_path: Optional[Union[str, Path]] = None,
    validation_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Recompute content/order locks; fail closed on any drift including swap/reorder."""
    if contract.get("contract_hash"):
        assert_direct_data_contract_self_consistent(contract)
    if len(train) != int(contract["train_count"]) or len(validation) != int(contract["validation_count"]):
        raise RuntimeError(
            f"Direct row counts changed: train {len(train)}/{contract['train_count']} "
            f"val {len(validation)}/{contract['validation_count']}"
        )
    t_uids = set(train["record_uid"].astype(str))
    v_uids = set(validation["record_uid"].astype(str))
    if t_uids & v_uids:
        raise RuntimeError(f"Direct train/validation UID overlap on reload: {len(t_uids & v_uids)}")
    from src.asr_full_data import is_frozen_split_label

    for name, df in (("train", train), ("validation", validation)):
        if "split" in df.columns and df["split"].map(is_frozen_split_label).any():
            raise RuntimeError(f"Direct {name} reload contains frozen-test split labels")
        if "source_split" in df.columns and df["source_split"].map(is_frozen_split_label).any():
            raise RuntimeError(f"Direct {name} reload contains frozen-test source_split labels")

    recomputed = {
        "train_uid_set_hash": compute_uid_set_hash(train),
        "validation_uid_set_hash": compute_uid_set_hash(validation),
        "train_ordered_uid_hash": compute_ordered_uid_hash(train),
        "validation_ordered_uid_hash": compute_ordered_uid_hash(validation),
        "train_pair_hash": compute_pair_hash(train),
        "validation_pair_hash": compute_pair_hash(validation),
        "train_ordered_row_hash": compute_ordered_row_hash(train),
        "validation_ordered_row_hash": compute_ordered_row_hash(validation),
    }
    if train_path is not None:
        recomputed["train_file_sha256"] = sha256_file(train_path)
    if validation_path is not None:
        recomputed["validation_file_sha256"] = sha256_file(validation_path)
    diffs = [k for k, v in recomputed.items() if str(contract.get(k)) != str(v)]
    if diffs:
        raise RuntimeError(f"Direct prepared frames drifted vs contract: {diffs}")
    return recomputed


def latest_prepare_pointer_path(durable_state_root: Union[str, Path]) -> Path:
    return Path(durable_state_root) / LATEST_PREPARE_POINTER


def load_latest_prepare_pointer(durable_state_root: Union[str, Path]) -> Dict[str, Any]:
    root = Path(durable_state_root).resolve()
    p = latest_prepare_pointer_path(root)
    if not p.is_file():
        raise RuntimeError("Direct prepare pointer missing; run FULL_STAGE='prepare' first")
    ptr = json.loads(p.read_text(encoding="utf-8"))
    state = Path(ptr["state_dir"]).resolve()
    try:
        state.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Prepare pointer state_dir {state} is not under {root}") from exc
    return ptr


def filesystem_device(path: Union[str, Path]) -> int:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    return int(target.stat().st_dev)


def same_filesystem(path_a: Union[str, Path], path_b: Union[str, Path]) -> bool:
    return filesystem_device(path_a) == filesystem_device(path_b)


def huggingface_hub_cache_dir(
    path: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Resolve the real Hugging Face hub cache (not ``HF_HOME`` itself)."""
    env_map = os.environ if env is None else env
    if path is not None:
        p = Path(path)
        if p.name == "hub" or any(p.glob("models--*")):
            return p
        return p / "hub"
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env_map.get(key):
            return Path(str(env_map[key]))
    if env_map.get("TRANSFORMERS_CACHE"):
        return Path(str(env_map["TRANSFORMERS_CACHE"]))
    if env_map.get("HF_HOME"):
        return Path(str(env_map["HF_HOME"])) / "hub"
    xdg = env_map.get("XDG_CACHE_HOME")
    if xdg:
        return Path(str(xdg)) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def huggingface_snapshot_dir(
    cache_dir: Union[str, Path], repo_id: str, revision: str
) -> Path:
    safe = "models--" + str(repo_id).replace("/", "--")
    return Path(cache_dir) / safe / "snapshots" / str(revision)


def huggingface_snapshot_is_present(
    cache_dir: Union[str, Path], repo_id: str, revision: str
) -> bool:
    snap = huggingface_snapshot_dir(cache_dir, repo_id, revision)
    if not snap.is_dir():
        return False
    markers = (
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if any((snap / name).is_file() for name in markers):
        return True
    return any(snap.glob("*.safetensors")) or any(snap.glob("pytorch_model*.bin"))


def missing_or_corrupt_wav_bytes(
    frames: Sequence[pd.DataFrame],
    *,
    audio_cache_dir: Union[str, Path],
    sample_rate: int = LOCKED_SAMPLE_RATE,
) -> Dict[str, Any]:
    """
    Local WAV bytes still needed: missing files or files that fail PCM verify.

    Already-valid hydrate is not counted again, so resume on a populated cache
    is not false-rejected.
    """
    cache = Path(audio_cache_dir)
    seen: set[str] = set()
    total = 0
    samples = 0
    n_missing = 0
    n_corrupt = 0
    n_ok = 0
    for frame in frames:
        if frame is None or not len(frame):
            continue
        if "n_samples" not in frame.columns:
            raise RuntimeError("eligible frame lacks n_samples (re-run FULL_STAGE=prepare)")
        has_sha = "sha256_pcm" in frame.columns
        for _, row in frame.iterrows():
            uid = str(row["record_uid"])
            if uid in seen:
                continue
            seen.add(uid)
            n = int(row["n_samples"])
            samples += n
            need = wav_bytes_for_samples(n)
            path = resolve_eligible_audio_path(row, [cache])
            if path is None:
                total += need
                n_missing += 1
                continue
            sha = str(row["sha256_pcm"]).strip() if has_sha else ""
            if not sha or sha.lower() in {"nan", "none"}:
                n_ok += 1
                continue
            good, _reason = local_audio_ok(
                path,
                expected_sha256_pcm=sha,
                expected_n_samples=n,
                expected_sample_rate=int(sample_rate),
            )
            if good:
                n_ok += 1
            else:
                total += need
                n_corrupt += 1
    return {
        "unique_records": len(seen),
        "n_samples": samples,
        "wav_bytes": total,
        "missing": n_missing,
        "corrupt": n_corrupt,
        "ok": n_ok,
        "audio_hours": samples / float(sample_rate) / 3600.0,
    }


def preflight_direct_local_disk(
    frames: Sequence[pd.DataFrame],
    *,
    audio_cache_dir: Union[str, Path],
    parquet_cache_dir: Union[str, Path],
    reserve_bytes: int = HYDRATE_DISK_RESERVE_BYTES,
    parquet_headroom_bytes: int = PARQUET_HEADROOM_BYTES,
    local_ckpt_dir: Optional[Union[str, Path]] = None,
    n_parameters: Optional[int] = None,
    copies: Optional[int] = None,
    save_total_limit: int = 2,
    hf_cache_dir: Optional[Union[str, Path]] = None,
    hf_models: Optional[Sequence[Tuple[str, str]]] = None,
    sample_rate: int = LOCKED_SAMPLE_RATE,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """
    Peak local-disk preflight on shared filesystems.

    Peak = missing/corrupt WAV + one parquet shard + local checkpoint copies
    (save_total_limit + 1, because Trainer writes the new checkpoint before
    rotating) + unresolved HF hub cache + reserve. Durable 4× copies are not
    a local-disk charge.
    """
    usable = [f for f in frames if f is not None and len(f)]
    wav = missing_or_corrupt_wav_bytes(
        usable, audio_cache_dir=audio_cache_dir, sample_rate=int(sample_rate)
    )
    wav_bytes = int(wav.get("wav_bytes") or 0)
    parquet_need = int(parquet_headroom_bytes)
    peak_copies = int(copies) if copies is not None else local_checkpoint_peak_copies(save_total_limit)
    checkpoint_peak = 0
    checkpoint_est: Optional[Dict[str, Any]] = None
    if n_parameters is not None:
        checkpoint_est = estimate_checkpoint_bytes(n_parameters=int(n_parameters))
        checkpoint_peak = int(checkpoint_est["checkpoint_bytes"]) * int(peak_copies)

    hf_need = 0
    hf_resolved: Optional[Path] = None
    hf_present: List[str] = []
    hf_missing: List[str] = []
    consider_hf = hf_cache_dir is not None or n_parameters is not None
    if consider_hf:
        hf_resolved = huggingface_hub_cache_dir(hf_cache_dir, env=env)
        models = list(hf_models) if hf_models is not None else list(DEFAULT_HF_MODELS)
        for repo_id, revision in models:
            label = f"{repo_id}@{revision}"
            if huggingface_snapshot_is_present(hf_resolved, repo_id, revision):
                hf_present.append(label)
            else:
                hf_missing.append(label)
        if hf_missing and n_parameters is not None:
            hf_need = int(n_parameters) * 8

    groups: Dict[int, Dict[str, Any]] = {}

    def _add(path: Union[str, Path], key: str, nbytes: int) -> None:
        if int(nbytes) <= 0:
            return
        p = Path(path)
        dev = filesystem_device(p)
        slot = groups.setdefault(dev, {"path": p, "needs": {}})
        slot["needs"][key] = int(slot["needs"].get(key, 0)) + int(nbytes)

    _add(audio_cache_dir, "wav", wav_bytes)
    _add(parquet_cache_dir, "parquet_shard_headroom", parquet_need)
    if local_ckpt_dir is not None:
        _add(local_ckpt_dir, "checkpoint_peak", checkpoint_peak)
    if hf_resolved is not None:
        _add(hf_resolved, "hf_model_cache", hf_need)

    reports = []
    for slot in groups.values():
        slot["needs"] = {k: v for k, v in slot["needs"].items() if int(v) > 0}
        reports.append(
            assert_local_disk_budget(
                slot["path"],
                needs=slot["needs"],
                reserve_bytes=int(reserve_bytes),
                label="direct local-disk peak",
            )
        )
    return {
        "wav": wav,
        "checkpoint": checkpoint_est,
        "peak_checkpoint_bytes": int(checkpoint_peak),
        "peak_copies": int(peak_copies),
        "hf_cache_dir": str(hf_resolved) if hf_resolved is not None else None,
        "hf_snapshots_present": hf_present,
        "hf_snapshots_missing": hf_missing,
        "same_filesystem_audio_parquet": same_filesystem(audio_cache_dir, parquet_cache_dir),
        "reports": reports,
    }


def preflight_hydrate_disk(
    frames: Sequence[pd.DataFrame],
    *,
    audio_cache_dir: Union[str, Path],
    parquet_cache_dir: Union[str, Path],
    reserve_bytes: int = HYDRATE_DISK_RESERVE_BYTES,
    parquet_headroom_bytes: int = PARQUET_HEADROOM_BYTES,
) -> Dict[str, Any]:
    report = preflight_direct_local_disk(
        frames,
        audio_cache_dir=audio_cache_dir,
        parquet_cache_dir=parquet_cache_dir,
        reserve_bytes=int(reserve_bytes),
        parquet_headroom_bytes=int(parquet_headroom_bytes),
    )
    report["audio"] = (report.get("reports") or [None])[0]
    report["parquet"] = report["audio"]
    return report


def hydrate_direct_audio(
    frames: Sequence[pd.DataFrame],
    *,
    asr_state_dir: Union[str, Path],
    dataset_id: str,
    parquet_revision: str,
    audio_cache_dir: Union[str, Path],
    parquet_cache_dir: Union[str, Path],
    sample_rate: int,
    tracker: Optional[DirectAccessTracker] = None,
) -> Dict[str, Any]:
    """
    Hydrate matched Direct rows shard-by-shard.

    After PCM verify, immediately delete the downloaded Parquet blob so WAV and
    full parquet cache never coexist.
    """
    preflight = preflight_hydrate_disk(
        frames, audio_cache_dir=audio_cache_dir, parquet_cache_dir=parquet_cache_dir
    )
    downloaded: Dict[str, str] = {}
    base_reader = make_hf_parquet_stream_reader(
        cache_dir=parquet_cache_dir, batch_size=16, downloaded=downloaded
    )

    def reader(ref: Any, indices: Any):
        # Record the shard at first access, including download/read failures.
        if tracker is not None:
            tracker.record_parquet(ref)
        return base_reader(ref, indices)

    def _cleanup(ref: Any) -> None:
        if tracker is not None:
            tracker.record_parquet(ref)
        key = getattr(ref, "shard_key", None) or str(ref)
        local = downloaded.pop(key, None)
        if local:
            cleanup_shard_download(local)

    try:
        report = hydrate_union_audio(
            list(frames),
            state_dir=asr_state_dir,
            dataset_id=dataset_id,
            parquet_revision=parquet_revision,
            shard_reader=reader,
            local_audio_dir=audio_cache_dir,
            target_sr=int(sample_rate),
            cleanup_shard=_cleanup,
        )
    finally:
        leftover = list(downloaded.values())
        downloaded.clear()
        for local in leftover:
            cleanup_shard_download(local)
    report["preflight"] = preflight
    report["parquet_blobs_remaining"] = 0
    return report
