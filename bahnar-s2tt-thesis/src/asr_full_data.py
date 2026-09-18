"""
Full-training data preparation for Notebook 03.

Shard-sequential Parquet reading, deterministic filtering, durable prepare state.
Does NOT write bulk WAVs onto durable storage — eligible CSVs/state go under
the contract-scoped durable state dir; optional audio cache stays on LOCAL_ROOT.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.asr_utils import (
    check_ctc_feasibility,
    classify_cache_status,
    is_forbidden_test_path,
)
from src.asr_runtime_paths import (
    EFFECTIVE_EXCLUSIONS_CSV,
    EFFECTIVE_MANIFEST_SUMMARY,
    EFFECTIVE_TRAIN_MANIFEST,
    EFFECTIVE_VAL_MANIFEST,
    OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
)
from src.data_utils import (
    encode_text_with_vocab,
    normalize_bahnar_ctc_v1,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_FULL_PREPARE = "SUCCESS_FULL_PREPARE"
STATUS_FAILED = "FAILED"

FULL_PREPARE_MIN_DURATION = 0.5
FULL_PREPARE_MAX_DURATION = 40.0

PAIR_KEY_TRAIN_EXCLUSION_REASON = "train_validation_pair_key_overlap"

ELIGIBLE_TRAIN_CSV = "full_train_eligible.csv"
ELIGIBLE_VAL_CSV = "full_validation_eligible.csv"
EXCLUSIONS_CSV = "full_data_exclusions.csv"
SUMMARY_JSON = "full_data_summary.json"
# Bumped for MAX=40 + pair_key train-drop policy + contract-scoped state dirs.
PREPARE_STATE_SCHEMA_VERSION = "full_prepare_v3_max40_pairkey_drop"
PREPARE_STATE_JSON = "full_prepare_state.json"

EXCLUSION_COLUMNS = [
    "split",
    "record_uid",
    "record_id",
    "group_id",
    "parquet_file",
    "shard_row_index",
    "reason",
    "detail",
]

ELIGIBLE_COLUMNS = [
    "record_uid",
    "record_id",
    "group_id",
    "recording_group_id",
    "pair_key",
    "source_split",
    "split",
    "parquet_file",
    "shard_row_index",
    "text_bahnar",
    "text_bahnar_norm",
    "duration_seconds",
    "processed_duration_seconds",
    "local_cache_relpath",
    "audio_source",  # reused_nb02_cache | prepared_local
    "sha256_pcm",
    "n_samples",
    "audio_pcm_pipeline_version",
]

# Single import surface for Notebook 03: the streaming prepare / hydrate /
# deterministic-PCM layers live in dedicated modules but are re-exported here.
from src.asr_full_pcm import (  # noqa: E402,F401  (re-export surface)
    AUDIO_PCM_PIPELINE_VERSION,
    DYNAMIC_PARQUET_REF,
    HfParquetRef,
    assert_disk_headroom,
    assert_local_disk_budget,
    assert_parquet_files_are_train_only,
    build_wav_bytes,
    fetch_parquet_shard_sizes,
    verify_pinned_parquet_snapshot,
    wav_bytes_for_samples,
    canonical_pcm16_from_array,
    AudioQaError,  # noqa: F401
    canonical_pcm16_from_bytes,
    cleanup_shard_download,
    delete_hf_cache_blob,
    disk_free_bytes,
    normalize_parquet_ref,
    parse_hf_parquet_url,
    read_pcm16_payload,
    safe_shard_key,
    write_wav_from_pcm16,
)
from src.asr_full_shards import (  # noqa: E402,F401  (re-export surface)
    MANIFEST_CONTENT_HASH_COLS,
    SIDECAR_REQUIRED_FIELDS,
    ShardPlan,
    ShardPrepareState,
    assert_sidecar_record_fields,
    assert_uid_set_accounting,
    build_shard_plans,
    compute_manifest_content_hash,
    expected_wav_bytes,
    hydrate_rows_audio,
    hydrate_shard_audio,
    hydrate_union_audio,
    load_shard_state,
    make_hf_parquet_stream_reader,
    local_audio_ok,
    plan_shard_hydrate,
    process_shard_streaming,
    run_full_prepare_streaming,
    save_shard_state,
    shard_index_exists,
    shard_index_paths,
    verify_eligible_audio_with_index,
    verify_shard_sidecar_index,
    write_shard_sidecar_index,
)


# ---------------------------------------------------------------------------
# Filtering (deterministic, no I/O)
# ---------------------------------------------------------------------------

def filter_by_duration(
    df: pd.DataFrame,
    *,
    min_duration: float = FULL_PREPARE_MIN_DURATION,
    max_duration: float = FULL_PREPARE_MAX_DURATION,
    duration_col: str = "duration_seconds",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, excluded) by inclusive duration gate on metadata."""
    if df is None or len(df) == 0:
        empty = df.iloc[0:0].copy() if df is not None else pd.DataFrame()
        return empty, empty
    durs = pd.to_numeric(df[duration_col], errors="coerce")
    ok = durs.notna() & np.isfinite(durs.to_numpy(dtype=float, copy=False))
    ok &= (durs >= float(min_duration)) & (durs <= float(max_duration))
    kept = df.loc[ok].copy().reset_index(drop=True)
    excl = df.loc[~ok].copy().reset_index(drop=True)
    return kept, excl


def filter_nonempty_normalized_text(
    df: pd.DataFrame,
    *,
    text_col: str = "text_bahnar",
    normalize_fn: Optional[Callable[[Any], str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, excluded) where kept has non-empty normalized transcript."""
    if normalize_fn is None:
        normalize_fn = normalize_bahnar_ctc_v1
    if df is None or len(df) == 0:
        empty = df.iloc[0:0].copy() if df is not None else pd.DataFrame()
        return empty, empty
    norms = [normalize_fn(t) for t in df[text_col].tolist()]
    ok = [(n is not None and str(n).strip() != "") for n in norms]
    out = df.copy()
    out["text_bahnar_norm"] = norms
    kept = out.loc[ok].copy().reset_index(drop=True)
    excl = out.loc[[not x for x in ok]].copy().reset_index(drop=True)
    return kept, excl


def check_split_overlaps(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cols: Sequence[str] = ("record_uid", "group_id", "recording_group_id", "pair_key"),
) -> Dict[str, int]:
    """
    Count overlaps between train/validation for each key column.

    Missing required columns fail closed (raise) — never silent -1.
    """
    missing = [c for c in cols if c not in train_df.columns or c not in val_df.columns]
    if missing:
        raise RuntimeError(
            f"Overlap check missing required columns (fail-closed): {missing}"
        )
    overlaps: Dict[str, int] = {}
    for col in cols:
        a = set(train_df[col].dropna().astype(str))
        b = set(val_df[col].dropna().astype(str))
        overlaps[col] = len(a & b)
    return overlaps


def pair_keys_overlapping_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    pair_col: str = "pair_key",
) -> Set[str]:
    """``pair_key`` values present in both splits (empty string keys ignored)."""
    if pair_col not in train_df.columns or pair_col not in val_df.columns:
        raise RuntimeError(
            f"pair_key overlap requires column {pair_col!r} on both frames (fail-closed)"
        )
    train_keys = {str(x) for x in train_df[pair_col].dropna().astype(str) if str(x)}
    val_keys = {str(x) for x in val_df[pair_col].dropna().astype(str) if str(x)}
    return train_keys & val_keys


def drop_train_pair_key_overlaps(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    pair_col: str = "pair_key",
    uid_col: str = "record_uid",
    reason: str = PAIR_KEY_TRAIN_EXCLUSION_REASON,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Remove from **train only** every row whose ``pair_key`` appears in validation.

    Validation is returned unchanged. Dropped train rows become an exclusions
    frame with ``reason`` set; the count is derived from the data, never hardcoded.
    Intra-train duplicate ``pair_key`` values that do not appear in validation
    are left alone.
    """
    overlapping = pair_keys_overlapping_validation(train_df, val_df, pair_col=pair_col)
    if not overlapping:
        empty = pd.DataFrame(columns=EXCLUSION_COLUMNS)
        return train_df.copy().reset_index(drop=True), empty, {
            "overlap_policy": OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
            "overlapping_pair_keys": 0,
            "train_rows_dropped": 0,
            "validation_rows": int(len(val_df)),
            "train_rows_kept": int(len(train_df)),
        }

    mask = train_df[pair_col].astype(str).isin(overlapping)
    dropped = train_df.loc[mask].copy()
    kept = train_df.loc[~mask].copy().reset_index(drop=True)
    excl_rows: List[Dict[str, Any]] = []
    for _, row in dropped.iterrows():
        excl_rows.append(
            {
                "split": "train",
                "record_uid": str(row.get(uid_col, "")),
                "record_id": str(row.get("record_id", "")),
                "group_id": str(row.get("group_id", "")),
                "parquet_file": str(row.get("parquet_file", "")),
                "shard_row_index": row.get("shard_row_index", ""),
                "reason": reason,
                "detail": f"{pair_col}={row.get(pair_col)}",
            }
        )
    excl = pd.DataFrame(excl_rows, columns=EXCLUSION_COLUMNS)
    report = {
        "overlap_policy": OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
        "overlapping_pair_keys": int(len(overlapping)),
        "train_rows_dropped": int(len(dropped)),
        "validation_rows": int(len(val_df)),
        "train_rows_kept": int(len(kept)),
        "validation_unchanged": True,
    }
    return kept, excl, report


def write_effective_manifests(
    *,
    state_dir: Union[str, Path],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_exclusions: Optional[pd.DataFrame] = None,
    source_train_path: Optional[str] = None,
    source_val_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persist effective train/validation manifests for Notebooks 03–05.

    Does not modify the original ``rq1_*.csv`` files. Validation is written as
    provided; train is the post-pair_key-drop frame.
    """
    from src.asr_full_shards import compute_manifest_content_hash

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    train_out = train_df.copy().reset_index(drop=True)
    val_out = val_df.copy().reset_index(drop=True)
    excl = (
        train_exclusions
        if train_exclusions is not None
        else pd.DataFrame(columns=EXCLUSION_COLUMNS)
    )
    train_path = state_dir / EFFECTIVE_TRAIN_MANIFEST
    val_path = state_dir / EFFECTIVE_VAL_MANIFEST
    excl_path = state_dir / EFFECTIVE_EXCLUSIONS_CSV
    _atomic_write_csv(train_path, train_out)
    _atomic_write_csv(val_path, val_out)
    _atomic_write_csv(excl_path, excl.reindex(columns=EXCLUSION_COLUMNS), EXCLUSION_COLUMNS)
    summary = {
        "overlap_policy": OVERLAP_POLICY_PAIR_KEY_TRAIN_DROP,
        "source_train_path": source_train_path,
        "source_validation_path": source_val_path,
        "effective_train_path": str(train_path),
        "effective_validation_path": str(val_path),
        "effective_exclusions_path": str(excl_path),
        "train_count": int(len(train_out)),
        "validation_count": int(len(val_out)),
        "exclusion_count": int(len(excl)),
        "train_manifest_content_hash": compute_manifest_content_hash(train_out),
        "validation_manifest_content_hash": compute_manifest_content_hash(val_out),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (state_dir / EFFECTIVE_MANIFEST_SUMMARY).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def estimate_hydrate_wav_bytes(eligible_df: pd.DataFrame) -> int:
    """Sum exact canonical WAV sizes from eligible ``n_samples`` (fail if missing)."""
    from src.asr_full_pcm import wav_bytes_for_samples

    if eligible_df is None or len(eligible_df) == 0:
        return 0
    if "n_samples" not in eligible_df.columns:
        raise RuntimeError(
            "Hydrate disk preflight requires n_samples on eligible rows (fail-closed)"
        )
    total = 0
    for raw in eligible_df["n_samples"].tolist():
        try:
            n = int(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid n_samples in eligible frame: {raw!r}") from exc
        if n < 0:
            raise RuntimeError(f"Negative n_samples in eligible frame: {n}")
        total += wav_bytes_for_samples(n)
    return int(total)


def assert_no_frozen_paths(paths: Iterable[Any]) -> None:
    assert_no_forbidden_paths(paths)


def assert_no_forbidden_paths(paths: Iterable[Any]) -> None:
    bad = [str(p) for p in paths if is_forbidden_test_path(str(p))]
    if bad:
        raise RuntimeError(f"Frozen-test path access forbidden: {bad[:5]}")

def assert_row_accounting(
    *,
    clean_count: int,
    eligible_count: int,
    exclusion_count: int,
    split: str,
) -> None:
    """SUCCESS accounting: eligible + exclusions must equal clean manifest count."""
    total = int(eligible_count) + int(exclusion_count)
    if total != int(clean_count):
        raise RuntimeError(
            f"Row accounting failed for split={split!r}: "
            f"eligible({eligible_count}) + exclusions({exclusion_count}) = {total} "
            f"!= clean_count({clean_count})"
        )


def vocab_fingerprint(vocab: Dict[str, int]) -> str:
    """Stable fingerprint of CTC vocab mapping."""
    import hashlib

    items = sorted((str(k), int(v)) for k, v in (vocab or {}).items())
    blob = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def manifest_contract_fingerprint(
    clean_df: pd.DataFrame,
    *,
    uid_col: str = "record_uid",
) -> Dict[str, Any]:
    """Hash + count for clean manifest binding into prepare resume state."""
    from src.data_utils import compute_ordered_uid_hash

    if uid_col not in clean_df.columns:
        return {"manifest_count": int(len(clean_df)), "manifest_uid_hash": ""}
    return {
        "manifest_count": int(len(clean_df)),
        "manifest_uid_hash": compute_ordered_uid_hash(clean_df, uid_col=uid_col),
    }


def verify_cached_wav_usable(
    path: Union[str, Path],
    *,
    record_uid: str,
    dataset_revision: str,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    expected_processing_version: str,
) -> Dict[str, Any]:
    """
    Re-validate a cached WAV for training after a session reset.

    Checks existence + sample rate + duration + finite + checksum + provenance
    via classify_cache_status (not mere path existence).
    """
    return classify_cache_status(
        Path(path),
        record_uid=record_uid,
        dataset_revision=dataset_revision,
        target_sr=target_sr,
        min_duration=min_duration,
        max_duration=max_duration,
        expected_processing_version=expected_processing_version,
    )


FROZEN_SPLIT_LABELS = frozenset({"test", "rq1_test", "frozen_test"})


def normalize_split_label(value: Any) -> str:
    """Fold case, whitespace and ``-``/``_`` so one label has one spelling."""
    token = str(value or "").strip().lower()
    for ch in (" ", "\t", "-"):
        token = token.replace(ch, "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def is_frozen_split_label(value: Any) -> bool:
    """
    Single frozen-split predicate shared by prepare, train and evaluate.

    ``Test``, ``rq1-test`` and ``frozen test`` all normalise to the same token, so
    the guard cannot be dodged by spelling.
    """
    return normalize_split_label(value) in FROZEN_SPLIT_LABELS


NB02_SIDECAR_REQUIRED_FIELDS = (
    "record_uid",
    "dataset_revision",
    "audio_processing_version",
    "target_sampling_rate",
    "cache_sha256",
)


def assert_sidecar_required_fields(
    sidecar: Optional[Dict[str, Any]],
    *,
    record_uid: str,
    dataset_revision: str,
    expected_processing_version: str,
    target_sr: int,
) -> None:
    """
    Fail-closed validation of an NB02 per-file cache sidecar.

    A sidecar without a valid ``cache_sha256`` cannot prove the WAV survived a
    session reset intact, so it is rejected rather than trusted.
    """
    from src.data_utils import is_valid_sha256

    if not sidecar:
        raise RuntimeError(f"Cache sidecar missing for uid={record_uid!r}")
    missing = [k for k in NB02_SIDECAR_REQUIRED_FIELDS if sidecar.get(k) in (None, "")]
    if missing:
        raise RuntimeError(f"Cache sidecar missing fields {missing} for uid={record_uid!r}")
    if str(sidecar.get("record_uid")) != str(record_uid):
        raise RuntimeError(
            f"Cache sidecar record_uid {sidecar.get('record_uid')!r} != {record_uid!r}"
        )
    if str(sidecar.get("dataset_revision")) != str(dataset_revision):
        raise RuntimeError(
            f"Cache sidecar dataset_revision {sidecar.get('dataset_revision')!r} "
            f"!= {dataset_revision!r} for uid={record_uid!r}"
        )
    if str(sidecar.get("audio_processing_version")) != str(expected_processing_version):
        raise RuntimeError(
            f"Cache sidecar audio_processing_version "
            f"{sidecar.get('audio_processing_version')!r} != {expected_processing_version!r}"
        )
    if int(sidecar.get("target_sampling_rate") or 0) != int(target_sr):
        raise RuntimeError(
            f"Cache sidecar target_sampling_rate {sidecar.get('target_sampling_rate')!r} "
            f"!= {target_sr!r} for uid={record_uid!r}"
        )
    if not is_valid_sha256(sidecar.get("cache_sha256")):
        raise RuntimeError(
            f"Cache sidecar cache_sha256 invalid for uid={record_uid!r}: "
            f"{sidecar.get('cache_sha256')!r}"
        )


# ---------------------------------------------------------------------------
# Shard grouping / sequential batch reading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prepare state
# ---------------------------------------------------------------------------

@dataclass
class FullPrepareState:
    dataset_id: str
    dataset_revision: str
    split: str
    completed_shards: List[str] = field(default_factory=list)
    pending_shards: List[str] = field(default_factory=list)
    n_eligible: int = 0
    n_excluded: int = 0
    finished: bool = False
    updated_at_utc: str = ""
    # Contract binding — invalidate resume if any of these change
    manifest_count: int = 0
    manifest_uid_hash: str = ""
    vocab_fp: str = ""
    processing_version: str = ""
    min_duration: float = FULL_PREPARE_MIN_DURATION
    max_duration: float = FULL_PREPARE_MAX_DURATION
    target_sr: int = 16000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FullPrepareState":
        return cls(
            dataset_id=str(d.get("dataset_id", "")),
            dataset_revision=str(d.get("dataset_revision", "")),
            split=str(d.get("split", "")),
            completed_shards=list(d.get("completed_shards") or []),
            pending_shards=list(d.get("pending_shards") or []),
            n_eligible=int(d.get("n_eligible") or 0),
            n_excluded=int(d.get("n_excluded") or 0),
            finished=bool(d.get("finished")),
            updated_at_utc=str(d.get("updated_at_utc") or ""),
            manifest_count=int(d.get("manifest_count") or 0),
            manifest_uid_hash=str(d.get("manifest_uid_hash") or ""),
            vocab_fp=str(d.get("vocab_fp") or ""),
            processing_version=str(d.get("processing_version") or ""),
            min_duration=float(d.get("min_duration") if d.get("min_duration") is not None else FULL_PREPARE_MIN_DURATION),
            max_duration=float(d.get("max_duration") if d.get("max_duration") is not None else FULL_PREPARE_MAX_DURATION),
            target_sr=int(d.get("target_sr") or 16000),
        )

    def matches_contract(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        split: str,
        manifest_count: int,
        manifest_uid_hash: str,
        vocab_fp: str,
        processing_version: str,
        min_duration: float,
        max_duration: float,
        target_sr: int,
    ) -> bool:
        return (
            self.dataset_id == str(dataset_id)
            and self.dataset_revision == str(dataset_revision)
            and self.split == str(split)
            and int(self.manifest_count) == int(manifest_count)
            and str(self.manifest_uid_hash) == str(manifest_uid_hash)
            and str(self.vocab_fp) == str(vocab_fp)
            and str(self.processing_version) == str(processing_version)
            and float(self.min_duration) == float(min_duration)
            and float(self.max_duration) == float(max_duration)
            and int(self.target_sr) == int(target_sr)
        )


def load_prepare_state(path: Path) -> Optional[FullPrepareState]:
    if not path.is_file():
        return None
    return FullPrepareState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_prepare_state(path: Path, state: FullPrepareState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at_utc = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_csv(path: Path, df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out = df.reindex(columns=list(columns)) if columns is not None else df
    out.to_csv(tmp, index=False)
    tmp.replace(path)


def _dedupe_records_by_uid(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep first occurrence per record_uid (deterministic for resume reloads)."""
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        uid = str(r.get("record_uid"))
        if uid in seen:
            continue
        seen.add(uid)
        out.append(r)
    return out


def exclusion_row(
    *,
    split: str,
    row: Any,
    reason: str,
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "split": split,
        "record_uid": str(row.get("record_uid")) if hasattr(row, "get") else None,
        "record_id": row.get("record_id") if hasattr(row, "get") else None,
        "group_id": row.get("group_id") if hasattr(row, "get") else None,
        "parquet_file": row.get("parquet_file") if hasattr(row, "get") else None,
        "shard_row_index": row.get("shard_row_index") if hasattr(row, "get") else None,
        "reason": reason,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Per-record audio eligibility (injectable I/O for tests)
# ---------------------------------------------------------------------------

def assert_eligible_frames_no_frozen_splits(df: pd.DataFrame, *, label: str = "eligible") -> None:
    """Fail-closed: eligible CSV must not contain frozen/test split labels."""
    for col in ("source_split", "split"):
        if col not in df.columns:
            continue
        bad = {
            raw for raw in df[col].dropna().astype(str).unique()
            if is_frozen_split_label(raw)
        }
        if bad:
            raise RuntimeError(f"Forbidden split values in {label}.{col}: {bad}")


def evaluate_record_ctc(
    text_bahnar: Any,
    duration_sec: float,
    *,
    vocab: Dict[str, int],
    target_sr: int,
    normalize_fn: Optional[Callable[[Any], str]] = None,
    encode_fn: Optional[Callable[..., List[int]]] = None,
) -> Dict[str, Any]:
    if normalize_fn is None:
        normalize_fn = normalize_bahnar_ctc_v1
    if encode_fn is None:
        encode_fn = encode_text_with_vocab
    norm = normalize_fn(text_bahnar)
    if norm is None or str(norm).strip() == "":
        return {"ok": False, "reason": "empty_normalized_reference", "norm": ""}
    tokens = encode_fn(norm, vocab)
    feas = check_ctc_feasibility(tokens, int(float(duration_sec) * int(target_sr)))
    if not feas.get("feasible"):
        return {"ok": False, "reason": feas.get("reason") or "ctc_infeasible", "norm": norm, "feas": feas}
    return {"ok": True, "reason": "ok", "norm": norm, "feas": feas}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prefilter_clean_split(
    clean_df: pd.DataFrame,
    *,
    split: str,
    min_duration: float = FULL_PREPARE_MIN_DURATION,
    max_duration: float = FULL_PREPARE_MAX_DURATION,
    normalize_fn: Optional[Callable[[Any], str]] = None,
) -> Dict[str, Any]:
    """
    Apply duration + nonempty-text filters before any parquet I/O.
    Returns candidates DataFrame and exclusion rows.
    """
    assert_no_forbidden_paths(
        [str(p) for p in clean_df.get("parquet_file", pd.Series(dtype=str)).dropna().astype(str).tolist()]
    )
    # Forbidden split names in columns (case / hyphen insensitive).
    for col in ("source_split", "split"):
        if col in clean_df.columns:
            bad = {
                raw for raw in clean_df[col].dropna().astype(str).unique()
                if is_frozen_split_label(raw)
            }
            if bad:
                raise RuntimeError(f"Forbidden split values in {col}: {bad}")

    dur_kept, dur_excl = filter_by_duration(
        clean_df, min_duration=min_duration, max_duration=max_duration
    )
    exclusions = [
        exclusion_row(split=split, row=r, reason="duration_out_of_range")
        for _, r in dur_excl.iterrows()
    ]
    txt_kept, txt_excl = filter_nonempty_normalized_text(
        dur_kept, normalize_fn=normalize_fn
    )
    exclusions.extend(
        exclusion_row(split=split, row=r, reason="empty_normalized_reference")
        for _, r in txt_excl.iterrows()
    )
    return {"candidates": txt_kept, "exclusions": exclusions}


def finalize_full_prepare(
    *,
    state_dir: Path,
    train_result: Dict[str, Any],
    val_result: Dict[str, Any],
    data_contract: Dict[str, Any],
    dataset_id: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    parquet_revision: Optional[str] = None,
    expected_train_clean_count: Optional[int] = None,
    expected_validation_clean_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Write durable FULL_STATE_DIR artifacts and derive status.

    The canonical ``data_contract`` (with its ``contract_hash``) is embedded in
    the summary so later stages can bind to it field by field instead of
    trusting a bare status string.

    SUCCESS_FULL_PREPARE only if:
      - both splits finished all shards
      - overlaps are zero (missing columns fail)
      - UID *set* accounting holds per split (eligible ⊎ exclusions == clean)
      - optional expected clean counts match
    """
    if not isinstance(data_contract, dict) or not data_contract.get("contract_hash"):
        raise ValueError("finalize_full_prepare requires a canonical data_contract")
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    train_df = train_result["eligible_df"]
    val_df = val_result["eligible_df"]
    excl = pd.concat(
        [train_result["exclusions_df"], val_result["exclusions_df"]],
        ignore_index=True,
    ) if (len(train_result["exclusions_df"]) or len(val_result["exclusions_df"])) else pd.DataFrame(columns=EXCLUSION_COLUMNS)

    # Fail-closed: never pad missing overlap columns
    required_ov = ("record_uid", "group_id", "recording_group_id", "pair_key")
    if len(train_df) == 0 or len(val_df) == 0:
        overlaps = {c: 0 for c in required_ov}
        overlaps_ok = False
    else:
        missing = [c for c in required_ov if c not in train_df.columns or c not in val_df.columns]
        if missing:
            raise RuntimeError(
                f"Overlap check missing required columns (fail-closed): {missing}"
            )
        overlaps = check_split_overlaps(train_df, val_df)
        overlaps_ok = all(v == 0 for v in overlaps.values())

    try:
        assert_eligible_frames_no_frozen_splits(train_df, label="full_train_eligible")
        assert_eligible_frames_no_frozen_splits(val_df, label="full_validation_eligible")
        frozen_eligible_ok = True
    except RuntimeError as exc:
        frozen_eligible_ok = False
        frozen_eligible_error = str(exc)
    else:
        frozen_eligible_error = None

    train_done = bool(train_result.get("all_shards_done"))
    val_done = bool(val_result.get("all_shards_done"))
    train_accounting = bool(train_result.get("accounting_ok", False))
    val_accounting = bool(val_result.get("accounting_ok", False))

    # Cross-check expected clean counts when provided
    counts_ok = True
    if expected_train_clean_count is not None:
        counts_ok = counts_ok and int(train_result.get("clean_count", -1)) == int(expected_train_clean_count)
    if expected_validation_clean_count is not None:
        counts_ok = counts_ok and int(val_result.get("clean_count", -1)) == int(expected_validation_clean_count)

    status = (
        STATUS_FULL_PREPARE
        if (
            train_done and val_done
            and overlaps_ok
            and train_accounting and val_accounting
            and counts_ok
            and frozen_eligible_ok
            and len(train_df) > 0 and len(val_df) > 0
        )
        else STATUS_FAILED
    )

    train_out = train_df.reindex(columns=ELIGIBLE_COLUMNS) if len(train_df) else pd.DataFrame(columns=ELIGIBLE_COLUMNS)
    val_out = val_df.reindex(columns=ELIGIBLE_COLUMNS) if len(val_df) else pd.DataFrame(columns=ELIGIBLE_COLUMNS)
    _atomic_write_csv(state_dir / ELIGIBLE_TRAIN_CSV, train_out, ELIGIBLE_COLUMNS)
    _atomic_write_csv(state_dir / ELIGIBLE_VAL_CSV, val_out, ELIGIBLE_COLUMNS)
    _atomic_write_csv(state_dir / EXCLUSIONS_CSV, excl, EXCLUSION_COLUMNS)

    st_train = train_result.get("state")
    summary = {
        "status": status,
        "dataset_id": dataset_id or (st_train.dataset_id if isinstance(st_train, FullPrepareState) else ""),
        "dataset_revision": dataset_revision or (
            st_train.dataset_revision if isinstance(st_train, FullPrepareState) else ""
        ),
        "train_eligible": int(len(train_out)),
        "validation_eligible": int(len(val_out)),
        "exclusions": int(len(excl)),
        "train_clean_count": int(train_result.get("clean_count") or 0),
        "validation_clean_count": int(val_result.get("clean_count") or 0),
        "train_manifest_uid_hash": train_result.get("manifest_uid_hash"),
        "validation_manifest_uid_hash": val_result.get("manifest_uid_hash"),
        "train_manifest_content_hash": train_result.get("manifest_content_hash"),
        "validation_manifest_content_hash": val_result.get("manifest_content_hash"),
        "parquet_revision": str(
            parquet_revision
            or train_result.get("parquet_revision")
            or (getattr(st_train, "parquet_revision", "") or "")
        ),
        "audio_pcm_pipeline_version": AUDIO_PCM_PIPELINE_VERSION,
        "train_uid_accounting": train_result.get("uid_accounting"),
        "validation_uid_accounting": val_result.get("uid_accounting"),
        "vocab_fp": train_result.get("vocab_fp"),
        "overlaps": overlaps,
        "overlaps_ok": overlaps_ok,
        "train_shards_done": train_done,
        "validation_shards_done": val_done,
        "train_accounting_ok": train_accounting,
        "validation_accounting_ok": val_accounting,
        "train_accounting_error": train_result.get("accounting_error"),
        "validation_accounting_error": val_result.get("accounting_error"),
        "expected_counts_ok": counts_ok,
        "frozen_eligible_ok": frozen_eligible_ok,
        "frozen_eligible_error": frozen_eligible_error,
        "data_contract": dict(data_contract),
        "contract_hash": str(data_contract["contract_hash"]),
        "prepare_state_schema_version": PREPARE_STATE_SCHEMA_VERSION,
        "audio_written": False,
        "note": (
            "Metadata-only prepare: eligible CSVs carry the canonical PCM hash; "
            "no WAV is kept, stages that need audio hydrate per shard."
        ),
    }

    (state_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    combined_state = {
        "status": status,
        "train": train_result["state"].to_dict() if hasattr(train_result["state"], "to_dict") else train_result["state"],
        "validation": val_result["state"].to_dict() if hasattr(val_result["state"], "to_dict") else val_result["state"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": summary["dataset_id"],
        "dataset_revision": summary["dataset_revision"],
    }
    tmp = state_dir / (PREPARE_STATE_JSON + ".tmp")
    tmp.write_text(json.dumps(combined_state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(state_dir / PREPARE_STATE_JSON)
    return summary


def prepare_contract_mismatches(
    summary: Dict[str, Any],
    expected_contract: Dict[str, Any],
) -> List[str]:
    """Field-by-field diff between a prepare summary's contract and the current one."""
    from src.asr_full_train import DATA_CONTRACT_KEYS

    recorded = summary.get("data_contract")
    if not isinstance(recorded, dict) or not recorded:
        return ["data_contract missing from summary"]
    problems: List[str] = []
    for key in DATA_CONTRACT_KEYS:
        want, got = expected_contract.get(key), recorded.get(key)
        if isinstance(want, float) or isinstance(got, float):
            same = want is not None and got is not None and float(want) == float(got)
        else:
            same = str(want) == str(got)
        if not same:
            problems.append(f"{key}: recorded={got!r} expected={want!r}")
    # The self-declared hash is never the authority; it only has to agree with
    # the fields, otherwise the summary was hand-edited.
    if str(recorded.get("contract_hash")) != str(expected_contract.get("contract_hash")):
        problems.append("contract_hash differs")
    version = summary.get("prepare_state_schema_version")
    if str(version) != PREPARE_STATE_SCHEMA_VERSION:
        problems.append(
            f"prepare_state_schema_version: recorded={version!r} "
            f"expected={PREPARE_STATE_SCHEMA_VERSION!r}"
        )
    return problems


def load_prepare_success(
    state_dir: Path,
    *,
    expected_contract: Dict[str, Any],
) -> bool:
    """
    True only for SUCCESS_FULL_PREPARE bound to the exact canonical contract.

    Every contract field must match, not just the status or a couple of hashes:
    a summary from another dataset, manifest content, vocab, audio pipeline,
    threshold, sample rate or model is not evidence that *this* run prepared.
    """
    summary_path = Path(state_dir) / SUMMARY_JSON
    if not summary_path.is_file():
        return False
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if data.get("status") != STATUS_FULL_PREPARE:
        return False
    return not prepare_contract_mismatches(data, expected_contract)


def derive_full_prepare_status(summary: Dict[str, Any]) -> str:
    return STATUS_FULL_PREPARE if summary.get("status") == STATUS_FULL_PREPARE else STATUS_FAILED
