"""
One-pass shard streaming prepare + per-shard audio hydrate for Notebook 03.

Design (metadata-only on durable storage):
  * Durable ``FULL_STATE_DIR`` holds ONLY state, eligible/exclusion CSVs
    and one sidecar index per shard. Never bulk WAV, never ~92k JSON files.
  * Bulk audio lives on local disk and is disposable: a new RunPod session
    re-hydrates it per shard straight from the pinned parquet snapshot and
    verifies bytes against the stored ``sha256_pcm`` — no re-running QA
    (duration gate / CTC feasibility / normalisation) for completed shards.
  * Each physical parquet shard is opened exactly once and serves BOTH the
    train and validation splits; rows stream in batches so a whole shard's
    audio never sits in RAM.
  * A shard is only marked complete after its sidecar index is written and
    verified, per-split UID accounting holds, and the state commit lands.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union,
)

import pandas as pd

from src.asr_full_pcm import (
    AUDIO_PCM_PIPELINE_VERSION,
    HfParquetRef,
    assert_disk_headroom,
    assert_local_disk_budget,
    assert_parquet_files_are_train_only,
    AudioQaError,
    canonical_pcm16_from_bytes,
    normalize_parquet_ref,
    read_pcm16_payload,
    safe_shard_key,
    sha256_hex,
    verify_pinned_parquet_snapshot,
    wav_bytes_for_samples,
    write_wav_from_pcm16,
)

AUDIO_INDEX_DIRNAME = "audio_index"
SHARD_INDEX_SUFFIX = ".jsonl"
# Room a shard download needs on top of the parquet itself: the ``.incomplete``
# temp file, decode buffers and filesystem slack.
SHARD_TEMP_FLOOR_BYTES = 256 * 1024 ** 2
SHARD_META_SUFFIX = ".meta.json"
SHARD_STATE_JSON = "full_prepare_shard_state.json"

SIDECAR_REQUIRED_FIELDS = (
    "record_uid",
    "sha256_pcm",
    "sha256_source",
    "n_samples",
    "sample_rate",
    "audio_pcm_pipeline_version",
    "parquet_revision",
    "dataset_revision",
    "shard_key",
)

# Row payload streamed out of a parquet shard.
ShardStreamReader = Callable[
    [HfParquetRef, Sequence[int]], Iterable[Tuple[int, Dict[str, Any]]]
]


# ---------------------------------------------------------------------------
# Manifest content hashing + exact UID set accounting
# ---------------------------------------------------------------------------

MANIFEST_CONTENT_HASH_COLS: Tuple[str, ...] = (
    "record_uid",
    "record_id",
    "pair_key",
    "group_id",
    "recording_group_id",
    "source_split",
    "split",
    "parquet_file",
    "shard_row_index",
    "text_bahnar",
    "duration_seconds",
)


def compute_manifest_content_hash(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = MANIFEST_CONTENT_HASH_COLS,
    uid_col: str = "record_uid",
) -> str:
    """
    Hash the manifest *content*, not only its UID list.

    A manifest that keeps the same UIDs but changes transcripts, durations or
    shard offsets must produce a different hash, otherwise resume/contract
    checks would accept silently different data.
    """
    if df is None:
        raise ValueError("manifest dataframe is required")
    present = [c for c in columns if c in df.columns]
    missing_required = [c for c in (uid_col, "text_bahnar", "duration_seconds") if c not in df.columns]
    if missing_required:
        raise RuntimeError(
            f"Manifest content hash missing required columns (fail-closed): {missing_required}"
        )
    work = df.loc[:, present].copy()
    for col in present:
        work[col] = work[col].astype(str)
    work = work.sort_values(list(present), kind="mergesort")
    payload = {
        "columns": present,
        "n_rows": int(len(work)),
        "rows": work.to_numpy().tolist(),
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return sha256_hex(blob.encode("utf-8"))


def assert_uid_set_accounting(
    *,
    clean_uids: Iterable[Any],
    eligible_uids: Iterable[Any],
    exclusion_uids: Iterable[Any],
    split: str,
) -> Dict[str, Any]:
    """
    Exact set accounting: eligible and exclusions partition the clean manifest.

    Counting alone (``len(a) + len(b) == n``) lets a duplicate cancel a missing
    row, so this compares sets and fails closed on overlap / missing / extra.
    """
    clean = {str(u) for u in clean_uids}
    elig = [str(u) for u in eligible_uids]
    excl = [str(u) for u in exclusion_uids]
    elig_set, excl_set = set(elig), set(excl)

    problems: List[str] = []
    if len(elig) != len(elig_set):
        problems.append(f"duplicate eligible uids ({len(elig) - len(elig_set)})")
    if len(excl) != len(excl_set):
        problems.append(f"duplicate exclusion uids ({len(excl) - len(excl_set)})")
    overlap = elig_set & excl_set
    if overlap:
        problems.append(f"uids in both eligible and exclusions ({len(overlap)}): {sorted(overlap)[:5]}")
    missing = clean - (elig_set | excl_set)
    if missing:
        problems.append(f"clean uids unaccounted ({len(missing)}): {sorted(missing)[:5]}")
    extra = (elig_set | excl_set) - clean
    if extra:
        problems.append(f"uids not in clean manifest ({len(extra)}): {sorted(extra)[:5]}")
    if problems:
        raise RuntimeError(f"UID set accounting failed for split={split!r}: " + "; ".join(problems))
    return {
        "split": split,
        "clean": len(clean),
        "eligible": len(elig_set),
        "exclusions": len(excl_set),
    }


# ---------------------------------------------------------------------------
# Per-shard sidecar index (durable on FULL_STATE_DIR, one file per shard)
# ---------------------------------------------------------------------------

def audio_index_dir(state_dir: Union[str, Path]) -> Path:
    return Path(state_dir) / AUDIO_INDEX_DIRNAME


def shard_index_paths(state_dir: Union[str, Path], shard_key: str) -> Tuple[Path, Path]:
    root = audio_index_dir(state_dir)
    token = safe_shard_key(shard_key)
    return root / f"{token}{SHARD_INDEX_SUFFIX}", root / f"{token}{SHARD_META_SUFFIX}"


def assert_sidecar_record_fields(record: Dict[str, Any]) -> None:
    """Fail-closed on an incomplete sidecar entry."""
    missing = [k for k in SIDECAR_REQUIRED_FIELDS if not str(record.get(k) or "").strip()]
    if missing:
        raise RuntimeError(
            f"Sidecar record missing required fields {missing}: uid={record.get('record_uid')!r}"
        )
    sha = str(record.get("sha256_pcm"))
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise RuntimeError(f"Sidecar sha256_pcm invalid: {sha!r}")
    if int(record.get("n_samples") or 0) <= 0:
        raise RuntimeError(f"Sidecar n_samples must be positive: {record.get('n_samples')!r}")


def write_shard_sidecar_index(
    state_dir: Union[str, Path],
    shard_key: str,
    records: Sequence[Dict[str, Any]],
    *,
    parquet_revision: str,
    dataset_revision: str,
) -> Dict[str, Any]:
    """
    Atomically write one JSONL sidecar index for a shard plus a hashed meta file.

    ~1.8k rows per shard for this dataset, i.e. 50 files instead of ~92k.
    """
    index_path, meta_path = shard_index_paths(state_dir, shard_key)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    seen: Set[str] = set()
    for rec in records:
        payload = dict(rec)
        payload.setdefault("shard_key", str(shard_key))
        payload.setdefault("parquet_revision", str(parquet_revision))
        payload.setdefault("dataset_revision", str(dataset_revision))
        payload.setdefault("audio_pcm_pipeline_version", AUDIO_PCM_PIPELINE_VERSION)
        assert_sidecar_record_fields(payload)
        uid = str(payload["record_uid"])
        if uid in seen:
            raise RuntimeError(f"Duplicate uid in shard sidecar index: {uid}")
        seen.add(uid)
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    body = ("\n".join(lines) + "\n") if lines else ""
    blob = body.encode("utf-8")

    tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_index.write_bytes(blob)
    os.replace(str(tmp_index), str(index_path))

    meta = {
        "shard_key": str(shard_key),
        "n_records": len(lines),
        "sha256_index": sha256_hex(blob),
        "parquet_revision": str(parquet_revision),
        "dataset_revision": str(dataset_revision),
        "audio_pcm_pipeline_version": AUDIO_PCM_PIPELINE_VERSION,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_meta), str(meta_path))
    return meta


def verify_shard_sidecar_index(
    state_dir: Union[str, Path],
    shard_key: str,
    *,
    expected_parquet_revision: Optional[str] = None,
    expected_dataset_revision: Optional[str] = None,
    expected_target_sr: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load + integrity-check a shard index. Returns {uid: record}. Fail-closed."""
    index_path, meta_path = shard_index_paths(state_dir, shard_key)
    if not index_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"Shard sidecar index missing for {shard_key!r} under {state_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    blob = index_path.read_bytes()
    if sha256_hex(blob) != str(meta.get("sha256_index")):
        raise RuntimeError(f"Shard sidecar index corrupt (hash mismatch) for {shard_key!r}")
    if str(meta.get("audio_pcm_pipeline_version")) != AUDIO_PCM_PIPELINE_VERSION:
        raise RuntimeError(
            f"Shard sidecar index pipeline version {meta.get('audio_pcm_pipeline_version')!r} "
            f"!= {AUDIO_PCM_PIPELINE_VERSION!r} for {shard_key!r}"
        )
    if expected_parquet_revision is not None and str(meta.get("parquet_revision")) != str(expected_parquet_revision):
        raise RuntimeError(
            f"Shard sidecar index parquet_revision {meta.get('parquet_revision')!r} "
            f"!= expected {expected_parquet_revision!r} for {shard_key!r}"
        )
    if expected_dataset_revision is not None and str(meta.get("dataset_revision")) != str(expected_dataset_revision):
        raise RuntimeError(
            f"Shard sidecar index dataset_revision {meta.get('dataset_revision')!r} "
            f"!= expected {expected_dataset_revision!r} for {shard_key!r}"
        )
    out: Dict[str, Dict[str, Any]] = {}
    for line in blob.decode("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        assert_sidecar_record_fields(rec)
        if expected_target_sr is not None and int(rec.get("sample_rate") or 0) != int(expected_target_sr):
            raise RuntimeError(
                f"Shard sidecar record sample_rate {rec.get('sample_rate')!r} "
                f"!= target {int(expected_target_sr)} for uid={rec.get('record_uid')!r} "
                f"in {shard_key!r}"
            )
        out[str(rec["record_uid"])] = rec
    # `or -1` would fold a legitimate n_records=0 into "missing", so an
    # all-excluded shard could never be verified. Absence and 0 are distinct.
    if "n_records" not in meta:
        raise RuntimeError(
            f"Shard sidecar index meta lacks n_records for {shard_key!r}"
        )
    try:
        n_expected = int(meta["n_records"])
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Shard sidecar index meta has non-numeric n_records "
            f"{meta.get('n_records')!r} for {shard_key!r}"
        ) from None
    if len(out) != n_expected:
        raise RuntimeError(
            f"Shard sidecar index count mismatch for {shard_key!r}: "
            f"{len(out)} != {n_expected}"
        )
    return out


def shard_index_exists(state_dir: Union[str, Path], shard_key: str) -> bool:
    index_path, meta_path = shard_index_paths(state_dir, shard_key)
    return index_path.is_file() and meta_path.is_file()


# ---------------------------------------------------------------------------
# Shard planning across both splits (single physical read per shard)
# ---------------------------------------------------------------------------

@dataclass
class ShardPlan:
    shard_key: str
    ref: HfParquetRef
    raw_parquet_file: str
    per_split: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def needed_indices(self) -> List[int]:
        idxs: Set[int] = set()
        for frame in self.per_split.values():
            vals = pd.to_numeric(frame["shard_row_index"], errors="coerce").dropna()
            idxs.update(int(v) for v in vals.tolist())
        return sorted(idxs)

    @property
    def n_rows(self) -> int:
        return sum(len(f) for f in self.per_split.values())


def build_shard_plans(
    candidates_by_split: Dict[str, pd.DataFrame],
    *,
    dataset_id: str,
    parquet_revision: str,
) -> List[ShardPlan]:
    """
    Group candidate rows of ALL splits by physical parquet shard.

    Train and validation of this dataset come from the same 50 shards, so a
    per-split loop would download and decode every shard twice.
    """
    plans: Dict[str, ShardPlan] = {}
    for split, frame in candidates_by_split.items():
        if frame is None or len(frame) == 0:
            continue
        if "parquet_file" not in frame.columns or "shard_row_index" not in frame.columns:
            raise RuntimeError(f"split={split!r} candidates missing parquet_file/shard_row_index")
        work = frame.copy()
        work["_shard_row"] = pd.to_numeric(work["shard_row_index"], errors="coerce")
        if work["_shard_row"].isna().any():
            bad = work.loc[work["_shard_row"].isna(), "record_uid"].astype(str).tolist()[:5]
            raise RuntimeError(f"split={split!r} has non-numeric shard_row_index (e.g. {bad})")
        for raw_pq, sub in work.groupby(work["parquet_file"].astype(str), sort=True):
            ref = normalize_parquet_ref(
                raw_pq, expected_repo_id=dataset_id, parquet_revision=parquet_revision
            )
            plan = plans.get(ref.shard_key)
            if plan is None:
                plan = ShardPlan(shard_key=ref.shard_key, ref=ref, raw_parquet_file=str(raw_pq))
                plans[ref.shard_key] = plan
            ordered = sub.sort_values(["_shard_row", "record_uid"]).drop(columns=["_shard_row"])
            plan.per_split[split] = ordered.reset_index(drop=True)
    return [plans[k] for k in sorted(plans)]


# ---------------------------------------------------------------------------
# Streaming shard processing
# ---------------------------------------------------------------------------

def process_shard_streaming(
    plan: ShardPlan,
    *,
    vocab: Dict[str, int],
    dataset_revision: str,
    parquet_revision: str,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    shard_reader: ShardStreamReader,
    local_audio_dir: Path,
    write_audio: bool = True,
) -> Dict[str, Any]:
    """
    Stream one shard once; emit eligible rows, exclusions and sidecar records.

    ``shard_reader`` yields ``(shard_row_index, payload)`` pairs where payload
    carries ``id`` and ``audio`` (raw encoded bytes). Rows are handled and
    released one at a time — the caller never materialises the shard in RAM.
    """
    from src.asr_full_data import evaluate_record_ctc, exclusion_row
    from src.data_utils import safe_cache_filename

    local_audio_dir = Path(local_audio_dir)
    if write_audio:
        local_audio_dir.mkdir(parents=True, exist_ok=True)

    # index rows by shard_row_index per split so a shared physical row can feed
    # both splits without a second read
    wanted: Dict[int, List[Tuple[str, Any]]] = {}
    for split, frame in plan.per_split.items():
        for _, row in frame.iterrows():
            wanted.setdefault(int(row["shard_row_index"]), []).append((split, row))

    eligible: Dict[str, List[Dict[str, Any]]] = {s: [] for s in plan.per_split}
    exclusions: Dict[str, List[Dict[str, Any]]] = {s: [] for s in plan.per_split}
    sidecars: List[Dict[str, Any]] = []
    seen_rows: Set[int] = set()

    for row_idx, payload in shard_reader(plan.ref, plan.needed_indices):
        idx = int(row_idx)
        if idx in seen_rows:
            raise RuntimeError(f"shard_reader yielded duplicate row {idx} for {plan.shard_key}")
        seen_rows.add(idx)
        targets = wanted.get(idx)
        if not targets:
            continue

        decoded: Optional[Dict[str, Any]] = None
        decode_error: Optional[str] = None
        raw_audio = payload.get("audio") if isinstance(payload, dict) else None
        if isinstance(raw_audio, dict):
            raw_audio = raw_audio.get("bytes")
        if not raw_audio:
            decode_error = "null_audio"
        else:
            try:
                decoded = canonical_pcm16_from_bytes(bytes(raw_audio), target_sr=int(target_sr))
            except AudioQaError as exc:
                decode_error = exc.reason
            except Exception as exc:  # noqa: BLE001 - recorded as exclusion
                decode_error = f"decode_error:{type(exc).__name__}"

        for split, row in targets:
            uid = str(row["record_uid"])
            rid = str(row["record_id"])
            payload_id = payload.get("id") if isinstance(payload, dict) else None
            if payload_id is not None and str(payload_id) != rid:
                exclusions[split].append(exclusion_row(
                    split=split, row=row, reason="record_id_mismatch",
                    detail=f"expected={rid} got={payload_id}",
                ))
                continue
            if decoded is None:
                exclusions[split].append(exclusion_row(
                    split=split, row=row, reason=f"audio_qa_{decode_error or 'fail'}",
                ))
                continue
            dur = float(decoded["duration_seconds"])
            if not (float(min_duration) <= dur <= float(max_duration)):
                exclusions[split].append(exclusion_row(
                    split=split, row=row, reason="decoded_duration_out_of_range",
                    detail=f"{dur:.3f}s",
                ))
                continue
            ctc = evaluate_record_ctc(
                row.get("text_bahnar"), dur, vocab=vocab, target_sr=int(target_sr)
            )
            if not ctc["ok"]:
                exclusions[split].append(exclusion_row(
                    split=split, row=row, reason=str(ctc["reason"]),
                ))
                continue

            rel = safe_cache_filename(uid)
            if write_audio:
                write_wav_from_pcm16(local_audio_dir / rel, decoded["pcm"], int(target_sr))
            eligible[split].append({
                **{c: row.get(c) for c in (
                    "record_uid", "record_id", "group_id", "recording_group_id", "pair_key",
                    "source_split", "split", "parquet_file", "shard_row_index", "text_bahnar",
                    "duration_seconds",
                )},
                "text_bahnar_norm": ctc["norm"],
                "processed_duration_seconds": dur,
                "local_cache_relpath": rel,
                "audio_source": "prepared_local",
                "sha256_pcm": decoded["sha256_pcm"],
                "n_samples": int(decoded["n_samples"]),
                "audio_pcm_pipeline_version": AUDIO_PCM_PIPELINE_VERSION,
            })
            sidecars.append({
                "record_uid": uid,
                "record_id": rid,
                "split": split,
                "shard_key": plan.shard_key,
                "shard_row_index": int(idx),
                "sha256_pcm": decoded["sha256_pcm"],
                "sha256_source": decoded["sha256_source"],
                "n_samples": int(decoded["n_samples"]),
                "sample_rate": int(decoded["sample_rate"]),
                "source_sample_rate": int(decoded["source_sample_rate"]),
                "source_subtype": str(decoded.get("source_subtype") or ""),
                "source_channels": int(decoded.get("source_channels") or 1),
                "resampled": bool(decoded.get("resampled")),
                "local_cache_relpath": rel,
                "audio_pcm_pipeline_version": AUDIO_PCM_PIPELINE_VERSION,
                "parquet_revision": str(parquet_revision),
                "dataset_revision": str(dataset_revision),
            })
        del decoded, raw_audio

    # rows the reader never produced
    for idx, targets in wanted.items():
        if idx in seen_rows:
            continue
        for split, row in targets:
            exclusions[split].append(exclusion_row(
                split=split, row=row, reason="audio_row_missing_in_shard",
            ))
    return {"eligible": eligible, "exclusions": exclusions, "sidecars": sidecars}


# ---------------------------------------------------------------------------
# Hydrate (rematerialize local WAV for an already-prepared shard)
# ---------------------------------------------------------------------------

def local_audio_ok(
    path: Union[str, Path],
    *,
    expected_sha256_pcm: str,
    expected_n_samples: Optional[int] = None,
    expected_sample_rate: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Byte-level check of a hydrated WAV against the stored canonical payload.

    The PCM hash covers samples only, so a file with the right bytes but a wrong
    header sample rate would otherwise pass and train at the wrong speed.
    """
    p = Path(path)
    if not p.is_file():
        return False, "missing"
    try:
        got = read_pcm16_payload(p)
    except ValueError as exc:  # non-mono and other contract violations
        return False, f"not_mono_or_invalid:{exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable:{type(exc).__name__}"
    if int(got["channels"]) != 1:
        return False, f"not_mono:{got['channels']}"
    if str(got["subtype"]).upper() != "PCM_16":
        return False, f"not_pcm16:{got['subtype']}"
    if got["sha256_pcm"] != str(expected_sha256_pcm):
        return False, "sha256_pcm_mismatch"
    if expected_n_samples is not None and int(got["n_samples"]) != int(expected_n_samples):
        return False, "n_samples_mismatch"
    if expected_sample_rate is not None and int(got["sample_rate"]) != int(expected_sample_rate):
        return False, f"sample_rate_mismatch:{got['sample_rate']}!={int(expected_sample_rate)}"
    return True, "ok"


def plan_shard_hydrate(
    state_dir: Union[str, Path],
    shard_key: str,
    *,
    local_audio_dir: Union[str, Path],
    uids: Optional[Iterable[str]] = None,
    expected_parquet_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Which UIDs of a completed shard still need local audio (no QA re-run)."""
    index = verify_shard_sidecar_index(
        state_dir, shard_key, expected_parquet_revision=expected_parquet_revision
    )
    wanted = {str(u) for u in uids} if uids is not None else set(index)
    unknown = sorted(wanted - set(index))
    if unknown:
        raise RuntimeError(
            f"UIDs absent from shard index {shard_key!r} (fail-closed): {unknown[:5]}"
        )
    root = Path(local_audio_dir)
    missing: List[str] = []
    reasons: Dict[str, str] = {}
    for uid in sorted(wanted):
        rec = index[uid]
        ok, reason = local_audio_ok(
            root / str(rec["local_cache_relpath"]),
            expected_sha256_pcm=str(rec["sha256_pcm"]),
            expected_n_samples=int(rec["n_samples"]),
            expected_sample_rate=int(rec["sample_rate"]),
        )
        if not ok:
            missing.append(uid)
            reasons[uid] = reason
    return {"index": index, "missing": missing, "reasons": reasons, "n_wanted": len(wanted)}


def hydrate_shard_audio(
    state_dir: Union[str, Path],
    plan: ShardPlan,
    *,
    local_audio_dir: Union[str, Path],
    shard_reader: ShardStreamReader,
    target_sr: int = 16000,
    uids: Optional[Iterable[str]] = None,
    expected_parquet_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Re-create local WAVs for one already-prepared shard, straight from parquet.

    Verifies each re-decoded payload against the stored ``sha256_pcm`` and fails
    closed on mismatch; ``sha256_source`` is compared too so the error message
    can tell "upstream bytes changed" from "our PCM pipeline changed".
    """
    hydrate_plan = plan_shard_hydrate(
        state_dir, plan.shard_key,
        local_audio_dir=local_audio_dir, uids=uids,
        expected_parquet_revision=expected_parquet_revision,
    )
    index = hydrate_plan["index"]
    missing = hydrate_plan["missing"]
    if not missing:
        return {"shard_key": plan.shard_key, "hydrated": 0, "verified": hydrate_plan["n_wanted"]}

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for uid in missing:
        rec = index[uid]
        by_row.setdefault(int(rec["shard_row_index"]), []).append(rec)
    root = Path(local_audio_dir)
    root.mkdir(parents=True, exist_ok=True)

    hydrated = 0
    for row_idx, payload in shard_reader(plan.ref, sorted(by_row)):
        recs = by_row.get(int(row_idx))
        if not recs:
            continue
        raw = payload.get("audio") if isinstance(payload, dict) else None
        if isinstance(raw, dict):
            raw = raw.get("bytes")
        if not raw:
            raise RuntimeError(
                f"Hydrate failed: no audio bytes for row {row_idx} of {plan.shard_key}"
            )
        decoded = canonical_pcm16_from_bytes(bytes(raw), target_sr=int(target_sr))
        for rec in recs:
            uid = str(rec["record_uid"])
            if decoded["sha256_pcm"] != str(rec["sha256_pcm"]):
                same_source = decoded.get("sha256_source") == rec.get("sha256_source")
                cause = (
                    "PCM pipeline drift (source bytes identical)"
                    if same_source else "upstream source bytes changed"
                )
                raise RuntimeError(
                    f"Hydrate SHA mismatch for uid={uid} in {plan.shard_key}: {cause}; "
                    f"expected_pcm={rec['sha256_pcm']} got={decoded['sha256_pcm']} "
                    f"pipeline={AUDIO_PCM_PIPELINE_VERSION}"
                )
            write_wav_from_pcm16(root / str(rec["local_cache_relpath"]), decoded["pcm"], int(target_sr))
            hydrated += 1
        del decoded, raw
    still_missing = [
        uid for uid in missing
        if not local_audio_ok(
            root / str(index[uid]["local_cache_relpath"]),
            expected_sha256_pcm=str(index[uid]["sha256_pcm"]),
        )[0]
    ]
    if still_missing:
        raise RuntimeError(
            f"Hydrate incomplete for {plan.shard_key}: {len(still_missing)} uids "
            f"still unusable (e.g. {still_missing[:5]})"
        )
    return {
        "shard_key": plan.shard_key,
        "hydrated": hydrated,
        "verified": hydrate_plan["n_wanted"],
    }


def expected_wav_bytes(frames: Sequence[pd.DataFrame]) -> Dict[str, Any]:
    """
    Exact local WAV footprint for the union of eligible frames, deduped by UID.

    Derived from the recorded ``n_samples`` of each record, so the number tracks
    the real artifacts instead of a hardcoded hours-to-GB guess.
    """
    seen: Set[str] = set()
    total = 0
    samples = 0
    for frame in frames:
        if frame is None or not len(frame):
            continue
        if "n_samples" not in frame.columns:
            raise RuntimeError("eligible frame lacks n_samples (re-run FULL_STAGE=prepare)")
        for uid, n in zip(frame["record_uid"], frame["n_samples"]):
            key = str(uid)
            if key in seen:
                continue
            seen.add(key)
            samples += int(n)
            total += wav_bytes_for_samples(n)
    return {
        "unique_records": len(seen),
        "n_samples": samples,
        "wav_bytes": total,
        "audio_hours": samples / 16000.0 / 3600.0,
    }


def verify_eligible_audio_with_index(
    df: pd.DataFrame,
    cache_roots: Sequence[Union[str, Path]],
    *,
    max_list: int = 8,
    target_sr: int = 16000,
) -> Dict[str, Any]:
    """
    Verify eligible rows against the ``sha256_pcm`` carried in the eligible CSV.

    O(1) per row and header-independent, unlike the NB02 per-file sidecar path.
    """
    if "sha256_pcm" not in df.columns:
        raise RuntimeError("eligible frame lacks sha256_pcm (re-run FULL_STAGE=prepare)")
    missing: List[str] = []
    bad: List[str] = []
    for _, row in df.iterrows():
        uid = str(row["record_uid"])
        rel = str(row.get("local_cache_relpath") or "")
        found = None
        for root in cache_roots:
            cand = Path(root) / rel
            if rel and cand.is_file():
                found = cand
                break
        if found is None:
            missing.append(uid)
            continue
        ok, reason = local_audio_ok(
            found,
            expected_sha256_pcm=str(row["sha256_pcm"]),
            expected_n_samples=int(row["n_samples"]) if "n_samples" in df.columns and pd.notna(row.get("n_samples")) else None,
            expected_sample_rate=int(target_sr),
        )
        if not ok:
            bad.append(f"{uid}:{reason}")
    if missing:
        raise RuntimeError(
            f"{len(missing)} eligible records have no local audio "
            f"(examples={missing[:max_list]}); hydrate the owning shards first."
        )
    if bad:
        raise RuntimeError(
            f"{len(bad)} eligible audio files failed PCM hash verification "
            f"(examples={bad[:max_list]})."
        )
    return {"checked": int(len(df)), "missing": 0, "corrupt": 0}


def hydrate_rows_audio(
    rows_df: pd.DataFrame,
    *,
    state_dir: Union[str, Path],
    dataset_id: str,
    parquet_revision: str,
    shard_reader: ShardStreamReader,
    local_audio_dir: Union[str, Path],
    target_sr: int = 16000,
    cleanup_shard: Optional[Callable[[HfParquetRef], Any]] = None,
) -> Dict[str, Any]:
    """
    Make local audio available for an arbitrary set of eligible rows.

    Used by resume-test / train after a session reset: only the shards that own
    the requested rows are downloaded, and only the missing files are rewritten.
    """
    plans = build_shard_plans(
        {"rows": rows_df}, dataset_id=dataset_id, parquet_revision=parquet_revision
    )
    reports: List[Dict[str, Any]] = []
    total = 0
    for plan in plans:
        uids = [str(u) for u in plan.per_split["rows"]["record_uid"].astype(str).tolist()]
        report = hydrate_shard_audio(
            state_dir, plan,
            local_audio_dir=local_audio_dir,
            shard_reader=shard_reader,
            target_sr=int(target_sr),
            uids=uids,
            expected_parquet_revision=parquet_revision,
        )
        reports.append(report)
        total += int(report["hydrated"])
        if cleanup_shard is not None:
            cleanup_shard(plan.ref)
    return {"shards": len(plans), "hydrated": total, "reports": reports}


def hydrate_union_audio(
    frames: Sequence[pd.DataFrame],
    *,
    state_dir: Union[str, Path],
    dataset_id: str,
    parquet_revision: str,
    shard_reader: ShardStreamReader,
    local_audio_dir: Union[str, Path],
    target_sr: int = 16000,
    cleanup_shard: Optional[Callable[[HfParquetRef], Any]] = None,
) -> Dict[str, Any]:
    """
    Hydrate the deduped union of several eligible frames in one pass.

    Train and validation are carved from the same physical shards, so hydrating
    them separately downloads every shard twice.
    """
    usable = [f for f in frames if f is not None and len(f)]
    if not usable:
        return {"shards": 0, "hydrated": 0, "unique_records": 0, "reports": []}
    union = pd.concat(usable, ignore_index=True)
    union = union.drop_duplicates(subset=["record_uid"], keep="first")
    report = hydrate_rows_audio(
        union,
        state_dir=state_dir,
        dataset_id=dataset_id,
        parquet_revision=parquet_revision,
        shard_reader=shard_reader,
        local_audio_dir=local_audio_dir,
        target_sr=int(target_sr),
        cleanup_shard=cleanup_shard,
    )
    report["unique_records"] = int(len(union))
    return report


def make_hf_parquet_stream_reader(
    *,
    cache_dir: Union[str, Path],
    batch_size: int = 32,
    id_column: str = "id",
    audio_column: str = "audio",
    download_fn: Optional[Callable[..., Any]] = None,
    downloaded: Optional[Dict[str, str]] = None,
) -> ShardStreamReader:
    """
    Streaming ``ShardStreamReader`` over Hugging Face parquet shards.

    Downloads with an explicit ``cache_dir`` and a properly split
    ``filename``/``revision`` (a full URL passed as ``filename`` 404s), then
    yields rows batch by batch so a shard's audio is never fully in RAM. The
    resolved local path of each shard is recorded in ``downloaded`` so the
    caller can delete the blob afterwards.
    """
    import pyarrow.parquet as pq

    cache_dir = Path(cache_dir)

    def _download(ref: HfParquetRef) -> Path:
        if download_fn is not None:
            return Path(download_fn(ref, cache_dir))
        from huggingface_hub import hf_hub_download

        cache_dir.mkdir(parents=True, exist_ok=True)
        return Path(hf_hub_download(
            repo_id=ref.repo_id,
            filename=ref.filename,
            repo_type=ref.repo_type,
            revision=ref.revision,
            cache_dir=str(cache_dir),
        ))

    def _reader(ref: HfParquetRef, needed: Sequence[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
        wanted = {int(i) for i in needed}
        if not wanted:
            return
        local = _download(ref)
        if downloaded is not None:
            downloaded[ref.shard_key] = str(local)
        handle = pq.ParquetFile(str(local))
        names = set(handle.schema_arrow.names)
        columns = [c for c in (id_column, audio_column) if c in names]
        if audio_column not in names:
            raise RuntimeError(
                f"Parquet shard {ref.filename} has no {audio_column!r} column (has {sorted(names)[:8]})"
            )
        cursor = 0
        remaining = len(wanted)
        for batch in handle.iter_batches(batch_size=int(batch_size), columns=columns):
            data = batch.to_pydict()
            n = batch.num_rows
            for offset in range(n):
                idx = cursor + offset
                if idx in wanted:
                    payload = {
                        "id": data.get(id_column, [None] * n)[offset] if id_column in data else None,
                        "audio": data[audio_column][offset],
                    }
                    yield idx, payload
                    remaining -= 1
            cursor += n
            del data, batch
            if remaining <= 0:
                break

    return _reader


# ---------------------------------------------------------------------------
# Shard-centric prepare state
# ---------------------------------------------------------------------------

@dataclass
class ShardPrepareState:
    dataset_id: str = ""
    dataset_revision: str = ""
    parquet_revision: str = ""
    vocab_fp: str = ""
    processing_version: str = ""
    audio_pcm_pipeline_version: str = AUDIO_PCM_PIPELINE_VERSION
    min_duration: float = 0.5
    max_duration: float = 30.0
    target_sr: int = 16000
    splits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    completed_shards: List[str] = field(default_factory=list)
    pending_shards: List[str] = field(default_factory=list)
    commit_seq: int = 0
    finished: bool = False
    updated_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "parquet_revision": self.parquet_revision,
            "vocab_fp": self.vocab_fp,
            "processing_version": self.processing_version,
            "audio_pcm_pipeline_version": self.audio_pcm_pipeline_version,
            "min_duration": float(self.min_duration),
            "max_duration": float(self.max_duration),
            "target_sr": int(self.target_sr),
            "splits": self.splits,
            "completed_shards": list(self.completed_shards),
            "pending_shards": list(self.pending_shards),
            "commit_seq": int(self.commit_seq),
            "finished": bool(self.finished),
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShardPrepareState":
        return cls(
            dataset_id=str(d.get("dataset_id") or ""),
            dataset_revision=str(d.get("dataset_revision") or ""),
            parquet_revision=str(d.get("parquet_revision") or ""),
            vocab_fp=str(d.get("vocab_fp") or ""),
            processing_version=str(d.get("processing_version") or ""),
            audio_pcm_pipeline_version=str(d.get("audio_pcm_pipeline_version") or ""),
            min_duration=float(d.get("min_duration") or 0.5),
            max_duration=float(d.get("max_duration") or 30.0),
            target_sr=int(d.get("target_sr") or 16000),
            splits=dict(d.get("splits") or {}),
            completed_shards=list(d.get("completed_shards") or []),
            pending_shards=list(d.get("pending_shards") or []),
            commit_seq=int(d.get("commit_seq") or 0),
            finished=bool(d.get("finished")),
            updated_at_utc=str(d.get("updated_at_utc") or ""),
        )

    def contract_fields(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "parquet_revision": self.parquet_revision,
            "vocab_fp": self.vocab_fp,
            "processing_version": self.processing_version,
            "audio_pcm_pipeline_version": self.audio_pcm_pipeline_version,
            "min_duration": float(self.min_duration),
            "max_duration": float(self.max_duration),
            "target_sr": int(self.target_sr),
            "splits": {
                k: {
                    "manifest_count": int(v.get("manifest_count") or 0),
                    "manifest_uid_hash": str(v.get("manifest_uid_hash") or ""),
                    "manifest_content_hash": str(v.get("manifest_content_hash") or ""),
                }
                for k, v in sorted(self.splits.items())
            },
        }


def load_shard_state(path: Union[str, Path]) -> Optional[ShardPrepareState]:
    p = Path(path)
    if not p.is_file():
        return None
    return ShardPrepareState.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_shard_state(path: Union[str, Path], state: ShardPrepareState) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at_utc = datetime.now(timezone.utc).isoformat()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(p))
    return p


# ---------------------------------------------------------------------------
# One-pass orchestration
# ---------------------------------------------------------------------------

def _split_partial_paths(state_dir: Path, split: str) -> Tuple[Path, Path]:
    d = state_dir / f"prepare_{split}"
    return d / "eligible_partial.csv", d / "exclusions_partial.csv"


PARTIAL_DIGEST_KEYS = ("sha256", "n_rows", "uid_hash")


def partial_csv_digest(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Fingerprint of one partial CSV: bytes, row count and UID set.

    UID accounting alone cannot notice an edited transcript or duration, so the
    byte hash is what makes a resumed partial provable rather than plausible.
    """
    p = Path(path)
    blob = p.read_bytes()
    frame = pd.read_csv(p)
    if "record_uid" in frame.columns and len(frame):
        uids = sorted(str(u) for u in frame["record_uid"].astype(str).tolist())
    else:
        uids = []
    return {
        "sha256": sha256_hex(blob),
        "n_rows": int(len(frame)),
        "uid_hash": sha256_hex("\n".join(uids).encode("utf-8")),
    }


def _partials_match_state(
    state_dir: Path, splits: Sequence[str], state: "ShardPrepareState"
) -> bool:
    """True only if every partial CSV still matches the digest the state committed."""
    for split in splits:
        recorded = dict((state.splits.get(split) or {}).get("partials") or {})
        if not recorded:
            return False
        elig_path, excl_path = _split_partial_paths(state_dir, split)
        for key, path in (("eligible", elig_path), ("exclusions", excl_path)):
            want = dict(recorded.get(key) or {})
            if not want or not path.is_file():
                return False
            try:
                got = partial_csv_digest(path)
            except Exception:
                return False
            if any(str(got.get(k)) != str(want.get(k)) for k in PARTIAL_DIGEST_KEYS):
                return False
    return True


def _unverifiable_completed_shards(
    state_dir: Path,
    shard_keys: Iterable[str],
    *,
    parquet_revision: str,
    dataset_revision: str,
    target_sr: int,
) -> Dict[str, str]:
    """
    Which completed shards can no longer be trusted, and why.

    Existence of the sidecar file proves nothing: a truncated upload, a wrong
    revision or a stale pipeline version must invalidate the shard here, at
    prepare time, instead of surfacing much later during hydrate.
    """
    bad: Dict[str, str] = {}
    for key in shard_keys:
        try:
            verify_shard_sidecar_index(
                state_dir, key,
                expected_parquet_revision=parquet_revision,
                expected_dataset_revision=dataset_revision,
                expected_target_sr=int(target_sr),
            )
        except Exception as exc:
            bad[str(key)] = str(exc)
    return bad


def run_full_prepare_streaming(
    *,
    splits: Dict[str, pd.DataFrame],
    state_dir: Union[str, Path],
    vocab: Dict[str, int],
    dataset_id: str,
    dataset_revision: str,
    parquet_revision: str,
    shard_reader: ShardStreamReader,
    local_audio_cache_dir: Union[str, Path],
    target_sr: int = 16000,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    expected_processing_version: str = "notebook02_audio_v3",
    resume: bool = True,
    shard_bytes: Optional[Mapping[str, int]] = None,
    shard_bytes_hint: int = 2 * 1024 ** 3,
    cleanup_shard: Optional[Callable[[HfParquetRef], Any]] = None,
    write_audio: bool = False,
) -> Dict[str, Any]:
    """
    Prepare every split in a single pass over the physical parquet shards.

    Resume semantics: a shard already in ``completed_shards`` with a verified
    sidecar index is never re-QA'd and never re-read; only pending shards are
    processed. A shard becomes complete only after index write + verify,
    per-shard UID accounting and the state commit all succeed.

    Prepare is metadata-only by default (``write_audio=False``): audio is
    decoded, QA'd and hashed but no WAV is kept, because local audio does not
    survive a session reset anyway. Stages that need audio hydrate it on demand.
    """
    from src.asr_full_data import (
        ELIGIBLE_COLUMNS,
        EXCLUSION_COLUMNS,
        _atomic_write_csv,
        _dedupe_records_by_uid,
        manifest_contract_fingerprint,
        prefilter_clean_split,
        vocab_fingerprint,
    )

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    local_audio_cache_dir = Path(local_audio_cache_dir)
    state_path = state_dir / SHARD_STATE_JSON
    vfp = vocab_fingerprint(vocab)

    prefiltered: Dict[str, Dict[str, Any]] = {}
    split_meta: Dict[str, Dict[str, Any]] = {}
    for split, clean_df in splits.items():
        pre = prefilter_clean_split(
            clean_df, split=split, min_duration=min_duration, max_duration=max_duration
        )
        prefiltered[split] = pre
        fp = manifest_contract_fingerprint(clean_df)
        split_meta[split] = {
            "manifest_count": int(len(clean_df)),
            "manifest_uid_hash": fp["manifest_uid_hash"],
            "manifest_content_hash": compute_manifest_content_hash(clean_df),
            "clean_uids": [str(u) for u in clean_df["record_uid"].astype(str).tolist()],
        }

    plans = build_shard_plans(
        {s: prefiltered[s]["candidates"] for s in splits},
        dataset_id=dataset_id,
        parquet_revision=parquet_revision,
    )
    shard_keys = [p.shard_key for p in plans]

    desired = ShardPrepareState(
        dataset_id=str(dataset_id),
        dataset_revision=str(dataset_revision),
        parquet_revision=str(parquet_revision),
        vocab_fp=vfp,
        processing_version=str(expected_processing_version),
        audio_pcm_pipeline_version=AUDIO_PCM_PIPELINE_VERSION,
        min_duration=float(min_duration),
        max_duration=float(max_duration),
        target_sr=int(target_sr),
        splits={
            s: {k: v for k, v in meta.items() if k != "clean_uids"}
            for s, meta in split_meta.items()
        },
        pending_shards=list(shard_keys),
    )

    state = load_shard_state(state_path) if resume else None
    if state is not None and state.contract_fields() != desired.contract_fields():
        state = None  # stale/mismatched resume — never reuse foreign partials
    if state is not None and state.completed_shards:
        # The partial CSVs carry every row committed so far, so if their bytes no
        # longer match the digest the state committed, no row is trustworthy and a
        # clean restart is the only safe move. (A single bad *shard* is recoverable
        # and handled further down; bad partials are not.)
        if not _partials_match_state(state_dir, list(splits), state):
            state = None
    if state is None:
        state = desired
        for split in splits:
            for path in _split_partial_paths(state_dir, split):
                if path.exists():
                    path.unlink()
        save_shard_state(state_path, state)

    accum_eligible: Dict[str, List[Dict[str, Any]]] = {}
    accum_exclusions: Dict[str, List[Dict[str, Any]]] = {}
    completed = set(state.completed_shards)
    for split in splits:
        elig_path, excl_path = _split_partial_paths(state_dir, split)
        if completed and elig_path.is_file():
            accum_eligible[split] = _dedupe_records_by_uid(
                pd.read_csv(elig_path).to_dict(orient="records")
            )
        else:
            accum_eligible[split] = []
        if completed and excl_path.is_file():
            accum_exclusions[split] = _dedupe_records_by_uid(
                pd.read_csv(excl_path).to_dict(orient="records")
            )
        else:
            accum_exclusions[split] = list(prefiltered[split]["exclusions"])

    # A completed shard may only be skipped while its sidecar index still
    # verifies against this run's provenance. One that no longer does is demoted
    # back to pending and its rows are pulled out of the accumulators, so the
    # normal loop redoes exactly that shard instead of trusting stale metadata.
    unverifiable = _unverifiable_completed_shards(
        state_dir, sorted(completed),
        parquet_revision=parquet_revision,
        dataset_revision=dataset_revision,
        target_sr=int(target_sr),
    )
    if unverifiable:
        uids_by_shard = {
            p.shard_key: {
                split: {str(u) for u in frame["record_uid"].astype(str).tolist()}
                for split, frame in p.per_split.items()
            }
            for p in plans
        }
        for key in unverifiable:
            completed.discard(key)
            for split in splits:
                drop = uids_by_shard.get(key, {}).get(split, set())
                if not drop:
                    continue
                accum_eligible[split] = [
                    r for r in accum_eligible[split] if str(r.get("record_uid")) not in drop
                ]
                accum_exclusions[split] = [
                    r for r in accum_exclusions[split] if str(r.get("record_uid")) not in drop
                ]
        state.completed_shards = sorted(completed)
        state.pending_shards = [k for k in shard_keys if k not in completed]
        save_shard_state(state_path, state)

    opened: Set[str] = set()
    for plan in plans:
        if plan.shard_key in completed:
            # Trusted from durable state that was verified above: no re-read, no
            # re-QA and no audio rehydrate inside prepare.
            continue

        if plan.shard_key in opened:
            raise RuntimeError(
                f"Physical shard opened more than once in a single pass: {plan.shard_key}"
            )
        # Re-checked per shard even when metadata-only: the parquet download
        # still needs room, and free space changes between shards.
        if shard_bytes is not None:
            if plan.ref.filename not in shard_bytes:
                raise RuntimeError(
                    f"No measured parquet size for {plan.ref.filename!r}; refusing to "
                    "download a shard whose disk cost is unknown"
                )
            need = int(shard_bytes[plan.ref.filename])
        else:
            need = int(shard_bytes_hint)
        assert_disk_headroom(
            local_audio_cache_dir,
            need_bytes=need + max(SHARD_TEMP_FLOOR_BYTES, need // 5),
            label=f"shard {plan.shard_key}",
        )
        opened.add(plan.shard_key)
        out = process_shard_streaming(
            plan,
            vocab=vocab,
            dataset_revision=dataset_revision,
            parquet_revision=parquet_revision,
            target_sr=int(target_sr),
            min_duration=float(min_duration),
            max_duration=float(max_duration),
            shard_reader=shard_reader,
            local_audio_dir=local_audio_cache_dir,
            write_audio=write_audio,
        )

        # Durable metadata BEFORE the shard counts as complete.
        write_shard_sidecar_index(
            state_dir, plan.shard_key, out["sidecars"],
            parquet_revision=parquet_revision,
            dataset_revision=dataset_revision,
        )
        verify_shard_sidecar_index(
            state_dir, plan.shard_key,
            expected_parquet_revision=parquet_revision,
            expected_dataset_revision=dataset_revision,
            expected_target_sr=int(target_sr),
        )
        for split, frame in plan.per_split.items():
            assert_uid_set_accounting(
                clean_uids=frame["record_uid"].astype(str).tolist(),
                eligible_uids=[r["record_uid"] for r in out["eligible"].get(split, [])],
                exclusion_uids=[r["record_uid"] for r in out["exclusions"].get(split, [])],
                split=f"{split}@{plan.shard_key}",
            )
            accum_eligible[split].extend(out["eligible"].get(split, []))
            accum_exclusions[split].extend(out["exclusions"].get(split, []))
            accum_eligible[split] = _dedupe_records_by_uid(accum_eligible[split])
            accum_exclusions[split] = _dedupe_records_by_uid(accum_exclusions[split])
            elig_path, excl_path = _split_partial_paths(state_dir, split)
            # Always with an explicit schema: a split with 0 eligible rows would
            # otherwise land as a header-less file that pandas refuses to read
            # back on the next session (EmptyDataError).
            _atomic_write_csv(
                elig_path,
                pd.DataFrame(accum_eligible[split], columns=ELIGIBLE_COLUMNS),
                ELIGIBLE_COLUMNS,
            )
            _atomic_write_csv(
                excl_path,
                pd.DataFrame(accum_exclusions[split], columns=EXCLUSION_COLUMNS),
                EXCLUSION_COLUMNS,
            )
        completed.add(plan.shard_key)
        state.completed_shards = sorted(completed)
        state.pending_shards = [k for k in shard_keys if k not in completed]
        state.commit_seq = int(state.commit_seq) + 1
        for split in splits:
            elig_path, excl_path = _split_partial_paths(state_dir, split)
            state.splits[split]["n_eligible"] = len(accum_eligible[split])
            state.splits[split]["n_excluded"] = len(accum_exclusions[split])
            # Committed together with the shard so a later session can prove the
            # partials are byte-identical to what this commit produced.
            state.splits[split]["partials"] = {
                "eligible": partial_csv_digest(elig_path),
                "exclusions": partial_csv_digest(excl_path),
                "commit_seq": int(state.commit_seq),
            }
        save_shard_state(state_path, state)
        if cleanup_shard is not None:
            cleanup_shard(plan.ref)

    state.pending_shards = [k for k in shard_keys if k not in completed]
    state.finished = not state.pending_shards
    save_shard_state(state_path, state)

    results: Dict[str, Any] = {}
    for split in splits:
        eligible_df = pd.DataFrame(accum_eligible[split], columns=ELIGIBLE_COLUMNS)
        excl_df = pd.DataFrame(accum_exclusions[split], columns=EXCLUSION_COLUMNS)
        accounting_ok, accounting_error, uid_report = True, None, None
        try:
            uid_report = assert_uid_set_accounting(
                clean_uids=split_meta[split]["clean_uids"],
                eligible_uids=eligible_df["record_uid"].tolist() if len(eligible_df) else [],
                exclusion_uids=excl_df["record_uid"].tolist() if len(excl_df) else [],
                split=split,
            )
        except RuntimeError as exc:
            accounting_ok, accounting_error = False, str(exc)
        results[split] = {
            "eligible_df": eligible_df,
            "exclusions_df": excl_df,
            "state": state,
            "all_shards_done": bool(state.finished),
            "clean_count": int(split_meta[split]["manifest_count"]),
            "accounting_ok": accounting_ok,
            "accounting_error": accounting_error,
            "uid_accounting": uid_report,
            "manifest_uid_hash": split_meta[split]["manifest_uid_hash"],
            "manifest_content_hash": split_meta[split]["manifest_content_hash"],
            "vocab_fp": vfp,
            "parquet_revision": str(parquet_revision),
        }
    results["_shards"] = {
        "plans": [p.shard_key for p in plans],
        "opened": sorted(opened),
        "completed": sorted(completed),
        "audio_written": bool(write_audio),
        "state": state,
    }
    return results
