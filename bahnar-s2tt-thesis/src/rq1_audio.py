"""Controlled frozen-test audio materialization and integrity verification for Notebook 06."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import pandas as pd

from src.asr_full_pcm import (
    canonical_pcm16_from_bytes,
    cleanup_shard_download,
    read_pcm16_payload,
    write_wav_from_pcm16,
)
from src.asr_full_shards import build_shard_plans, make_hf_parquet_stream_reader
from src.data_utils import safe_cache_filename
from src.rq1_contract import atomic_write_csv, sha256_file


def materialize_frozen_test_audio(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    parquet_revision: str,
    audio_dir: Union[str, Path],
    parquet_cache_dir: Union[str, Path],
    target_sr: int = 16000,
) -> Dict[str, Any]:
    """
    Materialize every frozen-test row exactly once; no scientific filtering.

    A decode/id mismatch is fatal rather than an exclusion because dropping test
    rows would invalidate paired C0-vs-D0 comparison.
    """
    audio_root = Path(audio_dir)
    audio_root.mkdir(parents=True, exist_ok=True)
    pq_root = Path(parquet_cache_dir)
    pq_root.mkdir(parents=True, exist_ok=True)
    plans = build_shard_plans({"test": frame}, dataset_id=dataset_id, parquet_revision=parquet_revision)
    downloaded: Dict[str, str] = {}
    reader = make_hf_parquet_stream_reader(cache_dir=pq_root, downloaded=downloaded)
    rows = []
    try:
        for plan in plans:
            wanted = {int(r["shard_row_index"]): r for _, r in plan.per_split["test"].iterrows()}
            seen = set()
            for idx, payload in reader(plan.ref, plan.needed_indices):
                idx = int(idx)
                if idx not in wanted:
                    continue
                if idx in seen:
                    raise RuntimeError(f"Duplicate parquet row {idx} in {plan.shard_key}")
                seen.add(idx)
                row = wanted[idx]
                uid = str(row["record_uid"])
                rid = str(row["record_id"])
                payload_id = payload.get("id") if isinstance(payload, dict) else None
                if payload_id is not None and str(payload_id) != rid:
                    raise RuntimeError(f"Frozen-test record_id mismatch uid={uid}: {payload_id!r} != {rid!r}")
                raw = payload.get("audio") if isinstance(payload, dict) else None
                if isinstance(raw, dict):
                    raw = raw.get("bytes")
                if not raw:
                    raise RuntimeError(f"Frozen-test null audio uid={uid}")
                decoded = canonical_pcm16_from_bytes(bytes(raw), target_sr=int(target_sr))
                rel = safe_cache_filename(uid)
                path = audio_root / rel
                write_wav_from_pcm16(path, decoded["pcm"], int(target_sr))
                rows.append(
                    {
                        "record_uid": uid,
                        "audio_path": str(path),
                        "sha256_pcm": decoded["sha256_pcm"],
                        "n_samples": int(decoded["n_samples"]),
                        "sample_rate": int(decoded["sample_rate"]),
                        "shard_key": plan.shard_key,
                        "shard_row_index": idx,
                    }
                )
            missing = sorted(set(wanted) - seen)
            if missing:
                raise RuntimeError(f"Frozen-test shard missing requested rows {plan.shard_key}: {missing[:5]}")
            local = downloaded.pop(plan.ref.shard_key, None)
            if local:
                cleanup_shard_download(local)
    finally:
        for p in list(downloaded.values()):
            try:
                cleanup_shard_download(p)
            except Exception:
                pass
        downloaded.clear()
    out = pd.DataFrame(rows)
    expected = frame["record_uid"].astype(str).tolist()
    if len(out) != len(frame) or set(out["record_uid"].astype(str)) != set(expected):
        raise RuntimeError("Frozen-test audio materialization UID accounting failed")
    out = out.set_index("record_uid").loc[expected].reset_index()
    return {
        "audio_frame": out,
        "audio_paths": dict(zip(out["record_uid"], out["audio_path"])),
        "shards": len(plans),
        "rows": len(out),
    }


def verify_audio_integrity_frame(
    integrity_df: pd.DataFrame,
    *,
    expected_uids: Sequence[str],
) -> Dict[str, Any]:
    """Re-read every WAV and compare against integrity rows."""
    got = integrity_df["record_uid"].astype(str).tolist()
    if got != list(expected_uids):
        raise RuntimeError("Audio integrity UID order mismatch vs frozen test")
    rows = []
    for _, row in integrity_df.iterrows():
        uid = str(row["record_uid"])
        path = Path(str(row["audio_path"]))
        if not path.is_file():
            raise RuntimeError(f"Missing audio WAV for {uid}: {path}")
        payload = read_pcm16_payload(path)
        if str(payload["sha256_pcm"]) != str(row["sha256_pcm"]):
            raise RuntimeError(f"PCM SHA mismatch for {uid}")
        if int(payload["n_samples"]) != int(row["n_samples"]):
            raise RuntimeError(f"n_samples mismatch for {uid}")
        if int(payload["sample_rate"]) != int(row["sample_rate"]):
            raise RuntimeError(f"sample_rate mismatch for {uid}")
        rows.append(
            {
                "record_uid": uid,
                "audio_path": str(path),
                "sha256_pcm": payload["sha256_pcm"],
                "n_samples": int(payload["n_samples"]),
                "sample_rate": int(payload["sample_rate"]),
            }
        )
    return {"ok": True, "rows": rows}


def ensure_verified_frozen_audio(
    test_df: pd.DataFrame,
    *,
    state_dir: Union[str, Path],
    dataset_id: str,
    parquet_revision: str,
    audio_dir: Union[str, Path],
    parquet_cache_dir: Union[str, Path],
    target_sr: int = 16000,
    rematerialize: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Ensure C0/D0 share the same verified audio.

    Writes ``rq1_audio_manifest.csv`` and ``rq1_audio_integrity.csv`` atomically.
    Missing/corrupt/mismatched WAV triggers rematerialize from pinned parquet.
    """
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    audio_manifest = state / "rq1_audio_manifest.csv"
    integrity_path = state / "rq1_audio_integrity.csv"
    expected = test_df["record_uid"].astype(str).tolist()

    def _materialize_and_persist() -> Dict[str, Any]:
        rep = materialize_frozen_test_audio(
            test_df,
            dataset_id=dataset_id,
            parquet_revision=parquet_revision,
            audio_dir=audio_dir,
            parquet_cache_dir=parquet_cache_dir,
            target_sr=target_sr,
        )
        af = rep["audio_frame"].copy()
        manifest_cols = [c for c in ("record_uid", "audio_path", "shard_key", "shard_row_index") if c in af.columns]
        integrity_cols = [
            c
            for c in ("record_uid", "audio_path", "sha256_pcm", "n_samples", "sample_rate", "shard_key", "shard_row_index")
            if c in af.columns
        ]
        atomic_write_csv(af[manifest_cols], audio_manifest)
        atomic_write_csv(af[integrity_cols], integrity_path)
        verify_audio_integrity_frame(pd.read_csv(integrity_path), expected_uids=expected)
        return {
            "audio_paths": dict(zip(af["record_uid"].astype(str), af["audio_path"].astype(str))),
            "audio_integrity_sha256": sha256_file(integrity_path),
            "rematerialized": True,
            "integrity_path": str(integrity_path),
            "manifest_path": str(audio_manifest),
        }

    if rematerialize is True:
        return _materialize_and_persist()

    if audio_manifest.is_file() and integrity_path.is_file():
        try:
            integrity = pd.read_csv(integrity_path)
            verify_audio_integrity_frame(integrity, expected_uids=expected)
            paths = dict(zip(integrity["record_uid"].astype(str), integrity["audio_path"].astype(str)))
            return {
                "audio_paths": paths,
                "audio_integrity_sha256": sha256_file(integrity_path),
                "rematerialized": False,
                "integrity_path": str(integrity_path),
                "manifest_path": str(audio_manifest),
            }
        except Exception:
            if rematerialize is False:
                raise
            # Fall through to rematerialize.
            pass
    return _materialize_and_persist()
