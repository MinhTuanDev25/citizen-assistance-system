"""
ASR Utilities for Notebook 03 — Bahnar ASR Baseline Training.

This module provides helpers for:
- Data collation (CTC padding)
- Pilot sampling (deterministic, group-aware)
- Audio preprocessing and validation
- Cache status classification
- CTC feasibility checking
- OOV analysis
- Metrics computation (CER, WER)
- Artifact management
- Environment info
- Notebook 03 orchestration (status, checkpoints, integrity)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SAMPLING_RATE = 16000

# CTC encoder output calculation constants for Wav2Vec2
WAV2VEC2_CONV_STRIDE_PRODUCT = 320  # product of all conv strides
WAV2VEC2_CONV_KERNEL_SPAN = 400     # receptive field of the conv stack

# ---------------------------------------------------------------------------
# Fixed schema columns for CSV exports (Notebook 03)
# ---------------------------------------------------------------------------

AUDIO_EXCLUSIONS_COLUMNS = [
    "run_id",
    "split",
    "record_uid",
    "record_id",
    "group_id",
    "reason",
    "exception_type",
    "source_duration_seconds",
    "processed_duration_seconds",
]

CTC_FEASIBILITY_EXCLUSIONS_COLUMNS = [
    "run_id",
    "split",
    "record_uid",
    "record_id",
    "group_id",
    "reason",
    "duration_seconds",
    "target_token_length",
    "min_required_frames",
    "estimated_output_frames",
]

VALIDATION_PREDICTIONS_COLUMNS = [
    "run_id",
    "checkpoint",
    "record_uid",
    "record_id",
    "group_id",
    "duration_seconds",
    "reference_raw",
    "prediction_raw",
    "reference_normalized",
    "prediction_normalized",
    "cer",
    "wer",
]

OOV_SUMMARY_COLUMNS = [
    "scope",
    "code_point",
    "char",
    "count",
    "n_records",
    "record_ratio",
]


def empty_dataframe(columns: List[str]) -> pd.DataFrame:
    """Create an empty DataFrame with the specified columns (for fixed-schema CSVs)."""
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Generate a unique run ID (UUID4)."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# SHA-256 file hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(preferred: str = "auto", run_mode: str = "pilot") -> torch.device:
    """
    Get the best available device for training.
    
    Args:
        preferred: "auto", "cpu", "cuda", or "mps"
        run_mode: "pilot" or "full" (for logging)
    
    Priority (when auto): CUDA > MPS > CPU
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Forbidden test path detection
# ---------------------------------------------------------------------------

def is_forbidden_test_path(path: Union[str, Path]) -> bool:
    """
    Check if a path references the frozen RQ1 test set.
    
    Returns True if the path contains 'rq1_test', 'frozen_test', or is
    a bare 'test.csv'/'test.parquet' (but NOT 'test_*.py' unit test files).
    """
    p = str(path).lower()
    
    # Explicit forbidden patterns
    if "rq1_test" in p or "frozen_test" in p:
        return True
    
    # Paths containing "/test/" directory
    if "/test/" in p:
        return True
    
    # Files ending with exactly "test.csv" or "test.parquet" (not "_test.csv" which is different)
    # But we need to match "test.csv" alone or "rq1_test.csv" etc.
    base = p.split("/")[-1] if "/" in p else p
    if base == "test.csv" or base == "test.parquet":
        return True
    if base.endswith("_test.csv") or base.endswith("_test.parquet"):
        return True
    
    return False


# ---------------------------------------------------------------------------
# CTC Data Collator
# ---------------------------------------------------------------------------

class CTCDataCollatorWithPadding:
    """
    Data collator for CTC training with dynamic padding.
    
    Pads input_values to max length in batch and labels to max label length,
    using -100 for label padding (ignored by CTC loss).
    """
    
    def __init__(self, processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Validate inputs
        if len(features) == 0:
            raise ValueError("Cannot collate empty batch")
        
        for i, f in enumerate(features):
            if "input_values" not in f:
                raise ValueError(f"Feature {i} missing 'input_values'")
            iv = f["input_values"]
            if len(iv) == 0:
                raise ValueError(f"Empty waveform in feature {i}")
            if not np.isfinite(iv).all():
                raise ValueError(f"NaN or Inf in waveform in feature {i}")
        
        # Separate input_values and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [f["labels"] for f in features]
        
        # Pad input_values
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels with -100
        max_label_len = max(len(l) for l in label_features)
        padded_labels = []
        for labels in label_features:
            padded = list(labels) + [-100] * (max_label_len - len(labels))
            padded_labels.append(padded)
        
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch


def validate_collator_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Validate a collated batch has expected structure and properties.
    
    Returns dict with: valid (bool), errors (list of str), and diagnostics.
    """
    errors: List[str] = []
    
    has_input_values = "input_values" in batch
    has_labels = "labels" in batch
    
    if not has_input_values:
        errors.append("Missing input_values")
    if not has_labels:
        errors.append("Missing labels")
    
    input_values_shape = None
    labels_shape = None
    labels_has_padding = False
    
    if has_input_values:
        iv = batch["input_values"]
        input_values_shape = list(iv.shape)  # list, not tuple
        if not torch.isfinite(iv).all():
            errors.append("NaN/Inf in input_values")
    
    if has_labels:
        lb = batch["labels"]
        labels_shape = list(lb.shape)  # list, not tuple
        labels_has_padding = bool((lb == -100).any())
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "has_input_values": has_input_values,
        "has_labels": has_labels,
        "input_values_shape": input_values_shape,
        "labels_shape": labels_shape,
        "labels_has_padding": labels_has_padding,
    }


# ---------------------------------------------------------------------------
# Waveform preprocessing and validation
# ---------------------------------------------------------------------------

def preprocess_waveform_for_training(
    waveform: Optional[np.ndarray],
    sample_rate: Optional[int],
    target_sample_rate: int = TARGET_SAMPLING_RATE,
) -> Optional[np.ndarray]:
    """
    Preprocess waveform: ensure float32, mono, resample if needed.
    
    Returns None if input is invalid (None, empty, invalid sample_rate, NaN values).
    This is a LOW-LEVEL helper. For model input, use preprocess_features_for_training.
    """
    # Validate inputs
    if waveform is None or len(waveform) == 0:
        return None
    if sample_rate is None or sample_rate <= 0:
        return None
    if not np.isfinite(waveform).all():
        return None
    
    # Ensure float32
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    
    # Ensure mono (average channels if stereo)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1) if waveform.shape[-1] <= 2 else waveform.mean(axis=0)
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        from scipy.signal import resample
        num_samples = int(len(waveform) * target_sample_rate / sample_rate)
        waveform = resample(waveform, num_samples).astype(np.float32)
    
    return waveform


def validate_waveform_for_training(
    waveform: Optional[np.ndarray],
    sample_rate: int = TARGET_SAMPLING_RATE,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> Dict[str, Any]:
    """
    Validate waveform meets training requirements.
    
    Returns dict with: ok (bool), reason (str), duration_sec (float|None).
    """
    # Handle None
    if waveform is None:
        return {"ok": False, "reason": "null_waveform", "duration_sec": None}
    
    # Handle empty
    if len(waveform) == 0:
        return {"ok": False, "reason": "empty", "duration_sec": None}
    
    # Check for non-finite values
    if not np.isfinite(waveform).all():
        return {"ok": False, "reason": "non_finite", "duration_sec": None}
    
    duration = len(waveform) / sample_rate if sample_rate > 0 else 0
    
    if duration < min_duration:
        return {"ok": False, "reason": "too_short", "duration_sec": duration}
    
    if duration > max_duration:
        return {"ok": False, "reason": "too_long", "duration_sec": duration}
    
    return {"ok": True, "reason": "ok", "duration_sec": duration}


def preprocess_features_for_training(
    processor,
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLING_RATE,
) -> np.ndarray:
    """
    Single canonical preprocessing path for ASR training/prediction.
    
    Uses the Wav2Vec2Processor for normalization.
    Returns 1D numpy array of input_values ready for the model.
    
    Raises ValueError if processor.feature_extractor.sampling_rate doesn't match target.
    """
    # Check feature extractor sampling rate matches target
    if hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
        fe_sr = processor.feature_extractor.sampling_rate
        if fe_sr != target_sample_rate:
            raise ValueError(
                f"Feature extractor sampling_rate ({fe_sr}) != target ({target_sample_rate})"
            )
    
    # Preprocess waveform (mono, float32, resample)
    waveform = preprocess_waveform_for_training(waveform, sample_rate, target_sample_rate)
    if waveform is None:
        raise ValueError("Invalid waveform after preprocessing")
    
    # Use processor for normalization (calls feature_extractor internally)
    inputs = processor(
        waveform,
        sampling_rate=target_sample_rate,
        return_tensors="np",
    )
    
    return inputs["input_values"][0]


# ---------------------------------------------------------------------------
# Pilot sampling (deterministic, group-aware)
# ---------------------------------------------------------------------------

def sample_pilot_data(
    df: pd.DataFrame,
    n_samples: int,
    seed: int,
    group_col: str = "group_id",
    uid_col: str = "record_uid",
) -> pd.DataFrame:
    """
    Deterministic, group-aware pilot sampler.

    Guarantees:
    - Returns exactly ``min(n_samples, n_unique_uids)`` rows.
    - Never returns more than requested.
    - Samples from multiple groups when possible.
    - Stable output order (sorted by group then uid within group).
    - No duplicate UIDs in output.
    
    Algorithm:
    1. Deduplicate by UID (keep first occurrence).
    2. Shuffle groups deterministically.
    3. Round-robin sample from groups until target reached.
    4. Sort output by group_id, then by original order within group.
    """
    if len(df) == 0 or n_samples <= 0:
        return df.iloc[0:0].copy()
    
    rng = np.random.default_rng(seed)
    
    # Deduplicate by UID
    df = df.drop_duplicates(subset=[uid_col], keep="first").copy()
    n_samples = min(n_samples, len(df))
    
    if n_samples == len(df):
        # Return all, but in deterministic group-aware order
        return df.sort_values([group_col, uid_col]).reset_index(drop=True)
    
    # Get groups and shuffle them
    groups = df[group_col].unique().tolist()
    rng.shuffle(groups)
    
    # Build index mapping: group -> list of row indices
    group_indices = {g: df[df[group_col] == g].index.tolist() for g in groups}
    for g in groups:
        rng.shuffle(group_indices[g])
    
    # Round-robin sampling
    selected_indices = []
    group_ptrs = {g: 0 for g in groups}
    
    while len(selected_indices) < n_samples:
        added_this_round = False
        for g in groups:
            if len(selected_indices) >= n_samples:
                break
            ptr = group_ptrs[g]
            if ptr < len(group_indices[g]):
                selected_indices.append(group_indices[g][ptr])
                group_ptrs[g] = ptr + 1
                added_this_round = True
        if not added_this_round:
            break
    
    result = df.loc[selected_indices].copy()
    return result.sort_values([group_col, uid_col]).reset_index(drop=True)


def verify_pilot_sample(
    sample_df: pd.DataFrame,
    source_df: pd.DataFrame,
    expected_n: int,
    uid_col: str = "record_uid",
    group_col: str = "group_id",
) -> Dict[str, Any]:
    """
    Verify a pilot sample meets requirements.
    
    Returns dict with: passed (bool), sample_size, n_groups, errors (list of str).
    """
    errors: List[str] = []
    sample_uids = set(sample_df[uid_col].astype(str))
    source_uids = set(source_df[uid_col].astype(str))
    
    sample_size = len(sample_df)
    n_groups = sample_df[group_col].nunique() if len(sample_df) > 0 else 0
    
    # Check size
    expected_size = min(expected_n, len(source_df))
    if sample_size != expected_size:
        errors.append(f"Size mismatch: expected {expected_size}, got {sample_size}")
    
    # Check UIDs are from source
    invalid_uids = sample_uids - source_uids
    if invalid_uids:
        errors.append(f"UIDs not in full data: {sorted(invalid_uids)[:5]}")
    
    # Check for duplicates
    if len(sample_df) != len(sample_uids):
        errors.append("Duplicate UIDs in sample")
    
    return {
        "passed": len(errors) == 0,
        "sample_size": sample_size,
        "n_groups": n_groups,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Cache status classification
# ---------------------------------------------------------------------------

def classify_cache_status(
    cache_path: Union[str, Path],
    *,
    record_uid: str,
    dataset_revision: str,
    target_sr: int = TARGET_SAMPLING_RATE,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    expected_duration: Optional[float] = None,
    expected_processing_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Strictly validate a cached WAV using Notebook 02 helpers and return a
    *specific* status category (never a generic ``cache_not_found``).

    Reason categories:
        ok
        cache_missing
        cache_sidecar_missing
        cache_bad_sr
        cache_not_mono
        cache_non_finite
        cache_sha_mismatch
        cache_provenance_mismatch
        cache_read_error:<ExceptionType>
        duration_too_short
        duration_too_long

    Returns dict: ``ok`` (bool), ``reason`` (str), ``duration_seconds`` (float|None),
    ``cache_sha256`` (str|None).
    """
    from src.data_utils import (
        validate_cache_wav,
        load_cache_sidecar,
        is_valid_sha256,
        AUDIO_PROCESSING_VERSION,
    )

    if expected_processing_version is None:
        expected_processing_version = AUDIO_PROCESSING_VERSION

    out: Dict[str, Any] = {
        "ok": False,
        "reason": "cache_missing",
        "duration_seconds": None,
        "cache_sha256": None,
    }
    path = Path(cache_path)
    if not path.is_file():
        return out

    qa = validate_cache_wav(
        path,
        target_sr=target_sr,
        min_duration=min_duration,
        max_duration=max_duration,
        expected_duration=expected_duration,
        recompute_sha=True,
    )
    out["duration_seconds"] = qa.get("cache_duration_sec")
    out["cache_sha256"] = qa.get("cache_sha256")

    reason = str(qa.get("cache_reason") or "")
    if reason.startswith("cache_read_error"):
        out["reason"] = reason
        return out

    # Structured checks (order: read -> finite -> sr -> mono -> duration).
    if not qa.get("cache_waveform_finite", False):
        out["reason"] = "cache_non_finite"
        return out
    if qa.get("cache_sr") is not None and int(qa["cache_sr"]) != int(target_sr):
        out["reason"] = "cache_bad_sr"
        return out
    if qa.get("cache_channels") is not None and int(qa["cache_channels"]) != 1:
        out["reason"] = "cache_not_mono"
        return out

    dur = qa.get("cache_duration_sec")
    if dur is not None:
        if dur < min_duration:
            out["reason"] = "duration_too_short"
            return out
        if dur > max_duration:
            out["reason"] = "duration_too_long"
            return out

    # Sidecar + provenance.
    sidecar = load_cache_sidecar(path)
    if sidecar is None:
        out["reason"] = "cache_sidecar_missing"
        return out

    # Provenance checks
    prov_ok = True
    if str(sidecar.get("record_uid", "")) != str(record_uid):
        prov_ok = False
    if str(sidecar.get("dataset_revision", "")) != str(dataset_revision):
        prov_ok = False
    if str(sidecar.get("audio_processing_version", "")) != str(expected_processing_version):
        prov_ok = False
    
    if not prov_ok:
        out["reason"] = "cache_provenance_mismatch"
        return out

    # SHA-256 check
    cache_sha = qa.get("cache_sha256")
    sidecar_sha = sidecar.get("cache_sha256")
    if cache_sha and sidecar_sha and cache_sha != sidecar_sha:
        out["reason"] = "cache_sha_mismatch"
        return out

    out["ok"] = True
    out["reason"] = "ok"
    return out


# ---------------------------------------------------------------------------
# CTC feasibility checking
# ---------------------------------------------------------------------------

def estimate_encoder_output_frames(num_samples: int) -> int:
    """
    Estimate the number of encoder output frames for a given number of input samples.
    
    Uses Wav2Vec2 conv stack parameters.
    """
    if num_samples < WAV2VEC2_CONV_KERNEL_SPAN:
        return 0
    return max(0, (num_samples - WAV2VEC2_CONV_KERNEL_SPAN) // WAV2VEC2_CONV_STRIDE_PRODUCT + 1)


def count_min_ctc_target_length(token_ids: List[int]) -> int:
    """
    Count the minimum CTC target length (accounting for blank insertion).
    
    CTC requires at least 2*n - 1 frames for n tokens with no consecutive duplicates,
    or more if there are consecutive duplicate tokens.
    """
    if not token_ids:
        return 0
    
    # Count required frames: each token + blank between consecutive duplicates
    min_frames = 1
    for i in range(1, len(token_ids)):
        min_frames += 1
        if token_ids[i] == token_ids[i - 1]:
            min_frames += 1  # need blank between duplicates
    
    return min_frames


def check_ctc_feasibility(
    token_ids: List[int],
    num_samples: int,
) -> Dict[str, Any]:
    """
    Check if a sample is CTC-feasible (enough encoder frames for the target).
    
    Returns dict with feasibility result and diagnostic info.
    """
    est_frames = estimate_encoder_output_frames(num_samples)
    min_required = count_min_ctc_target_length(token_ids)
    
    # Empty target is not feasible (nothing to train on)
    if len(token_ids) == 0:
        return {
            "feasible": False,
            "reason": "empty_target",
            "estimated_output_frames": est_frames,
            "min_required_frames": min_required,
            "target_token_length": 0,
            "num_samples": num_samples,
        }
    
    feasible = est_frames >= min_required
    
    return {
        "feasible": feasible,
        "reason": "ok" if feasible else "insufficient_frames_for_ctc_target",
        "estimated_output_frames": est_frames,
        "min_required_frames": min_required,
        "target_token_length": len(token_ids),
        "num_samples": num_samples,
    }


# ---------------------------------------------------------------------------
# OOV analysis
# ---------------------------------------------------------------------------

def analyze_oov(
    texts: List[str],
    vocab: Dict[str, int],
    scope: str = "unknown",
    *,
    normalize_fn: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Analyze out-of-vocabulary characters in a list of texts.

    Texts are normalized with ``normalize_bahnar_ctc_v1`` (or ``normalize_fn``)
    *before* OOV counting, matching CTC training targets. Uppercase / punctuation
    that normalization removes or casefolds must not be reported as OOV.

    Returns DataFrame with OOV character statistics (does NOT modify vocab).
    Whitespace is skipped (space is typically normalized to "|" delimiter in CTC).
    """
    if normalize_fn is None:
        from src.data_utils import normalize_bahnar_ctc_v1 as normalize_fn

    vocab_chars = set(vocab.keys())
    oov_counts: Dict[str, int] = {}
    oov_records: Dict[str, int] = {}
    total_records = len(texts)

    for text in texts:
        if text is None:
            continue
        norm = normalize_fn(text)
        if norm is None:
            continue
        text_oov = set()
        for char in str(norm):
            # Skip whitespace (space normalized to word delimiter in CTC)
            if char.isspace():
                continue
            if char not in vocab_chars:
                oov_counts[char] = oov_counts.get(char, 0) + 1
                text_oov.add(char)
        for char in text_oov:
            oov_records[char] = oov_records.get(char, 0) + 1

    if not oov_counts:
        return empty_dataframe(OOV_SUMMARY_COLUMNS)

    rows = []
    for char, count in sorted(oov_counts.items(), key=lambda x: -x[1]):
        rows.append({
            "scope": scope,
            "code_point": f"U+{ord(char):04X}",
            "char": char,
            "count": count,
            "n_records": oov_records.get(char, 0),
            "record_ratio": oov_records.get(char, 0) / total_records if total_records > 0 else 0,
        })

    return pd.DataFrame(rows, columns=OOV_SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# Metrics computation (CER, WER)
# ---------------------------------------------------------------------------

def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis."""
    from jiwer import cer
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(cer(reference, hypothesis))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    from jiwer import wer
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(wer(reference, hypothesis))


def compute_batch_metrics(
    references: List[str],
    hypotheses: List[str],
    allow_empty_reference: bool = False,
) -> Dict[str, Any]:
    """
    Compute CER and WER for a batch of references and hypotheses.
    
    Args:
        references: List of reference strings.
        hypotheses: List of hypothesis strings.
        allow_empty_reference: If False (default), raise ValueError if any
            reference is empty/None. Training/evaluation should use False.
    
    Returns:
        Dict with corpus-level and per-sample metrics.
    
    Raises:
        ValueError: If lengths don't match or if empty references found
            when allow_empty_reference=False.
    """
    from jiwer import cer, wer
    
    if len(references) != len(hypotheses):
        raise ValueError(
            f"length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
        )
    
    # Check for empty references
    if not allow_empty_reference:
        for i, ref in enumerate(references):
            if ref is None or str(ref).strip() == "":
                raise ValueError(f"Empty reference at index {i} (allow_empty_reference=False)")
    
    # Handle empty case
    if len(references) == 0:
        return {"cer": 0.0, "wer": 0.0, "n": 0, "per_sample_cer": [], "per_sample_wer": []}
    
    # Compute per-sample metrics for ALL pairs (even empty refs when allowed)
    per_sample_cer = []
    per_sample_wer = []
    for ref, hyp in zip(references, hypotheses):
        ref_str = str(ref) if ref is not None else ""
        hyp_str = str(hyp) if hyp is not None else ""
        if ref_str.strip() == "":
            # Empty reference: CER/WER is 0 if hyp also empty, else 1
            sample_cer = 0.0 if hyp_str.strip() == "" else 1.0
            sample_wer = 0.0 if hyp_str.strip() == "" else 1.0
        else:
            sample_cer = float(cer(ref_str, hyp_str))
            sample_wer = float(wer(ref_str, hyp_str))
        per_sample_cer.append(sample_cer)
        per_sample_wer.append(sample_wer)
    
    # Filter out pairs with empty references for corpus-level computation
    valid_pairs = [(r, h) for r, h in zip(references, hypotheses) 
                   if r is not None and str(r).strip() != ""]
    
    if not valid_pairs:
        corpus_cer = 0.0
        corpus_wer = 0.0
    else:
        refs = [str(r) for r, h in valid_pairs]
        hyps = [str(h) if h else "" for r, h in valid_pairs]
        corpus_cer = float(cer(refs, hyps))
        corpus_wer = float(wer(refs, hyps))
    
    return {
        "cer": corpus_cer,
        "wer": corpus_wer,
        "n": len(references),  # total count, not filtered
        "per_sample_cer": per_sample_cer,
        "per_sample_wer": per_sample_wer,
    }


# ---------------------------------------------------------------------------
# Artifact manifest creation
# ---------------------------------------------------------------------------

def create_artifact_manifest(
    artifact_dir: Path,
    patterns: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create manifest of artifacts with paths, sizes, and SHA-256 hashes.
    
    Args:
        artifact_dir: Directory containing artifacts
        patterns: Optional glob patterns to include (default: all files)
    
    Returns:
        List of artifact info dicts
    """
    if patterns is None:
        patterns = ["**/*"]
    
    artifacts = []
    for pattern in patterns:
        for path in artifact_dir.glob(pattern):
            if path.is_file():
                try:
                    artifacts.append({
                        "path": str(path.relative_to(artifact_dir)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    })
                except Exception as e:
                    artifacts.append({
                        "path": str(path.relative_to(artifact_dir)),
                        "size_bytes": None,
                        "sha256": None,
                        "error": str(e),
                    })
    
    return sorted(artifacts, key=lambda x: x["path"])


# ---------------------------------------------------------------------------
# Notebook 03 prerequisites verification
# ---------------------------------------------------------------------------

def verify_notebook03_prerequisites(
    contract_path: Union[str, Path],
    tokenizer_dir: Union[str, Path],
    expected_dataset_revision: str,
    expected_train_clean_count: int,
    expected_validation_clean_count: int,
) -> Dict[str, Any]:
    """
    Verify that Notebook 02 outputs exist and match expectations.
    
    Returns dict with: passed (bool), errors (list), vocab_size, revision_match, etc.
    """
    errors: List[str] = []
    
    contract_path = Path(contract_path)
    tokenizer_dir = Path(tokenizer_dir)
    
    contract_exists = False
    tokenizer_exists = False
    revision_match = False
    train_count_match = False
    validation_count_match = False
    vocab_size = None
    
    # Check contract exists
    if contract_path.is_file():
        contract_exists = True
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            
            # Check dataset revision
            if contract.get("dataset_revision") == expected_dataset_revision:
                revision_match = True
            else:
                errors.append(f"Revision mismatch: {contract.get('dataset_revision')} != {expected_dataset_revision}")
            
            # Check counts
            train_count = contract.get("train", {}).get("clean_count", 0)
            val_count = contract.get("validation", {}).get("clean_count", 0)
            
            if train_count == expected_train_clean_count:
                train_count_match = True
            else:
                errors.append(f"Train count mismatch: {train_count} != {expected_train_clean_count}")
            
            if val_count == expected_validation_clean_count:
                validation_count_match = True
            else:
                errors.append(f"Validation count mismatch: {val_count} != {expected_validation_clean_count}")
                
        except Exception as e:
            errors.append(f"Contract parse error: {e}")
    else:
        errors.append(f"Contract not found: {contract_path}")
    
    # Check tokenizer exists
    vocab_path = tokenizer_dir / "vocab.json"
    if vocab_path.is_file():
        tokenizer_exists = True
        try:
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab_size = len(vocab)
        except Exception:
            vocab_size = None
    else:
        errors.append(f"Tokenizer not found: {vocab_path}")
    
    # Overall pass/fail
    passed = (
        contract_exists and
        tokenizer_exists and
        revision_match and
        train_count_match and
        validation_count_match
    )
    
    return {
        "passed": passed,
        "errors": errors,
        "contract_exists": contract_exists,
        "tokenizer_exists": tokenizer_exists,
        "revision_match": revision_match,
        "train_count_match": train_count_match,
        "validation_count_match": validation_count_match,
        "vocab_size": vocab_size,
    }


# ---------------------------------------------------------------------------
# Environment Info
# ---------------------------------------------------------------------------

def get_environment_info() -> Dict[str, Any]:
    """Collect environment information for reproducibility."""
    import torch
    import transformers
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    return info


# ---------------------------------------------------------------------------
# Notebook 03 orchestration helpers (pilot cache pool, status, integrity)
# ---------------------------------------------------------------------------

EMPTY_NORMALIZED_REFERENCE = "empty_normalized_reference"

FROZEN_TEST_TOKENS = ("test", "rq1_test", "frozen_test")


def validate_prediction_lengths(
    pred_len: int,
    label_len: int,
    meta_len: int,
    dataset_len: int,
) -> None:
    """
    Assert prediction/label/metadata/dataset counts are all equal.

    Raises ValueError on ANY mismatch so predictions can never be silently
    truncated (no ``min(...)``).
    """
    if not (pred_len == label_len == meta_len == dataset_len):
        raise ValueError(
            "Prediction length mismatch (no silent truncation allowed): "
            f"predictions={pred_len}, labels={label_len}, "
            f"metadata={meta_len}, dataset={dataset_len}"
        )


def filter_empty_normalized_references(
    df: pd.DataFrame,
    *,
    text_col: str,
    normalize_fn: Optional[Any] = None,
    uid_col: str = "record_uid",
    id_col: str = "record_id",
    group_col: str = "group_id",
    duration_col: str = "processed_duration_seconds",
    fallback_duration_col: str = "duration_seconds",
    run_id: str = "",
    split: str = "",
) -> "tuple[pd.DataFrame, list]":
    """
    Split ``df`` into (kept, excluded) where excluded rows have an empty
    normalized transcript. Excluded rows are returned as dicts matching
    ``AUDIO_EXCLUSIONS_COLUMNS`` with reason ``empty_normalized_reference``.
    """
    if normalize_fn is None:
        from src.data_utils import normalize_bahnar_ctc_v1 as normalize_fn

    keep_mask = []
    excluded: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        norm = normalize_fn(row.get(text_col))
        if norm is not None and str(norm).strip() != "":
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            dur = row.get(duration_col)
            if dur is None:
                dur = row.get(fallback_duration_col)
            excluded.append({
                "run_id": run_id, "split": split,
                "record_uid": str(row.get(uid_col)), "record_id": row.get(id_col),
                "group_id": row.get(group_col), "reason": EMPTY_NORMALIZED_REFERENCE,
                "exception_type": None,
                "source_duration_seconds": row.get(fallback_duration_col),
                "processed_duration_seconds": dur,
            })
    kept = df[pd.Series(keep_mask, index=df.index)].copy() if len(df) else df.copy()
    return kept, excluded


def resolve_positive_duration(
    row: Any,
    *,
    duration_col: str = "processed_duration_seconds",
    fallback_duration_col: str = "duration_seconds",
    require: bool = False,
    uid: str = "",
    split: str = "",
) -> Optional[float]:
    """
    Resolve a usable duration from primary then fallback columns.

    Accepts only numeric, finite values strictly greater than 0.
    Does **not** use ``a or b`` (NaN is truthy in Python and would block fallback).
    If ``require`` and neither column yields a valid duration, raises ValueError.
    """
    for col in (duration_col, fallback_duration_col):
        if col is None:
            continue
        raw = row.get(col) if hasattr(row, "get") else row[col] if col in getattr(row, "index", ()) else None
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    if require:
        raise ValueError(
            f"[{split}] record_uid={uid}: no numeric finite duration > 0 "
            f"in columns ({duration_col}, {fallback_duration_col})"
        )
    return None


def resolve_training_duration_seconds(
    train_runtime: Any,
    perf_counter_seconds: Any,
    *,
    global_step: Any = 1,
) -> Dict[str, Any]:
    """
    Choose the official training duration for reporting.

    Prefer HuggingFace ``train_result.metrics["train_runtime"]`` when it is a
    positive finite number. Keep the local ``time.perf_counter()`` delta in
    ``perf_counter_seconds`` for diagnostics and as fallback (note: perf_counter
    may pause during Mac sleep). ``step_time_seconds`` is always derived from
    the official duration and ``global_step``.
    """
    def _positive_finite(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(f) and f > 0.0:
            return f
        return None

    perf = _positive_finite(perf_counter_seconds)
    hf = _positive_finite(train_runtime)
    if hf is not None:
        official = hf
        source = "train_runtime"
    elif perf is not None:
        official = perf
        source = "perf_counter_fallback"
    else:
        official = 0.0
        source = "unavailable"

    try:
        steps = int(global_step) if global_step is not None else 1
    except (TypeError, ValueError):
        steps = 1
    steps = max(1, steps)
    step_time = official / steps if official > 0.0 else 0.0

    return {
        "training_duration_seconds": float(official),
        "train_runtime_seconds": float(hf) if hf is not None else None,
        "perf_counter_seconds": float(perf) if perf is not None else None,
        "duration_source": source,
        "step_time_seconds": float(step_time),
        "global_step": steps,
    }


def build_eligible_pool_and_sample(
    pool_df: pd.DataFrame,
    target_n: int,
    split: str,
    seed: int,
    *,
    vocab: Dict[str, int],
    normalize_fn: Optional[Any] = None,
    ctc_check_fn: Optional[Any] = None,
    encode_fn: Optional[Any] = None,
    target_sr: int = TARGET_SAMPLING_RATE,
    text_col: str = "text_bahnar",
    uid_col: str = "record_uid",
    id_col: str = "record_id",
    group_col: str = "group_id",
    duration_col: str = "processed_duration_seconds",
    fallback_duration_col: str = "duration_seconds",
    run_id: str = "",
) -> Dict[str, Any]:
    """
    Build eligible pool by filtering empty refs + CTC-invalid, then group-aware sample.

    Steps:
    1. Filter entire pool for empty normalized references → exclusions
    2. Filter remaining for CTC feasibility → exclusions
    3. Create eligible_pool from records passing both checks
    4. Check len(eligible_pool) >= target_n, raise if not
    5. Call sample_pilot_data(eligible_pool, target_n, ...) for group-aware selection
    6. Return active_df, exclusions, and metadata (group counts)

    Returns dict with:
        active_df: Selected records (exactly target_n if pool sufficient)
        empty_ref_exclusions: List of dicts for audio_exclusions.csv
        ctc_exclusions: List of dicts for ctc_exclusions.csv
        eligible_pool_size: int
        eligible_group_count: int
        active_group_count: int
        group_coverage_ratio: float
        expected_active_groups: int
    """
    if normalize_fn is None:
        from src.data_utils import normalize_bahnar_ctc_v1 as normalize_fn
    if ctc_check_fn is None:
        ctc_check_fn = check_ctc_feasibility
    if encode_fn is None:
        from src.data_utils import encode_text_with_vocab as encode_fn

    empty_exclusions: List[Dict[str, Any]] = []
    ctc_exclusions: List[Dict[str, Any]] = []
    eligible_uids: List[str] = []
    dur_map: Dict[str, float] = {}

    # Step 1-2: Filter entire pool for empty refs AND CTC feasibility
    for _, row in pool_df.iterrows():
        uid = str(row.get(uid_col))
        text = row.get(text_col)
        norm = normalize_fn(text)

        # Empty reference check
        if norm is None or str(norm).strip() == "":
            dur = resolve_positive_duration(
                row,
                duration_col=duration_col,
                fallback_duration_col=fallback_duration_col,
                require=False,
                uid=uid,
                split=split,
            )
            empty_exclusions.append({
                "run_id": run_id, "split": split,
                "record_uid": uid, "record_id": row.get(id_col),
                "group_id": row.get(group_col), "reason": EMPTY_NORMALIZED_REFERENCE,
                "exception_type": None,
                "source_duration_seconds": resolve_positive_duration(
                    row,
                    duration_col=fallback_duration_col,
                    fallback_duration_col=fallback_duration_col,
                    require=False,
                    uid=uid,
                    split=split,
                ),
                "processed_duration_seconds": dur,
            })
            continue

        # CTC feasibility check — encode the *normalized* transcript
        dur = resolve_positive_duration(
            row,
            duration_col=duration_col,
            fallback_duration_col=fallback_duration_col,
            require=True,
            uid=uid,
            split=split,
        )
        token_ids = encode_fn(norm, vocab)
        num_samples = int(dur * target_sr)
        feas = ctc_check_fn(token_ids, num_samples)

        if not feas["feasible"]:
            ctc_exclusions.append({
                "run_id": run_id, "split": split,
                "record_uid": uid, "record_id": row.get(id_col),
                "group_id": row.get(group_col), "reason": feas["reason"],
                "duration_seconds": dur,
                "target_token_length": feas["target_token_length"],
                "min_required_frames": feas["min_required_frames"],
                "estimated_output_frames": feas["estimated_output_frames"],
            })
            continue

        # Record passes both checks
        eligible_uids.append(uid)
        dur_map[uid] = dur

    # Step 3: Create eligible_pool
    eligible_pool = pool_df[pool_df[uid_col].astype(str).isin(set(eligible_uids))].copy()
    eligible_pool[duration_col] = eligible_pool[uid_col].astype(str).map(dur_map)

    eligible_pool_size = len(eligible_pool)
    eligible_group_count = int(eligible_pool[group_col].nunique()) if eligible_pool_size > 0 else 0

    # Step 4: Check sufficiency
    if eligible_pool_size < target_n:
        raise RuntimeError(
            f"[{split}] Eligible pool insufficient for target. "
            f"cache_pool={len(pool_df)}, eligible_pool={eligible_pool_size}, target={target_n}, "
            f"empty_ref_exclusions={len(empty_exclusions)}, ctc_exclusions={len(ctc_exclusions)}. "
            "No Parquet download allowed; extend NB02 caching."
        )

    # Step 5: Group-aware sampling from eligible pool (NOT from full pool)
    active_df = sample_pilot_data(
        eligible_pool,
        target_n,
        seed=seed,
        group_col=group_col,
        uid_col=uid_col,
    )

    # Verify selection
    assert len(active_df) == target_n, f"Expected {target_n} records, got {len(active_df)}"
    active_uids = set(active_df[uid_col].astype(str))
    assert len(active_uids) == target_n, "Duplicate UIDs in active selection"
    assert active_uids.issubset(set(eligible_uids)), "Active records not all in eligible pool"

    # Step 6: Group coverage metrics + assert (applies to train and validation)
    active_group_count = int(active_df[group_col].nunique())
    expected_active_groups = min(target_n, eligible_group_count)
    group_coverage_ratio = (
        active_group_count / eligible_group_count if eligible_group_count > 0 else 0.0
    )
    # When group_id is present and non-null on eligible rows, round-robin must hit
    # min(target_n, eligible_group_count) distinct groups.
    if eligible_group_count > 0 and active_df[group_col].notna().all():
        assert active_group_count == expected_active_groups, (
            f"[{split}] group coverage mismatch: active_group_count={active_group_count} "
            f"!= expected_active_groups={expected_active_groups} "
            f"(target_n={target_n}, eligible_group_count={eligible_group_count})"
        )

    return {
        "active_df": active_df,
        "empty_ref_exclusions": empty_exclusions,
        "ctc_exclusions": ctc_exclusions,
        "eligible_pool_size": eligible_pool_size,
        "eligible_group_count": eligible_group_count,
        "active_group_count": active_group_count,
        "expected_active_groups": expected_active_groups,
        "group_coverage_ratio": group_coverage_ratio,
    }


def compute_records_and_hours(
    df: pd.DataFrame,
    *,
    duration_col: str = "processed_duration_seconds",
    fallback_duration_col: str = "duration_seconds",
) -> Dict[str, Any]:
    """Recompute record count and total hours from the FINAL active DataFrame."""
    n = int(len(df))
    if n == 0:
        return {"records": 0, "hours": 0.0}
    if duration_col in df.columns:
        durs = pd.to_numeric(df[duration_col], errors="coerce")
        if fallback_duration_col in df.columns:
            durs = durs.fillna(pd.to_numeric(df[fallback_duration_col], errors="coerce"))
    elif fallback_duration_col in df.columns:
        durs = pd.to_numeric(df[fallback_duration_col], errors="coerce")
    else:
        return {"records": n, "hours": 0.0}
    total_sec = float(durs.fillna(0.0).sum())
    return {"records": n, "hours": total_sec / 3600.0}


def checkpoint_belongs_to_run(checkpoint_path: Union[str, Path, None],
                              run_checkpoint_dir: Union[str, Path]) -> bool:
    """True iff ``checkpoint_path`` resolves inside ``run_checkpoint_dir``."""
    if not checkpoint_path:
        return False
    try:
        cp = Path(checkpoint_path).resolve()
        root = Path(run_checkpoint_dir).resolve()
    except (OSError, ValueError):
        return False
    return str(cp) == str(root) or str(cp).startswith(str(root) + os.sep)


def require_best_checkpoint(best_model_checkpoint: Optional[str]) -> str:
    """
    Return the trainer's best_model_checkpoint or raise. No fallback to the
    highest-step checkpoint is allowed.
    """
    if not best_model_checkpoint:
        raise RuntimeError(
            "trainer.state.best_model_checkpoint is missing; refusing to fall "
            "back to the highest-step checkpoint."
        )
    return str(best_model_checkpoint)


def derive_pilot_status(run_mode: str, pilot_passed: bool) -> str:
    """
    STATUS derivation with the invariant that SUCCESS_PILOT implies
    pilot_passed. Full mode never reports SUCCESS here (locked separately).
    """
    if run_mode == "pilot":
        return "SUCCESS_PILOT" if pilot_passed else "FAILED"
    return "FAILED"


def assert_full_mode_supported(run_mode: str, full_training_implemented: bool) -> None:
    """Fail-fast if full mode is selected before the scalable pipeline exists."""
    if run_mode == "full" and not full_training_implemented:
        raise RuntimeError(
            "Full training pipeline is not implemented yet.\n"
            "Do not treat the 20-step pilot as full training."
        )


def check_cache_pool_sufficiency(pool_size: int, target_size: int, split: str) -> Dict[str, Any]:
    """
    Verify a verified-cache pool has enough records for the pilot target.
    Returns a dict with ok/deficit/message (no download is ever attempted).
    """
    deficit = max(0, int(target_size) - int(pool_size))
    ok = int(pool_size) >= int(target_size)
    msg = (
        f"[{split}] verified cache pool has {pool_size} records; "
        f"target={target_size}; deficit={deficit}."
    )
    if not ok:
        msg += (
            " Refusing to download Parquet to backfill the pilot. "
            "Re-run Notebook 02 audio caching or lower the pilot target."
        )
    return {"ok": ok, "pool_size": int(pool_size), "target": int(target_size),
            "deficit": deficit, "split": split, "message": msg}


def detect_frozen_leakage(
    *,
    opened_paths: List[str],
    loaded_splits: List[str],
    source_splits: List[str],
    split_values: List[str],
    forbidden_tokens: tuple = FROZEN_TEST_TOKENS,
) -> List[str]:
    """
    Return a list of human-readable hits if any opened path / loaded split /
    source_split / split value references frozen-test data. Empty => safe.
    """
    hits: List[str] = []
    toks = {t.lower() for t in forbidden_tokens}
    for p in opened_paths:
        low = str(p).lower()
        if is_forbidden_test_path(p) or any(t in low for t in ("rq1_test", "frozen_test")):
            hits.append(f"path:{p}")
    for s in loaded_splits:
        if str(s).lower() in toks:
            hits.append(f"split:{s}")
    for s in source_splits:
        if str(s).lower() in toks:
            hits.append(f"source_split:{s}")
    for s in split_values:
        if str(s).lower() in toks:
            hits.append(f"split_value:{s}")
    return hits


def verify_artifacts_run_id(dirs: List[Union[str, Path]], run_id: str) -> List[str]:
    """
    Cross-file integrity: every JSON with a ``run_id`` field must equal
    ``run_id``; every non-empty CSV with a ``run_id`` column must contain only
    ``run_id``. Returns a list of mismatch messages (empty => all consistent).
    """
    issues: List[str] = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(obj, dict) and "run_id" in obj and obj.get("run_id") is not None:
                if str(obj["run_id"]) != str(run_id):
                    issues.append(f"{path.name}: run_id={obj['run_id']} != {run_id}")
        for path in sorted(d.rglob("*.csv")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "run_id" in df.columns and len(df) > 0:
                others = set(df["run_id"].dropna().astype(str)) - {str(run_id)}
                if others:
                    issues.append(f"{path.name}: unexpected run_ids {sorted(others)}")
    return issues
