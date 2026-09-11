"""Audio decode / quality checks for RQ1 manifests."""

from __future__ import annotations

from typing import Any


HARD_FAILURE_REASONS = frozenset(
    {
        "null_array",
        "empty",
        "bad_sampling_rate",
        "invalid_shape",
        "non_finite_values",
        "too_short",
        "too_long",
    }
)

QUALITY_WARNING_REASONS = frozenset(
    {
        "near_silent",
        "heavy_clipping",
        "duration_mismatch",
    }
)


def _fail(
    reason: str,
    *,
    sampling_rate=None,
    duration_sec=None,
    clip_ratio=None,
    rms=None,
    quality_warnings=None,
    duration_difference_sec=None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "hard_ok": False,
        "reason": reason,
        "duration_sec": duration_sec,
        "sampling_rate": sampling_rate,
        "clip_ratio": clip_ratio,
        "rms": rms,
        "quality_warnings": list(quality_warnings or []),
        "duration_difference_sec": duration_difference_sec,
    }


def parse_sampling_rate(sampling_rate) -> int | None:
    """
    Safely parse a sampling rate to a positive int.
    Returns None for None/0/negative/nan/inf/non-numeric — never raises.
    """
    import numpy as np

    if sampling_rate is None:
        return None
    try:
        if isinstance(sampling_rate, (bytes, bytearray)):
            sampling_rate = sampling_rate.decode("utf-8", errors="strict")
        if isinstance(sampling_rate, str):
            text = sampling_rate.strip()
            if not text:
                return None
            value = float(text)
        else:
            value = float(sampling_rate)
        if not np.isfinite(value):
            return None
        if value <= 0:
            return None
        # Reject non-integer rates that are not whole numbers (e.g. 16000.5)
        if abs(value - round(value)) > 1e-9:
            return None
        return int(round(value))
    except (TypeError, ValueError, OverflowError, UnicodeDecodeError):
        return None


def _normalize_pcm_to_float(wav, np):
    """Normalize numeric array to float64 in approx [-1, 1] before channel mix."""
    if not np.issubdtype(wav.dtype, np.number):
        raise ValueError("invalid_shape")
    if np.issubdtype(wav.dtype, np.floating):
        return wav.astype(np.float64, copy=False)
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        as_f = wav.astype(np.float64)
        if info.min >= 0:
            # Unsigned PCM: center at midpoint then scale to [-1, 1]
            mid = (float(info.max) + 1.0) / 2.0
            return (as_f - mid) / mid
        denom = float(max(abs(info.min), info.max))
        if denom == 0:
            raise ValueError("invalid_shape")
        return as_f / denom
    raise ValueError("invalid_shape")


def _mix_to_mono(wav, np):
    """Mix channel axis after PCM normalization. Accepts mono or 2-D stereo layouts."""
    if wav.ndim == 0:
        raise ValueError("invalid_shape")
    if wav.ndim == 1:
        return wav
    if wav.ndim != 2:
        raise ValueError("invalid_shape")

    rows, cols = wav.shape
    # (channels, samples) when channels is small and samples is large
    if rows <= 8 and cols > rows:
        return wav.mean(axis=0)
    if cols <= 8 and rows > cols:
        # (samples, channels)
        return wav.mean(axis=1)
    if rows <= 8 and cols <= 8:
        # Ambiguous tiny matrix: prefer mean over last axis (samples, channels)
        return wav.mean(axis=-1)
    # Both dims large — assume (samples, channels) if second dim smaller
    return wav.mean(axis=1) if cols < rows else wav.mean(axis=0)


def waveform_to_mono_float32(array) -> "Any":
    """Public canonical PCM→mono float32 in [-1, 1] (normalize integer before mix)."""
    import numpy as np

    return _to_mono_float(np.asarray(array), np).astype(np.float32)


def _to_mono_float(wav, np):
    """
    Convert waveform to mono float64 in [-1, 1].

    Order:
    1) asarray
    2) shape / numeric dtype checks
    3) integer PCM normalize (signed or unsigned)
    4) then mix channels to mono
    """
    wav = np.asarray(wav)
    if wav.ndim == 0:
        raise ValueError("invalid_shape")
    if wav.size == 0:
        raise ValueError("empty")
    if not np.issubdtype(wav.dtype, np.number):
        raise ValueError("invalid_shape")
    if wav.ndim > 2:
        raise ValueError("invalid_shape")

    normalized = _normalize_pcm_to_float(wav, np)
    return _mix_to_mono(normalized, np)


def check_waveform(
    array,
    sampling_rate: int | None,
    *,
    min_duration: float = 0.05,
    max_duration: float = 120.0,
    max_clip_ratio: float = 0.05,
    expected_duration: float | None = None,
    duration_mismatch_tol: float = 0.35,
    near_silent_rms: float = 1e-5,
) -> dict[str, Any]:
    """
    Return QA fields for a decoded mono/stereo waveform.

    `ok` / hard failures exclude records from frozen eval.
    Quality warnings (near_silent, heavy_clipping, duration_mismatch) do NOT
    flip hard_ok/ok to False — they are reported in quality_warnings and
    mirrored in `reason` when there is no hard failure (first warning or "ok").
    """
    import numpy as np

    if array is None:
        return _fail("null_array", sampling_rate=sampling_rate)

    sr = parse_sampling_rate(sampling_rate)
    if sr is None:
        return _fail("bad_sampling_rate", sampling_rate=sampling_rate)

    try:
        wav = np.asarray(array)
    except Exception:
        return _fail("invalid_shape", sampling_rate=sampling_rate)

    if wav.size == 0:
        return _fail("empty", sampling_rate=sr, duration_sec=0.0)

    try:
        mono = _to_mono_float(wav, np)
    except ValueError as exc:
        reason = "empty" if str(exc) == "empty" else "invalid_shape"
        return _fail(reason, sampling_rate=sr, duration_sec=0.0 if reason == "empty" else None)

    if not np.isfinite(mono).all():
        return _fail("non_finite_values", sampling_rate=sr)

    duration = float(len(mono) / sr)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    clip_ratio = float(np.mean(np.abs(mono) >= 0.99)) if peak > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0

    if duration < min_duration:
        return _fail(
            "too_short",
            sampling_rate=sr,
            duration_sec=duration,
            clip_ratio=clip_ratio,
            rms=rms,
        )
    if duration > max_duration:
        return _fail(
            "too_long",
            sampling_rate=sr,
            duration_sec=duration,
            clip_ratio=clip_ratio,
            rms=rms,
        )

    warnings: list[str] = []
    duration_difference_sec = None
    if expected_duration is not None:
        try:
            exp = float(expected_duration)
            if np.isfinite(exp):
                duration_difference_sec = float(duration - exp)
                if abs(duration_difference_sec) > duration_mismatch_tol:
                    warnings.append("duration_mismatch")
        except (TypeError, ValueError, OverflowError):
            pass

    if rms < near_silent_rms:
        warnings.append("near_silent")
    if clip_ratio > max_clip_ratio:
        warnings.append("heavy_clipping")

    reason = warnings[0] if warnings else "ok"
    return {
        "ok": True,
        "hard_ok": True,
        "reason": reason,
        "duration_sec": duration,
        "sampling_rate": sr,
        "clip_ratio": clip_ratio,
        "rms": rms,
        "quality_warnings": warnings,
        "duration_difference_sec": duration_difference_sec,
    }
