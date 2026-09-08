"""Unit tests for audio_utils (no network)."""

from __future__ import annotations

import numpy as np

from src.audio_utils import check_waveform, parse_sampling_rate


def test_float_mono_ok():
    sr = 16000
    wav = np.zeros(sr, dtype=np.float32)
    wav[100:200] = 0.1
    qa = check_waveform(wav, sr)
    assert qa["hard_ok"] is True
    assert qa["ok"] is True
    assert qa["reason"] == "ok"
    assert abs(qa["duration_sec"] - 1.0) < 1e-6


def test_stereo_samples_channels():
    sr = 8000
    wav = np.zeros((sr, 2), dtype=np.float32)
    wav[10:50, :] = 0.2
    qa = check_waveform(wav, sr)
    assert qa["hard_ok"] is True
    assert abs(qa["duration_sec"] - 1.0) < 1e-6


def test_stereo_channels_samples():
    sr = 8000
    wav = np.zeros((2, sr), dtype=np.float32)
    wav[:, 10:50] = 0.2
    qa = check_waveform(wav, sr)
    assert qa["hard_ok"] is True
    assert abs(qa["duration_sec"] - 1.0) < 1e-6


def test_int16_pcm_normalized():
    sr = 16000
    wav = np.zeros(sr, dtype=np.int16)
    wav[100:300] = 10000
    qa = check_waveform(wav, sr)
    assert qa["hard_ok"] is True
    assert 0.0 < qa["rms"] < 1.0


def test_int16_stereo_samples_channels():
    sr = 8000
    wav = np.zeros((sr, 2), dtype=np.int16)
    wav[100:500, :] = 10000
    qa = check_waveform(wav, sr)
    assert qa["hard_ok"] is True
    assert abs(qa["duration_sec"] - 1.0) < 1e-6
    assert 0.0 <= qa["rms"] <= 1.0
    assert "heavy_clipping" not in qa["quality_warnings"]


def test_int16_stereo_channels_samples():
    sr = 8000
    wav = np.zeros((2, sr), dtype=np.int16)
    wav[:, 100:500] = 10000
    qa = check_waveform(wav, sr)
    assert qa["hard_ok"] is True
    assert abs(qa["duration_sec"] - 1.0) < 1e-6
    assert 0.0 <= qa["rms"] <= 1.0
    assert "heavy_clipping" not in qa["quality_warnings"]


def test_unsigned_pcm_normalized():
    sr = 8000
    wav = np.full(sr, 128, dtype=np.uint8)  # midpoint ≈ silence after centering
    wav[100:300] = 200
    qa = check_waveform(wav, sr)
    assert qa["hard_ok"] is True
    assert 0.0 <= qa["rms"] <= 1.0


def test_null_and_empty():
    assert check_waveform(None, 16000)["reason"] == "null_array"
    assert check_waveform(np.array([]), 16000)["reason"] == "empty"
    assert check_waveform(None, 16000)["hard_ok"] is False


def test_bad_sampling_rate():
    wav = np.zeros(100, dtype=np.float32)
    assert check_waveform(wav, None)["reason"] == "bad_sampling_rate"
    assert check_waveform(wav, 0)["reason"] == "bad_sampling_rate"


def test_sampling_rate_nan():
    wav = np.zeros(1600, dtype=np.float32)
    wav[0] = 0.1
    qa = check_waveform(wav, np.nan)
    assert qa["hard_ok"] is False
    assert qa["reason"] == "bad_sampling_rate"


def test_sampling_rate_inf():
    wav = np.zeros(1600, dtype=np.float32)
    wav[0] = 0.1
    qa = check_waveform(wav, np.inf)
    assert qa["reason"] == "bad_sampling_rate"


def test_sampling_rate_string():
    wav = np.zeros(1600, dtype=np.float32)
    wav[0] = 0.1
    assert check_waveform(wav, "bad")["reason"] == "bad_sampling_rate"
    assert check_waveform(wav, "")["reason"] == "bad_sampling_rate"


def test_sampling_rate_negative():
    wav = np.zeros(1600, dtype=np.float32)
    wav[0] = 0.1
    assert check_waveform(wav, -16000)["reason"] == "bad_sampling_rate"


def test_sampling_rate_float_integer_value():
    wav = np.zeros(16000, dtype=np.float32)
    wav[100:200] = 0.1
    qa = check_waveform(wav, 16000.0)
    assert qa["hard_ok"] is True
    assert qa["sampling_rate"] == 16000
    assert parse_sampling_rate(16000.0) == 16000


def test_nan_inf_hard_fail():
    wav = np.zeros(1600, dtype=np.float32)
    wav[10] = np.nan
    assert check_waveform(wav, 16000)["reason"] == "non_finite_values"
    wav2 = np.zeros(1600, dtype=np.float32)
    wav2[10] = np.inf
    assert check_waveform(wav2, 16000)["reason"] == "non_finite_values"


def test_too_short_too_long():
    short = np.zeros(10, dtype=np.float32)
    assert check_waveform(short, 16000, min_duration=0.05)["reason"] == "too_short"
    long = np.zeros(16000 * 200, dtype=np.float32)
    long[0] = 0.1
    assert check_waveform(long, 16000, max_duration=120.0)["reason"] == "too_long"


def test_near_silent_is_warning_not_hard_fail():
    wav = np.zeros(16000, dtype=np.float32)
    qa = check_waveform(wav, 16000)
    assert qa["hard_ok"] is True
    assert qa["ok"] is True
    assert "near_silent" in qa["quality_warnings"]
    assert qa["reason"] == "near_silent"


def test_heavy_clipping_is_warning_not_hard_fail():
    wav = np.ones(16000, dtype=np.float32)
    qa = check_waveform(wav, 16000, max_clip_ratio=0.05)
    assert qa["hard_ok"] is True
    assert qa["ok"] is True
    assert "heavy_clipping" in qa["quality_warnings"]


def test_duration_mismatch_warning():
    wav = np.zeros(16000, dtype=np.float32)
    wav[0] = 0.1
    qa = check_waveform(wav, 16000, expected_duration=3.0, duration_mismatch_tol=0.35)
    assert qa["hard_ok"] is True
    assert "duration_mismatch" in qa["quality_warnings"]
    assert qa["duration_difference_sec"] is not None


def test_invalid_shape():
    wav = np.zeros((2, 2, 2), dtype=np.float32)
    assert check_waveform(wav, 16000)["reason"] == "invalid_shape"
