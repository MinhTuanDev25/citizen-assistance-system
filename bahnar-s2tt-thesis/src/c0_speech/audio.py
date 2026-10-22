"""Strict in-memory WAV validation for the Speech API.

Decodes with soundfile (same library as C0 CLI) but rejects stereo instead of
mixing channels. Never opens a client-supplied filesystem path.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np

from src.c0_cascaded import SAMPLE_RATE
from src.c0_speech.errors import SpeechAudioError

WAV_MAGIC_RIFF = b"RIFF"
WAV_MAGIC_WAVE = b"WAVE"


def decode_wav_bytes(
    data: bytes,
    *,
    max_bytes: int,
    max_duration_seconds: float,
    reader: Any = None,
) -> tuple[np.ndarray, int]:
    if data is None or len(data) == 0:
        raise SpeechAudioError("audio_empty", "Audio is empty")
    if len(data) > int(max_bytes):
        raise SpeechAudioError("payload_too_large", "Upload exceeds maximum size", status=413)
    if len(data) < 12 or data[:4] != WAV_MAGIC_RIFF or data[8:12] != WAV_MAGIC_WAVE:
        raise SpeechAudioError("audio_unsupported", "Only 16 kHz mono WAV is supported")
    if reader is None:
        import soundfile as sf

        reader = sf.read
    try:
        wav, sr = reader(io.BytesIO(data), dtype="float32", always_2d=True)
    except SpeechAudioError:
        raise
    except Exception as exc:
        raise SpeechAudioError("audio_invalid", "Audio could not be decoded") from exc
    arr = np.asarray(wav)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise SpeechAudioError("audio_empty", "Audio is empty")
    if arr.shape[1] != 1:
        raise SpeechAudioError("audio_channels", "Audio must be mono")
    if int(sr) != SAMPLE_RATE:
        raise SpeechAudioError("audio_sample_rate", "Audio sample rate must be 16000 Hz")
    mono = np.asarray(arr[:, 0], dtype=np.float32)
    if mono.size == 0:
        raise SpeechAudioError("audio_empty", "Audio is empty")
    if not np.isfinite(mono).all():
        raise SpeechAudioError("audio_invalid", "Audio could not be decoded")
    seconds = float(len(mono)) / float(SAMPLE_RATE)
    if seconds > float(max_duration_seconds):
        raise SpeechAudioError("audio_too_long", "Audio exceeds maximum duration")
    return mono, SAMPLE_RATE
