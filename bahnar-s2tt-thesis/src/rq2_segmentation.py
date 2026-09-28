"""RQ2 U_real speech segmentation.

Language-independent, CPU-only speech segmentation for the VOV4 pseudo-label
pool (NB11). The frame-level speech decision uses WebRTC VAD; segment *planning*
is pure integer/float arithmetic over a boolean speech mask plus a per-frame
energy vector, so it is unit-testable without the optional ``webrtcvad`` binary.

Design rules (frozen-once contract):

* Sample-exact, half-open segment intervals ``[start_sample, end_sample)``.
* No temporal overlap: segments come from disjoint frame ranges.
* ``duration_seconds = (end_sample - start_sample) / sample_rate``.
* Long-region split picks, in order: (1) nearest non-speech boundary within
  ``boundary_search_seconds`` of the target, (2) deterministic lowest-energy
  frame inside that window, (3) hard cut only if no candidate is possible.
* A short final tail is merged into the previous chunk when the merged duration
  stays ``<= max``; otherwise it is dropped with reason ``TOO_SHORT``.
* Every dropped candidate carries an audit reason.
* ``segment_uid`` binds source identity + source PCM hash + exact offsets +
  the segmentation contract hash (which includes ``energy_fallback``).

Nothing here performs ASR, reads any transcript, or loads a model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_WEBRTC_VALID_FRAME_MS = (10, 20, 30)
_WEBRTC_VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)

SEGMENTATION_CONTRACT_SCHEMA_VERSION = "rq2-seg-1.1"

# Planning-level drop reasons (segment QA extends these downstream).
DROP_TOO_SHORT = "TOO_SHORT"
DROP_TOO_LONG = "TOO_LONG"
DROP_LOW_SPEECH_FRACTION = "LOW_SPEECH_FRACTION"


@dataclass(frozen=True)
class SegmentationConfig:
    vad_frame_ms: int = 30
    vad_mode: int = 2
    min_segment_seconds: float = 3.0
    target_segment_seconds: float = 25.0
    max_segment_seconds: float = 40.0
    merge_silence_gap_seconds: float = 0.4
    boundary_search_seconds: float = 2.0
    min_speech_fraction: float = 0.60
    energy_fallback: bool = True
    sample_rate: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2  # PCM16
    frozen: bool = False

    def __post_init__(self) -> None:
        if self.vad_frame_ms not in _WEBRTC_VALID_FRAME_MS:
            raise ValueError(f"vad_frame_ms must be one of {_WEBRTC_VALID_FRAME_MS}, got {self.vad_frame_ms}")
        if self.sample_rate not in _WEBRTC_VALID_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {_WEBRTC_VALID_SAMPLE_RATES}, got {self.sample_rate}")
        if not 0 <= self.vad_mode <= 3:
            raise ValueError(f"vad_mode must be in 0..3, got {self.vad_mode}")
        if self.channels != 1 or self.sample_width_bytes != 2:
            raise ValueError("RQ2 segmentation only handles mono PCM16")
        if not (0.0 < self.min_segment_seconds <= self.target_segment_seconds <= self.max_segment_seconds):
            raise ValueError("require 0 < min <= target <= max segment seconds")
        if not 0.0 <= self.min_speech_fraction <= 1.0:
            raise ValueError("min_speech_fraction must be in [0, 1]")
        if self.merge_silence_gap_seconds < 0 or self.boundary_search_seconds < 0:
            raise ValueError("gap/boundary search seconds must be >= 0")

    @property
    def frame_len_samples(self) -> int:
        return int(self.sample_rate * self.vad_frame_ms // 1000)

    def seconds_to_frames(self, seconds: float) -> int:
        return int(round(seconds * 1000.0 / self.vad_frame_ms))

    def contract_payload(self) -> dict:
        return {
            "schema_version": SEGMENTATION_CONTRACT_SCHEMA_VERSION,
            "vad_frame_ms": self.vad_frame_ms,
            "vad_mode": self.vad_mode,
            "min_segment_seconds": round(float(self.min_segment_seconds), 6),
            "target_segment_seconds": round(float(self.target_segment_seconds), 6),
            "max_segment_seconds": round(float(self.max_segment_seconds), 6),
            "merge_silence_gap_seconds": round(float(self.merge_silence_gap_seconds), 6),
            "boundary_search_seconds": round(float(self.boundary_search_seconds), 6),
            "min_speech_fraction": round(float(self.min_speech_fraction), 6),
            "energy_fallback": bool(self.energy_fallback),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
        }


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def segmentation_contract_sha256(config: SegmentationConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(config.contract_payload())).hexdigest()


@dataclass(frozen=True)
class Segment:
    start_sample: int
    end_sample: int
    start_frame: int
    end_frame: int
    vad_speech_fraction: float

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample

    def duration_seconds(self, sample_rate: int) -> float:
        return (self.end_sample - self.start_sample) / float(sample_rate)


@dataclass(frozen=True)
class DroppedCandidate:
    start_sample: int
    end_sample: int
    start_frame: int
    end_frame: int
    reason: str
    vad_speech_fraction: float


@dataclass(frozen=True)
class SegmentPlan:
    segments: List[Segment]
    drops: List[DroppedCandidate]


# --------------------------------------------------------------------------- #
# Lazy dependency handling                                                     #
# --------------------------------------------------------------------------- #
def has_webrtcvad() -> bool:
    try:
        import webrtcvad  # noqa: F401
    except Exception:
        return False
    return True


def require_webrtcvad():
    try:
        import webrtcvad
    except Exception as exc:  # pragma: no cover - only without the dep
        raise RuntimeError(
            "webrtcvad is required for full speech segmentation. Install it "
            "(pip install webrtcvad) or inject a precomputed speech mask + energy "
            "vector into plan_segments() for offline tests."
        ) from exc
    return webrtcvad


def compute_speech_flags(pcm_int16: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Per-frame boolean speech mask via WebRTC VAD. Requires ``webrtcvad``."""
    webrtcvad = require_webrtcvad()
    pcm = np.asarray(pcm_int16, dtype="<i2")
    frame_len = config.frame_len_samples
    n_frames = len(pcm) // frame_len
    vad = webrtcvad.Vad(config.vad_mode)
    flags = np.zeros(n_frames, dtype=bool)
    raw = pcm.tobytes()
    frame_nbytes = frame_len * config.sample_width_bytes
    for i in range(n_frames):
        flags[i] = vad.is_speech(raw[i * frame_nbytes:(i + 1) * frame_nbytes], config.sample_rate)
    return flags


def compute_frame_energies(pcm_int16: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Per-frame RMS energy (float, [0,1]) aligned to the speech mask. Pure."""
    pcm = np.asarray(pcm_int16, dtype="<i2").astype(np.float64) / 32768.0
    frame_len = config.frame_len_samples
    n_frames = len(pcm) // frame_len
    energies = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        chunk = pcm[i * frame_len:(i + 1) * frame_len]
        energies[i] = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
    return energies


# --------------------------------------------------------------------------- #
# Pure segment planning                                                        #
# --------------------------------------------------------------------------- #
def _merge_speech_regions(speech_flags: np.ndarray, merge_gap_frames: int) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    n = len(speech_flags)
    i = 0
    while i < n:
        if not speech_flags[i]:
            i += 1
            continue
        start = i
        end = i + 1
        while end < n:
            if speech_flags[end]:
                end += 1
                continue
            gap = end
            while gap < n and not speech_flags[gap]:
                gap += 1
            if gap < n and (gap - end) <= merge_gap_frames:
                end = gap
                continue
            break
        regions.append((start, end))
        i = end
    return regions


def _choose_cut(
    cursor: int,
    region_end: int,
    speech_flags: np.ndarray,
    frame_energies: np.ndarray,
    config: SegmentationConfig,
) -> int:
    """Deterministic cut frame for a long region starting at ``cursor``."""
    target = max(1, config.seconds_to_frames(config.target_segment_seconds))
    max_frames = max(1, config.seconds_to_frames(config.max_segment_seconds))
    search = config.seconds_to_frames(config.boundary_search_seconds)
    ideal = cursor + target

    low = max(cursor + 1, ideal - search)
    high = min(region_end - 1, ideal + search)

    # (1) nearest non-speech boundary within the search window
    best_silence = None
    for candidate in range(low, high + 1):
        if not speech_flags[candidate]:
            if best_silence is None or abs(candidate - ideal) < abs(best_silence - ideal):
                best_silence = candidate
    if best_silence is not None:
        cut = best_silence
    elif config.energy_fallback and high >= low:
        # (2) deterministic lowest-energy frame in the window (tie -> earliest)
        window = range(low, high + 1)
        cut = min(window, key=lambda c: (float(frame_energies[c]), c))
    else:
        # (3) hard cut at the target
        cut = ideal

    if cut - cursor > max_frames:
        cut = cursor + max_frames
    cut = min(cut, region_end)
    if cut <= cursor:
        cut = min(cursor + 1, region_end)
    return cut


def _split_region(
    region_start: int,
    region_end: int,
    speech_flags: np.ndarray,
    frame_energies: np.ndarray,
    config: SegmentationConfig,
) -> List[Tuple[int, int]]:
    max_frames = max(1, config.seconds_to_frames(config.max_segment_seconds))
    min_frames = max(1, config.seconds_to_frames(config.min_segment_seconds))
    chunks: List[Tuple[int, int]] = []
    cursor = region_start
    while region_end - cursor > max_frames:
        cut = _choose_cut(cursor, region_end, speech_flags, frame_energies, config)
        chunks.append((cursor, cut))
        cursor = cut
    if cursor < region_end:
        chunks.append((cursor, region_end))

    # Short final tail: merge into the previous chunk if the merged length fits.
    if len(chunks) >= 2:
        last_start, last_end = chunks[-1]
        if (last_end - last_start) < min_frames:
            prev_start, prev_end = chunks[-2]
            if (last_end - prev_start) <= max_frames:
                chunks[-2] = (prev_start, last_end)
                chunks.pop()
    return chunks


def plan_segments(
    speech_flags: Sequence[bool],
    frame_energies: Sequence[float],
    n_samples: int,
    config: SegmentationConfig,
) -> SegmentPlan:
    """Turn a frame-level speech mask + energy vector into a segment plan.

    Returns kept :class:`Segment` objects (sample-exact, non-overlapping,
    ``min <= duration <= max``, ``speech_fraction >= min``) and a list of
    :class:`DroppedCandidate` with audit reasons.
    """
    flags = np.asarray(list(speech_flags), dtype=bool)
    energies = np.asarray(list(frame_energies), dtype=np.float64)
    if len(energies) < len(flags):
        # pad with zeros if energies are short (defensive; should match length)
        energies = np.pad(energies, (0, len(flags) - len(energies)))
    frame_len = config.frame_len_samples
    merge_gap_frames = config.seconds_to_frames(config.merge_silence_gap_seconds)

    segments: List[Segment] = []
    drops: List[DroppedCandidate] = []
    for region_start, region_end in _merge_speech_regions(flags, merge_gap_frames):
        for f_start, f_end in _split_region(region_start, region_end, flags, energies, config):
            start_sample = f_start * frame_len
            end_sample = min(n_samples, f_end * frame_len)
            n_frames = f_end - f_start
            if end_sample <= start_sample or n_frames <= 0:
                continue
            duration = (end_sample - start_sample) / float(config.sample_rate)
            speech_frames = int(np.count_nonzero(flags[f_start:f_end]))
            speech_fraction = round(speech_frames / float(n_frames), 6)

            if duration + 1e-9 < config.min_segment_seconds:
                drops.append(DroppedCandidate(int(start_sample), int(end_sample), int(f_start), int(f_end), DROP_TOO_SHORT, speech_fraction))
                continue
            if duration > config.max_segment_seconds + 1e-9:
                drops.append(DroppedCandidate(int(start_sample), int(end_sample), int(f_start), int(f_end), DROP_TOO_LONG, speech_fraction))
                continue
            if speech_fraction + 1e-9 < config.min_speech_fraction:
                drops.append(DroppedCandidate(int(start_sample), int(end_sample), int(f_start), int(f_end), DROP_LOW_SPEECH_FRACTION, speech_fraction))
                continue
            segments.append(Segment(int(start_sample), int(end_sample), int(f_start), int(f_end), speech_fraction))

    assert_no_overlap(segments)
    return SegmentPlan(segments=segments, drops=drops)


def assert_no_overlap(segments: Sequence[Segment]) -> None:
    prev_end = -1
    for seg in segments:
        if seg.start_sample < 0 or seg.end_sample <= seg.start_sample:
            raise AssertionError(f"invalid segment interval: {seg}")
        if seg.start_sample < prev_end:
            raise AssertionError(f"temporal overlap at sample {seg.start_sample} < {prev_end}")
        prev_end = seg.end_sample


def segment_uid(
    source_id: str,
    source_pcm16_sha256: str,
    start_sample: int,
    end_sample: int,
    segmentation_contract_sha: str,
) -> str:
    if not source_id or not source_pcm16_sha256 or not segmentation_contract_sha:
        raise ValueError("segment_uid requires source_id, source_pcm16_sha256 and contract sha")
    canonical = "\x1f".join(
        [str(source_id), str(source_pcm16_sha256), str(int(start_sample)), str(int(end_sample)), str(segmentation_contract_sha)]
    )
    return "SEG_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def slice_pcm(pcm_int16: np.ndarray, start_sample: int, end_sample: int) -> np.ndarray:
    return np.asarray(pcm_int16, dtype="<i2")[start_sample:end_sample]
