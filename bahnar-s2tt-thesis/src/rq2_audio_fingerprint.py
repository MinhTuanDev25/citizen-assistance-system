"""RQ2 acoustic / perceptual audio fingerprinting.

Layers:

* **Exact identity** — SHA-256 of the canonical PCM16 bytes.
* **Perceptual identity** — a Chromaprint fingerprint (via ``fpcalc`` as a
  subprocess) plus a two-stage matcher: shingle inverted-index candidate
  retrieval, then alignment-aware verification with partial-overlap support.

The matcher math is pure NumPy/Python and is unit-testable with synthetic
fingerprint arrays, so offline tests never need the ``fpcalc`` binary. Only the
step that turns an audio file into a fingerprint requires ``fpcalc``.

This module never reads transcripts, runs a speech model, or computes
BLEU/CER/WER. It only compares audio identity.

``OverlapConfig.frozen`` gates the *threshold*, which must be frozen on
synthetic re-encode/trim/gain/different pairs — never on the frozen-test
similarity distribution.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

OVERLAP_CONTRACT_SCHEMA_VERSION = "rq2-overlap-1.1"
FPCALC_BINARY = "fpcalc"

# Precomputed popcount table for uint8 (fast, deterministic bit similarity).
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


@dataclass(frozen=True)
class OverlapConfig:
    fpcalc_binary: str = FPCALC_BINARY
    fingerprint_length_seconds: int = 120
    # Chromaprint emits ~1 item per this many seconds (used for matched_duration).
    seconds_per_fingerprint_item: float = 0.1238
    # Two-stage matcher knobs.
    shingle_k: int = 8
    min_shared_shingles: int = 1
    min_overlap_items: int = 20
    # Decision boundary in [0, 1]. PLACEHOLDER until frozen.
    similarity_threshold: float = 0.90
    frozen: bool = False

    def contract_payload(self) -> dict:
        return {
            "schema_version": OVERLAP_CONTRACT_SCHEMA_VERSION,
            "fingerprint_length_seconds": int(self.fingerprint_length_seconds),
            "seconds_per_fingerprint_item": round(float(self.seconds_per_fingerprint_item), 6),
            "shingle_k": int(self.shingle_k),
            "min_shared_shingles": int(self.min_shared_shingles),
            "min_overlap_items": int(self.min_overlap_items),
            "similarity_threshold": round(float(self.similarity_threshold), 6),
            "retrieval": "masked-byte-bands-v1",
            "band_count": 4,
        }


@dataclass(frozen=True)
class MatchEvidence:
    candidate_uid: str
    reference_uid: str
    reference_split: str
    matched_duration_seconds: float
    alignment_offset: int
    similarity: float
    match_type: str  # "u_u" or "protected"


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def overlap_contract_sha256(config: OverlapConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(config.contract_payload())).hexdigest()


# --------------------------------------------------------------------------- #
# Exact identity                                                               #
# --------------------------------------------------------------------------- #
def pcm16_sha256(pcm_int16: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(pcm_int16, dtype="<i2").tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Lazy dependency handling                                                     #
# --------------------------------------------------------------------------- #
def has_fpcalc(binary: str = FPCALC_BINARY) -> bool:
    return shutil.which(binary) is not None


def require_fpcalc(binary: str = FPCALC_BINARY) -> str:
    path = shutil.which(binary)
    if path is None:  # pragma: no cover - only without the dep
        raise RuntimeError(
            f"the '{binary}' binary (Chromaprint) is required for perceptual "
            "fingerprinting. Install it (brew install chromaprint) or inject "
            "precomputed fingerprints for offline tests."
        )
    return path


def compute_fingerprint(wav_path, config: OverlapConfig) -> List[int]:
    """Raw Chromaprint fingerprint (list of uint32) for a WAV file. Needs fpcalc."""
    binary = require_fpcalc(config.fpcalc_binary)
    cmd = [binary, "-raw", "-json", "-length", str(int(config.fingerprint_length_seconds)), str(wav_path)]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            f"fpcalc failed ({completed.returncode}) for {wav_path}: "
            f"{completed.stderr.decode(errors='replace')[-400:]}"
        )
    payload = json.loads(completed.stdout.decode("utf-8"))
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, list) or not fingerprint:
        raise RuntimeError(f"fpcalc returned an empty fingerprint for {wav_path}")
    return [int(x) & 0xFFFFFFFF for x in fingerprint]


def fingerprint_sha256(fingerprint: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(list(fingerprint), dtype="<u4").tobytes()).hexdigest()


def is_valid_fingerprint(fingerprint) -> bool:
    return isinstance(fingerprint, (list, tuple, np.ndarray)) and len(fingerprint) > 0


# --------------------------------------------------------------------------- #
# Alignment-aware verification (pure)                                          #
# --------------------------------------------------------------------------- #
def _bit_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    xor = np.bitwise_xor(a.astype("<u4"), b.astype("<u4")).view(np.uint8)
    diff_bits = int(_POPCOUNT8[xor].sum())
    return 1.0 - (diff_bits / float(len(a) * 32))


def compare_fingerprints_detailed(
    fp_a: Sequence[int],
    fp_b: Sequence[int],
    config: OverlapConfig,
) -> Tuple[float, int, int]:
    """Best-offset similarity. Returns (similarity, alignment_offset, overlap_items).

    Slides the shorter fingerprint across the longer one; supports partial
    temporal overlap. Only overlaps with ``>= min_overlap_items`` are scored.
    The offset is expressed relative to ``fp_a`` (offset of b within a).
    """
    a = np.asarray(list(fp_a), dtype="<u4")
    b = np.asarray(list(fp_b), dtype="<u4")
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0, 0
    swapped = False
    if len(a) > len(b):
        a, b = b, a
        swapped = True
    la, lb = len(a), len(b)
    min_items = max(1, int(config.min_overlap_items))
    best_score, best_offset, best_overlap = 0.0, 0, 0
    for offset in range(-(la - 1), lb):
        a_lo = max(0, -offset)
        a_hi = min(la, lb - offset)
        overlap = a_hi - a_lo
        if overlap < min_items:
            continue
        score = _bit_similarity(a[a_lo:a_hi], b[a_lo + offset:a_hi + offset])
        if score > best_score:
            best_score, best_offset, best_overlap = score, offset, overlap
    if swapped:
        best_offset = -best_offset
    return best_score, int(best_offset), int(best_overlap)


def compare_fingerprints(fp_a: Sequence[int], fp_b: Sequence[int], config: OverlapConfig) -> float:
    return compare_fingerprints_detailed(fp_a, fp_b, config)[0]


def matched_duration_seconds(overlap_items: int, config: OverlapConfig) -> float:
    return round(int(overlap_items) * float(config.seconds_per_fingerprint_item), 6)


def is_perceptual_match(fp_a: Sequence[int], fp_b: Sequence[int], config: OverlapConfig) -> bool:
    if not config.frozen:
        raise RuntimeError(
            "OverlapConfig is not frozen; freeze the similarity threshold on "
            "synthetic pairs before making perceptual match decisions."
        )
    return compare_fingerprints(fp_a, fp_b, config) >= config.similarity_threshold


# --------------------------------------------------------------------------- #
# Two-stage matcher: shingle inverted index + verification                     #
# --------------------------------------------------------------------------- #
def fingerprint_shingles(fingerprint: Sequence[int], k: int) -> List[Tuple[int, ...]]:
    fp = [int(x) & 0xFFFFFFFF for x in fingerprint]
    if len(fp) < k:
        return [tuple(fp)] if fp else []
    return [tuple(fp[i:i + k]) for i in range(len(fp) - k + 1)]


# Each mask drops one byte. A 1-bit change still matches the band that drops that byte.
_BAND_MASKS = (0x00FFFFFF, 0xFF00FFFF, 0xFFFF00FF, 0xFFFFFF00)


def fingerprint_query_keys(fingerprint: Sequence[int], config: OverlapConfig) -> List[tuple]:
    """Exact shingles plus masked-band keys. Shared by both index directions."""
    keys = [("exact", sh) for sh in fingerprint_shingles(fingerprint, config.shingle_k)]
    keys.extend(fingerprint_band_keys(fingerprint))
    return keys


def fingerprint_band_keys(fingerprint: Sequence[int]) -> List[Tuple[str, int, int]]:
    keys = []
    for value in fingerprint:
        item = int(value) & 0xFFFFFFFF
        for band, mask in enumerate(_BAND_MASKS):
            keys.append(("band", band, item & mask))
    return keys


class FingerprintIndex:
    """Inverted index over fingerprint shingles for candidate retrieval."""

    def __init__(self, config: OverlapConfig):
        self.config = config
        self._shingle_to_ids: Dict[tuple, set] = {}
        self._fingerprints: Dict[str, List[int]] = {}
        self._meta: Dict[str, dict] = {}

    def add(self, item_id: str, fingerprint: Sequence[int], meta: Optional[dict] = None) -> None:
        if not is_valid_fingerprint(fingerprint):
            raise ValueError(f"invalid fingerprint for {item_id}")
        fp = [int(x) & 0xFFFFFFFF for x in fingerprint]
        self._fingerprints[item_id] = fp
        self._meta[item_id] = dict(meta or {})
        for sh in fingerprint_shingles(fp, self.config.shingle_k):
            self._shingle_to_ids.setdefault(("exact", sh), set()).add(item_id)
        for key in fingerprint_band_keys(fp):
            self._shingle_to_ids.setdefault(key, set()).add(item_id)

    @property
    def entry_count(self) -> int:
        """Approximate inverted-index size (posting keys, not scientific matches)."""
        return len(self._shingle_to_ids)

    def fingerprint_of(self, item_id: str) -> List[int]:
        return self._fingerprints[item_id]

    def meta_of(self, item_id: str) -> dict:
        return self._meta.get(item_id, {})

    def candidates(self, fingerprint: Sequence[int], *, exclude: Optional[set] = None) -> List[str]:
        exclude = exclude or set()
        counts: Dict[str, int] = {}
        query_keys = fingerprint_query_keys(fingerprint, self.config)
        for key in query_keys:
            for item_id in self._shingle_to_ids.get(key, ()):
                if item_id in exclude:
                    continue
                counts[item_id] = counts.get(item_id, 0) + 1
        min_shared = max(1, int(self.config.min_shared_shingles))
        eligible = [(cid, c) for cid, c in counts.items() if c >= min_shared]
        # deterministic order: more shared shingles first, then id
        eligible.sort(key=lambda t: (-t[1], t[0]))
        return [cid for cid, _ in eligible]


class SegmentCandidateIndex:
    """Candidate index over the smaller U set.

    ``candidates_for_reference`` returns the same U ids that
    ``FingerprintIndex.candidates`` would return for that U fingerprint when the
    reference fingerprint is the indexed side. The score is the multiplicity of
    U keys that are present in the reference, so the candidate set does not
    change when the index direction is reversed.
    """

    def __init__(self, config: OverlapConfig):
        self.config = config
        self._fingerprints: Dict[str, List[int]] = {}
        self._key_mult: Dict[tuple, Dict[str, int]] = {}

    def add(self, item_id: str, fingerprint: Sequence[int]) -> None:
        if not is_valid_fingerprint(fingerprint):
            raise ValueError(f"invalid fingerprint for {item_id}")
        fp = [int(x) & 0xFFFFFFFF for x in fingerprint]
        self._fingerprints[item_id] = fp
        for key, mult in Counter(fingerprint_query_keys(fp, self.config)).items():
            self._key_mult.setdefault(key, {})[item_id] = int(mult)

    @property
    def entry_count(self) -> int:
        return len(self._key_mult)

    def fingerprint_of(self, item_id: str) -> List[int]:
        return self._fingerprints[item_id]

    def candidates_for_reference(self, fingerprint: Sequence[int]) -> List[str]:
        present = set(fingerprint_query_keys(fingerprint, self.config))
        counts: Dict[str, int] = {}
        for key in present:
            for item_id, mult in self._key_mult.get(key, {}).items():
                counts[item_id] = counts.get(item_id, 0) + mult
        min_shared = max(1, int(self.config.min_shared_shingles))
        eligible = [(cid, c) for cid, c in counts.items() if c >= min_shared]
        eligible.sort(key=lambda t: (-t[1], t[0]))
        return [cid for cid, _ in eligible]
