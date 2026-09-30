"""RQ2 U_clean: segment QA, dedup, contamination protection, freeze.

Thin, composable helpers that NB11 orchestrates. Hard rules enforced here:

* Only ``ELIGIBLE_SOURCE`` + ``technical_status == PASS`` sources from NB10 are
  consumed; the source contract is validated and WAV hashes are re-verified
  (fail closed).
* Protected references (RQ1 G_train / G_validation / frozen G_test) are matched
  on **audio identity only** — the loader projects a small allow-list of columns
  and refuses transcript / reference / prediction / metric columns.
* The similarity threshold is never tuned on the frozen test; ``OverlapConfig``
  must be frozen (on synthetic pairs) before any real matching decision.
* Success requires explicit pipeline-completion evidence (references resolved,
  NB10 locked, fingerprint coverage complete, protection complete). Missing
  evidence -> FAIL, never a silent SUCCESS.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
import wave
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.rq2_audio_fingerprint import (
    FingerprintIndex,
    SegmentCandidateIndex,
    MatchEvidence,
    OverlapConfig,
    compare_fingerprints_detailed,
    fingerprint_sha256,
    is_valid_fingerprint,
    matched_duration_seconds,
    overlap_contract_sha256,
    pcm16_sha256,
)
from src.rq2_segmentation import (
    SegmentationConfig,
    Segment,
    SegmentPlan,
    DROP_TOO_SHORT,
    DROP_TOO_LONG,
    DROP_LOW_SPEECH_FRACTION,
    compute_frame_energies,
    plan_segments,
    segment_uid,
    segmentation_contract_sha256,
    slice_pcm,
)

U_CLEAN_SCHEMA_VERSION = "rq2-uclean-1.1"

SUCCESS_STATUS = "SUCCESS_RQ2_U_CLEAN_FROZEN"
FAIL_STATUS = "FAIL_RQ2_U_CLEAN"

NB10_SUCCESS_STATUS = "SUCCESS_RQ2_VOV4_FULL_SOURCE_POOL"

RETAINED_STATUS = "U_CLEAN_RETAINED"
EXCLUDED_EXACT_DUPLICATE = "EXCLUDED_EXACT_DUPLICATE"
EXCLUDED_PERCEPTUAL_DUPLICATE = "EXCLUDED_PERCEPTUAL_DUPLICATE"
EXCLUDED_PROTECTED_OVERLAP = "EXCLUDED_PROTECTED_OVERLAP"
EXCLUDED_TOO_SHORT = "EXCLUDED_TOO_SHORT"
EXCLUDED_TOO_LONG = "EXCLUDED_TOO_LONG"
EXCLUDED_LOW_SPEECH_FRACTION = "EXCLUDED_LOW_SPEECH_FRACTION"
EXCLUDED_SILENT = "EXCLUDED_SILENT"
EXCLUDED_INVALID_PCM = "EXCLUDED_INVALID_PCM"
EXCLUDED_SEGMENT_WRITE_FAILED = "EXCLUDED_SEGMENT_WRITE_FAILED"

# QA-level exclusions never enter dedup/protection (they are dropped upstream).
QA_EXCLUSIONS = (
    EXCLUDED_TOO_SHORT,
    EXCLUDED_TOO_LONG,
    EXCLUDED_LOW_SPEECH_FRACTION,
    EXCLUDED_SILENT,
    EXCLUDED_INVALID_PCM,
    EXCLUDED_SEGMENT_WRITE_FAILED,
)

_DROP_REASON_TO_STATUS = {
    DROP_TOO_SHORT: EXCLUDED_TOO_SHORT,
    DROP_TOO_LONG: EXCLUDED_TOO_LONG,
    DROP_LOW_SPEECH_FRACTION: EXCLUDED_LOW_SPEECH_FRACTION,
}

PROTECTED_SPLITS = ("g_train", "g_validation", "frozen_test")

_SPLIT_TO_FLAG = {
    "g_train": "overlap_g_train",
    "g_validation": "overlap_g_validation",
    "frozen_test": "overlap_frozen_test",
}

# NB11 must never read/propagate these from any protected-reference manifest.
FORBIDDEN_REFERENCE_TOKENS = (
    "text_bahnar", "text_vi", "text_en", "text_",
    "reference_text", "prediction", "hypothesis", "transcript",
    "bleu", "cer", "wer", "metric", "score", "target_text", "label_text",
)

REQUIRED_SOURCE_COLUMNS = (
    "source_id",
    "article_url",
    "article_date",
    "source_section",
    "content_class",
    "phase",
    "wav_local_path",
    "wav_sha256",
    "pcm16_sha256",
    "media_url_sha256",
    "duration_seconds",
    "source_level_status",
    "technical_status",
)

# Filenames Notebook 06 already writes under Rq1RuntimePaths.state_dir.
RQ1_FINAL_CONTRACT_FILENAME = "rq1_final_contract.json"
RQ1_TEST_CONTRACT_FILENAME = "rq1_test_contract.json"
# SHA fields already stored by build_rq1_test_contract. Paths are not.
RQ1_SPLIT_SHA_FIELDS = {
    "g_train": "train_manifest_sha256",
    "g_validation": "validation_manifest_sha256",
    "frozen_test": "manifest_sha256",
}

U_CLEAN_MANIFEST_COLUMNS = [
    "segment_uid",
    "canonical_segment_uid",
    "source_id",
    "source_group_id",
    "article_url",
    "media_url_sha256",
    "article_date",
    "source_section",
    "content_class",
    "phase",
    "source_wav_local_path",
    "source_wav_sha256",
    "source_pcm16_sha256",
    "segment_local_path",
    "segment_wav_sha256",
    "segment_pcm16_sha256",
    "start_sample",
    "end_sample",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "vad_speech_fraction",
    "peak_abs",
    "rms",
    "peak_dbfs",
    "rms_dbfs",
    "silence_fraction_energy",
    "fingerprint_sha256",
    "u_clean_status",
]

# Extra columns kept internally / in QA + exclusions exports (not in manifest).
_AUDIT_COLUMNS = [
    "exact_duplicate",
    "perceptual_duplicate",
    "overlap_g_train",
    "overlap_g_validation",
    "overlap_frozen_test",
    "exclusion_reason",
]

FORBIDDEN_PATH_MARKERS = ("/Users/", "/home/", "/workspace/", "C:/Users/", "C:\\Users\\")

_SILENCE_DBFS = -50.0
_SILENCE_FRAME_MS = 30
# A retained segment that is essentially silent by energy is excluded as SILENT.
_SILENT_FRACTION_THRESHOLD = 0.95

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# Project root & portability                                                   #
# --------------------------------------------------------------------------- #
def find_project_root(start: Optional[Path] = None) -> Path:
    node = Path(start or Path.cwd()).resolve()
    for cand in [node, *node.parents]:
        if (cand / "requirements.txt").is_file() and (cand / "src").is_dir() and (cand / "notebooks").is_dir():
            return cand
    raise RuntimeError(f"Cannot locate bahnar-s2tt-thesis root from {node}")


def to_project_relative(path, project_root: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(project_root).resolve()))
    except Exception:
        return str(p)


def resolve_project_path(value: str, project_root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (Path(project_root) / p)


def is_portable_path_value(value: str) -> bool:
    text = str(value or "")
    if not text:
        return True
    if Path(text).is_absolute():
        return False
    return not any(marker in text for marker in FORBIDDEN_PATH_MARKERS)


def assert_portable_artifact_paths(rows: Sequence[dict], fields: Sequence[str]) -> None:
    for row in rows:
        for f in fields:
            value = row.get(f, "")
            if value and not is_portable_path_value(value):
                raise RuntimeError(f"Non-portable path in field {f!r}: {value!r}")


def _is_valid_sha256(value: str) -> bool:
    return bool(_HEX64.match(str(value or "").strip().lower()))


def _dbfs(value: float) -> float:
    v = max(float(value), 1e-10)
    return round(20.0 * math.log10(v), 4)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Atomic IO                                                                    #
# --------------------------------------------------------------------------- #
def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class UCleanConfig:
    project_root: Path
    segmentation: SegmentationConfig
    overlap: OverlapConfig
    run_full_pipeline: bool = False
    nb10_manifest_path: Optional[Path] = None
    nb10_summary_path: Optional[Path] = None
    nb10_contract_path: Optional[Path] = None
    out_dir: Optional[Path] = None
    protected_reference_resolver: "Optional[ProtectedReferenceResolver]" = None

    @property
    def segmentation_config_frozen(self) -> bool:
        return bool(self.segmentation.frozen)

    @property
    def overlap_config_frozen(self) -> bool:
        return bool(self.overlap.frozen)

    def segmentation_contract_sha(self) -> str:
        return segmentation_contract_sha256(self.segmentation)

    def overlap_contract_sha(self) -> str:
        return overlap_contract_sha256(self.overlap)


def build_config(
    *,
    project_root: Optional[Path] = None,
    run_full_pipeline: bool = False,
    segmentation: Optional[SegmentationConfig] = None,
    overlap: Optional[OverlapConfig] = None,
    protected_reference_resolver: "Optional[ProtectedReferenceResolver]" = None,
) -> UCleanConfig:
    root = Path(project_root) if project_root is not None else find_project_root()
    seg = segmentation or SegmentationConfig()
    ov = overlap or OverlapConfig()
    out_dir = root / "artifacts" / "rq2" / "u_clean"
    nb10_dir = root / "artifacts" / "rq2" / "vov4_full"
    return UCleanConfig(
        project_root=root,
        segmentation=seg,
        overlap=ov,
        run_full_pipeline=run_full_pipeline,
        nb10_manifest_path=nb10_dir / "source_pool_manifest.csv",
        nb10_summary_path=nb10_dir / "summary.json",
        nb10_contract_path=nb10_dir / "contract.json",
        out_dir=out_dir,
        protected_reference_resolver=protected_reference_resolver,
    )


# --------------------------------------------------------------------------- #
# Protected reference (audio identity only)                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProtectedReferenceEntry:
    reference_uid: str
    split: str
    audio_locator: str
    group_id: str = ""
    duration_seconds: float = 0.0
    source_sha256: str = ""

    def __post_init__(self) -> None:
        if self.split not in PROTECTED_SPLITS:
            raise ValueError(f"split must be one of {PROTECTED_SPLITS}, got {self.split!r}")


@dataclass(frozen=True)
class ReferenceColumnMapping:
    """Maps manifest source columns to audio-identity fields. No text columns."""

    reference_uid: str
    audio_locator: str
    split: Optional[str] = None
    group_id: Optional[str] = None
    duration_seconds: Optional[str] = None
    source_sha256: Optional[str] = None

    def source_columns(self) -> List[str]:
        cols = [self.reference_uid, self.audio_locator]
        for opt in (self.split, self.group_id, self.duration_seconds, self.source_sha256):
            if opt:
                cols.append(opt)
        return cols


PROTECTED_AUDIO_IDENTITY_COLUMNS = ("reference_uid", "split", "audio_locator", "source_sha256")


def default_protected_manifest_mapping() -> ReferenceColumnMapping:
    """Map the canonical RQ1 manifests. Those files have no per-audio SHA256 column."""
    return ReferenceColumnMapping(
        reference_uid="record_uid",
        audio_locator="audio_path",
        duration_seconds="duration_seconds",
        group_id="group_id",
    )


def require_protected_source_sha256(entries: Sequence[ProtectedReferenceEntry]) -> None:
    """Full-run gate: every protected entry must carry a real audio SHA256."""
    for entry in entries:
        if not _is_valid_sha256(str(entry.source_sha256 or "")):
            raise RuntimeError(
                f"protected reference {entry.split}:{entry.reference_uid} has no source_sha256"
            )


def load_protected_audio_identity_index(path, expected_sha256: str) -> Dict[Tuple[str, str], dict]:
    """Load a hash-locked identity index. Transcript columns are not read."""
    from src.rq1_contract import sha256_file
    import pandas as pd

    index_path = Path(path)
    expected = str(expected_sha256 or "").strip().lower()
    if not _is_valid_sha256(expected):
        raise RuntimeError("protected audio identity index requires a SHA256 pin")
    if not index_path.is_file():
        raise RuntimeError(f"protected audio identity index is missing: {index_path}")
    actual = sha256_file(index_path)
    if actual != expected:
        raise RuntimeError("protected audio identity index SHA256 mismatch")
    frame = pd.read_csv(index_path, usecols=list(PROTECTED_AUDIO_IDENTITY_COLUMNS))
    assert_no_forbidden_reference_columns(list(frame.columns))
    index: Dict[Tuple[str, str], dict] = {}
    for _, row in frame.iterrows():
        split = str(row["split"] or "")
        uid = str(row["reference_uid"] or "")
        locator = str(row["audio_locator"] or "")
        digest = str(row["source_sha256"] or "").strip().lower()
        if split not in PROTECTED_SPLITS or not uid or not locator:
            raise RuntimeError("protected audio identity row is incomplete")
        if not _is_valid_sha256(digest):
            raise RuntimeError(f"protected audio identity {split}:{uid} has no source_sha256")
        key = (split, uid)
        if key in index:
            raise RuntimeError(f"duplicate protected audio identity for {split}:{uid}")
        index[key] = {"audio_locator": locator, "source_sha256": digest}
    if not index:
        raise RuntimeError("protected audio identity index is empty")
    return index


def bind_protected_audio_identity(
    entries: Sequence[ProtectedReferenceEntry],
    index: Dict[Tuple[str, str], dict],
) -> List[ProtectedReferenceEntry]:
    """Attach the hash-locked SHA256 to each manifest entry. Locator must match."""
    bound: List[ProtectedReferenceEntry] = []
    for entry in entries:
        item = index.get((entry.split, entry.reference_uid))
        if item is None:
            raise RuntimeError(
                f"protected audio identity missing for {entry.split}:{entry.reference_uid}"
            )
        if item["audio_locator"] != entry.audio_locator:
            raise RuntimeError(
                f"protected audio locator mismatch for {entry.split}:{entry.reference_uid}"
            )
        bound.append(replace(entry, source_sha256=item["source_sha256"]))
    require_protected_source_sha256(bound)
    return bound


def assert_no_forbidden_reference_columns(columns: Sequence[str]) -> None:
    lowered = [str(c).lower() for c in columns]
    for col in lowered:
        for token in FORBIDDEN_REFERENCE_TOKENS:
            if token in col:
                raise RuntimeError(
                    f"protected-reference column {col!r} matches forbidden token "
                    f"{token!r}; NB11 must load audio identity only"
                )


def load_reference_identity_frame(path, mapping: ReferenceColumnMapping):
    """Load ONLY the audio-identity columns from a manifest. Fails closed."""
    import pandas as pd

    source_cols = mapping.source_columns()
    assert_no_forbidden_reference_columns(source_cols)
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, usecols=source_cols)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path, columns=source_cols)
    else:
        raise RuntimeError(f"unsupported protected manifest format: {suffix or path!r}")
    assert_no_forbidden_reference_columns(list(frame.columns))
    return frame


class ProtectedReferenceResolver:
    """Interface: resolve() -> audio-identity entries for all 3 splits."""

    def resolve(self) -> List[ProtectedReferenceEntry]:  # pragma: no cover - interface
        raise NotImplementedError

    def provenance(self) -> dict:
        return {"resolver": type(self).__name__}


class SyntheticProtectedReferenceResolver(ProtectedReferenceResolver):
    """Deterministic in-memory resolver for synthetic tests / fixtures."""

    def __init__(self, entries: Sequence[ProtectedReferenceEntry], audio_root: Optional[object] = None):
        self._entries = list(entries)
        self._audio_root = Path(audio_root) if audio_root is not None else None

    def resolve(self) -> List[ProtectedReferenceEntry]:
        return list(self._entries)

    def resolve_audio_path(self, entry: ProtectedReferenceEntry) -> Path:
        if self._audio_root is None:
            raise RuntimeError("synthetic resolver has no audio_root; cannot resolve protected audio")
        return _durable_audio_path(self._audio_root, entry.audio_locator)

    def provenance(self) -> dict:
        return {"resolver": type(self).__name__, "n_entries": len(self._entries)}


class ManifestProtectedReferenceResolver(ProtectedReferenceResolver):
    """Resolve from explicit per-split manifest paths + a column mapping."""

    def __init__(self, split_to_path: Dict[str, object], mapping: ReferenceColumnMapping):
        missing = [s for s in PROTECTED_SPLITS if s not in split_to_path]
        if missing:
            raise ValueError(f"missing protected splits: {missing}")
        self._split_to_path = dict(split_to_path)
        self._mapping = mapping

    def resolve(self) -> List[ProtectedReferenceEntry]:
        entries: List[ProtectedReferenceEntry] = []
        m = self._mapping
        for split in PROTECTED_SPLITS:
            frame = load_reference_identity_frame(self._split_to_path[split], m)
            for _, row in frame.iterrows():
                entries.append(
                    ProtectedReferenceEntry(
                        reference_uid=str(row[m.reference_uid]),
                        split=split,
                        audio_locator=str(row[m.audio_locator]),
                        group_id=str(row[m.group_id]) if m.group_id else "",
                        duration_seconds=float(row[m.duration_seconds]) if m.duration_seconds else 0.0,
                        source_sha256=str(row[m.source_sha256]) if m.source_sha256 else "",
                    )
                )
        return entries

    def provenance(self) -> dict:
        return {
            "resolver": type(self).__name__,
            "split_paths": {s: str(p) for s, p in self._split_to_path.items()},
        }


def _durable_audio_path(durable_root, locator: str) -> Path:
    text = str(locator or "").strip()
    if not text or Path(text).is_absolute():
        raise RuntimeError(f"protected audio locator must be relative to durable_root, got {locator!r}")
    root = Path(durable_root).resolve()
    resolved = (root / text).resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"protected audio path escapes durable_root: {locator!r}")
    return resolved


def _logical_manifest_path(path, durable_root, project_root) -> Tuple[str, str]:
    """Return (root_name, relative posix path). Absolute machine paths are not stored."""
    resolved = Path(path).resolve()
    roots = (
        ("durable", Path(durable_root).resolve()),
        ("project", Path(project_root).resolve()),
    )
    for name, root in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        if relative.as_posix() in ("", "."):
            break
        return name, relative.as_posix()
    raise RuntimeError(f"manifest path is outside durable_root and project_root: {path}")


def verify_rq1_contract_hash(payload: dict, hash_field: str, expected_hash: str) -> str:
    """Recompute a contract hash with RQ1's ``sha256_json``. A stale declared hash fails."""
    from src.rq1_contract import sha256_json

    declared = str(payload.get(hash_field) or "").strip().lower()
    body = {key: value for key, value in payload.items() if key != hash_field}
    recomputed = sha256_json(body)
    expected = str(expected_hash or "").strip().lower()
    if recomputed != declared or recomputed != expected:
        raise RuntimeError(
            f"RQ1 contract hash mismatch for {hash_field}: "
            f"recomputed {recomputed} declared {declared} expected {expected}"
        )
    return recomputed


class DurableCanonicalReferenceResolver(ProtectedReferenceResolver):
    """Resolve FINAL RQ1 audio identity through the existing durable runtime.

    The contract file is ``Rq1RuntimePaths.state_dir(hash) / rq1_final_contract.json``,
    the path Notebook 06 already writes. Its hash is recomputed with ``sha256_json``.
    Split manifest bytes are hash-locked to the SHA fields in the sibling
    ``rq1_test_contract.json``. The final contract does not store paths, so the
    caller supplies those locations in ``reference_index``.
    """

    def __init__(
        self,
        *,
        durable_root: Optional[object],
        contract_hash: Optional[str],
        reference_index: Optional[Dict[str, object]],
        mapping: Optional[ReferenceColumnMapping],
        project_root: Optional[object] = None,
        audio_identity_index: Optional[object] = None,
        audio_identity_sha256: Optional[str] = None,
    ):
        self._durable_root = durable_root
        self._project_root = project_root
        self._contract_hash = str(contract_hash or "").strip().lower()
        self._reference_index = reference_index
        self._mapping = mapping
        self._audio_identity_index = audio_identity_index
        self._audio_identity_sha256 = str(audio_identity_sha256 or "").strip().lower()
        self.expected_counts: Dict[str, int] = {}
        self._manifest_paths: Dict[str, Path] = {}
        self._split_provenance: Dict[str, dict] = {}
        self._test_contract_hash = ""

    def _load_contract(self) -> dict:
        from src.rq1_contract import sha256_file
        from src.rq1_runtime_paths import resolve_rq1_runtime_paths

        if not self._durable_root or not self._project_root or not self._contract_hash or not self._mapping:
            raise RuntimeError(
                "protected reference is unavailable: durable_root, project_root, "
                "contract_hash and column mapping must come from the RQ1 runtime (fail closed)."
            )
        runtime = resolve_rq1_runtime_paths(
            project_root=self._project_root,
            durable_root=self._durable_root,
        )
        state_dir = runtime.state_dir(self._contract_hash)
        contract_path = state_dir / RQ1_FINAL_CONTRACT_FILENAME
        if not contract_path.is_file():
            raise RuntimeError(f"RQ1 final contract is missing: {contract_path}")
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        verify_rq1_contract_hash(payload, "rq1_final_contract_hash", self._contract_hash)
        test_hash = str(payload.get("rq1_test_contract_hash") or "").strip().lower()
        test_path = state_dir / RQ1_TEST_CONTRACT_FILENAME
        if not test_path.is_file():
            raise RuntimeError(f"RQ1 test contract is missing: {test_path}")
        test_payload = json.loads(test_path.read_text(encoding="utf-8"))
        verify_rq1_contract_hash(test_payload, "rq1_test_contract_hash", test_hash)
        self._test_contract_hash = test_hash
        if not self._reference_index:
            raise RuntimeError(
                "RQ1 final contract does not store manifest paths; an explicit "
                "reference_index is required (fail closed)."
            )
        paths: Dict[str, Path] = {}
        counts: Dict[str, int] = {}
        provenance: Dict[str, dict] = {}
        for split, sha_field in RQ1_SPLIT_SHA_FIELDS.items():
            expected_sha = str(test_payload.get(sha_field) or "").strip().lower()
            if not _is_valid_sha256(expected_sha):
                raise RuntimeError(f"RQ1 test contract is missing a SHA pin for {split}")
            if split not in self._reference_index or not str(self._reference_index.get(split) or "").strip():
                raise RuntimeError(f"reference_index is missing a manifest locator for {split}")
            manifest_path = Path(self._reference_index[split])
            if not manifest_path.is_file():
                raise RuntimeError(f"RQ1 manifest missing for {split}: {manifest_path}")
            actual_sha = sha256_file(manifest_path)
            if actual_sha != expected_sha:
                raise RuntimeError(f"RQ1 manifest SHA256 mismatch for {split}")
            logical_root, relative = _logical_manifest_path(
                manifest_path, self._durable_root, self._project_root,
            )
            paths[split] = manifest_path
            if split == "frozen_test":
                if "test_count" not in test_payload:
                    raise RuntimeError("RQ1 test contract is missing test_count")
                counts[split] = int(test_payload["test_count"])
            provenance[split] = {
                "manifest_root": logical_root,
                "manifest_relative_path": relative,
                "manifest_sha256": actual_sha,
            }
        self._manifest_paths = paths
        self.expected_counts = counts
        self._split_provenance = provenance
        return payload

    def resolve(self) -> List[ProtectedReferenceEntry]:
        self._load_contract()
        if not self._audio_identity_index or not self._audio_identity_sha256:
            raise RuntimeError(
                "canonical RQ1 manifests have no per-audio SHA256; a hash-locked "
                "protected audio identity index is required"
            )
        identity = load_protected_audio_identity_index(
            self._audio_identity_index, self._audio_identity_sha256,
        )
        delegate = ManifestProtectedReferenceResolver(self._manifest_paths, self._mapping)
        entries = bind_protected_audio_identity(delegate.resolve(), identity)
        actual = Counter(entry.split for entry in entries)
        for split, expected in self.expected_counts.items():
            if actual.get(split, 0) != expected:
                raise RuntimeError(
                    f"protected split {split} count {actual.get(split, 0)} != contract n_rows {expected}"
                )
        for split in PROTECTED_SPLITS:
            self._split_provenance[split]["n_rows"] = int(actual.get(split, 0))
        return entries

    def resolve_audio_path(self, entry: ProtectedReferenceEntry) -> Path:
        if not self._durable_root:
            raise RuntimeError("durable_root is not set")
        return _durable_audio_path(self._durable_root, entry.audio_locator)

    def provenance(self) -> dict:
        return {
            "resolver": type(self).__name__,
            "rq1_final_contract_hash": str(self._contract_hash or ""),
            "rq1_test_contract_hash": str(self._test_contract_hash or ""),
            "protected_audio_identity_sha256": str(self._audio_identity_sha256 or ""),
            "splits": {split: dict(info) for split, info in self._split_provenance.items()},
        }


def build_durable_protected_resolver(
    *,
    project_root,
    reference_index: Dict[str, object],
    mapping: ReferenceColumnMapping,
    durable_root=None,
    audio_identity_index=None,
    audio_identity_sha256: Optional[str] = None,
) -> DurableCanonicalReferenceResolver:
    """Bind an explicit manifest index to the RQ1 final contract hash.

    ``LATEST_UNLOCKED.json`` supplies ``final_contract_hash`` only. Its
    ``state_dir`` field is ignored; the state directory is recomputed by
    ``Rq1RuntimePaths.state_dir``.
    """
    from src.rq1_runtime_paths import resolve_rq1_runtime_paths

    if not reference_index:
        raise RuntimeError(
            "RQ1 final contract does not store manifest paths; an explicit "
            "reference_index is required (fail closed)."
        )
    runtime = resolve_rq1_runtime_paths(project_root=project_root, durable_root=durable_root)
    latest_path = runtime.durable_state_root / "LATEST_UNLOCKED.json"
    if not latest_path.is_file():
        raise RuntimeError("RQ1 LATEST_UNLOCKED.json is missing; cannot resolve the final contract hash")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    contract_hash = str(latest.get("final_contract_hash") or "").strip().lower()
    if not _is_valid_sha256(contract_hash):
        raise RuntimeError("LATEST_UNLOCKED.final_contract_hash is missing or invalid")
    return DurableCanonicalReferenceResolver(
        durable_root=runtime.durable_root,
        project_root=runtime.project_root,
        contract_hash=contract_hash,
        reference_index=dict(reference_index),
        mapping=mapping,
        audio_identity_index=audio_identity_index,
        audio_identity_sha256=audio_identity_sha256,
    )


def summarize_reference_entries(entries: Sequence[ProtectedReferenceEntry]) -> dict:
    """Per-split counts + a stable fingerprint of the reference identity set."""
    per_split = {s: 0 for s in PROTECTED_SPLITS}
    canonical = []
    for e in entries:
        per_split[e.split] = per_split.get(e.split, 0) + 1
        canonical.append((e.split, e.reference_uid, e.audio_locator, e.source_sha256))
    canonical.sort()
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"per_split": per_split, "n_total": len(entries), "reference_set_sha256": digest}


# --------------------------------------------------------------------------- #
# Audio IO & QA                                                                #
# --------------------------------------------------------------------------- #
def read_wav_pcm16(path) -> dict:
    """Read a 16 kHz mono PCM16 WAV. Fails closed on any other format."""
    raw = Path(path).read_bytes()
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        n_frames = handle.getnframes()
        frames = handle.readframes(n_frames)
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise RuntimeError(
            f"expected 16kHz mono PCM16, got rate={sample_rate} ch={channels} width={sample_width}"
        )
    pcm = np.frombuffer(frames, dtype="<i2")
    return {
        "pcm": pcm,
        "sample_rate": sample_rate,
        "channels": channels,
        "n_samples": len(pcm),
        "wav_sha256": hashlib.sha256(raw).hexdigest(),
        "pcm16_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
    }


def write_wav_pcm16_atomic(path, pcm_int16: np.ndarray, sample_rate: int = 16000) -> str:
    """Atomically write a mono PCM16 WAV and return its container SHA256."""
    import io

    pcm = np.asarray(pcm_int16, dtype="<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    data = buffer.getvalue()
    atomic_write_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def compute_segment_acoustics(pcm_int16: np.ndarray, sample_rate: int = 16000) -> dict:
    pcm = np.asarray(pcm_int16, dtype="<i2").astype(np.float64) / 32768.0
    if len(pcm) == 0:
        return {
            "peak_abs": 0.0, "rms": 0.0, "peak_dbfs": _dbfs(0.0),
            "rms_dbfs": _dbfs(0.0), "silence_fraction_energy": 1.0,
        }
    peak_abs = float(np.max(np.abs(pcm)))
    rms = float(np.sqrt(np.mean(np.square(pcm))))
    frame_len = int(sample_rate * _SILENCE_FRAME_MS // 1000)
    if frame_len <= 0:
        frame_len = len(pcm)
    n_frames = max(1, len(pcm) // frame_len)
    silent = 0
    threshold = 10.0 ** (_SILENCE_DBFS / 20.0)
    for i in range(n_frames):
        chunk = pcm[i * frame_len:(i + 1) * frame_len]
        if len(chunk) == 0:
            continue
        frame_rms = float(np.sqrt(np.mean(np.square(chunk))))
        if frame_rms <= threshold:
            silent += 1
    return {
        "peak_abs": round(peak_abs, 8),
        "rms": round(rms, 8),
        "peak_dbfs": _dbfs(peak_abs),
        "rms_dbfs": _dbfs(rms),
        "silence_fraction_energy": round(silent / float(n_frames), 8),
    }


# --------------------------------------------------------------------------- #
# Inputs & pipeline-completion state                                           #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineCompletion:
    references_resolved: bool = False
    reference_fingerprint_coverage_complete: bool = False
    segment_fingerprint_coverage_complete: bool = False
    u_u_protection_complete: bool = False
    protected_overlap_check_complete: bool = False
    nb10_locked: bool = False
    segments_written_verified: bool = False
    artifacts_reverified: bool = False

    @property
    def fingerprint_coverage_complete(self) -> bool:
        return self.segment_fingerprint_coverage_complete and self.reference_fingerprint_coverage_complete

    @property
    def protection_complete(self) -> bool:
        return (
            self.references_resolved
            and self.reference_fingerprint_coverage_complete
            and self.segment_fingerprint_coverage_complete
            and self.u_u_protection_complete
            and self.protected_overlap_check_complete
        )

    def as_dict(self) -> dict:
        return {
            "references_resolved": self.references_resolved,
            "reference_fingerprint_coverage_complete": self.reference_fingerprint_coverage_complete,
            "segment_fingerprint_coverage_complete": self.segment_fingerprint_coverage_complete,
            "u_u_protection_complete": self.u_u_protection_complete,
            "protected_overlap_check_complete": self.protected_overlap_check_complete,
            "fingerprint_coverage_complete": self.fingerprint_coverage_complete,
            "protection_complete": self.protection_complete,
            "nb10_locked": self.nb10_locked,
            "segments_written_verified": self.segments_written_verified,
            "artifacts_reverified": self.artifacts_reverified,
        }


@dataclass
class UCleanContext:
    config: UCleanConfig
    eligible_sources: List[dict] = field(default_factory=list)
    nb10_status: str = ""
    nb10_locks: Dict[str, str] = field(default_factory=dict)
    protected_entries: List[ProtectedReferenceEntry] = field(default_factory=list)
    reference_summary: Dict[str, object] = field(default_factory=dict)
    reference_provenance: Dict[str, object] = field(default_factory=dict)
    completion: PipelineCompletion = field(default_factory=PipelineCompletion)
    match_evidence: List[MatchEvidence] = field(default_factory=list)


def _read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_source_contract(rows: Sequence[dict]) -> None:
    """Item 12: fail closed on an ill-formed / ineligible NB10 source row."""
    seen_ids = set()
    for row in rows:
        missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in row]
        if missing:
            raise RuntimeError(f"source row missing required columns: {missing}")
        sid = str(row.get("source_id") or "").strip()
        if not sid:
            raise RuntimeError("source_id is empty")
        if sid in seen_ids:
            raise RuntimeError(f"duplicate source_id: {sid}")
        seen_ids.add(sid)
        if str(row.get("source_level_status") or "") != "ELIGIBLE_SOURCE":
            raise RuntimeError(f"source {sid} is not ELIGIBLE_SOURCE")
        if str(row.get("technical_status") or "") != "PASS":
            raise RuntimeError(f"source {sid} technical_status != PASS")
        wav_rel = str(row.get("wav_local_path") or "")
        if not wav_rel or not is_portable_path_value(wav_rel):
            raise RuntimeError(f"source {sid} wav_local_path missing/non-portable: {wav_rel!r}")
        if not _is_valid_sha256(row.get("wav_sha256")):
            raise RuntimeError(f"source {sid} has missing/invalid wav_sha256")
        if not _is_valid_sha256(row.get("pcm16_sha256")):
            raise RuntimeError(f"source {sid} has missing/invalid pcm16_sha256")
        if not _is_valid_sha256(row.get("media_url_sha256")):
            raise RuntimeError(f"source {sid} has missing/invalid media_url_sha256")
        for field in ("article_url", "article_date", "source_section", "content_class", "phase"):
            if not str(row.get(field) or "").strip():
                raise RuntimeError(f"source {sid} is missing provenance field {field}")
        try:
            duration = float(row.get("duration_seconds") or 0.0)
        except (TypeError, ValueError):
            raise RuntimeError(f"source {sid} has non-numeric duration_seconds")
        if duration <= 0:
            raise RuntimeError(f"source {sid} duration_seconds must be > 0")


def lock_nb10_upstream(config: UCleanConfig) -> Dict[str, str]:
    """Item 4: require 3 NB10 files + SUCCESS status; return SHA256 locks."""
    paths = {
        "summary.json": config.nb10_summary_path,
        "source_pool_manifest.csv": config.nb10_manifest_path,
        "contract.json": config.nb10_contract_path,
    }
    locks: Dict[str, str] = {}
    for name, path in paths.items():
        if not path or not Path(path).is_file():
            raise RuntimeError(f"NB10 upstream artifact missing: {name}")
        locks[name] = _sha256_file(path)
    summary = _read_json(config.nb10_summary_path)
    status = str(summary.get("status") or "")
    if status != NB10_SUCCESS_STATUS:
        raise RuntimeError(f"NB10 is not {NB10_SUCCESS_STATUS}: {status!r}")
    locks["nb10_status"] = status
    return locks


def resolve_protected_references(config: UCleanConfig, context: UCleanContext) -> None:
    """Item 3: resolve all 3 splits, populate context, fail closed if incomplete."""
    resolver = config.protected_reference_resolver
    if resolver is None:
        raise RuntimeError("no protected_reference_resolver configured; cannot protect references")
    entries = resolver.resolve()
    present = {e.split for e in entries}
    missing = [s for s in PROTECTED_SPLITS if s not in present]
    if missing:
        raise RuntimeError(f"protected references incomplete; missing splits: {missing}")
    require_protected_source_sha256(entries)
    context.protected_entries = list(entries)
    context.reference_summary = summarize_reference_entries(entries)
    context.reference_provenance = resolver.provenance()
    context.completion.references_resolved = True


def validate_inputs(
    config: UCleanConfig,
    *,
    source_rows: Optional[Sequence[dict]] = None,
    resolve_references: Optional[bool] = None,
) -> UCleanContext:
    """Load NB10 outputs; keep only ELIGIBLE_SOURCE + PASS; validate contract.

    In full-pipeline mode NB10 is locked (3 files + SUCCESS), the source
    contract is validated, and protected references are resolved.
    ``source_rows`` / ``resolve_references`` may be overridden for tests.
    """
    context = UCleanContext(config=config)

    if config.run_full_pipeline:
        context.nb10_locks = lock_nb10_upstream(config)
        context.nb10_status = context.nb10_locks.get("nb10_status", "")
        context.completion.nb10_locked = True

    if source_rows is None:
        manifest_path = config.nb10_manifest_path
        if manifest_path and Path(manifest_path).is_file():
            import pandas as pd

            frame = pd.read_csv(manifest_path, dtype=str).fillna("")
            source_rows = frame.to_dict(orient="records")
        else:
            source_rows = []

    eligible = [
        dict(r)
        for r in source_rows
        if str(r.get("source_level_status") or "") == "ELIGIBLE_SOURCE"
        and str(r.get("technical_status") or "") == "PASS"
    ]
    validate_source_contract(eligible)
    context.eligible_sources = eligible

    should_resolve = config.run_full_pipeline if resolve_references is None else resolve_references
    if should_resolve:
        resolve_protected_references(config, context)

    return context


def verify_source_hashes(row: dict, project_root: Path) -> None:
    """Re-verify the source WAV container + PCM hash. Fails closed (item 12)."""
    wav_rel = row.get("wav_local_path") or ""
    if not wav_rel:
        raise RuntimeError(f"source {row.get('source_id')} has no wav_local_path")
    expected_wav = str(row.get("wav_sha256") or "").strip().lower()
    expected_pcm = str(row.get("pcm16_sha256") or "").strip().lower()
    if not _is_valid_sha256(expected_wav) or not _is_valid_sha256(expected_pcm):
        raise RuntimeError(f"source {row.get('source_id')} has missing/invalid hashes")
    wav_path = resolve_project_path(wav_rel, project_root)
    info = read_wav_pcm16(wav_path)
    if info["wav_sha256"] != expected_wav:
        raise RuntimeError(f"wav_sha256 mismatch for {row.get('source_id')}")
    if info["pcm16_sha256"] != expected_pcm:
        raise RuntimeError(f"pcm16_sha256 mismatch for {row.get('source_id')}")


# --------------------------------------------------------------------------- #
# Build segments (planning + QA)                                               #
# --------------------------------------------------------------------------- #
def _base_segment_row(source_row: dict, config: UCleanConfig, source_pcm_sha: str) -> dict:
    article_url = source_row.get("article_url", "")
    return {
        "source_id": source_row.get("source_id", ""),
        "source_group_id": source_row.get("source_id", ""),
        "article_url": article_url,
        "media_url_sha256": str(source_row.get("media_url_sha256") or "").strip().lower(),
        "article_date": source_row.get("article_date", ""),
        "source_section": source_row.get("source_section", ""),
        "content_class": source_row.get("content_class", ""),
        "phase": source_row.get("phase", ""),
        "source_wav_local_path": source_row.get("wav_local_path", ""),
        "source_wav_sha256": source_row.get("wav_sha256", ""),
        "source_pcm16_sha256": source_pcm_sha,
        "segment_local_path": "",
        "segment_wav_sha256": "",
        "segment_pcm16_sha256": "",
        "fingerprint_sha256": "",
        "canonical_segment_uid": "",
        "peak_abs": 0.0,
        "rms": 0.0,
        "peak_dbfs": _dbfs(0.0),
        "rms_dbfs": _dbfs(0.0),
        "silence_fraction_energy": 1.0,
        "exact_duplicate": False,
        "perceptual_duplicate": False,
        "overlap_g_train": False,
        "overlap_g_validation": False,
        "overlap_frozen_test": False,
        "u_clean_status": RETAINED_STATUS,
        "exclusion_reason": "",
    }


def _finalize_uid(row: dict, source_pcm_sha: str, contract_sha: str) -> None:
    row["segment_uid"] = segment_uid(
        str(row["source_id"]), source_pcm_sha,
        int(row["start_sample"]), int(row["end_sample"]), contract_sha,
    )
    if not row["canonical_segment_uid"]:
        row["canonical_segment_uid"] = row["segment_uid"]


def build_segments_for_source(
    source_row: dict,
    pcm_int16: np.ndarray,
    speech_flags: Sequence[bool],
    config: UCleanConfig,
    *,
    frame_energies: Optional[Sequence[float]] = None,
) -> List[dict]:
    """Plan segments for one source, run QA, and attach identity fields.

    Returns every candidate row (retained + QA-excluded), each auditable via a
    stable ``segment_uid``. ``frame_energies`` is derived from ``pcm_int16`` when
    not injected.
    """
    seg_cfg = config.segmentation
    contract_sha = config.segmentation_contract_sha()
    n_samples = len(pcm_int16)
    source_pcm_sha = pcm16_sha256(pcm_int16)
    if frame_energies is None:
        frame_energies = compute_frame_energies(pcm_int16, seg_cfg)

    plan: SegmentPlan = plan_segments(speech_flags, frame_energies, n_samples, seg_cfg)
    rows: List[dict] = []

    for seg in plan.segments:
        seg_pcm = slice_pcm(pcm_int16, seg.start_sample, seg.end_sample)
        row = _base_segment_row(source_row, config, source_pcm_sha)
        row.update({
            "start_sample": int(seg.start_sample),
            "end_sample": int(seg.end_sample),
            "start_seconds": round(seg.start_sample / float(seg_cfg.sample_rate), 6),
            "end_seconds": round(seg.end_sample / float(seg_cfg.sample_rate), 6),
            "duration_seconds": round(seg.duration_seconds(seg_cfg.sample_rate), 6),
            "vad_speech_fraction": seg.vad_speech_fraction,
        })
        _finalize_uid(row, source_pcm_sha, contract_sha)

        # Segment QA (item 9): INVALID_PCM / SILENT are excluded, else retained.
        if len(seg_pcm) == 0:
            _exclude(row, EXCLUDED_INVALID_PCM)
        else:
            acoustics = compute_segment_acoustics(seg_pcm, seg_cfg.sample_rate)
            row.update({
                "segment_pcm16_sha256": pcm16_sha256(seg_pcm),
                "peak_abs": acoustics["peak_abs"],
                "rms": acoustics["rms"],
                "peak_dbfs": acoustics["peak_dbfs"],
                "rms_dbfs": acoustics["rms_dbfs"],
                "silence_fraction_energy": acoustics["silence_fraction_energy"],
            })
            if acoustics["silence_fraction_energy"] >= _SILENT_FRACTION_THRESHOLD or acoustics["peak_abs"] <= 0.0:
                _exclude(row, EXCLUDED_SILENT)
        rows.append(row)

    for drop in plan.drops:
        row = _base_segment_row(source_row, config, source_pcm_sha)
        row.update({
            "start_sample": int(drop.start_sample),
            "end_sample": int(drop.end_sample),
            "start_seconds": round(drop.start_sample / float(seg_cfg.sample_rate), 6),
            "end_seconds": round(drop.end_sample / float(seg_cfg.sample_rate), 6),
            "duration_seconds": round((drop.end_sample - drop.start_sample) / float(seg_cfg.sample_rate), 6),
            "vad_speech_fraction": drop.vad_speech_fraction,
        })
        _finalize_uid(row, source_pcm_sha, contract_sha)
        _exclude(row, _DROP_REASON_TO_STATUS.get(drop.reason, EXCLUDED_TOO_SHORT))
        rows.append(row)

    return rows


# --------------------------------------------------------------------------- #
# Dedup + contamination protection                                             #
# --------------------------------------------------------------------------- #
def _exclude(row: dict, reason: str) -> None:
    row["u_clean_status"] = reason
    row["exclusion_reason"] = reason


def _owner_key(row: dict) -> tuple:
    """Item 10: deterministic owner = smallest (source_id, start_sample, uid)."""
    return (str(row["source_id"]), int(row["start_sample"]), str(row["segment_uid"]))


def exact_deduplicate(segments: List[dict]) -> List[dict]:
    """Mark exact PCM duplicates and persist canonical_segment_uid."""
    by_hash: Dict[str, List[dict]] = {}
    for row in segments:
        if row["u_clean_status"] != RETAINED_STATUS:
            continue
        by_hash.setdefault(row["segment_pcm16_sha256"], []).append(row)
    for group in by_hash.values():
        owner = min(group, key=_owner_key)
        for row in group:
            row["canonical_segment_uid"] = owner["segment_uid"]
            if row is owner:
                continue
            row["exact_duplicate"] = True
            _exclude(row, EXCLUDED_EXACT_DUPLICATE)
    return segments


def perceptual_deduplicate(
    segments: List[dict],
    fingerprints_by_uid: Dict[str, Sequence[int]],
    config: UCleanConfig,
) -> List[MatchEvidence]:
    """Scalable U-U perceptual dedup via shingle index (item 11)."""
    if not config.overlap_config_frozen:
        raise RuntimeError("perceptual dedup requires a frozen OverlapConfig")
    ov = config.overlap
    retained = [r for r in segments if r["u_clean_status"] == RETAINED_STATUS]
    retained.sort(key=_owner_key)
    index = FingerprintIndex(ov)
    uid_to_row: Dict[str, dict] = {}
    evidence: List[MatchEvidence] = []
    for row in retained:
        uid = row["segment_uid"]
        fp = fingerprints_by_uid.get(uid)
        if fp is None:
            continue  # coverage gate handles missing fingerprints elsewhere
        best = None
        for cand_uid in index.candidates(fp):
            score, offset, overlap = compare_fingerprints_detailed(fp, index.fingerprint_of(cand_uid), ov)
            if score >= ov.similarity_threshold and (best is None or score > best[0]):
                best = (score, offset, overlap, cand_uid)
        if best is not None:
            score, offset, overlap, owner_uid = best
            row["perceptual_duplicate"] = True
            row["canonical_segment_uid"] = uid_to_row[owner_uid]["canonical_segment_uid"]
            _exclude(row, EXCLUDED_PERCEPTUAL_DUPLICATE)
            evidence.append(MatchEvidence(
                candidate_uid=uid, reference_uid=owner_uid, reference_split="u_real",
                matched_duration_seconds=matched_duration_seconds(overlap, ov),
                alignment_offset=offset, similarity=round(score, 6), match_type="u_u",
            ))
        else:
            index.add(uid, fp)
            uid_to_row[uid] = row
    return evidence


def protect_against_references(
    segments: List[dict],
    segment_fingerprints_by_uid: Dict[str, Sequence[int]],
    reference_fingerprints: Sequence[dict],
    config: UCleanConfig,
    benchmark_sink: Optional[dict] = None,
) -> List[MatchEvidence]:
    """Flag & exclude segments overlapping any protected split (item 11).

    ``reference_fingerprints``: list of {"uid", "split", "fingerprint"}.
    Uses a shingle inverted-index for candidate retrieval, then aligned verify.
    """
    if not config.overlap_config_frozen:
        raise RuntimeError("overlap protection requires a frozen OverlapConfig")
    index, build_seconds = build_u_candidate_index(segments, segment_fingerprints_by_uid, config)
    if benchmark_sink is not None:
        benchmark_sink["index_build_elapsed_seconds"] = round(build_seconds, 6)
        benchmark_sink["index_entry_count"] = int(index.entry_count)
        benchmark_sink["n_references_fingerprinted"] = 0
    evidence: List[MatchEvidence] = []
    batch: List[dict] = []
    for ref in reference_fingerprints:
        batch.append(ref)
        if len(batch) >= 32:
            evidence.extend(match_protected_batch(segments, index, batch, config))
            if benchmark_sink is not None:
                benchmark_sink["n_references_fingerprinted"] = int(benchmark_sink["n_references_fingerprinted"]) + len(batch)
            batch = []
    if batch:
        evidence.extend(match_protected_batch(segments, index, batch, config))
        if benchmark_sink is not None:
            benchmark_sink["n_references_fingerprinted"] = int(benchmark_sink["n_references_fingerprinted"]) + len(batch)
    return evidence


def build_u_candidate_index(
    segments: Sequence[dict],
    segment_fingerprints_by_uid: Dict[str, Sequence[int]],
    config: UCleanConfig,
) -> Tuple[SegmentCandidateIndex, float]:
    """Index retained U fingerprints. Protected references are not stored here."""
    started = time.perf_counter()
    index = SegmentCandidateIndex(config.overlap)
    for row in segments:
        if row.get("u_clean_status") != RETAINED_STATUS:
            continue
        fp = segment_fingerprints_by_uid.get(row["segment_uid"])
        if not is_valid_fingerprint(fp):
            continue
        index.add(str(row["segment_uid"]), fp)
    return index, time.perf_counter() - started


def match_protected_batch(
    segments: List[dict],
    index: SegmentCandidateIndex,
    reference_batch: Sequence[dict],
    config: UCleanConfig,
) -> List[MatchEvidence]:
    """Query one protected batch against the U index and verify alignments."""
    ov = config.overlap
    by_uid = {str(row["segment_uid"]): row for row in segments}
    evidence: List[MatchEvidence] = []
    for ref in reference_batch:
        ref_fp = ref.get("fingerprint")
        if not is_valid_fingerprint(ref_fp):
            continue
        split = str(ref.get("split") or "")
        for uid in index.candidates_for_reference(ref_fp):
            row = by_uid.get(uid)
            status = row.get("u_clean_status") if row is not None else ""
            if row is None or status not in (RETAINED_STATUS, EXCLUDED_PROTECTED_OVERLAP):
                continue
            score, offset, overlap = compare_fingerprints_detailed(index.fingerprint_of(uid), ref_fp, ov)
            if score < ov.similarity_threshold:
                continue
            flag = _SPLIT_TO_FLAG.get(split)
            if flag:
                row[flag] = True
            evidence.append(MatchEvidence(
                candidate_uid=uid,
                reference_uid=str(ref.get("uid") or ""),
                reference_split=split,
                matched_duration_seconds=matched_duration_seconds(overlap, ov),
                alignment_offset=offset,
                similarity=round(score, 6),
                match_type="protected",
            ))
            _exclude(row, EXCLUDED_PROTECTED_OVERLAP)
    return evidence


def references_cover_splits(entries: Sequence[ProtectedReferenceEntry]) -> bool:
    counts = {split: 0 for split in PROTECTED_SPLITS}
    for entry in entries:
        counts[entry.split] = counts.get(entry.split, 0) + 1
    return bool(entries) and all(counts[split] > 0 for split in PROTECTED_SPLITS)


def reference_fingerprints_cover(entries: Sequence[ProtectedReferenceEntry], reference_fingerprints: Optional[Sequence[dict]]) -> bool:
    """Every protected entry has exactly one valid fingerprint for its split."""
    if not references_cover_splits(entries):
        return False
    by_uid: Dict[str, dict] = {}
    for ref in reference_fingerprints or []:
        uid = str(ref.get("uid") or "")
        if not uid or uid in by_uid or not is_valid_fingerprint(ref.get("fingerprint")):
            return False
        by_uid[uid] = ref
    if len(by_uid) != len(entries):
        return False
    for entry in entries:
        ref = by_uid.get(entry.reference_uid)
        if ref is None or str(ref.get("split") or "") != entry.split:
            return False
    return True


def verify_protected_reference_audio(entry: ProtectedReferenceEntry, audio_path) -> None:
    """Fail closed before fingerprinting a protected reference."""
    path = Path(audio_path)
    if not str(entry.reference_uid or "").strip():
        raise RuntimeError("protected reference uid is empty")
    if not path.is_file():
        raise RuntimeError(f"protected audio missing for {entry.reference_uid}: {path}")
    expected = str(entry.source_sha256 or "").strip().lower()
    if not _is_valid_sha256(expected):
        raise RuntimeError(f"protected reference {entry.reference_uid} has no source_sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"protected audio hash mismatch for {entry.reference_uid}")
    if entry.duration_seconds > 0 and path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 0
        if rate <= 0:
            raise RuntimeError(f"protected audio has no sample rate: {entry.reference_uid}")
        duration = frames / float(rate)
        if abs(duration - float(entry.duration_seconds)) > 0.05:
            raise RuntimeError(f"protected audio duration mismatch for {entry.reference_uid}")


def probe_protected_audio_identity(entries: Sequence[ProtectedReferenceEntry], resolver, per_split: int = 1) -> List[dict]:
    """Identity-only preflight. The report has no transcript fields."""
    seen: Dict[str, int] = {}
    report: List[dict] = []
    for entry in entries:
        seen[entry.split] = seen.get(entry.split, 0) + 1
        if seen[entry.split] > per_split:
            continue
        sha_available = _is_valid_sha256(str(entry.source_sha256 or ""))
        resolvable = False
        matches = False
        try:
            path = Path(resolver.resolve_audio_path(entry))
            resolvable = path.is_file()
            if resolvable and sha_available:
                matches = hashlib.sha256(path.read_bytes()).hexdigest() == str(entry.source_sha256).strip().lower()
        except Exception:
            resolvable = False
            matches = False
        report.append({
            "split": entry.split,
            "reference_uid": entry.reference_uid,
            "locator_resolvable": resolvable,
            "expected_sha_available": sha_available,
            "hash_matches": matches,
        })
    return report


def _fingerprint_coverage_complete(segments: Sequence[dict], fingerprints_by_uid: Dict[str, Sequence[int]]) -> bool:
    """Every segment that survived exact dedup must have a fingerprint."""
    for row in segments:
        status = row["u_clean_status"]
        if status == RETAINED_STATUS or status in (EXCLUDED_PERCEPTUAL_DUPLICATE, EXCLUDED_PROTECTED_OVERLAP):
            if fingerprints_by_uid.get(row["segment_uid"]) is None:
                return False
    return True


def protect_and_deduplicate(
    segments: List[dict],
    context: UCleanContext,
    *,
    segment_fingerprints_by_uid: Optional[Dict[str, Sequence[int]]] = None,
    reference_fingerprints: Optional[Sequence[dict]] = None,
) -> List[dict]:
    """Exact dedup -> perceptual U-U dedup -> protected-overlap exclusion.

    Sets ``context.completion`` flags honestly: protection is only complete when
    the overlap config is frozen, references are resolved, fingerprint coverage
    is complete, and both perceptual + protected passes have run.
    """
    exact_deduplicate(segments)
    context.completion.references_resolved = references_cover_splits(context.protected_entries)

    fps = segment_fingerprints_by_uid
    overlap_ready = context.config.overlap_config_frozen and fps is not None
    if not overlap_ready:
        context.completion.segment_fingerprint_coverage_complete = False
        context.completion.reference_fingerprint_coverage_complete = False
        context.completion.u_u_protection_complete = False
        context.completion.protected_overlap_check_complete = False
        return segments

    segment_coverage = _fingerprint_coverage_complete(segments, fps)
    context.completion.segment_fingerprint_coverage_complete = segment_coverage
    if segment_coverage:
        ev1 = perceptual_deduplicate(segments, fps, context.config)
        context.completion.u_u_protection_complete = True
    else:
        ev1 = []
        context.completion.u_u_protection_complete = False

    reference_coverage = reference_fingerprints_cover(context.protected_entries, reference_fingerprints)
    context.completion.reference_fingerprint_coverage_complete = reference_coverage
    if segment_coverage:
        bench: Dict[str, object] = {}
        ev2 = protect_against_references(
            segments, fps, reference_fingerprints or [], context.config, benchmark_sink=bench,
        )
        provenance = dict(context.reference_provenance)
        provenance["protected_index_benchmark"] = bench
        context.reference_provenance = provenance
        context.completion.protected_overlap_check_complete = bool(reference_coverage)
    else:
        ev2 = []
        context.completion.protected_overlap_check_complete = False
    context.match_evidence = list(ev1) + list(ev2)
    return segments


# --------------------------------------------------------------------------- #
# Validation (fail closed)                                                     #
# --------------------------------------------------------------------------- #
def retained_segments(segments: Sequence[dict]) -> List[dict]:
    return [r for r in segments if r.get("u_clean_status") == RETAINED_STATUS]


def validate_u_clean(segments: Sequence[dict], context: UCleanContext) -> None:
    cfg = context.config
    seg_cfg = cfg.segmentation
    retained = retained_segments(segments)

    if cfg.run_full_pipeline:
        if not cfg.segmentation_config_frozen:
            raise RuntimeError("full run requires SEGMENTATION_CONFIG_FROZEN")
        if not cfg.overlap_config_frozen:
            raise RuntimeError("full run requires OVERLAP_CONFIG_FROZEN")
        if not context.eligible_sources:
            raise RuntimeError("full run requires at least one eligible source")
        if not retained:
            raise RuntimeError("full run produced an empty U_clean")
        if not context.completion.references_resolved:
            raise RuntimeError("full run requires resolved protected references for all 3 splits")
        if not context.completion.nb10_locked:
            raise RuntimeError("full run requires a locked NB10 upstream")
        if not context.completion.segment_fingerprint_coverage_complete:
            raise RuntimeError("full run requires complete segment fingerprint coverage")
        if not context.completion.reference_fingerprint_coverage_complete:
            raise RuntimeError("full run requires a fingerprint for every protected reference")
        if not context.completion.u_u_protection_complete:
            raise RuntimeError("full run requires completed U-U protection")
        if not context.completion.protected_overlap_check_complete:
            raise RuntimeError("full run requires a completed protected-overlap check")
        if not context.completion.segments_written_verified:
            raise RuntimeError("full run requires verified segment files")
        verify_retained_segment_files(segments, context)

    seen_uid = set()
    seen_pcm = set()
    per_source: Dict[str, List[dict]] = {}
    for row in retained:
        uid = row["segment_uid"]
        if uid in seen_uid:
            raise RuntimeError(f"duplicate segment_uid retained: {uid}")
        seen_uid.add(uid)
        pcm_sha = row["segment_pcm16_sha256"]
        if not pcm_sha:
            raise RuntimeError(f"retained segment without pcm sha: {uid}")
        if pcm_sha in seen_pcm:
            raise RuntimeError(f"retained exact-duplicate PCM: {uid}")
        seen_pcm.add(pcm_sha)

        duration = float(row["duration_seconds"])
        if duration + 1e-9 < seg_cfg.min_segment_seconds or duration > seg_cfg.max_segment_seconds + 1e-9:
            raise RuntimeError(f"segment {uid} duration {duration} out of range")
        expected = (int(row["end_sample"]) - int(row["start_sample"])) / float(seg_cfg.sample_rate)
        if abs(expected - duration) > 1e-6:
            raise RuntimeError(f"segment {uid} duration arithmetic inconsistent")
        if float(row["vad_speech_fraction"]) + 1e-9 < seg_cfg.min_speech_fraction:
            raise RuntimeError(f"segment {uid} speech fraction below minimum")

        if row.get("exact_duplicate") or row.get("perceptual_duplicate"):
            raise RuntimeError(f"retained duplicate flag set: {uid}")
        if row.get("overlap_g_train") or row.get("overlap_g_validation") or row.get("overlap_frozen_test"):
            raise RuntimeError(f"retained protected-overlap: {uid}")

        per_source.setdefault(row["source_id"], []).append(row)

    for source_id, rows in per_source.items():
        ordered = sorted(rows, key=lambda r: int(r["start_sample"]))
        prev_end = -1
        for row in ordered:
            start = int(row["start_sample"])
            end = int(row["end_sample"])
            if end <= start:
                raise RuntimeError(f"invalid interval for {row['segment_uid']}")
            if start < prev_end:
                raise RuntimeError(f"temporal overlap within source {source_id}")
            prev_end = end

    assert_portable_artifact_paths(retained, ["source_wav_local_path", "segment_local_path"])


# --------------------------------------------------------------------------- #
# Manifest / QA / summary / contract                                           #
# --------------------------------------------------------------------------- #
def u_clean_rows(segments: Sequence[dict]) -> List[dict]:
    """RETAINED-only manifest rows (item 6)."""
    return [
        {col: row.get(col, "") for col in U_CLEAN_MANIFEST_COLUMNS}
        for row in retained_segments(segments)
    ]


def exclusion_rows(segments: Sequence[dict]) -> List[dict]:
    cols = ["segment_uid", "source_id", "start_sample", "end_sample", "duration_seconds", "u_clean_status", "exclusion_reason", "canonical_segment_uid"]
    return [
        {c: row.get(c, "") for c in cols}
        for row in segments
        if row.get("u_clean_status") != RETAINED_STATUS
    ]


def segment_qa_rows(segments: Sequence[dict]) -> List[dict]:
    cols = ["segment_uid", "source_id", "start_sample", "end_sample", "duration_seconds", "vad_speech_fraction", "peak_dbfs", "rms_dbfs", "silence_fraction_energy", "u_clean_status"]
    return [{c: row.get(c, "") for c in cols} for row in segments]


def overlap_audit_rows(context: UCleanContext) -> List[dict]:
    return [
        {
            "candidate_uid": e.candidate_uid,
            "reference_uid": e.reference_uid,
            "reference_split": e.reference_split,
            "matched_duration_seconds": e.matched_duration_seconds,
            "alignment_offset": e.alignment_offset,
            "similarity": e.similarity,
            "match_type": e.match_type,
        }
        for e in context.match_evidence
    ]


def _status_counts(segments: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in segments:
        counts[row["u_clean_status"]] = counts.get(row["u_clean_status"], 0) + 1
    return counts


def build_summary(segments: Sequence[dict], context: UCleanContext) -> dict:
    cfg = context.config
    retained = retained_segments(segments)
    counts = _status_counts(segments)
    total_retained_seconds = sum(float(r["duration_seconds"]) for r in retained)

    per_source: Dict[str, dict] = {}
    for row in retained:
        entry = per_source.setdefault(row["source_id"], {"n_segments": 0, "seconds": 0.0})
        entry["n_segments"] += 1
        entry["seconds"] = round(entry["seconds"] + float(row["duration_seconds"]), 3)

    invariants_ok = True
    try:
        validate_u_clean(segments, context)
    except Exception:
        invariants_ok = False

    completion = context.completion
    nonempty = bool(context.eligible_sources) and bool(segments) and bool(retained) and total_retained_seconds > 0
    gates = {
        "run_full_pipeline": cfg.run_full_pipeline,
        "segmentation_config_frozen": cfg.segmentation_config_frozen,
        "overlap_config_frozen": cfg.overlap_config_frozen,
        "references_resolved": completion.references_resolved,
        "reference_fingerprint_coverage_complete": completion.reference_fingerprint_coverage_complete,
        "segment_fingerprint_coverage_complete": completion.segment_fingerprint_coverage_complete,
        "u_u_protection_complete": completion.u_u_protection_complete,
        "protected_overlap_check_complete": completion.protected_overlap_check_complete,
        "nb10_locked": completion.nb10_locked,
        "fingerprint_coverage_complete": completion.fingerprint_coverage_complete,
        "protection_complete": completion.protection_complete,
        "segments_written_verified": completion.segments_written_verified,
        "artifacts_reverified": completion.artifacts_reverified,
        "nonempty_u_clean": nonempty,
        "u_clean_invariants": invariants_ok,
    }
    can_succeed = all(gates.values())
    status = SUCCESS_STATUS if can_succeed else FAIL_STATUS
    durations = sorted(float(row["duration_seconds"]) for row in retained)

    def _percentile(q: float) -> float:
        if not durations:
            return 0.0
        index = min(len(durations) - 1, max(0, int(round(q * (len(durations) - 1)))))
        return round(durations[index], 6)

    generated_seconds = sum(float(row["duration_seconds"]) for row in segments)
    source_seconds = 0.0
    for source in context.eligible_sources:
        try:
            source_seconds += float(source.get("duration_seconds") or 0.0)
        except (TypeError, ValueError):
            source_seconds += 0.0
    qa_rejected = sum(1 for row in segments if row.get("u_clean_status") in QA_EXCLUSIONS)

    return {
        "status": status,
        "schema_version": U_CLEAN_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_full_pipeline": cfg.run_full_pipeline,
        "segmentation_config_frozen": cfg.segmentation_config_frozen,
        "overlap_config_frozen": cfg.overlap_config_frozen,
        "segmentation_contract_sha256": cfg.segmentation_contract_sha(),
        "overlap_contract_sha256": cfg.overlap_contract_sha(),
        "nb10_status": context.nb10_status,
        "nb10_locks": dict(context.nb10_locks),
        "reference_summary": dict(context.reference_summary),
        "reference_provenance": dict(context.reference_provenance),
        "protected_reference_fingerprints": dict(context.reference_summary),
        "completion": completion.as_dict(),
        "n_source_input": len(context.eligible_sources),
        "source_input_hours": round(source_seconds / 3600.0, 6),
        "n_eligible_sources": len(context.eligible_sources),
        "n_segments_generated": len(segments),
        "generated_segment_hours": round(generated_seconds / 3600.0, 6),
        "n_segments_total": len(segments),
        "n_segment_qa_pass": len(segments) - qa_rejected,
        "n_segment_qa_rejected": qa_rejected,
        "n_segments_retained": len(retained),
        "n_u_clean": len(retained),
        "u_clean_hours": round(total_retained_seconds / 3600.0, 6),
        "status_counts": counts,
        "n_exact_duplicates": counts.get(EXCLUDED_EXACT_DUPLICATE, 0),
        "n_excluded_exact_duplicate": counts.get(EXCLUDED_EXACT_DUPLICATE, 0),
        "n_perceptual_duplicates_u": counts.get(EXCLUDED_PERCEPTUAL_DUPLICATE, 0),
        "n_excluded_perceptual_duplicate": counts.get(EXCLUDED_PERCEPTUAL_DUPLICATE, 0),
        "n_excluded_protected_overlap": counts.get(EXCLUDED_PROTECTED_OVERLAP, 0),
        "n_overlap_g_train": sum(1 for row in segments if row.get("overlap_g_train")),
        "n_overlap_g_validation": sum(1 for row in segments if row.get("overlap_g_validation")),
        "n_overlap_frozen_test": sum(1 for row in segments if row.get("overlap_frozen_test")),
        "n_match_evidence": len(context.match_evidence),
        "min_segment_seconds": round(durations[0], 6) if durations else 0.0,
        "median_segment_seconds": _percentile(0.5),
        "p90_segment_seconds": _percentile(0.9),
        "max_segment_seconds": round(durations[-1], 6) if durations else 0.0,
        "retained_audio_seconds": round(total_retained_seconds, 3),
        "retained_audio_hours": round(total_retained_seconds / 3600.0, 6),
        "source_count_u_clean": len(per_source),
        "per_source": per_source,
        "per_source_segment_counts": {source_id: entry["n_segments"] for source_id, entry in per_source.items()},
        "per_source_hours": {source_id: round(entry["seconds"] / 3600.0, 6) for source_id, entry in per_source.items()},
        "gates": gates,
    }


def build_contract(context: UCleanContext) -> dict:
    cfg = context.config
    return {
        "schema_version": U_CLEAN_SCHEMA_VERSION,
        "segmentation_contract": cfg.segmentation.contract_payload(),
        "segmentation_contract_sha256": cfg.segmentation_contract_sha(),
        "segmentation_config_frozen": cfg.segmentation_config_frozen,
        "overlap_contract": cfg.overlap.contract_payload(),
        "overlap_contract_sha256": cfg.overlap_contract_sha(),
        "overlap_config_frozen": cfg.overlap_config_frozen,
        "protected_splits": list(PROTECTED_SPLITS),
        "protected_reference_policy": "audio_identity_only",
        "reference_provenance": dict(context.reference_provenance),
        "reference_summary": dict(context.reference_summary),
        "nb10_locks": dict(context.nb10_locks),
        "forbidden_reference_tokens": list(FORBIDDEN_REFERENCE_TOKENS),
        "consumes": "nb10 ELIGIBLE_SOURCE + technical PASS only",
        "manifest_columns": list(U_CLEAN_MANIFEST_COLUMNS),
        "frozen_test_usage": {
            "text_accessed": False,
            "model_inference": False,
            "metric_evaluation": False,
            "audio_identity_only": True,
        },
    }


def compatibility_key(context: UCleanContext) -> dict:
    """Item 15: what a resumed run must match before reusing cached work."""
    cfg = context.config
    return {
        "schema_version": U_CLEAN_SCHEMA_VERSION,
        "segmentation_contract_sha256": cfg.segmentation_contract_sha(),
        "overlap_contract_sha256": cfg.overlap_contract_sha(),
        "nb10_locks": dict(context.nb10_locks),
        "reference_set_sha256": context.reference_summary.get("reference_set_sha256", ""),
    }


def assert_checkpoint_compatible(cached_key: dict, current_key: dict) -> None:
    if cached_key != current_key:
        raise RuntimeError(
            "checkpoint is stale/incompatible; upstream hashes or contracts changed "
            "(fail closed). Delete the checkpoint and re-run."
        )


# --------------------------------------------------------------------------- #
# Full-run artifact writing (item 5 & 7)                                       #
# --------------------------------------------------------------------------- #
def verify_retained_segment_files(segments: Sequence[dict], context: UCleanContext) -> None:
    """Reopen every retained WAV and check container, PCM, and duration."""
    cfg = context.config
    for row in retained_segments(segments):
        rel = str(row.get("segment_local_path") or "")
        if not rel:
            raise RuntimeError(f"retained segment has no file: {row.get('segment_uid')}")
        path = resolve_project_path(rel, cfg.project_root)
        if not path.is_file():
            raise RuntimeError(f"retained segment file missing: {rel}")
        info = read_wav_pcm16(path)
        if info["wav_sha256"] != str(row.get("segment_wav_sha256") or ""):
            raise RuntimeError(f"segment wav sha mismatch: {row.get('segment_uid')}")
        if info["pcm16_sha256"] != str(row.get("segment_pcm16_sha256") or ""):
            raise RuntimeError(f"segment pcm sha mismatch: {row.get('segment_uid')}")
        expected_samples = int(row["end_sample"]) - int(row["start_sample"])
        if info["n_samples"] != expected_samples:
            raise RuntimeError(f"segment sample count mismatch: {row.get('segment_uid')}")
        duration = info["n_samples"] / float(info["sample_rate"])
        if abs(duration - float(row["duration_seconds"])) > 1e-6:
            raise RuntimeError(f"segment file duration mismatch: {row.get('segment_uid')}")
    context.completion.artifacts_reverified = True


def write_and_verify_segments(segments: Sequence[dict], context: UCleanContext) -> None:
    """Write retained segment WAVs, or reuse a cached file whose hashes still match.

    A single new write failure excludes that row. A cached path that is missing
    or whose hash does not match fails the run closed. Completion is true when
    every row that is still retained has a verified file.
    """
    cfg = context.config
    seg_cfg = cfg.segmentation
    seg_root = Path(cfg.out_dir) / "segments"
    try:
        seg_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        context.completion.segments_written_verified = False
        raise
    source_cache: Dict[str, np.ndarray] = {}
    for row in list(segments):
        if row["u_clean_status"] != RETAINED_STATUS:
            continue
        existing = str(row.get("segment_local_path") or "")
        if existing:
            reused = False
            try:
                info = read_wav_pcm16(resolve_project_path(existing, cfg.project_root))
                pcm_ok = info["pcm16_sha256"] == row.get("segment_pcm16_sha256")
                wav_ok = (not row.get("segment_wav_sha256")) or info["wav_sha256"] == row.get("segment_wav_sha256")
                reused = pcm_ok and wav_ok
            except Exception:
                reused = False
            if reused:
                continue
            row["segment_local_path"] = ""
            row["segment_wav_sha256"] = ""
        sid = str(row["source_id"])
        try:
            if sid not in source_cache:
                source_cache[sid] = read_wav_pcm16(
                    resolve_project_path(row["source_wav_local_path"], cfg.project_root)
                )["pcm"]
        except Exception as exc:
            raise RuntimeError(f"canonical source invalid for {sid}") from exc
        try:
            seg_pcm = slice_pcm(source_cache[sid], int(row["start_sample"]), int(row["end_sample"]))
            abs_path = seg_root / sid / (row["segment_uid"] + ".wav")
            wav_sha = write_wav_pcm16_atomic(abs_path, seg_pcm, seg_cfg.sample_rate)
            reopened = read_wav_pcm16(abs_path)
            if reopened["pcm16_sha256"] != row["segment_pcm16_sha256"]:
                raise RuntimeError("reopen pcm16 mismatch")
            row["segment_local_path"] = to_project_relative(abs_path, cfg.project_root)
            row["segment_wav_sha256"] = wav_sha
        except Exception:
            _exclude(row, EXCLUDED_SEGMENT_WRITE_FAILED)
    unverified = [
        row for row in retained_segments(segments)
        if not row.get("segment_local_path") or not row.get("segment_wav_sha256")
    ]
    context.completion.segments_written_verified = not unverified


def require_full_run_dependencies() -> None:
    """Fail before any segmentation or fingerprinting if a full-run dependency is missing."""
    from src.rq2_audio_fingerprint import has_fpcalc
    from src.rq2_segmentation import has_webrtcvad

    missing = []
    if not has_webrtcvad():
        missing.append("webrtcvad")
    if not has_fpcalc():
        missing.append("fpcalc")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        missing.append("pyarrow")
    if missing:
        raise RuntimeError(
            "full NB11 requires: webrtcvad, fpcalc, pyarrow; missing: " + ", ".join(missing)
        )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def checkpoint_root(context: UCleanContext) -> Path:
    return Path(context.config.out_dir) / "checkpoint"


def _empty_checkpoint_state(context: UCleanContext) -> dict:
    return {
        "compatibility_key": compatibility_key(context),
        "sources": {},
        "segment_shards": [],
        "segment_fingerprint_shards": [],
        "protected_fingerprint_shards": [],
        "next_segment_shard_id": 0,
        "next_segment_fingerprint_shard_id": 0,
        "next_protected_fingerprint_shard_id": 0,
    }


def _read_checkpoint_state(context: UCleanContext) -> dict:
    path = checkpoint_root(context) / "state.json"
    if not path.is_file():
        return _empty_checkpoint_state(context)
    state = json.loads(path.read_text(encoding="utf-8"))
    assert_checkpoint_compatible(state.get("compatibility_key") or {}, compatibility_key(context))
    state.setdefault("sources", {})
    state.setdefault("segment_shards", [])
    state.setdefault("segment_fingerprint_shards", [])
    state.setdefault("protected_fingerprint_shards", [])
    state.setdefault("next_segment_shard_id", 0)
    state.setdefault("next_segment_fingerprint_shard_id", 0)
    state.setdefault("next_protected_fingerprint_shard_id", 0)
    return state


def _write_checkpoint_state(context: UCleanContext, state: dict) -> None:
    root = checkpoint_root(context)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(root / "state.json", json.dumps(state, ensure_ascii=False, indent=2))


def _shard_uids(shards: Sequence[dict]) -> set:
    found = set()
    for shard in shards:
        for uid in shard.get("uids") or []:
            found.add(str(uid))
    return found


def _allocate_shard_name(directory: Path, suffix: str, state: dict, counter_key: str) -> str:
    """Next shard id is max(files on disk, persisted counter) + 1. Never reuse a hole."""
    directory.mkdir(parents=True, exist_ok=True)
    highest = int(state.get(counter_key) or 0)
    for path in directory.glob("part-*." + suffix):
        token = path.name[len("part-"):-(len(suffix) + 1)]
        if token.isdigit():
            highest = max(highest, int(token))
    number = highest + 1
    name = "part-%06d.%s" % (number, suffix)
    if (directory / name).exists():
        raise RuntimeError(f"shard destination already exists: {name}")
    state[counter_key] = number
    return name


def _write_parquet_shard(directory: Path, rows: Sequence[dict], name: str) -> str:
    import pandas as pd

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("full NB11 requires: webrtcvad, fpcalc, pyarrow; missing: pyarrow") from exc
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / name
    if dest.exists():
        raise RuntimeError(f"shard destination already exists: {name}")
    tmp = directory / (name + ".tmp")
    pd.DataFrame(list(rows)).to_parquet(tmp, index=False)
    os.replace(tmp, dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def _forget_shard_uids(shards: Sequence[dict], uids: set) -> List[dict]:
    kept = []
    for shard in shards:
        remain = [uid for uid in (shard.get("uids") or []) if str(uid) not in uids]
        if not remain:
            continue
        updated = dict(shard)
        updated["uids"] = remain
        kept.append(updated)
    return kept


def mark_source_segmentation_complete(context: UCleanContext, source_id: str, segment_rows: Sequence[dict]) -> None:
    """Record a finished source even when it produced zero segments."""
    state = _read_checkpoint_state(context)
    sid = str(source_id)
    if state["sources"].get(sid, {}).get("segmentation_status") == "COMPLETE":
        return
    rows = [_json_ready(row) for row in segment_rows if str(row.get("source_id") or "") == sid]
    if rows:
        directory = checkpoint_root(context) / "segments"
        name = _allocate_shard_name(directory, "jsonl", state, "next_segment_shard_id")
        dest = directory / name
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        atomic_write_text(dest, payload)
        state["segment_shards"].append({
            "name": name,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "source_id": sid,
            "n_segments": len(rows),
        })
    state["sources"][sid] = {"segmentation_status": "COMPLETE", "n_segments": len(rows)}
    _write_checkpoint_state(context, state)


def append_fingerprint_shard(context: UCleanContext, kind: str, rows: Sequence[dict]) -> Optional[str]:
    """Append one fingerprint shard. Existing shard files are not rewritten."""
    if kind not in ("segment_fingerprints", "protected_fingerprints"):
        raise RuntimeError(f"unknown fingerprint shard kind: {kind}")
    state = _read_checkpoint_state(context)
    key = "segment_fingerprint_shards" if kind == "segment_fingerprints" else "protected_fingerprint_shards"
    uid_field = "segment_uid" if kind == "segment_fingerprints" else "reference_uid"
    known = _shard_uids(state[key])
    fresh = []
    for row in rows:
        uid = str(row.get(uid_field) or row.get("uid") or "")
        fp = row.get("fingerprint")
        if not uid or uid in known or not is_valid_fingerprint(fp):
            continue
        stored = {
            uid_field: uid,
            "fingerprint": [int(item) for item in fp],
            "fingerprint_sha256": fingerprint_sha256(fp),
        }
        if kind == "segment_fingerprints":
            stored["audio_sha256"] = str(row.get("audio_sha256") or "")
        else:
            stored["split"] = str(row.get("split") or "")
            stored["source_audio_sha256"] = str(row.get("source_sha256") or row.get("source_audio_sha256") or "")
        fresh.append(stored)
        known.add(uid)
    if not fresh:
        return None
    directory = checkpoint_root(context) / kind
    counter_key = "next_segment_fingerprint_shard_id" if kind == "segment_fingerprints" else "next_protected_fingerprint_shard_id"
    name = _allocate_shard_name(directory, "parquet", state, counter_key)
    digest = _write_parquet_shard(directory, fresh, name)
    state[key].append({
        "name": name,
        "sha256": digest,
        "uids": [row[uid_field] for row in fresh],
    })
    _write_checkpoint_state(context, state)
    return name


def save_incremental_checkpoint(
    context: UCleanContext,
    *,
    completed_source_ids: Optional[Sequence[str]] = None,
    segments: Optional[Sequence[dict]] = None,
    segment_fingerprints_by_uid: Optional[Dict[str, Sequence[int]]] = None,
    reference_fingerprints: Optional[Sequence[dict]] = None,
) -> Path:
    """Append newly finished pieces. Completed shard files stay untouched."""
    grouped: Dict[str, List[dict]] = {}
    for row in segments or []:
        grouped.setdefault(str(row.get("source_id") or ""), []).append(row)
    source_ids = list(completed_source_ids) if completed_source_ids is not None else list(grouped)
    for sid in source_ids:
        if not sid:
            continue
        mark_source_segmentation_complete(context, sid, grouped.get(sid, []))
    if segment_fingerprints_by_uid:
        append_fingerprint_shard(
            context,
            "segment_fingerprints",
            [{"segment_uid": uid, "fingerprint": fp} for uid, fp in segment_fingerprints_by_uid.items()],
        )
    if reference_fingerprints:
        append_fingerprint_shard(context, "protected_fingerprints", list(reference_fingerprints))
    return checkpoint_root(context) / "state.json"


def _publish_consistent_generation(staging: Path, out_dir: Path, relative_files: Sequence[str]) -> str:
    """Snapshot a complete generation before the flat publish. CURRENT moves only after COMPLETE.json."""
    gen_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    generation = out_dir / "generations" / gen_id
    generation.mkdir(parents=True, exist_ok=False)
    for rel in relative_files:
        src = staging / rel
        dest = generation / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    atomic_write_text(
        generation / "COMPLETE.json",
        json.dumps({"generation": gen_id, "files": list(relative_files)}, ensure_ascii=False, indent=2),
    )
    atomic_write_text(out_dir / "CURRENT", gen_id + "\n")
    return gen_id


def _commit_staged_artifacts(staging: Path, out_dir: Path, relative_files: Sequence[str]) -> None:
    """Move staged files into place. summary.json is last. Failure restores the previous files."""
    _publish_consistent_generation(staging, out_dir, relative_files)
    backup = out_dir / ".publish_backup"
    if backup.exists():
        shutil.rmtree(backup)
    replaced: List[str] = []
    summary_name = "summary.json"
    ordered = [rel for rel in relative_files if rel != summary_name]
    if summary_name in relative_files:
        ordered.append(summary_name)
    try:
        for rel in ordered:
            src = staging / rel
            dest = out_dir / rel
            if not src.is_file() or src.stat().st_size <= 0:
                raise RuntimeError(f"staged artifact missing: {rel}")
            if dest.is_file():
                backed = backup / rel
                backed.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, backed)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dest)
            replaced.append(rel)
    except Exception:
        for rel in reversed(replaced):
            dest = out_dir / rel
            backed = backup / rel
            if backed.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backed, dest)
            elif dest.is_file():
                dest.unlink()
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def write_u_clean_artifacts(
    segments: Sequence[dict],
    context: UCleanContext,
    *,
    segment_fingerprints_by_uid: Optional[Dict[str, Sequence[int]]] = None,
    reference_fingerprints: Optional[Sequence[dict]] = None,
) -> dict:
    """Stage every artifact, then publish. summary.json is replaced only after the rest verify."""
    import pandas as pd

    cfg = context.config
    if not cfg.run_full_pipeline:
        return {"written": False, "reason": "RUN_FULL_PIPELINE is False"}

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for sub in ("fingerprints", "qa", "manifests"):
            (staging / sub).mkdir(parents=True, exist_ok=True)
        manifest = u_clean_rows(segments)
        assert_portable_artifact_paths(manifest, ["source_wav_local_path", "segment_local_path"])
        pd.DataFrame(manifest, columns=U_CLEAN_MANIFEST_COLUMNS).to_csv(
            staging / "u_clean_manifest.csv", index=False,
        )
        atomic_write_text(
            staging / "u_clean_manifest.jsonl",
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest),
        )
        pd.DataFrame(exclusion_rows(segments)).to_csv(staging / "qa" / "exclusions.csv", index=False)
        pd.DataFrame(segment_qa_rows(segments)).to_csv(staging / "qa" / "segment_qa.csv", index=False)
        pd.DataFrame(overlap_audit_rows(context)).to_csv(staging / "qa" / "overlap_audit.csv", index=False)
        source_rows = [
            {
                "source_id": s.get("source_id", ""),
                "wav_local_path": s.get("wav_local_path", ""),
                "wav_sha256": s.get("wav_sha256", ""),
                "pcm16_sha256": s.get("pcm16_sha256", ""),
                "duration_seconds": s.get("duration_seconds", ""),
            }
            for s in context.eligible_sources
        ]
        pd.DataFrame(source_rows).to_csv(staging / "manifests" / "source_segments.csv", index=False)
        summary = build_summary(segments, context)
        atomic_write_text(staging / "contract.json", json.dumps(build_contract(context), ensure_ascii=False, indent=2))
        fingerprint_meta = {}
        if segment_fingerprints_by_uid:
            pcm_by_uid = {
                str(row.get("segment_uid") or ""): str(row.get("segment_pcm16_sha256") or "")
                for row in segments
            }
            append_fingerprint_shard(
                context,
                "segment_fingerprints",
                [
                    {
                        "segment_uid": uid,
                        "fingerprint": fp,
                        "audio_sha256": pcm_by_uid.get(str(uid), ""),
                    }
                    for uid, fp in segment_fingerprints_by_uid.items()
                ],
            )
        if reference_fingerprints:
            append_fingerprint_shard(context, "protected_fingerprints", list(reference_fingerprints))
        if segment_fingerprints_by_uid is not None or reference_fingerprints is not None:
            fingerprint_meta = write_fingerprint_tables(
                segments,
                segment_fingerprints_by_uid or {},
                reference_fingerprints or [],
                context,
                out_dir=staging,
            )
        if reference_fingerprints is None:
            shard_dir = checkpoint_root(context) / "protected_fingerprints"
            if shard_dir.is_dir() and any(shard_dir.glob("part-*.parquet")):
                protected_sha = write_protected_parquet_from_shards(
                    context, staging / "fingerprints" / "protected_reference.parquet",
                )
                fingerprint_meta["protected_reference_parquet_sha256"] = protected_sha
                fingerprint_meta.setdefault("u_segments_parquet_sha256", "")
        atomic_write_text(staging / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        relative = [
            "u_clean_manifest.csv",
            "u_clean_manifest.jsonl",
            "qa/exclusions.csv",
            "qa/segment_qa.csv",
            "qa/overlap_audit.csv",
            "manifests/source_segments.csv",
            "contract.json",
            "summary.json",
        ]
        if fingerprint_meta:
            relative[7:7] = [
                "fingerprints/u_segments.parquet",
                "fingerprints/protected_reference.parquet",
            ]
        _commit_staged_artifacts(staging, out_dir, relative)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {"written": True, "out_dir": str(out_dir), "status": summary["status"]}


def write_protected_parquet_from_shards(context: UCleanContext, dest_path) -> str:
    """Compact verified protected shards into one parquet, one shard at a time."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    writer = None
    contract_sha = context.config.overlap_contract_sha()
    try:
        for batch in iter_protected_fingerprint_shards(context):
            rows = [
                {
                    "reference_uid": row["uid"],
                    "split": row["split"],
                    "source_audio_sha256": row["source_sha256"],
                    "fingerprint": [int(item) for item in row["fingerprint"]],
                    "fingerprint_sha256": fingerprint_sha256(row["fingerprint"]),
                    "fingerprint_contract_sha256": contract_sha,
                }
                for row in batch
            ]
            if not rows:
                continue
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
        if writer is None:
            empty = pa.table({
                "reference_uid": pa.array([], type=pa.string()),
                "split": pa.array([], type=pa.string()),
                "source_audio_sha256": pa.array([], type=pa.string()),
                "fingerprint": pa.array([], type=pa.list_(pa.int64())),
                "fingerprint_sha256": pa.array([], type=pa.string()),
                "fingerprint_contract_sha256": pa.array([], type=pa.string()),
            })
            pq.write_table(empty, tmp)
        else:
            writer.close()
            writer = None
        os.replace(tmp, dest)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists() and not dest.exists():
            tmp.unlink()
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def write_fingerprint_tables(
    segments: Sequence[dict],
    segment_fingerprints_by_uid: Dict[str, Sequence[int]],
    reference_fingerprints: Sequence[dict],
    context: UCleanContext,
    out_dir=None,
) -> dict:
    """Persist segment and protected-reference fingerprints for audit and resume."""
    import pandas as pd

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("full NB11 requires: webrtcvad, fpcalc, pyarrow; missing: pyarrow") from exc
    root = Path(out_dir) if out_dir is not None else Path(context.config.out_dir)
    fp_dir = root / "fingerprints"
    fp_dir.mkdir(parents=True, exist_ok=True)
    contract_sha = context.config.overlap_contract_sha()
    segment_rows = []
    for row in segments:
        fp = segment_fingerprints_by_uid.get(row["segment_uid"])
        if not is_valid_fingerprint(fp):
            continue
        segment_rows.append({
            "segment_uid": row["segment_uid"],
            "audio_sha256": row.get("segment_pcm16_sha256") or "",
            "fingerprint": [int(item) for item in fp],
            "fingerprint_sha256": fingerprint_sha256(fp),
            "fingerprint_contract_sha256": contract_sha,
        })
    reference_rows = []
    for ref in reference_fingerprints:
        fp = ref.get("fingerprint")
        if not is_valid_fingerprint(fp):
            continue
        reference_rows.append({
            "reference_uid": str(ref.get("uid") or ""),
            "split": str(ref.get("split") or ""),
            "source_audio_sha256": str(ref.get("source_sha256") or ""),
            "fingerprint": [int(item) for item in fp],
            "fingerprint_sha256": fingerprint_sha256(fp),
            "fingerprint_contract_sha256": contract_sha,
        })
    segment_path = fp_dir / "u_segments.parquet"
    reference_path = fp_dir / "protected_reference.parquet"
    pd.DataFrame(segment_rows).to_parquet(segment_path, index=False)
    pd.DataFrame(reference_rows).to_parquet(reference_path, index=False)
    return {
        "u_segments_parquet_sha256": hashlib.sha256(segment_path.read_bytes()).hexdigest(),
        "protected_reference_parquet_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }


def _verified_fingerprint_items(items: Sequence[dict], uid_field: str) -> List[dict]:
    kept = []
    for item in items:
        fp = item.get("fingerprint")
        if not is_valid_fingerprint(fp):
            continue
        if fingerprint_sha256(fp) != str(item.get("fingerprint_sha256") or ""):
            continue
        kept.append(item)
    return kept


def _load_fingerprint_shard(path: Path, kind: str) -> List[dict]:
    import pandas as pd

    frame = pd.read_parquet(path)
    rows = []
    uid_field = "segment_uid" if kind == "segment_fingerprints" else "reference_uid"
    for _, row in frame.iterrows():
        fp = [int(item) for item in list(row["fingerprint"])]
        if fingerprint_sha256(fp) != str(row["fingerprint_sha256"]):
            continue
        item = {"fingerprint": fp, "uid": str(row[uid_field])}
        if kind == "segment_fingerprints":
            item["segment_uid"] = item["uid"]
            item["audio_sha256"] = str(row["audio_sha256"] or "")
        else:
            item["split"] = str(row["split"])
            item["source_sha256"] = str(row["source_audio_sha256"])
        rows.append(item)
    return rows


def load_resumable_checkpoint(context: UCleanContext, path=None, load_protected_fingerprints: bool = True) -> dict:
    """Load sharded resume state. A bad shard is dropped; a stale contract fails closed."""
    state = _read_checkpoint_state(context)
    root = checkpoint_root(context)
    loaded = {
        "completed_source_ids": [
            sid for sid, info in state["sources"].items()
            if info.get("segmentation_status") == "COMPLETE"
        ],
        "source_status": dict(state["sources"]),
        "segments": [],
        "segment_fingerprints_by_uid": {},
        "segment_fingerprint_audio_sha256": {},
        "reference_fingerprints": [],
        "discarded_segment_shards": [],
        "discarded_reference_shards": [],
        "discarded_segment_fingerprints": 0,
        "discarded_reference_fingerprints": 0,
    }
    kept_segment_shards = []
    for shard in state["segment_shards"]:
        file_path = root / "segments" / shard["name"]
        try:
            if hashlib.sha256(file_path.read_bytes()).hexdigest() != shard["sha256"]:
                raise RuntimeError("hash")
            for line in file_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    loaded["segments"].append(json.loads(line))
            kept_segment_shards.append(shard)
        except Exception:
            loaded["discarded_segment_shards"].append(shard["name"])
            sid = str(shard.get("source_id") or "")
            if sid:
                loaded["segments"] = [
                    row for row in loaded["segments"] if str(row.get("source_id") or "") != sid
                ]
                state["sources"].pop(sid, None)
    kept_fp = []
    for kind, key, discard_names, discard_count in (
        ("segment_fingerprints", "segment_fingerprint_shards", "discarded_segment_shards", "discarded_segment_fingerprints"),
        ("protected_fingerprints", "protected_fingerprint_shards", "discarded_reference_shards", "discarded_reference_fingerprints"),
    ):
        kept = kept_fp if kind == "segment_fingerprints" else None
        bucket = []
        for shard in state[key]:
            file_path = root / kind / shard["name"]
            try:
                if hashlib.sha256(file_path.read_bytes()).hexdigest() != shard["sha256"]:
                    raise RuntimeError("hash")
                rows = _load_fingerprint_shard(file_path, kind)
                if kind == "segment_fingerprints":
                    for row in rows:
                        loaded["segment_fingerprints_by_uid"][row["uid"]] = row["fingerprint"]
                        loaded["segment_fingerprint_audio_sha256"][row["uid"]] = row.get("audio_sha256") or ""
                    kept_fp.append(shard)
                else:
                    expected = {
                        entry.reference_uid: str(entry.source_sha256 or "").strip().lower()
                        for entry in context.protected_entries
                    }
                    matched = []
                    for row in rows:
                        stored = str(row.get("source_sha256") or "").strip().lower()
                        if expected and (expected.get(row["uid"]) != stored or not _is_valid_sha256(stored)):
                            loaded["discarded_reference_fingerprints"] += 1
                            continue
                        matched.append(row)
                        if load_protected_fingerprints:
                            loaded["reference_fingerprints"].append({
                                "uid": row["uid"],
                                "split": row["split"],
                                "source_sha256": row["source_sha256"],
                                "fingerprint": row["fingerprint"],
                            })
                    if matched:
                        updated = dict(shard)
                        updated["uids"] = [row["uid"] for row in matched]
                        bucket.append(updated)
            except Exception:
                loaded[discard_names].append(shard["name"])
                loaded[discard_count] += len(shard.get("uids") or [])
        if kind == "protected_fingerprints":
            state[key] = bucket
    state["segment_shards"] = kept_segment_shards
    state["segment_fingerprint_shards"] = kept_fp
    _drop_unbound_cached_fingerprints(context, loaded, state)
    loaded["completed_source_ids"] = [
        sid for sid, info in state["sources"].items()
        if info.get("segmentation_status") == "COMPLETE"
    ]
    loaded["source_status"] = dict(state["sources"])
    if (
        loaded["discarded_segment_shards"]
        or loaded["discarded_reference_shards"]
        or loaded["discarded_segment_fingerprints"]
        or loaded["discarded_reference_fingerprints"]
    ):
        _write_checkpoint_state(context, state)
    loaded.pop("segment_fingerprint_audio_sha256", None)
    return loaded


def _drop_unbound_cached_fingerprints(context: UCleanContext, loaded: dict, state: dict) -> None:
    """Drop cached fingerprints whose stored audio hash does not match the current audio."""
    pcm_by_uid = {
        str(row.get("segment_uid") or ""): str(row.get("segment_pcm16_sha256") or "").strip().lower()
        for row in loaded["segments"]
    }
    stale_segments = []
    kept_fps = {}
    for uid, fp in loaded["segment_fingerprints_by_uid"].items():
        stored = str(loaded["segment_fingerprint_audio_sha256"].get(uid) or "").strip().lower()
        if not _is_valid_sha256(stored):
            stale_segments.append(str(uid))
            continue
        if str(uid) in pcm_by_uid and stored != pcm_by_uid[str(uid)]:
            stale_segments.append(str(uid))
            continue
        kept_fps[uid] = fp
    loaded["segment_fingerprints_by_uid"] = kept_fps
    loaded["discarded_segment_fingerprints"] += len(stale_segments)
    if stale_segments:
        state["segment_fingerprint_shards"] = _forget_shard_uids(
            state["segment_fingerprint_shards"], set(stale_segments),
        )


def protected_fingerprint_coverage_complete(context: UCleanContext) -> bool:
    """True when every protected entry has one verified shard fingerprint for its split.

    Shards are read one at a time. Only uid and split are retained between shards.
    """
    entries = list(context.protected_entries)
    if not references_cover_splits(entries):
        return False
    seen: Dict[str, str] = {}
    for batch in iter_protected_fingerprint_shards(context):
        for row in batch:
            uid = str(row.get("uid") or "")
            if not uid or uid in seen or not is_valid_fingerprint(row.get("fingerprint")):
                return False
            seen[uid] = str(row.get("split") or "")
    if len(seen) != len(entries):
        return False
    for entry in entries:
        if seen.get(entry.reference_uid) != entry.split:
            return False
    return True


def protected_reference_uids(context: UCleanContext) -> set:
    """UIDs already stored in verified protected shards, without loading fingerprint arrays."""
    state = _read_checkpoint_state(context)
    return _shard_uids(state["protected_fingerprint_shards"])


def iter_protected_fingerprint_shards(context: UCleanContext):
    """Yield one verified protected shard at a time."""
    state = _read_checkpoint_state(context)
    expected = {
        entry.reference_uid: str(entry.source_sha256 or "").strip().lower()
        for entry in context.protected_entries
    }
    root = checkpoint_root(context) / "protected_fingerprints"
    for shard in list(state["protected_fingerprint_shards"]):
        allowed = {str(uid) for uid in (shard.get("uids") or [])}
        file_path = root / shard["name"]
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path.is_file() else ""
        if digest != shard.get("sha256"):
            continue
        rows = _load_fingerprint_shard(file_path, "protected_fingerprints")
        batch = []
        for row in rows:
            if allowed and row["uid"] not in allowed:
                continue
            stored = str(row.get("source_sha256") or "").strip().lower()
            if expected and (expected.get(row["uid"]) != stored or not _is_valid_sha256(stored)):
                continue
            batch.append({
                "uid": row["uid"],
                "split": row["split"],
                "source_sha256": row["source_sha256"],
                "fingerprint": row["fingerprint"],
            })
        if batch:
            yield batch


def scan_nb11_safety(paths: Sequence[object]) -> List[str]:
    """Reject ASR, MT, training, or metric calls in NB11 sources."""
    snippets = (
        "import " + "transformers",
        "from " + "transformers",
        "import " + "whisper",
        "faster_" + "whisper",
        "import " + "sacrebleu",
        "sacrebleu" + ".",
        "model" + ".fit(",
        "model" + ".train(",
        "trainer" + ".train(",
    )
    hits = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            for snippet in snippets:
                if snippet in line:
                    hits.append(f"{path}:{number}:{snippet}")
    if hits:
        raise RuntimeError("NB11 static safety scan failed: " + "; ".join(hits[:8]))
    return hits
