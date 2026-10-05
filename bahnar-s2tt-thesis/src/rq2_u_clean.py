"""RQ2 U_clean: segment QA, dedup, contamination protection, freeze.

Thin, composable helpers that NB11 orchestrates. Hard rules enforced here:

* Only ``ELIGIBLE_SOURCE`` + ``technical_status == PASS`` sources from NB10 are
  consumed; the source contract is validated and WAV hashes are re-verified
  (fail closed).
* Protected references (RQ1 G_train / G_validation / frozen G_test) are matched
  on **audio identity only** — the loader projects a small allow-list of columns
  and refuses transcript / reference / prediction / metric columns.
* The similarity threshold is never tuned on the frozen test. The production
  notebook's frozen ``OverlapConfig`` is authoritative. That threshold was
  calibrated on controlled real VOV4 transformations and negative pairs.
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
import tempfile
import time
import uuid
import wave
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
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
    canonical_json_bytes,
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
EXCLUDED_PERCEPTUAL_INELIGIBLE = "EXCLUDED_PERCEPTUAL_INELIGIBLE"

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


def to_u_clean_relative(path, config) -> str:
    """Store an NB11 segment path relative to ``config.out_dir``."""
    root = Path(config.out_dir).resolve()
    absolute = Path(path).resolve()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("segment path escapes U_clean out_dir: %s" % path) from exc
    text = relative.as_posix()
    if not text or Path(text).is_absolute() or ".." in Path(text).parts:
        raise RuntimeError("segment path escapes U_clean out_dir: %s" % path)
    return text


def resolve_u_clean_path(value, config) -> Path:
    """Open a stored segment path. Absolute paths and traversal fail closed."""
    text = str(value or "").strip()
    if not text:
        raise RuntimeError("segment_local_path is empty")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("segment_local_path must stay inside U_clean out_dir: %s" % text)
    root = Path(config.out_dir).resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("segment_local_path must stay inside U_clean out_dir: %s" % text) from exc
    return resolved


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


def _is_pinned_revision(value: str) -> bool:
    """Hugging Face snapshot SHAs are 40 hex characters, not file hashes."""
    text = str(value or "").strip().lower()
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text) and not text.startswith("refs/")


# RQ1 prepare writes sidecar ``split`` as train/validation. NB11 manifests use
# the protected names. This is a fixed alias, not a directory heuristic.
_AUDIO_INDEX_SPLIT_ALIASES = {
    "g_train": "g_train",
    "train": "g_train",
    "g_validation": "g_validation",
    "validation": "g_validation",
    "frozen_test": "frozen_test",
}


def _parse_required_row_index(value, label: str) -> int:
    """Parse a shard row index. Zero is valid; only a real absence fails."""
    if isinstance(value, bool) or value is None:
        raise RuntimeError(f"{label} is missing shard_row_index")
    if isinstance(value, float):
        if math.isnan(value) or not value.is_integer():
            raise RuntimeError(f"{label} has invalid shard_row_index {value!r}")
        index = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            raise RuntimeError(f"{label} is missing shard_row_index")
        try:
            index = int(text)
        except ValueError:
            raise RuntimeError(f"{label} has invalid shard_row_index {value!r}")
    else:
        try:
            index = int(value)
        except (TypeError, ValueError):
            raise RuntimeError(f"{label} has invalid shard_row_index {value!r}")
    if index < 0:
        raise RuntimeError(f"{label} has invalid shard_row_index {value!r}")
    return index


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


U_CLEAN_RELATIVE_DIR = Path("artifacts") / "rq2" / "u_clean"


def resolve_u_clean_output(
    project_root,
    *,
    durable_root=None,
    env: Optional[dict] = None,
) -> dict:
    """Resolve NB11's artifact directory from the shared durable-runtime root.

    The code checkout and the durable artifact root are different directories
    on RunPod. When that durable tree already holds ``checkpoint/state.json``,
    or when ``BAHNAR_DURABLE_ROOT`` / ``durable_root`` is set, U_clean writes
    there. Otherwise the project-local artifacts directory is kept so offline
    tests do not depend on the RunPod default.
    """
    from src.rq1_runtime_paths import resolve_rq1_runtime_paths

    project = Path(project_root).resolve()
    runtime = resolve_rq1_runtime_paths(
        project_root=project, durable_root=durable_root, env=env,
    )
    durable = Path(runtime.durable_root).resolve()
    project_out = project / U_CLEAN_RELATIVE_DIR
    durable_out = durable / U_CLEAN_RELATIVE_DIR
    durable_checkpoint = durable_out / "checkpoint"
    explicit = durable_root is not None or bool((env if env is not None else os.environ).get("BAHNAR_DURABLE_ROOT"))
    use_durable = explicit or (durable_checkpoint / "state.json").is_file()
    selected = durable_out if use_durable else project_out
    return {
        "project_root": project,
        "durable_root": durable,
        "out_dir": selected,
        "checkpoint_root": selected / "checkpoint",
        "durable_checkpoint_state": durable_checkpoint / "state.json",
    }


def assert_nb11_out_dir_uses_durable_checkpoint(project_root, out_dir, *, durable_root=None, env: Optional[dict] = None) -> dict:
    """Fail closed if a run would ignore an existing durable U_clean checkpoint."""
    layout = resolve_u_clean_output(project_root, durable_root=durable_root, env=env)
    selected = Path(out_dir).resolve()
    required = Path(layout["out_dir"]).resolve()
    durable_state = Path(layout["durable_checkpoint_state"])
    if durable_state.is_file() and selected != required:
        raise RuntimeError(
            "NB11 out_dir ignores the existing durable checkpoint at %s; "
            "resolved out_dir must be %s, got %s"
            % (durable_state, required, selected)
        )
    return layout


def build_config(
    *,
    project_root: Optional[Path] = None,
    run_full_pipeline: bool = False,
    segmentation: Optional[SegmentationConfig] = None,
    overlap: Optional[OverlapConfig] = None,
    protected_reference_resolver: "Optional[ProtectedReferenceResolver]" = None,
    durable_root=None,
    out_dir: Optional[Path] = None,
) -> UCleanConfig:
    root = Path(project_root) if project_root is not None else find_project_root()
    seg = segmentation or SegmentationConfig()
    ov = overlap or OverlapConfig()
    layout = resolve_u_clean_output(root, durable_root=durable_root)
    if out_dir is None:
        out_dir = layout["out_dir"]
    else:
        out_dir = Path(out_dir)
    assert_nb11_out_dir_uses_durable_checkpoint(root, out_dir, durable_root=durable_root)
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
    manifest_audio_locator: str = ""
    parquet_file: str = ""
    shard_key: str = ""
    shard_row_index: int = -1
    dataset_revision: str = ""
    parquet_revision: str = ""
    pcm_pipeline_version: str = ""
    sha256_pcm: str = ""
    sha256_source: str = ""
    n_samples: int = 0
    sample_rate: int = 0
    record_id: str = ""

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

# Derived NB11 index. It is not an RQ1 artifact. sha256_pcm is the canonical
# PCM identity; sha256_source/n_samples/sample_rate may be blank until a
# manifest-only row is reconstructed from pinned Parquet.
DERIVED_PROTECTED_IDENTITY_COLUMNS = (
    "reference_uid",
    "split",
    "manifest_audio_locator",
    "parquet_file",
    "shard_key",
    "shard_row_index",
    "dataset_revision",
    "parquet_revision",
    "pcm_pipeline_version",
    "sha256_pcm",
    "sha256_source",
    "n_samples",
    "sample_rate",
    "record_id",
)

EXPECTED_PROTECTED_MANIFEST_COUNTS = {
    "g_train": 102698,
    "g_validation": 11132,
    "frozen_test": 215,
}

CANONICAL_PROTECTED_MANIFESTS = {
    "g_train": "data/manifests/rq1_train.csv",
    "g_validation": "data/manifests/rq1_validation.csv",
    "frozen_test": "data/manifests/rq1_test.csv",
}

_MANIFEST_IDENTITY_COLUMNS = (
    "record_uid",
    "record_id",
    "audio_path",
    "parquet_file",
    "shard_row_index",
)


def default_protected_manifest_mapping() -> ReferenceColumnMapping:
    """Map the canonical RQ1 manifests. Those files have no per-audio SHA256 column."""
    return ReferenceColumnMapping(
        reference_uid="record_uid",
        audio_locator="audio_path",
        duration_seconds="duration_seconds",
        group_id="group_id",
    )


def _entry_can_reconstruct(entry: ProtectedReferenceEntry) -> bool:
    return bool(str(entry.parquet_file or "").strip()) and int(entry.shard_row_index) >= 0


def require_protected_source_sha256(entries: Sequence[ProtectedReferenceEntry]) -> None:
    """Every protected entry needs a PCM SHA or pinned Parquet coordinates."""
    for entry in entries:
        if _is_valid_sha256(str(entry.source_sha256 or "")) or _entry_can_reconstruct(entry):
            continue
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
        locator = str(item.get("manifest_audio_locator") or item["audio_locator"])
        if locator != entry.audio_locator:
            raise RuntimeError(
                f"protected audio locator mismatch for {entry.split}:{entry.reference_uid}"
            )
        pcm = str(item.get("sha256_pcm") or item["source_sha256"] or "").strip().lower()
        bound.append(replace(
            entry,
            source_sha256=pcm,
            manifest_audio_locator=locator,
            parquet_file=str(item.get("parquet_file") or ""),
            shard_key=str(item.get("shard_key") or ""),
            shard_row_index=int(item.get("shard_row_index") if item.get("shard_row_index") not in (None, "") else -1),
            dataset_revision=str(item.get("dataset_revision") or ""),
            parquet_revision=str(item.get("parquet_revision") or ""),
            pcm_pipeline_version=str(item.get("pcm_pipeline_version") or ""),
            sha256_pcm=pcm if _is_valid_sha256(pcm) else "",
            sha256_source=str(item.get("sha256_source") or "").strip().lower(),
            n_samples=int(item.get("n_samples") or 0),
            sample_rate=int(item.get("sample_rate") or 0),
            record_id=str(item.get("record_id") or ""),
        ))
    require_protected_source_sha256(bound)
    return bound


def canonical_reference_index(project_root) -> Dict[str, str]:
    """Project-relative locators for the three canonical RQ1 manifests."""
    root = Path(project_root)
    index = {}
    for split, relative in CANONICAL_PROTECTED_MANIFESTS.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"canonical RQ1 manifest missing for {split}: {relative}")
        index[split] = relative
    return index


def _shard_key_from_parquet_file(parquet_file: str) -> str:
    text = str(parquet_file or "").strip()
    if text.lower().startswith("http://") or text.lower().startswith("https://"):
        from src.asr_full_pcm import parse_hf_parquet_url
        return parse_hf_parquet_url(text).filename
    return text.lstrip("/")


def _load_manifest_identity_rows(path, split: str) -> List[dict]:
    import pandas as pd

    frame = pd.read_csv(path, usecols=list(_MANIFEST_IDENTITY_COLUMNS))
    assert_no_forbidden_reference_columns(list(frame.columns))
    rows = []
    seen = set()
    for _, row in frame.iterrows():
        uid = str(row["record_uid"] or "").strip()
        if not uid:
            raise RuntimeError(f"{split} manifest row has an empty record_uid")
        if uid in seen:
            raise RuntimeError(f"duplicate protected uid in {split} manifest: {uid}")
        seen.add(uid)
        parquet_file = str(row["parquet_file"] or "").strip()
        try:
            shard_row_index = int(row["shard_row_index"])
        except (TypeError, ValueError):
            shard_row_index = -1
        if not parquet_file or shard_row_index < 0:
            raise RuntimeError(f"missing shard row for {split}:{uid}")
        rows.append({
            "reference_uid": uid,
            "split": split,
            "manifest_audio_locator": str(row["audio_path"] or "").strip(),
            "parquet_file": parquet_file,
            "shard_key": _shard_key_from_parquet_file(parquet_file),
            "shard_row_index": shard_row_index,
            "record_id": str(row["record_id"] or "").strip(),
        })
    return rows


def _load_audio_index_jsonl(directory) -> Dict[str, dict]:
    from src.asr_full_pcm import AUDIO_PCM_PIPELINE_VERSION

    root = Path(directory)
    if not root.is_dir():
        raise RuntimeError(f"RQ1 audio index directory is missing: {root}")
    found: Dict[str, dict] = {}
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            uid = str(payload.get("record_uid") or "").strip()
            if not uid:
                raise RuntimeError(f"audio index row missing record_uid in {path.name}")
            if uid in found:
                raise RuntimeError(f"duplicate audio index uid: {uid}")
            pipeline = str(payload.get("audio_pcm_pipeline_version") or "")
            if pipeline != AUDIO_PCM_PIPELINE_VERSION:
                raise RuntimeError(
                    f"audio index pipeline for {uid} is {pipeline!r}, expected {AUDIO_PCM_PIPELINE_VERSION}"
                )
            digest = str(payload.get("sha256_pcm") or "").strip().lower()
            if not _is_valid_sha256(digest):
                raise RuntimeError(f"audio index sha256_pcm invalid for {uid}")
            source = str(payload.get("sha256_source") or "").strip().lower()
            if not _is_valid_sha256(source):
                raise RuntimeError(f"audio index sha256_source invalid for {uid}")
            shard_key = str(payload.get("shard_key") or "").strip()
            if not shard_key:
                raise RuntimeError(f"audio index shard_key missing for {uid}")
            raw_split = str(payload.get("split") or "").strip()
            protected_split = _AUDIO_INDEX_SPLIT_ALIASES.get(raw_split)
            if protected_split is None:
                raise RuntimeError(f"audio index split invalid for {uid}: {raw_split!r}")
            parquet_revision = str(payload.get("parquet_revision") or "").strip().lower()
            dataset_revision = str(payload.get("dataset_revision") or "").strip().lower()
            if not parquet_revision or not dataset_revision:
                raise RuntimeError(f"audio index revision missing for {uid}")
            if "shard_row_index" not in payload:
                raise RuntimeError(f"audio index shard_row_index missing for {uid}")
            found[uid] = {
                "identity_source": "audio_index",
                "split": protected_split,
                "sha256_pcm": digest,
                "sha256_source": source,
                "n_samples": int(payload.get("n_samples") or 0),
                "sample_rate": int(payload.get("sample_rate") or 0),
                "shard_key": shard_key,
                "shard_row_index": _parse_required_row_index(payload.get("shard_row_index"), f"audio index {uid}"),
                "parquet_revision": parquet_revision,
                "dataset_revision": dataset_revision,
                "pcm_pipeline_version": pipeline,
                "local_cache_relpath": str(payload.get("local_cache_relpath") or "").strip(),
            }
    return found


def _load_frozen_integrity(path) -> Dict[str, dict]:
    import pandas as pd

    frame = pd.read_csv(path)
    assert_no_forbidden_reference_columns(list(frame.columns))
    if "shard_row_index" not in frame.columns:
        raise RuntimeError("frozen audio integrity is missing shard_row_index")
    found = {}
    for _, row in frame.iterrows():
        uid = str(row.get("record_uid") or "").strip()
        if not uid:
            raise RuntimeError("frozen audio integrity row missing record_uid")
        if uid in found:
            raise RuntimeError(f"duplicate frozen audio integrity uid: {uid}")
        digest = str(row.get("sha256_pcm") or "").strip().lower()
        if not _is_valid_sha256(digest):
            raise RuntimeError(f"frozen audio integrity sha256_pcm invalid for {uid}")
        found[uid] = {
            "identity_source": "frozen_integrity",
            "sha256_pcm": digest,
            "sha256_source": str(row["sha256_source"]).strip().lower() if "sha256_source" in frame.columns else "",
            "n_samples": int(row.get("n_samples") or 0),
            "sample_rate": int(row.get("sample_rate") or 0),
            "shard_key": str(row.get("shard_key") or ""),
            "shard_row_index": _parse_required_row_index(row.get("shard_row_index"), f"frozen audio integrity {uid}"),
        }
    return found


def build_derived_protected_identity_rows(
    *,
    reference_index: Dict[str, object],
    project_root,
    dataset_id: str,
    dataset_revision: str,
    parquet_revision: str,
    audio_index_dir,
    frozen_integrity_csv,
    expected_counts: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """Join canonical manifests to read-only RQ1 identity metadata.

    Rows that RQ1 excluded from its optimizer index stay in this universe.
    Their PCM hash is filled later by on-demand reconstruction. This function
    does not write or open any RQ1 artifact for update.
    """
    from src.asr_full_pcm import AUDIO_PCM_PIPELINE_VERSION

    root = Path(project_root)
    counts = dict(expected_counts or EXPECTED_PROTECTED_MANIFEST_COUNTS)
    pinned_dataset = str(dataset_revision or "").strip().lower()
    pinned_parquet = str(parquet_revision or "").strip().lower()
    if not dataset_id or not _is_pinned_revision(pinned_dataset) or not _is_pinned_revision(pinned_parquet):
        raise RuntimeError("derived identity index requires dataset id and pinned revisions")
    indexed = _load_audio_index_jsonl(audio_index_dir)
    frozen = _load_frozen_integrity(frozen_integrity_csv)
    rows: List[dict] = []
    seen = set()
    for split in PROTECTED_SPLITS:
        locator = reference_index.get(split)
        if not locator:
            raise RuntimeError(f"reference_index is missing a manifest locator for {split}")
        path = Path(locator)
        if not path.is_absolute():
            path = root / path
        manifest_rows = _load_manifest_identity_rows(path, split)
        if split in counts and len(manifest_rows) != int(counts[split]):
            raise RuntimeError(
                f"protected split {split} count {len(manifest_rows)} != expected {counts[split]}"
            )
        for item in manifest_rows:
            uid = item["reference_uid"]
            key = (split, uid)
            if key in seen:
                raise RuntimeError(f"duplicate protected audio identity for {split}:{uid}")
            seen.add(key)
            meta = frozen.get(uid) if split == "frozen_test" else indexed.get(uid)
            if meta is None and split == "frozen_test":
                meta = indexed.get(uid)
            if meta is not None:
                _verify_protected_identity_meta(
                    meta, item, split, uid,
                    pinned_dataset=pinned_dataset,
                    pinned_parquet=pinned_parquet,
                    pipeline=AUDIO_PCM_PIPELINE_VERSION,
                )
            rows.append({
                "reference_uid": uid,
                "split": split,
                "manifest_audio_locator": item["manifest_audio_locator"],
                "parquet_file": item["parquet_file"],
                "shard_key": item["shard_key"],
                "shard_row_index": int(item["shard_row_index"]),
                "dataset_revision": pinned_dataset,
                "parquet_revision": pinned_parquet,
                "pcm_pipeline_version": AUDIO_PCM_PIPELINE_VERSION,
                "sha256_pcm": "" if meta is None else meta["sha256_pcm"],
                "sha256_source": "" if meta is None else meta.get("sha256_source") or "",
                "n_samples": 0 if meta is None else int(meta.get("n_samples") or 0),
                "sample_rate": 0 if meta is None else int(meta.get("sample_rate") or 0),
                "record_id": item["record_id"],
            })
    rows.sort(key=lambda item: (item["split"], item["reference_uid"]))
    assert_no_forbidden_reference_columns(DERIVED_PROTECTED_IDENTITY_COLUMNS)
    return rows


def write_derived_protected_identity_index(rows: Sequence[dict], dest) -> str:
    """Atomically write the derived index and return its file SHA-256."""
    import pandas as pd

    assert_no_forbidden_reference_columns(DERIVED_PROTECTED_IDENTITY_COLUMNS)
    frame = pd.DataFrame(list(rows), columns=list(DERIVED_PROTECTED_IDENTITY_COLUMNS))
    if list(frame.columns) != list(DERIVED_PROTECTED_IDENTITY_COLUMNS):
        raise RuntimeError("derived identity index columns drifted")
    payload = frame.to_csv(index=False)
    header = payload.splitlines()[0].lower() if payload else ""
    for token in FORBIDDEN_REFERENCE_TOKENS:
        if token in header:
            raise RuntimeError(f"derived identity index contains forbidden token {token!r}")
    dest_path = Path(dest)
    atomic_write_text(dest_path, payload)
    digest = _sha256_file(dest_path)
    atomic_write_text(dest_path.with_suffix(dest_path.suffix + ".sha256"), digest + "\n")
    return digest


def load_derived_protected_identity_index(path, expected_sha256: str) -> Dict[Tuple[str, str], dict]:
    """Load the derived index. Transcript columns are rejected."""
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
    frame = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    columns = list(frame.columns)
    assert_no_forbidden_reference_columns(columns)
    if columns != list(DERIVED_PROTECTED_IDENTITY_COLUMNS):
        raise RuntimeError(f"derived identity index schema mismatch: {columns}")
    index: Dict[Tuple[str, str], dict] = {}
    for _, row in frame.iterrows():
        split = str(row["split"] or "")
        uid = str(row["reference_uid"] or "")
        locator = str(row["manifest_audio_locator"] or "")
        digest = str(row["sha256_pcm"] or "").strip().lower()
        if split not in PROTECTED_SPLITS or not uid or not locator:
            raise RuntimeError("protected audio identity row is incomplete")
        if digest and not _is_valid_sha256(digest):
            raise RuntimeError(f"protected audio identity {split}:{uid} has no source_sha256")
        shard_row_index = _parse_required_row_index(row["shard_row_index"], f"{split}:{uid}")
        if not str(row["parquet_file"] or "").strip():
            raise RuntimeError(f"missing shard row for {split}:{uid}")
        key = (split, uid)
        if key in index:
            raise RuntimeError(f"duplicate protected audio identity for {split}:{uid}")
        index[key] = {
            "audio_locator": locator,
            "manifest_audio_locator": locator,
            "source_sha256": digest,
            "parquet_file": str(row["parquet_file"] or ""),
            "shard_key": str(row["shard_key"] or ""),
            "shard_row_index": shard_row_index,
            "dataset_revision": str(row["dataset_revision"] or ""),
            "parquet_revision": str(row["parquet_revision"] or ""),
            "pcm_pipeline_version": str(row["pcm_pipeline_version"] or ""),
            "sha256_pcm": digest,
            "sha256_source": str(row["sha256_source"] or "").strip().lower(),
            "n_samples": int(row["n_samples"] or 0),
            "sample_rate": int(row["sample_rate"] or 0),
            "record_id": str(row["record_id"] or ""),
        }
    if not index:
        raise RuntimeError("protected audio identity index is empty")
    return index


def discover_full_audio_index_dir(durable_root) -> Path:
    """Locate the RQ1 full-state audio index without choosing among contracts.

    The RQ1 final contract does not store the ``full_state/contract_*`` directory
    that owns ``audio_index``. There is no pinned path to bind, so this accepts
    the directory only when exactly one exists. It does not rank by mtime, name,
    or lexical order.
    """
    root = Path(durable_root) / "bahnar_s2tt" / "full_state"
    matches = [path for path in root.glob("contract_*/audio_index") if path.is_dir()]
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches[:5])
        raise RuntimeError(
            "RQ1 final contract does not pin the full_state audio_index directory; "
            f"refusing to choose among {len(matches)} matches under {root}"
            + (f" ({found})" if found else "")
        )
    return matches[0]


def _verify_protected_identity_meta(
    meta: dict,
    item: dict,
    split: str,
    uid: str,
    *,
    pinned_dataset: str,
    pinned_parquet: str,
    pipeline: str,
) -> None:
    """Bind a sidecar or frozen-integrity row to one manifest identity."""
    label = f"{split}:{uid}"
    row_index = meta.get("shard_row_index")
    if row_index is None:
        raise RuntimeError(f"{label} is missing shard_row_index")
    if int(row_index) != int(item["shard_row_index"]):
        raise RuntimeError(
            f"shard_row_index mismatch for {label}: sidecar {int(row_index)} != manifest {int(item['shard_row_index'])}"
        )
    if meta.get("identity_source") == "audio_index":
        if meta.get("split") != split:
            raise RuntimeError(f"split mismatch for {label}")
        if meta.get("shard_key") != item["shard_key"]:
            raise RuntimeError(f"shard key mismatch for {label}")
        if not _is_valid_sha256(str(meta.get("sha256_source") or "")):
            raise RuntimeError(f"audio index sha256_source invalid for {label}")
        if not _is_valid_sha256(str(meta.get("sha256_pcm") or "")):
            raise RuntimeError(f"audio index sha256_pcm invalid for {label}")
        if meta.get("dataset_revision") != pinned_dataset:
            raise RuntimeError(f"wrong dataset revision for {label}")
        if meta.get("parquet_revision") != pinned_parquet:
            raise RuntimeError(f"wrong parquet revision for {label}")
        if meta.get("pcm_pipeline_version") != pipeline:
            raise RuntimeError(f"wrong pcm pipeline for {label}")
        return
    if meta.get("shard_key") and meta["shard_key"] != item["shard_key"]:
        raise RuntimeError(f"shard key mismatch for {label}")
    if meta.get("parquet_revision") and meta["parquet_revision"] != pinned_parquet:
        raise RuntimeError(f"wrong parquet revision for {label}")
    if meta.get("dataset_revision") and meta["dataset_revision"] != pinned_dataset:
        raise RuntimeError(f"wrong dataset revision for {label}")
    if meta.get("pcm_pipeline_version") and meta["pcm_pipeline_version"] != pipeline:
        raise RuntimeError(f"wrong pcm pipeline for {label}")


def ensure_derived_protected_identity_index(
    *,
    project_root,
    reference_index: Dict[str, object],
    durable_root=None,
    dest=None,
    audio_index_dir=None,
    frozen_integrity_csv=None,
    expected_counts: Optional[Dict[str, int]] = None,
) -> dict:
    """Build the NB11 identity index from read-only RQ1 manifests and sidecars.

    The CSV and its ``.sha256`` sidecar are NB11 artifacts under the durable
    U_clean root. RQ1 contracts, manifests, and the audio index are opened
    for reading only.
    """
    from src.rq1_runtime_paths import resolve_rq1_runtime_paths

    runtime = resolve_rq1_runtime_paths(project_root=project_root, durable_root=durable_root)
    latest_path = runtime.durable_state_root / "LATEST_UNLOCKED.json"
    if not latest_path.is_file():
        raise RuntimeError("RQ1 LATEST_UNLOCKED.json is missing; cannot build the protected identity index")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    contract_hash = str(latest.get("final_contract_hash") or "").strip().lower()
    state_dir = runtime.state_dir(contract_hash)
    final_payload = json.loads((state_dir / RQ1_FINAL_CONTRACT_FILENAME).read_text(encoding="utf-8"))
    verify_rq1_contract_hash(final_payload, "rq1_final_contract_hash", contract_hash)
    test_path = state_dir / RQ1_TEST_CONTRACT_FILENAME
    test_payload = json.loads(test_path.read_text(encoding="utf-8"))
    verify_rq1_contract_hash(
        test_payload, "rq1_test_contract_hash", str(final_payload.get("rq1_test_contract_hash") or ""),
    )
    dataset_id = str(test_payload.get("dataset_id") or final_payload.get("dataset_id") or "")
    dataset_revision = str(test_payload.get("dataset_revision") or final_payload.get("dataset_revision") or "")
    parquet_revision = str(
        test_payload.get("parquet_revision")
        or test_payload.get("parquet_commit_sha")
        or final_payload.get("parquet_revision")
        or ""
    )
    index_dir = Path(audio_index_dir) if audio_index_dir else discover_full_audio_index_dir(runtime.durable_root)
    integrity = Path(frozen_integrity_csv) if frozen_integrity_csv else state_dir / "rq1_audio_integrity.csv"
    if dest:
        dest_path = Path(dest)
    else:
        dest_path = resolve_u_clean_output(project_root, durable_root=durable_root)["out_dir"] / "protected_audio_identity.csv"
    rows = build_derived_protected_identity_rows(
        reference_index=reference_index,
        project_root=project_root,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        parquet_revision=parquet_revision,
        audio_index_dir=index_dir,
        frozen_integrity_csv=integrity,
        expected_counts=expected_counts,
    )
    digest = write_derived_protected_identity_index(rows, dest_path)
    out_root = resolve_u_clean_output(project_root, durable_root=durable_root)["out_dir"].resolve()
    try:
        relative = dest_path.resolve().relative_to(out_root).as_posix()
    except ValueError:
        relative = dest_path.name
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError("derived identity index escaped the U_clean out_dir")
    return {"relative_path": relative, "path": str(dest_path.resolve()), "sha256": digest, "n_rows": len(rows)}


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


def _load_bound_identity_index(path, expected_sha256: str) -> Dict[Tuple[str, str], dict]:
    header = Path(path).read_text(encoding="utf-8").splitlines()[:1]
    if header and "manifest_audio_locator" in header[0]:
        return load_derived_protected_identity_index(path, expected_sha256)
    return load_protected_audio_identity_index(path, expected_sha256)


@dataclass(frozen=True)
class MaterializedProtectedAudio:
    entry: ProtectedReferenceEntry
    path: Path
    sha256_pcm: str
    sha256_source: str
    n_samples: int
    sample_rate: int


class ProtectedParquetMaterializer:
    """Reconstruct protected audio from a pinned Parquet shard into a temp WAV.

    One shard download serves every requested row in that shard. Temp WAVs and
    the shard download are removed when iteration finishes. Nothing is written
    under the RQ1 durable state.
    """

    def __init__(self, *, dataset_id: str, cache_dir, download_fn=None, reader_factory=None):
        self._dataset_id = str(dataset_id or "")
        self._cache_dir = Path(cache_dir)
        self._download_fn = download_fn
        self._reader_factory = reader_factory

    def materialize_audio(self, entry: ProtectedReferenceEntry):
        @contextmanager
        def _one():
            generator = self.iter_materialized_audio([entry])
            try:
                item = next(generator)
            except StopIteration:
                raise RuntimeError(f"protected audio was not materialized for {entry.reference_uid}")
            try:
                yield item
            finally:
                generator.close()
        return _one()

    def iter_materialized_audio(self, entries: Sequence[ProtectedReferenceEntry]):
        from src.asr_full_pcm import (
            AUDIO_PCM_PIPELINE_VERSION,
            canonical_pcm16_from_bytes,
            cleanup_shard_download,
            normalize_parquet_ref,
            write_wav_from_pcm16,
        )
        from src.asr_full_shards import make_hf_parquet_stream_reader

        groups: Dict[str, List[ProtectedReferenceEntry]] = {}
        for entry in entries:
            if not _entry_can_reconstruct(entry):
                raise RuntimeError(f"missing shard row for {entry.split}:{entry.reference_uid}")
            if entry.pcm_pipeline_version and entry.pcm_pipeline_version != AUDIO_PCM_PIPELINE_VERSION:
                raise RuntimeError(
                    f"wrong pcm pipeline for {entry.reference_uid}: {entry.pcm_pipeline_version}"
                )
            groups.setdefault(entry.shard_key or _shard_key_from_parquet_file(entry.parquet_file), []).append(entry)
        for shard_key in sorted(groups):
            group = groups[shard_key]
            revisions = {entry.parquet_revision for entry in group if entry.parquet_revision}
            if len(revisions) != 1:
                raise RuntimeError(f"wrong parquet revision for shard {shard_key}")
            pinned = next(iter(revisions))
            by_index: Dict[int, ProtectedReferenceEntry] = {}
            for entry in group:
                index = int(entry.shard_row_index)
                if index in by_index:
                    raise RuntimeError(f"duplicate shard row {index} in {shard_key}")
                by_index[index] = entry
            ref = normalize_parquet_ref(
                group[0].parquet_file,
                expected_repo_id=self._dataset_id,
                parquet_revision=pinned,
            )
            downloaded: Dict[str, str] = {}
            if self._reader_factory is not None:
                reader = self._reader_factory(downloaded)
            else:
                reader = make_hf_parquet_stream_reader(
                    cache_dir=self._cache_dir,
                    downloaded=downloaded,
                    download_fn=self._download_fn,
                )
            produced = set()
            try:
                for idx, payload in reader(ref, sorted(by_index)):
                    entry = by_index.get(int(idx))
                    if entry is None:
                        continue
                    if int(idx) in produced:
                        raise RuntimeError(f"duplicate shard row {idx} in {shard_key}")
                    produced.add(int(idx))
                    item = _materialize_one_payload(entry, payload, canonical_pcm16_from_bytes, write_wav_from_pcm16)
                    try:
                        yield item
                    finally:
                        if item.path.exists():
                            item.path.unlink()
                missing = sorted(set(by_index) - produced)
                if missing:
                    raise RuntimeError(f"missing shard row {missing[:5]} in {shard_key}")
            finally:
                for local in list(downloaded.values()):
                    cleanup_shard_download(local)
                downloaded.clear()


def _materialize_one_payload(entry, payload, decode, write_wav) -> MaterializedProtectedAudio:
    payload_id = payload.get("id") if isinstance(payload, dict) else None
    if entry.record_id and payload_id is not None and str(payload_id) != entry.record_id:
        raise RuntimeError(
            f"protected record_id mismatch uid={entry.reference_uid}: {payload_id!r} != {entry.record_id!r}"
        )
    raw = payload.get("audio") if isinstance(payload, dict) else None
    if isinstance(raw, dict):
        raw = raw.get("bytes")
    if not raw:
        raise RuntimeError(f"protected audio payload empty for {entry.reference_uid}")
    decoded = decode(bytes(raw), target_sr=16000)
    source_hash = str(decoded.get("sha256_source") or "").strip().lower()
    pcm_hash = str(decoded.get("sha256_pcm") or "").strip().lower()
    if entry.sha256_source and source_hash != entry.sha256_source:
        raise RuntimeError(f"sha256_source mismatch for {entry.reference_uid}")
    if entry.sha256_pcm and pcm_hash != entry.sha256_pcm:
        raise RuntimeError(f"sha256_pcm mismatch for {entry.reference_uid}")
    if not _is_valid_sha256(pcm_hash):
        raise RuntimeError(f"decoded protected audio has no sha256_pcm for {entry.reference_uid}")
    n_samples = int(decoded.get("n_samples") or 0)
    sample_rate = int(decoded.get("sample_rate") or 0)
    if entry.n_samples and n_samples != int(entry.n_samples):
        raise RuntimeError(f"n_samples mismatch for {entry.reference_uid}")
    if entry.sample_rate and sample_rate != int(entry.sample_rate):
        raise RuntimeError(f"sample_rate mismatch for {entry.reference_uid}")
    handle = tempfile.NamedTemporaryFile(prefix="nb11-protected-", suffix=".wav", delete=False)
    handle.close()
    path = Path(handle.name)
    try:
        write_wav(path, decoded["pcm"], sample_rate or 16000)
    except Exception:
        if path.exists():
            path.unlink()
        raise
    return MaterializedProtectedAudio(
        entry=entry,
        path=path,
        sha256_pcm=pcm_hash,
        sha256_source=source_hash,
        n_samples=n_samples,
        sample_rate=sample_rate,
    )


def _parquet_materializer_from_entries(entries, *, dataset_id: str, cache_dir) -> ProtectedParquetMaterializer:
    return ProtectedParquetMaterializer(dataset_id=dataset_id, cache_dir=cache_dir)


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
        self._pinned_parquet_revision = str(test_payload.get("parquet_revision") or "").strip().lower()
        self._pinned_dataset_revision = str(test_payload.get("dataset_revision") or "").strip().lower()
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
            if not manifest_path.is_absolute():
                manifest_path = Path(self._project_root) / manifest_path
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
        identity = _load_bound_identity_index(
            self._audio_identity_index, self._audio_identity_sha256,
        )
        delegate = ManifestProtectedReferenceResolver(self._manifest_paths, self._mapping)
        entries = bind_protected_audio_identity(delegate.resolve(), identity)
        pinned_parquet = str(self._pinned_parquet_revision or "")
        pinned_dataset = str(self._pinned_dataset_revision or "")
        for entry in entries:
            if pinned_parquet and entry.parquet_revision and entry.parquet_revision != pinned_parquet:
                raise RuntimeError(f"wrong parquet revision for {entry.split}:{entry.reference_uid}")
            if pinned_dataset and entry.dataset_revision and entry.dataset_revision != pinned_dataset:
                raise RuntimeError(f"wrong dataset revision for {entry.split}:{entry.reference_uid}")
            if entry.pcm_pipeline_version and entry.pcm_pipeline_version != "full_pcm16_le_v1":
                raise RuntimeError(f"wrong pcm pipeline for {entry.split}:{entry.reference_uid}")
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
        raise RuntimeError(
            "protected audio is reconstructed from pinned Parquet; "
            "use materialize_audio instead of a durable_root file path"
        )

    def materialize_audio(self, entry: ProtectedReferenceEntry):
        return _parquet_materializer_from_entries(
            [entry],
            dataset_id=self._dataset_id(),
            cache_dir=self._parquet_cache_dir(),
        ).materialize_audio(entry)

    def iter_materialized_audio(self, entries: Sequence[ProtectedReferenceEntry]):
        return _parquet_materializer_from_entries(
            entries,
            dataset_id=self._dataset_id(),
            cache_dir=self._parquet_cache_dir(),
        ).iter_materialized_audio(entries)

    def _dataset_id(self) -> str:
        payload = self._load_contract()
        dataset_id = str(payload.get("dataset_id") or "")
        if not dataset_id:
            test_path = self._state_dir() / RQ1_TEST_CONTRACT_FILENAME
            test_payload = json.loads(test_path.read_text(encoding="utf-8"))
            dataset_id = str(test_payload.get("dataset_id") or "")
        if not dataset_id:
            raise RuntimeError("RQ1 contract is missing dataset_id")
        return dataset_id

    def _state_dir(self) -> Path:
        from src.rq1_runtime_paths import resolve_rq1_runtime_paths

        runtime = resolve_rq1_runtime_paths(
            project_root=self._project_root,
            durable_root=self._durable_root,
        )
        return runtime.state_dir(self._contract_hash)

    def _parquet_cache_dir(self) -> Path:
        from src.rq1_runtime_paths import resolve_rq1_runtime_paths

        runtime = resolve_rq1_runtime_paths(
            project_root=self._project_root,
            durable_root=self._durable_root,
        )
        return runtime.hf_parquet_cache_dir

    def provenance(self) -> dict:
        return {
            "resolver": type(self).__name__,
            "rq1_final_contract_hash": str(self._contract_hash or ""),
            "rq1_test_contract_hash": str(self._test_contract_hash or ""),
            "protected_audio_identity_sha256": str(self._audio_identity_sha256 or ""),
            "parquet_revision": str(getattr(self, "_pinned_parquet_revision", "") or ""),
            "dataset_revision": str(getattr(self, "_pinned_dataset_revision", "") or ""),
            "pcm_pipeline_version": "full_pcm16_le_v1",
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


def apply_u_perceptual_eligibility(segments: List[dict], config: UCleanConfig) -> int:
    """Exclude retained segments that cannot emit min_overlap_items fingerprints.

    This runs after exact PCM dedup and before WAV writing. It does not change
    the segmentation contract: segments between 3.0s and 5.1s remain planned,
    then leave U_clean here.
    """
    sample_rate = int(config.segmentation.sample_rate)
    min_items = int(config.overlap.min_overlap_items)
    excluded = 0
    for row in segments:
        if row.get("u_clean_status") != RETAINED_STATUS:
            continue
        n_samples = int(row["end_sample"]) - int(row["start_sample"])
        if _duration_reaches_perceptual_minimum(n_samples, sample_rate):
            continue
        row["u_perceptual_min_duration_seconds"] = U_PERCEPTUAL_MIN_DURATION_SECONDS
        row["perceptual_eligibility_min_overlap_items"] = min_items
        row["u_perceptual_policy"] = U_PERCEPTUAL_POLICY
        row["u_perceptual_fpcalc_version"] = PROTECTED_PERCEPTUAL_FPCALC_VERSION
        _exclude(row, EXCLUDED_PERCEPTUAL_INELIGIBLE)
        excluded += 1
    return excluded


def migrate_cached_u_fingerprints(
    segments: List[dict],
    context: UCleanContext,
    cached_fps: Optional[Dict[str, Sequence[int]]] = None,
) -> Dict[str, List[int]]:
    """Drop cached U fingerprints that cannot be reused. Segmentation shards stay.

    A segment shorter than 5.1s is excluded and its cached fingerprint is ignored.
    A segment of at least 5.1s whose cached fingerprint has fewer than
    min_overlap_items items loses only that UID, so the next fingerprint pass
    recomputes it. Valid fingerprints and protected-reference state stay.
    """
    apply_u_perceptual_eligibility(segments, context.config)
    sample_rate = int(context.config.segmentation.sample_rate)
    min_items = int(context.config.overlap.min_overlap_items)
    by_uid = {str(row.get("segment_uid") or ""): row for row in segments}
    reusable: Dict[str, List[int]] = {}
    drop = set()
    for uid, fp in (cached_fps or {}).items():
        key = str(uid)
        row = by_uid.get(key)
        n_items = len(fp) if is_valid_fingerprint(fp) else 0
        if row is None:
            if n_items >= min_items:
                reusable[key] = [int(item) for item in fp]
            continue
        n_samples = int(row["end_sample"]) - int(row["start_sample"])
        below_duration = not _duration_reaches_perceptual_minimum(n_samples, sample_rate)
        if below_duration or n_items < min_items:
            drop.add(key)
            continue
        reusable[key] = [int(item) for item in fp]
    if drop:
        state = _read_checkpoint_state(context)
        state["segment_fingerprint_shards"] = _forget_shard_uids(
            state["segment_fingerprint_shards"], drop,
        )
        _write_checkpoint_state(context, state)
    return reusable


def fingerprint_retained_u_segments(
    segments: List[dict],
    context: UCleanContext,
    fingerprint_fn,
    cached_fps: Optional[Dict[str, Sequence[int]]] = None,
) -> Dict[str, List[int]]:
    """Fingerprint retained U segments. Ineligible rows are already excluded.

    A cached fingerprint is reused only when it already has min_overlap_items
    items. fpcalc errors, empty fingerprints, and shorter results fail closed.
    """
    cached = cached_fps or {}
    config = context.config
    sample_rate = int(config.segmentation.sample_rate)
    min_items = int(config.overlap.min_overlap_items)
    out: Dict[str, List[int]] = {}
    pending: List[dict] = []
    for row in segments:
        if row.get("u_clean_status") != RETAINED_STATUS:
            continue
        uid = str(row["segment_uid"])
        n_samples = int(row["end_sample"]) - int(row["start_sample"])
        if not _duration_reaches_perceptual_minimum(n_samples, sample_rate):
            raise RuntimeError(
                f"U segment {uid} is below perceptual eligibility and must be excluded "
                f"before fingerprinting"
            )
        fp = cached.get(uid)
        n_items = len(fp) if is_valid_fingerprint(fp) else 0
        reused = n_items >= min_items
        if not reused:
            fp = fingerprint_fn(row)
            n_items = len(fp) if is_valid_fingerprint(fp) else 0
            if n_items < min_items:
                duration = n_samples / float(sample_rate)
                raise RuntimeError(
                    f"U fingerprint below min_overlap_items for {uid} "
                    f"duration_seconds={duration:.6f} items={n_items} required={min_items}"
                )
            pending.append({
                "segment_uid": uid,
                "fingerprint": fp,
                "audio_sha256": str(row.get("segment_pcm16_sha256") or ""),
            })
            if len(pending) >= 32:
                append_fingerprint_shard(context, "segment_fingerprints", pending)
                pending = []
        stored = [int(item) for item in fp]
        out[uid] = stored
        row["fingerprint_sha256"] = fingerprint_sha256(stored)
    if pending:
        append_fingerprint_shard(context, "segment_fingerprints", pending)
    return out


def _ordered_uu_rows(segments: Sequence[dict]) -> List[dict]:
    """Retained rows in the exact sequential order U-U dedup scans."""
    retained = [row for row in segments if row["u_clean_status"] == RETAINED_STATUS]
    retained.sort(key=_owner_key)
    return retained


def _decide_perceptual_row(
    row: dict,
    index: FingerprintIndex,
    uid_to_row: Dict[str, dict],
    fingerprints_by_uid: Dict[str, Sequence[int]],
    overlap_config,
) -> dict:
    """One sequential U-U step. Candidate universe is the owner index so far."""
    uid = str(row["segment_uid"])
    fp = fingerprints_by_uid.get(uid)
    if not _u_fingerprint_indexable(fp, overlap_config):
        return {
            "segment_uid": uid,
            "decision": "NOT_INDEXABLE",
            "canonical_segment_uid": str(row.get("canonical_segment_uid") or ""),
            "n_comparisons": 0,
            "evidence": None,
        }
    best = None
    n_comparisons = 0
    for cand_uid in index.candidates(fp):
        n_comparisons += 1
        score, offset, overlap = compare_fingerprints_detailed(
            fp, index.fingerprint_of(cand_uid), overlap_config,
        )
        if score >= overlap_config.similarity_threshold and (best is None or score > best[0]):
            best = (score, offset, overlap, cand_uid)
    if best is not None:
        score, offset, overlap, owner_uid = best
        row["perceptual_duplicate"] = True
        row["canonical_segment_uid"] = uid_to_row[owner_uid]["canonical_segment_uid"]
        _exclude(row, EXCLUDED_PERCEPTUAL_DUPLICATE)
        evidence = {
            "candidate_uid": uid,
            "reference_uid": str(owner_uid),
            "reference_split": "u_real",
            "matched_duration_seconds": matched_duration_seconds(overlap, overlap_config),
            "alignment_offset": int(offset),
            "similarity": round(score, 6),
            "match_type": "u_u",
        }
        return {
            "segment_uid": uid,
            "decision": "PERCEPTUAL_DUPLICATE",
            "canonical_segment_uid": str(row["canonical_segment_uid"]),
            "n_comparisons": n_comparisons,
            "evidence": evidence,
        }
    index.add(uid, fp)
    uid_to_row[uid] = row
    return {
        "segment_uid": uid,
        "decision": "RETAINED_OWNER",
        "canonical_segment_uid": str(row.get("canonical_segment_uid") or ""),
        "n_comparisons": n_comparisons,
        "evidence": None,
    }


def perceptual_deduplicate(
    segments: List[dict],
    fingerprints_by_uid: Dict[str, Sequence[int]],
    config: UCleanConfig,
) -> List[MatchEvidence]:
    """Scalable U-U perceptual dedup via shingle index (item 11)."""
    if not config.overlap_config_frozen:
        raise RuntimeError("perceptual dedup requires a frozen OverlapConfig")
    overlap_config = config.overlap
    index = FingerprintIndex(overlap_config)
    uid_to_row: Dict[str, dict] = {}
    evidence: List[MatchEvidence] = []
    for row in _ordered_uu_rows(segments):
        decision = _decide_perceptual_row(row, index, uid_to_row, fingerprints_by_uid, overlap_config)
        if decision["evidence"] is not None:
            evidence.append(MatchEvidence(**decision["evidence"]))
    return evidence


UU_DEDUP_SCHEMA_VERSION = "rq2-uu-dedup-1"
_UU_DECISIONS = ("RETAINED_OWNER", "PERCEPTUAL_DUPLICATE", "NOT_INDEXABLE")


def uu_dedup_dir(context: UCleanContext) -> Path:
    return checkpoint_root(context) / "u_u_dedup"


def _uu_identity(context: UCleanContext, ordered_rows: Sequence[dict], fingerprints_by_uid: Dict[str, Sequence[int]]) -> dict:
    """Binds a U-U checkpoint without touching the global checkpoint compatibility key."""
    key = compatibility_key(context)
    uids = [str(row["segment_uid"]) for row in ordered_rows]
    fingerprint_pairs = []
    for row in ordered_rows:
        uid = str(row["segment_uid"])
        fp = fingerprints_by_uid.get(uid)
        fingerprint_pairs.append([
            uid,
            fingerprint_sha256(fp) if is_valid_fingerprint(fp) else "",
        ])
    return {
        "schema_version": UU_DEDUP_SCHEMA_VERSION,
        "compatibility_key": key,
        "compatibility_sha256": hashlib.sha256(canonical_json_bytes(key)).hexdigest(),
        "overlap_contract_sha256": context.config.overlap_contract_sha(),
        "ordered_uid_sha256": hashlib.sha256(canonical_json_bytes(uids)).hexdigest(),
        "fingerprint_identity_sha256": hashlib.sha256(canonical_json_bytes(fingerprint_pairs)).hexdigest(),
    }


def _uu_jsonl(rows: Sequence[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _uu_evidence_rows(decisions: Sequence[dict]) -> List[dict]:
    return [dict(decision["evidence"]) for decision in decisions if decision.get("evidence")]


def _uu_counts(decisions: Sequence[dict]) -> dict:
    owners = 0
    duplicates = 0
    skipped = 0
    comparisons = 0
    for decision in decisions:
        kind = decision.get("decision")
        comparisons += int(decision.get("n_comparisons") or 0)
        if kind == "RETAINED_OWNER":
            owners += 1
        elif kind == "PERCEPTUAL_DUPLICATE":
            duplicates += 1
        elif kind == "NOT_INDEXABLE":
            skipped += 1
    return {
        "n_retained_owners": owners,
        "n_perceptual_duplicates": duplicates,
        "n_not_indexable": skipped,
        "n_candidate_comparisons": comparisons,
    }


def _format_duration(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "unknown"
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%d:%02d" % (minutes, secs)


def _print_uu_progress(state: dict, *, resumed: bool = False) -> None:
    processed = int(state["n_processed"])
    total = int(state["n_total"])
    elapsed = float(state["elapsed_seconds"])
    rate = (processed / elapsed) if elapsed > 0 else 0.0
    remaining = max(total - processed, 0)
    eta = (remaining / rate) if rate > 0 else float("inf")
    prefix = "U-U DEDUP resume" if resumed else "U-U DEDUP"
    print(
        "%s\nprocessed: %d / %d\nretained owners: %d\nperceptual duplicates: %d\n"
        "candidate comparisons: %d\nelapsed: %s\nrate: %.3f rows/s\nETA: %s"
        % (
            prefix,
            processed,
            total,
            int(state["n_retained_owners"]),
            int(state["n_perceptual_duplicates"]),
            int(state["n_candidate_comparisons"]),
            _format_duration(elapsed),
            rate,
            _format_duration(eta),
        ),
        flush=True,
    )


def _commit_uu_checkpoint(directory: Path, state: dict, decisions: Sequence[dict]) -> None:
    """Decisions and evidence become durable before state.json advances."""
    directory.mkdir(parents=True, exist_ok=True)
    decision_rows = [dict(row) for row in decisions]
    evidence_rows = _uu_evidence_rows(decision_rows)
    decisions_text = _uu_jsonl(decision_rows)
    evidence_text = _uu_jsonl(evidence_rows)
    atomic_write_text(directory / "decisions.jsonl", decisions_text)
    atomic_write_text(directory / "evidence.jsonl", evidence_text)
    counts = _uu_counts(decision_rows)
    state.update(counts)
    state["n_processed"] = len(decision_rows)
    state["next_ordinal"] = len(decision_rows)
    state["status"] = "COMPLETE" if len(decision_rows) == int(state["n_total"]) else "IN_PROGRESS"
    state["decisions_sha256"] = hashlib.sha256(decisions_text.encode("utf-8")).hexdigest()
    state["evidence_sha256"] = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
    atomic_write_text(directory / "state.json", json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_uu_jsonl(
    path: Path,
    expected_sha: str,
    label: str,
    *,
    prefix_rows: Optional[int] = None,
) -> List[dict]:
    """Return the committed JSONL prefix. ``prefix_rows`` may be zero.

    A zero-length prefix is the SHA256 of an empty file. Later lines in a
    replaced file belong to an uncommitted checkpoint and are ignored.
    """
    if not path.is_file():
        raise RuntimeError("U-U checkpoint is corrupt: missing %s" % label)
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if prefix_rows is None:
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_sha:
            raise RuntimeError("U-U checkpoint is corrupt: %s" % label)
        prefix_rows = len(lines)
    if prefix_rows < 0 or len(lines) < prefix_rows:
        raise RuntimeError("U-U checkpoint state is ahead of %s" % label)
    raw_prefix = "".join(line + "\n" for line in lines[:prefix_rows])
    if hashlib.sha256(raw_prefix.encode("utf-8")).hexdigest() != expected_sha:
        raise RuntimeError("U-U checkpoint is corrupt: %s" % label)
    try:
        return [json.loads(line) for line in lines[:prefix_rows]]
    except json.JSONDecodeError as exc:
        raise RuntimeError("U-U checkpoint is corrupt: %s" % label) from exc


def _assert_uu_state_counters(state: dict, decisions: Sequence[dict]) -> None:
    """Committed counters must match the decision prefix. Do not repair them."""
    counts = _uu_counts(decisions)
    expected = {
        "n_processed": len(decisions),
        "next_ordinal": len(decisions),
        "n_retained_owners": counts["n_retained_owners"],
        "n_perceptual_duplicates": counts["n_perceptual_duplicates"],
        "n_not_indexable": counts["n_not_indexable"],
        "n_candidate_comparisons": counts["n_candidate_comparisons"],
    }
    for name, value in expected.items():
        try:
            actual = int(state[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("U-U checkpoint counters do not match decisions: %s" % name) from exc
        if actual != value:
            raise RuntimeError(
                "U-U checkpoint counters do not match decisions: %s state=%s decisions=%s"
                % (name, actual, value)
            )


def _load_uu_checkpoint(directory: Path, identity: dict, ordered_rows: Sequence[dict]) -> Optional[dict]:
    state_path = directory / "state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("U-U checkpoint is corrupt: state.json") from exc
    if not isinstance(state, dict):
        raise RuntimeError("U-U checkpoint is corrupt: state.json")
    if state.get("schema_version") != identity["schema_version"]:
        raise RuntimeError("U-U checkpoint is stale/incompatible: schema_version")
    for field_name in (
        "compatibility_sha256",
        "overlap_contract_sha256",
        "ordered_uid_sha256",
        "fingerprint_identity_sha256",
    ):
        if state.get(field_name) != identity[field_name]:
            raise RuntimeError("U-U checkpoint is stale/incompatible: %s" % field_name)
    if state.get("compatibility_key") != identity["compatibility_key"]:
        raise RuntimeError("U-U checkpoint is stale/incompatible: compatibility_key")
    status = state.get("status")
    if status not in ("IN_PROGRESS", "COMPLETE"):
        raise RuntimeError("U-U checkpoint is corrupt: status")
    try:
        n_processed = int(state["n_processed"])
        n_total = int(state["n_total"])
        next_ordinal = int(state["next_ordinal"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("U-U checkpoint is corrupt: counters") from exc
    if n_total != len(ordered_rows) or n_processed < 0 or n_processed > n_total or next_ordinal != n_processed:
        raise RuntimeError("U-U checkpoint is corrupt: ordinal prefix")
    if status == "COMPLETE" and n_processed != n_total:
        raise RuntimeError("U-U checkpoint is corrupt: COMPLETE prefix is short")
    decisions = _parse_uu_jsonl(
        directory / "decisions.jsonl", str(state.get("decisions_sha256") or ""), "decisions", prefix_rows=n_processed,
    )
    if len(decisions) != n_processed:
        raise RuntimeError("U-U checkpoint is corrupt: decisions length")
    for ordinal, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise RuntimeError("U-U checkpoint is corrupt: decision")
        if int(decision.get("ordinal", -1)) != ordinal:
            raise RuntimeError("U-U checkpoint ordinals are not contiguous")
        if str(decision.get("segment_uid") or "") != str(ordered_rows[ordinal]["segment_uid"]):
            raise RuntimeError("U-U checkpoint UID prefix does not match the current retained order")
        if decision.get("decision") not in _UU_DECISIONS:
            raise RuntimeError("U-U checkpoint is corrupt: decision")
    _assert_uu_state_counters(state, decisions)
    expected_evidence = _uu_evidence_rows(decisions)
    evidence = _parse_uu_jsonl(
        directory / "evidence.jsonl",
        str(state.get("evidence_sha256") or ""),
        "evidence",
        prefix_rows=len(expected_evidence),
    )
    if evidence != expected_evidence:
        raise RuntimeError("U-U checkpoint is corrupt: evidence does not match decisions")
    state["decisions"] = decisions
    return state


def _restore_uu_decision(row: dict, decision: dict, index: FingerprintIndex, uid_to_row: Dict[str, dict], fingerprints_by_uid, overlap_config) -> None:
    kind = decision["decision"]
    uid = str(row["segment_uid"])
    if kind == "NOT_INDEXABLE":
        return
    if kind == "RETAINED_OWNER":
        fp = fingerprints_by_uid.get(uid)
        if not _u_fingerprint_indexable(fp, overlap_config):
            raise RuntimeError("U-U checkpoint owner is no longer indexable: %s" % uid)
        if str(row.get("canonical_segment_uid") or "") != str(decision.get("canonical_segment_uid") or ""):
            raise RuntimeError("U-U checkpoint owner canonical_segment_uid drifted: %s" % uid)
        index.add(uid, fp)
        uid_to_row[uid] = row
        return
    if kind == "PERCEPTUAL_DUPLICATE":
        row["perceptual_duplicate"] = True
        row["canonical_segment_uid"] = decision["canonical_segment_uid"]
        _exclude(row, EXCLUDED_PERCEPTUAL_DUPLICATE)
        return
    raise RuntimeError("U-U checkpoint is corrupt: decision")


def _uu_match_evidence(decisions: Sequence[dict]) -> List[MatchEvidence]:
    return [MatchEvidence(**dict(decision["evidence"])) for decision in decisions if decision.get("evidence")]


def perceptual_deduplicate_resumable(
    segments: List[dict],
    fingerprints_by_uid: Dict[str, Sequence[int]],
    context: UCleanContext,
    *,
    checkpoint_every: int = 250,
    progress_interval_seconds: float = 45.0,
) -> List[MatchEvidence]:
    """Sequential U-U dedup with a durable prefix checkpoint.

    A resumed run restores decisions in order, rebuilds the owner index only
    from the retained-owner prefix, then continues at the next ordinal. A
    COMPLETE checkpoint is reapplied with no new fingerprint comparisons.
    """
    config = context.config
    if not config.overlap_config_frozen:
        raise RuntimeError("perceptual dedup requires a frozen OverlapConfig")
    overlap_config = config.overlap
    ordered = _ordered_uu_rows(segments)
    identity = _uu_identity(context, ordered, fingerprints_by_uid)
    directory = uu_dedup_dir(context)
    loaded = _load_uu_checkpoint(directory, identity, ordered)
    every = max(1, int(checkpoint_every))
    interval = float(progress_interval_seconds)
    index = FingerprintIndex(overlap_config)
    uid_to_row: Dict[str, dict] = {}
    if loaded is None:
        decisions: List[dict] = []
        base_elapsed = 0.0
        resumed = False
    else:
        decisions = [dict(row) for row in loaded["decisions"]]
        for decision in decisions:
            _restore_uu_decision(
                ordered[int(decision["ordinal"])], decision, index, uid_to_row, fingerprints_by_uid, overlap_config,
            )
        base_elapsed = float(loaded.get("elapsed_seconds") or 0.0)
        resumed = True
        if loaded.get("status") == "COMPLETE":
            state = dict(loaded)
            state.pop("decisions", None)
            _print_uu_progress(state, resumed=True)
            return _uu_match_evidence(decisions)
    started = time.perf_counter()
    last_commit = started
    if resumed:
        snapshot = _uu_progress_state(identity, decisions, len(ordered), base_elapsed, "IN_PROGRESS")
        _print_uu_progress(snapshot, resumed=True)
    for ordinal in range(len(decisions), len(ordered)):
        decision = _decide_perceptual_row(
            ordered[ordinal], index, uid_to_row, fingerprints_by_uid, overlap_config,
        )
        decision["ordinal"] = ordinal
        decisions.append(decision)
        now = time.perf_counter()
        done = len(decisions) == len(ordered)
        due = done or (len(decisions) % every == 0) or ((now - last_commit) >= interval)
        if not due:
            continue
        elapsed = base_elapsed + (now - started)
        state = _uu_progress_state(identity, decisions, len(ordered), elapsed, "IN_PROGRESS")
        _commit_uu_checkpoint(directory, state, decisions)
        last_commit = time.perf_counter()
        _print_uu_progress(state, resumed=False)
    return _uu_match_evidence(decisions)


def _uu_progress_state(identity: dict, decisions: Sequence[dict], n_total: int, elapsed: float, status: str) -> dict:
    counts = _uu_counts(decisions)
    state = dict(identity)
    state.update(counts)
    state["status"] = status
    state["n_total"] = int(n_total)
    state["n_processed"] = len(decisions)
    state["next_ordinal"] = len(decisions)
    state["elapsed_seconds"] = round(float(elapsed), 6)
    return state


def require_verified_segment_files(context: UCleanContext) -> None:
    """Stop a full run before fingerprinting or U-U dedup when segment files are unverified."""
    if not context.completion.segments_written_verified:
        raise RuntimeError("full run requires verified segment files")


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
    by_uid = {str(row["segment_uid"]): row for row in segments}
    evidence: List[MatchEvidence] = []
    batch: List[dict] = []
    for ref in reference_fingerprints:
        batch.append(ref)
        if len(batch) >= 32:
            evidence.extend(match_protected_batch(segments, index, batch, config, by_uid=by_uid))
            if benchmark_sink is not None:
                benchmark_sink["n_references_fingerprinted"] = int(benchmark_sink["n_references_fingerprinted"]) + len(batch)
            batch = []
    if batch:
        evidence.extend(match_protected_batch(segments, index, batch, config, by_uid=by_uid))
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
        if not _u_fingerprint_indexable(fp, config.overlap):
            continue
        index.add(str(row["segment_uid"]), fp)
    return index, time.perf_counter() - started


def _eligible_protected_match_uids(by_uid: Dict[str, dict]) -> frozenset:
    """Statuses the protected matcher is allowed to compare.

    A protected hit sets EXCLUDED_PROTECTED_OVERLAP and leaves the segment in
    later comparisons. Exact and perceptual exclusions are already absent from
    the candidate index, so read-only matching over this set is equivalent to
    checking status inside the historical per-reference loop.
    """
    allowed = (RETAINED_STATUS, EXCLUDED_PROTECTED_OVERLAP)
    return frozenset(
        uid for uid, row in by_uid.items()
        if row.get("u_clean_status") in allowed
    )


def compute_protected_matches(
    index: SegmentCandidateIndex,
    reference_batch: Sequence[dict],
    overlap: OverlapConfig,
    eligible_uids,
) -> dict:
    """Read-only protected comparison. Does not mutate segments or exclusions."""
    eligible = set(eligible_uids)
    evidence: List[MatchEvidence] = []
    n_comparisons = 0
    for ref in reference_batch:
        ref_fp = ref.get("fingerprint")
        if not is_valid_fingerprint(ref_fp):
            continue
        split = str(ref.get("split") or "")
        for uid in index.candidates_for_reference(ref_fp):
            if uid not in eligible:
                continue
            n_comparisons += 1
            score, offset, overlap_items = compare_fingerprints_detailed(
                index.fingerprint_of(uid), ref_fp, overlap,
            )
            if score < overlap.similarity_threshold:
                continue
            evidence.append(MatchEvidence(
                candidate_uid=uid,
                reference_uid=str(ref.get("uid") or ""),
                reference_split=split,
                matched_duration_seconds=matched_duration_seconds(overlap_items, overlap),
                alignment_offset=int(offset),
                similarity=round(float(score), 6),
                match_type="protected",
            ))
    return {
        "evidence": evidence,
        "n_references_processed": len(reference_batch),
        "n_candidate_comparisons": n_comparisons,
        "n_matches": len(evidence),
    }


def apply_protected_match_evidence(evidence: Sequence[MatchEvidence], by_uid: Dict[str, dict]) -> None:
    """Apply overlap flags and protected-overlap exclusion in evidence order."""
    for item in evidence:
        row = by_uid.get(item.candidate_uid)
        if row is None:
            continue
        flag = _SPLIT_TO_FLAG.get(item.reference_split)
        if flag:
            row[flag] = True
        _exclude(row, EXCLUDED_PROTECTED_OVERLAP)


def match_protected_batch(
    segments: List[dict],
    index: SegmentCandidateIndex,
    reference_batch: Sequence[dict],
    config: UCleanConfig,
    *,
    by_uid: Optional[Dict[str, dict]] = None,
) -> List[MatchEvidence]:
    """Query one protected batch against the U index and verify alignments."""
    if by_uid is None:
        by_uid = {str(row["segment_uid"]): row for row in segments}
    computed = compute_protected_matches(
        index, reference_batch, config.overlap, _eligible_protected_match_uids(by_uid),
    )
    apply_protected_match_evidence(computed["evidence"], by_uid)
    return computed["evidence"]


_PROTECTED_MATCH_RUNTIME: Dict[str, object] = {}
_LAST_MATCH_WORKER_PIDS: List[int] = []
_LAST_MATCH_MAX_INFLIGHT = 0
_LAST_MATCH_PULLED = 0
_LAST_MATCH_PULLED_AT_POOL_START = 0
_LAST_MATCH_PULLED_AT_FIRST_SUBMIT = 0
_PROTECTED_MATCH_KIND = "rq2_protected_match_shard_v1"
_PROTECTED_MATCH_FIELDS = (
    "kind",
    "status",
    "ordinal",
    "input_shard_name",
    "input_shard_sha256",
    "input_batch_sha256",
    "overlap_contract_sha256",
    "u_segment_uid_sha256",
    "u_fingerprint_identity_sha256",
    "compatibility",
    "compatibility_sha256",
    "result_sha256",
    "n_references_processed",
    "n_candidate_comparisons",
    "n_matches",
    "evidence",
)


def protected_match_worker_count(explicit: Optional[int] = None) -> int:
    """Runtime worker count. This is not a scientific contract field."""
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int):
            raise RuntimeError("protected match worker count must be a positive integer")
        count = explicit
    else:
        raw = os.environ.get("BAHNAR_NB11_MATCH_WORKERS", "").strip()
        if raw:
            if not raw.isdigit():
                raise RuntimeError("BAHNAR_NB11_MATCH_WORKERS must be a positive integer")
            count = int(raw)
        else:
            count = min(32, max(1, os.cpu_count() or 1))
    if count < 1:
        raise RuntimeError("BAHNAR_NB11_MATCH_WORKERS must be a positive integer")
    return count


def _u_candidate_identity(index: SegmentCandidateIndex) -> Tuple[str, str]:
    uids = sorted(index._fingerprints)
    uid_sha = hashlib.sha256(canonical_json_bytes(uids)).hexdigest()
    pairs = [[uid, list(index.fingerprint_of(uid))] for uid in uids]
    fingerprint_sha = hashlib.sha256(canonical_json_bytes(pairs)).hexdigest()
    return uid_sha, fingerprint_sha


def _base_match_compatibility(context: UCleanContext, index: SegmentCandidateIndex) -> dict:
    """U and overlap identity. Computed once per orchestration call."""
    key = compatibility_key(context)
    uid_sha, fingerprint_sha = _u_candidate_identity(index)
    return {
        "overlap_contract_sha256": key["overlap_contract_sha256"],
        "u_segment_uid_sha256": uid_sha,
        "u_fingerprint_identity_sha256": fingerprint_sha,
        "nb10_locks": key["nb10_locks"],
        "reference_set_sha256": key["reference_set_sha256"],
        "rq1_final_contract_hash": key["rq1_final_contract_hash"],
        "rq1_test_contract_hash": key["rq1_test_contract_hash"],
        "protected_audio_identity_sha256": key["protected_audio_identity_sha256"],
        "parquet_revision": key["parquet_revision"],
        "pcm_pipeline_version": key["pcm_pipeline_version"],
    }


def _shard_match_compatibility(base: dict, input_shard_sha256: str, input_batch_sha256: str) -> dict:
    """Per-shard identity. Does not walk or hash the U index."""
    compatibility = dict(base)
    compatibility["input_shard_sha256"] = str(input_shard_sha256)
    compatibility["input_batch_sha256"] = str(input_batch_sha256)
    return compatibility


def _protected_input_batch_sha256(batch: Sequence[dict]) -> str:
    """Hash the exact rows the matcher will see, in iterator order."""
    rows = []
    for row in batch:
        fp_sha = str(row.get("fingerprint_sha256") or "")
        if not fp_sha:
            fp_sha = fingerprint_sha256(row.get("fingerprint") or [])
        rows.append({
            "uid": str(row.get("uid") or ""),
            "split": str(row.get("split") or ""),
            "source_sha256": str(row.get("source_sha256") or ""),
            "fingerprint_sha256": fp_sha,
        })
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _evidence_rows(evidence: Sequence[MatchEvidence]) -> List[dict]:
    return [
        {
            "candidate_uid": item.candidate_uid,
            "reference_uid": item.reference_uid,
            "reference_split": item.reference_split,
            "matched_duration_seconds": item.matched_duration_seconds,
            "alignment_offset": int(item.alignment_offset),
            "similarity": item.similarity,
            "match_type": item.match_type,
        }
        for item in evidence
    ]


def _match_evidence_from_row(row: dict) -> MatchEvidence:
    return MatchEvidence(
        candidate_uid=str(row["candidate_uid"]),
        reference_uid=str(row["reference_uid"]),
        reference_split=str(row["reference_split"]),
        matched_duration_seconds=float(row["matched_duration_seconds"]),
        alignment_offset=int(row["alignment_offset"]),
        similarity=float(row["similarity"]),
        match_type=str(row["match_type"]),
    )


def _protected_match_dir(context: UCleanContext) -> Path:
    return checkpoint_root(context) / "protected_matches"


def _protected_match_path(context: UCleanContext, ordinal: int) -> Path:
    return _protected_match_dir(context) / ("match-%06d.json" % int(ordinal))


def _result_sha256(evidence_rows: Sequence[dict]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(evidence_rows))).hexdigest()


def _compatibility_sha256(compatibility: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(compatibility)).hexdigest()


def _read_protected_match_shard(
    context: UCleanContext,
    ordinal: int,
    expected: dict,
    batch_length: int,
) -> Optional[dict]:
    """Return a complete compatible result, or None so the shard is recomputed.

    A partial, corrupt, or stale file is not reused. The caller replaces it
    only after a new verified result has been written.
    """
    path = _protected_match_path(context, ordinal)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    if any(field not in document for field in _PROTECTED_MATCH_FIELDS):
        return None
    if document.get("kind") != _PROTECTED_MATCH_KIND or document.get("status") != "COMPLETE":
        return None
    if int(document.get("ordinal")) != int(ordinal):
        return None
    compatibility = document.get("compatibility")
    if compatibility != expected:
        return None
    if document.get("input_shard_sha256") != expected["input_shard_sha256"]:
        return None
    if document.get("input_batch_sha256") != expected["input_batch_sha256"]:
        return None
    if document.get("overlap_contract_sha256") != expected["overlap_contract_sha256"]:
        return None
    if document.get("u_segment_uid_sha256") != expected["u_segment_uid_sha256"]:
        return None
    if document.get("u_fingerprint_identity_sha256") != expected["u_fingerprint_identity_sha256"]:
        return None
    if document.get("compatibility_sha256") != _compatibility_sha256(expected):
        return None
    evidence = document.get("evidence")
    if not isinstance(evidence, list):
        return None
    for row in evidence:
        if not isinstance(row, dict):
            return None
        if any(key not in row for key in (
            "candidate_uid",
            "reference_uid",
            "reference_split",
            "matched_duration_seconds",
            "alignment_offset",
            "similarity",
            "match_type",
        )):
            return None
    if document.get("result_sha256") != _result_sha256(evidence):
        return None
    try:
        n_matches = int(document["n_matches"])
        n_references = int(document["n_references_processed"])
        n_comparisons = int(document["n_candidate_comparisons"])
    except (TypeError, ValueError):
        return None
    if n_matches != len(evidence) or n_references < 0 or n_comparisons < 0:
        return None
    if n_references != int(batch_length):
        return None
    if (
        type(document["n_matches"]) is not int
        or type(document["n_references_processed"]) is not int
        or type(document["n_candidate_comparisons"]) is not int
    ):
        return None
    return document


def _match_shard_record(context: UCleanContext, document: dict) -> dict:
    ordinal = int(document["ordinal"])
    return {
        "ordinal": ordinal,
        "name": _protected_match_path(context, ordinal).name,
        "input_shard_name": document["input_shard_name"],
        "input_shard_sha256": document["input_shard_sha256"],
        "input_batch_sha256": document["input_batch_sha256"],
        "result_sha256": document["result_sha256"],
        "compatibility_sha256": document["compatibility_sha256"],
        "n_references_processed": document["n_references_processed"],
        "n_candidate_comparisons": document["n_candidate_comparisons"],
        "n_matches": document["n_matches"],
        "status": "COMPLETE",
    }


def _snapshot_protected_match_checkpoint(context: UCleanContext, results: Dict[int, dict]) -> None:
    """Write match bookkeeping once, after every shard has a verified file.

    Resume does not read this snapshot. match-*.json is the source of truth,
    so a crash before this write still reuses committed shards.
    """
    records = [_match_shard_record(context, results[ordinal]) for ordinal in sorted(results)]
    directory = _protected_match_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        directory / "manifest.json",
        json.dumps({"shards": records}, ensure_ascii=False, indent=2, sort_keys=True),
    )
    state = _read_checkpoint_state(context)
    state["protected_match_shards"] = records
    _write_checkpoint_state(context, state)


def _write_match_progress(context: UCleanContext, progress: dict) -> None:
    directory = _protected_match_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        directory / "progress.json",
        json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True),
    )


def _format_match_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    remaining = int(max(0, round(seconds)))
    hours, rem = divmod(remaining, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "%dh %dm" % (hours, minutes)
    if minutes:
        return "%dm %ds" % (minutes, secs)
    return "%ds" % secs


def _emit_match_progress(
    context: UCleanContext,
    *,
    done: int,
    total: int,
    references: int,
    total_references: int,
    matches: int,
    started: float,
    computed_references: int,
    status: str = "running",
    workers: int = 0,
    reused: int = 0,
    submitted: int = 0,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = references / elapsed
    if computed_references > 0 and total_references >= references:
        remaining = max(total_references - references, 0)
        eta = remaining / (computed_references / elapsed)
    else:
        eta = None
    progress = {
        "status": status,
        "workers": int(workers),
        "total_shards": int(total),
        "completed_shards": int(done),
        "reused_shards": int(reused),
        "submitted_shards": int(submitted),
        "references_processed": int(references),
        "matches": int(matches),
        "elapsed_seconds": round(elapsed, 6),
        "eta_seconds": None if eta is None else round(eta, 6),
        "n_match_shards_completed": int(done),
        "n_match_shards_total": int(total),
        "n_matches": int(matches),
        "n_references_processed": int(references),
        "n_references_total": int(total_references),
        "refs_per_second": round(rate, 6),
    }
    _write_match_progress(context, progress)
    print(
        "Protected match: %d/%d shards\nstatus: %s\nworkers: %d\nsubmitted: %d\nreused: %d\nreferences: %d/%d\nmatches: %d\nelapsed: %.1fs\nrate: %.1f refs/s\nETA: %s"
        % (
            done, total, status, workers, submitted, reused,
            references, total_references, matches, elapsed, rate, _format_match_eta(eta),
        ),
        flush=True,
    )


def _init_protected_match_worker(index, overlap, eligible, ready_barrier=None) -> None:
    _PROTECTED_MATCH_RUNTIME["index"] = index
    _PROTECTED_MATCH_RUNTIME["overlap"] = overlap
    _PROTECTED_MATCH_RUNTIME["eligible"] = eligible
    _PROTECTED_MATCH_RUNTIME["ready_barrier"] = ready_barrier
    _PROTECTED_MATCH_RUNTIME["barrier_passed"] = False


def _match_ready_barrier(workers: int):
    """Optional test latch. It is not a scientific contract field."""
    if os.environ.get("BAHNAR_NB11_MATCH_READY_BARRIER", "").strip() != "1":
        return None
    import multiprocessing
    return multiprocessing.Barrier(int(workers))


def _match_worker_result(job: dict, computed: dict) -> dict:
    return {
        "ordinal": int(job["ordinal"]),
        "input_shard_name": job["input_shard_name"],
        "input_shard_sha256": job["input_shard_sha256"],
        "input_batch_sha256": job["input_batch_sha256"],
        "n_references_processed": computed["n_references_processed"],
        "n_candidate_comparisons": computed["n_candidate_comparisons"],
        "n_matches": computed["n_matches"],
        "evidence": computed["evidence"],
        "worker_pid": os.getpid(),
    }


def _run_protected_match_worker(job: dict) -> dict:
    barrier = _PROTECTED_MATCH_RUNTIME.get("ready_barrier")
    if barrier is not None and not _PROTECTED_MATCH_RUNTIME.get("barrier_passed"):
        barrier.wait()
        _PROTECTED_MATCH_RUNTIME["barrier_passed"] = True
    computed = compute_protected_matches(
        _PROTECTED_MATCH_RUNTIME["index"],
        job["batch"],
        _PROTECTED_MATCH_RUNTIME["overlap"],
        _PROTECTED_MATCH_RUNTIME["eligible"],
    )
    return _match_worker_result(job, computed)


def _compute_protected_match_job(job, index, overlap, eligible) -> dict:
    computed = compute_protected_matches(index, job["batch"], overlap, eligible)
    return _match_worker_result(job, computed)


def _protected_match_max_inflight(workers: int) -> int:
    return max(1, int(workers) * 2)


def _completed_match_futures(inflight: Dict[object, dict]):
    done, _pending = wait(set(inflight), return_when=FIRST_COMPLETED)
    return list(done)


def _shutdown_match_executor(executor, inflight, *, wait_for_running: bool) -> None:
    for future in list(inflight):
        future.cancel()
    inflight.clear()
    executor.shutdown(wait=wait_for_running, cancel_futures=True)


def _note_match_pull(pulled: int) -> None:
    global _LAST_MATCH_PULLED
    _LAST_MATCH_PULLED = int(pulled)


def _protected_match_document(base: dict, payload: dict) -> dict:
    compatibility = _shard_match_compatibility(
        base, payload["input_shard_sha256"], payload["input_batch_sha256"],
    )
    evidence_rows = _evidence_rows(payload["evidence"])
    result_sha = _result_sha256(evidence_rows)
    compatibility_sha = _compatibility_sha256(compatibility)
    return {
        "kind": _PROTECTED_MATCH_KIND,
        "status": "COMPLETE",
        "ordinal": int(payload["ordinal"]),
        "input_shard_name": payload["input_shard_name"],
        "input_shard_sha256": payload["input_shard_sha256"],
        "input_batch_sha256": payload["input_batch_sha256"],
        "overlap_contract_sha256": compatibility["overlap_contract_sha256"],
        "u_segment_uid_sha256": compatibility["u_segment_uid_sha256"],
        "u_fingerprint_identity_sha256": compatibility["u_fingerprint_identity_sha256"],
        "compatibility": compatibility,
        "compatibility_sha256": compatibility_sha,
        "result_sha256": result_sha,
        "n_references_processed": int(payload["n_references_processed"]),
        "n_candidate_comparisons": int(payload["n_candidate_comparisons"]),
        "n_matches": int(payload["n_matches"]),
        "evidence": evidence_rows,
    }


def _commit_protected_match_shard(context: UCleanContext, document: dict, batch_length: int) -> dict:
    """Persist one verified result. The match file itself is the resume record."""
    path = _protected_match_path(context, int(document["ordinal"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    loaded = _read_protected_match_shard(
        context, int(document["ordinal"]), document["compatibility"], batch_length,
    )
    if loaded is None or loaded.get("result_sha256") != document["result_sha256"]:
        raise RuntimeError("protected match shard failed verification: %s" % path.name)
    if loaded.get("status") != "COMPLETE":
        raise RuntimeError("protected match shard was not complete: %s" % path.name)
    return loaded


def _accept_reused_match(stats: dict, ordinal: int, document: dict, batch_length: int) -> None:
    stats["results"][int(ordinal)] = document
    stats["reused"] += 1
    stats["references"] += int(batch_length)
    stats["matches"] += int(document["n_matches"])
    stats["comparisons"] += int(document["n_candidate_comparisons"])


def _accept_computed_match(stats: dict, payload: dict, batch_length: int, context, base) -> None:
    stats["worker_pids"].append(int(payload.pop("worker_pid")))
    ordinal = int(payload["ordinal"])
    document = _protected_match_document(base, payload)
    committed = _commit_protected_match_shard(context, document, batch_length)
    stats["results"][ordinal] = committed
    stats["computed"] += 1
    stats["references"] += int(batch_length)
    stats["matches"] += int(committed["n_matches"])
    stats["comparisons"] += int(committed["n_candidate_comparisons"])
    stats["computed_references"] += int(batch_length)


def _protected_metadata_reference_total(state: dict) -> int:
    """Reference count from checkpoint UID lists. Parquet files are not opened."""
    return sum(len(shard.get("uids") or []) for shard in state.get("protected_fingerprint_shards") or [])


def _stream_match_progress(context, stats, *, started, total_meta, total_references, workers, status) -> None:
    _emit_match_progress(
        context,
        done=len(stats["results"]),
        total=total_meta,
        references=stats["references"],
        total_references=total_references,
        matches=stats["matches"],
        started=started,
        computed_references=stats["computed_references"],
        status=status,
        workers=workers,
        reused=stats["reused"],
        submitted=stats["submitted"],
    )


def match_protected_references_resumable(
    segments: List[dict],
    index: SegmentCandidateIndex,
    context: UCleanContext,
    *,
    workers: Optional[int] = None,
) -> dict:
    """Stream protected shards into a bounded worker window, then merge by ordinal.

    The process pool is opened before the protected-shard iterator is consumed.
    Workers only compute evidence. Flags and exclusions are applied after the
    results are ordered by shard ordinal. Worker count cannot change the result.
    """
    global _LAST_MATCH_WORKER_PIDS
    global _LAST_MATCH_MAX_INFLIGHT
    global _LAST_MATCH_PULLED
    global _LAST_MATCH_PULLED_AT_POOL_START
    global _LAST_MATCH_PULLED_AT_FIRST_SUBMIT
    if not context.config.overlap_config_frozen:
        raise RuntimeError("overlap protection requires a frozen OverlapConfig")
    started = time.perf_counter()
    worker_count = protected_match_worker_count(workers)
    by_uid = {str(row["segment_uid"]): row for row in segments}
    eligible = _eligible_protected_match_uids(by_uid)
    base = _base_match_compatibility(context, index)
    shard_state = _read_checkpoint_state(context)
    total_meta = len(shard_state["protected_fingerprint_shards"])
    total_references = _protected_metadata_reference_total(shard_state)
    stats = {
        "results": {},
        "reused": 0,
        "computed": 0,
        "submitted": 0,
        "references": 0,
        "matches": 0,
        "comparisons": 0,
        "computed_references": 0,
        "worker_pids": [],
        "jobs": 0,
    }
    _LAST_MATCH_PULLED = 0
    _LAST_MATCH_PULLED_AT_POOL_START = 0
    _LAST_MATCH_PULLED_AT_FIRST_SUBMIT = 0
    _LAST_MATCH_MAX_INFLIGHT = 0
    _stream_match_progress(
        context, stats, started=started, total_meta=total_meta, total_references=total_references,
        workers=worker_count, status="running",
    )
    units = _iter_protected_match_units(context, base)
    overlap = context.config.overlap

    def consume_reuse(unit) -> None:
        _accept_reused_match(stats, unit["ordinal"], unit["document"], unit["batch_length"])
        _stream_match_progress(
            context, stats, started=started, total_meta=total_meta, total_references=total_references,
            workers=worker_count, status="running",
        )

    def consume_payload(payload, batch_length) -> None:
        _accept_computed_match(stats, payload, batch_length, context, base)
        _stream_match_progress(
            context, stats, started=started, total_meta=total_meta, total_references=total_references,
            workers=worker_count, status="running",
        )

    if worker_count <= 1:
        for unit in units:
            _note_match_pull(stats["jobs"] + 1)
            stats["jobs"] += 1
            if unit["action"] == "reuse":
                consume_reuse(unit)
                continue
            stats["submitted"] += 1
            payload = _compute_protected_match_job(unit["job"], index, overlap, eligible)
            consume_payload(payload, len(unit["job"]["batch"]))
    else:
        max_inflight = _protected_match_max_inflight(worker_count)
        _LAST_MATCH_PULLED_AT_POOL_START = int(stats["jobs"])
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_protected_match_worker,
            initargs=(index, overlap, eligible, _match_ready_barrier(worker_count)),
        )
        inflight = {}
        source = iter(units)
        exhausted = False
        closed = False

        def close_pool(wait_for_running: bool) -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            _shutdown_match_executor(executor, inflight, wait_for_running=wait_for_running)

        def submit_job(job) -> None:
            global _LAST_MATCH_MAX_INFLIGHT
            global _LAST_MATCH_PULLED_AT_FIRST_SUBMIT
            if stats["submitted"] == 0:
                _LAST_MATCH_PULLED_AT_FIRST_SUBMIT = int(stats["jobs"])
            future = executor.submit(_run_protected_match_worker, job)
            inflight[future] = job
            stats["submitted"] += 1
            if len(inflight) > _LAST_MATCH_MAX_INFLIGHT:
                _LAST_MATCH_MAX_INFLIGHT = len(inflight)

        def pull_available() -> None:
            nonlocal exhausted
            while len(inflight) < max_inflight and not exhausted:
                try:
                    unit = next(source)
                except StopIteration:
                    exhausted = True
                    return
                stats["jobs"] += 1
                _note_match_pull(stats["jobs"])
                if unit["action"] == "reuse":
                    consume_reuse(unit)
                    continue
                submit_job(unit["job"])

        try:
            pull_available()
            while inflight:
                for future in _completed_match_futures(inflight):
                    job = inflight.pop(future)
                    consume_payload(future.result(), len(job["batch"]))
                pull_available()
        except BaseException:
            close_pool(True)
            raise
        else:
            close_pool(True)
    evidence: List[MatchEvidence] = []
    for ordinal in sorted(stats["results"]):
        for row in stats["results"][ordinal]["evidence"]:
            evidence.append(_match_evidence_from_row(row))
    apply_protected_match_evidence(evidence, by_uid)
    _snapshot_protected_match_checkpoint(context, stats["results"])
    _LAST_MATCH_WORKER_PIDS = list(stats["worker_pids"])
    _stream_match_progress(
        context, stats, started=started, total_meta=total_meta, total_references=total_references,
        workers=worker_count, status="complete",
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    benchmark = {
        "workers": worker_count,
        "n_match_shards_total": int(stats["jobs"]),
        "n_match_shards_reused": int(stats["reused"]),
        "n_match_shards_computed": int(stats["computed"]),
        "n_references_processed": int(stats["references"]),
        "n_candidate_comparisons": int(stats["comparisons"]),
        "n_matches": len(evidence),
        "elapsed_seconds": round(elapsed, 6),
        "refs_per_second": round(stats["references"] / elapsed, 6),
    }
    return {
        "evidence": evidence,
        "benchmark": benchmark,
        "n_references_processed": stats["references"],
    }


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
        sha_available = _is_valid_sha256(str(entry.sha256_pcm or entry.source_sha256 or ""))
        resolvable = False
        matches = False
        try:
            if _entry_can_reconstruct(entry) and hasattr(resolver, "materialize_audio"):
                with resolver.materialize_audio(entry) as item:
                    resolvable = Path(item.path).is_file()
                    matches = sha_available and item.sha256_pcm == str(entry.sha256_pcm or entry.source_sha256).strip().lower()
                    if not sha_available:
                        matches = _is_valid_sha256(item.sha256_pcm)
            else:
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


def _u_fingerprint_indexable(fingerprint, overlap_config) -> bool:
    """A U fingerprint can enter a matcher only when it can form a legal alignment."""
    return is_valid_fingerprint(fingerprint) and len(fingerprint) >= int(overlap_config.min_overlap_items)


def _fingerprint_coverage_complete(segments: Sequence[dict], fingerprints_by_uid: Dict[str, Sequence[int]], config: UCleanConfig) -> bool:
    """Every segment that still requires perceptual protection has a long-enough fingerprint."""
    min_items = int(config.overlap.min_overlap_items)
    for row in segments:
        status = row["u_clean_status"]
        if status == RETAINED_STATUS or status in (EXCLUDED_PERCEPTUAL_DUPLICATE, EXCLUDED_PROTECTED_OVERLAP):
            fp = fingerprints_by_uid.get(row["segment_uid"])
            if not is_valid_fingerprint(fp) or len(fp) < min_items:
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

    segment_coverage = _fingerprint_coverage_complete(segments, fps, context.config)
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
            raise RuntimeError(
                "full run requires every protected reference to be accounted for "
                "(fingerprint or SHORT_NOT_PERCEPTUALLY_ELIGIBLE)"
            )
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
        n_samples = int(row["end_sample"]) - int(row["start_sample"])
        expected = n_samples / float(seg_cfg.sample_rate)
        if abs(expected - duration) > 1e-6:
            raise RuntimeError(f"segment {uid} duration arithmetic inconsistent")
        if not _duration_reaches_perceptual_minimum(n_samples, int(seg_cfg.sample_rate)):
            raise RuntimeError(
                f"retained segment {uid} is below U perceptual eligibility "
                f"duration_samples={n_samples} required_seconds={U_PERCEPTUAL_MIN_DURATION_SECONDS}"
            )
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
    cols = [
        "segment_uid", "source_id", "start_sample", "end_sample", "duration_seconds",
        "u_clean_status", "exclusion_reason", "canonical_segment_uid",
        "u_perceptual_min_duration_seconds", "perceptual_eligibility_min_overlap_items",
    ]
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
        "n_u_perceptual_ineligible": counts.get(EXCLUDED_PERCEPTUAL_INELIGIBLE, 0),
        "u_perceptual_eligibility": {
            "n_u_perceptual_ineligible": counts.get(EXCLUDED_PERCEPTUAL_INELIGIBLE, 0),
            "u_perceptual_min_duration_seconds": U_PERCEPTUAL_MIN_DURATION_SECONDS,
            "min_overlap_items": int(cfg.overlap.min_overlap_items),
            "policy": U_PERCEPTUAL_POLICY,
            "fpcalc_version": PROTECTED_PERCEPTUAL_FPCALC_VERSION,
        },
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
        "rq1_final_contract_hash": str(context.reference_provenance.get("rq1_final_contract_hash") or ""),
        "rq1_test_contract_hash": str(context.reference_provenance.get("rq1_test_contract_hash") or ""),
        "protected_audio_identity_sha256": str(context.reference_provenance.get("protected_audio_identity_sha256") or ""),
        "parquet_revision": str(context.reference_provenance.get("parquet_revision") or ""),
        "pcm_pipeline_version": str(context.reference_provenance.get("pcm_pipeline_version") or ""),
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
        path = resolve_u_clean_path(rel, cfg)
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

    A valid cached WAV is reused without rewrite. If the in-memory WAV SHA is
    blank after resume, it is hydrated from the physical file only after PCM
    identity (and duration) already match. A cached path whose PCM or existing
    WAV SHA disagrees is not blessed; the canonical source is sliced again.
    A single new write failure excludes that row. Completion is true only when
    every row that is still retained has a verified path and WAV SHA.
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
        if not existing:
            candidate = "segments/%s/%s.wav" % (row["source_id"], row["segment_uid"])
            if resolve_u_clean_path(candidate, cfg).is_file():
                existing = candidate
                row["segment_local_path"] = candidate
        if existing:
            existing_path = resolve_u_clean_path(existing, cfg)
            reused = False
            try:
                info = read_wav_pcm16(existing_path)
                pcm_ok = info["pcm16_sha256"] == row.get("segment_pcm16_sha256")
                expected_wav = str(row.get("segment_wav_sha256") or "")
                wav_ok = (info["wav_sha256"] == expected_wav) if expected_wav else True
                expected_samples = int(row["end_sample"]) - int(row["start_sample"])
                samples_ok = info["n_samples"] == expected_samples
                duration_ok = abs(
                    info["n_samples"] / float(info["sample_rate"]) - float(row["duration_seconds"])
                ) <= 1e-6
                reused = bool(pcm_ok and wav_ok and samples_ok and duration_ok)
                if reused and not expected_wav:
                    row["segment_wav_sha256"] = info["wav_sha256"]
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
            row["segment_local_path"] = to_u_clean_relative(abs_path, cfg)
            row["segment_wav_sha256"] = wav_sha
        except Exception:
            _exclude(row, EXCLUDED_SEGMENT_WRITE_FAILED)
    unverified = [
        row for row in retained_segments(segments)
        if not row.get("segment_local_path") or not row.get("segment_wav_sha256")
    ]
    context.completion.segments_written_verified = not unverified
    if cfg.run_full_pipeline and unverified:
        raise RuntimeError("full run requires verified segment files")


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
        "protected_short_eligibility": {},
        "protected_exact_identity": {},
        "protected_materialized_pcm": {},
        "protected_match_shards": [],
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
    state.setdefault("protected_short_eligibility", {})
    state.setdefault("protected_exact_identity", {})
    state.setdefault("protected_materialized_pcm", {})
    state.setdefault("protected_match_shards", [])
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
    """Load sharded resume state. A bad shard is dropped; a stale contract fails closed.

    ``load_protected_fingerprints=False`` is the NB11 production resume mode.
    It loads segments, segment fingerprints, and source status, and leaves
    protected fingerprint parquet files unread. Those shards are verified
    later, one at a time, inside protected matching. The default remains an
    eager protected verification for callers that require it.
    """
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
    fingerprint_kinds = [
        ("segment_fingerprints", "segment_fingerprint_shards", "discarded_segment_shards", "discarded_segment_fingerprints"),
    ]
    if load_protected_fingerprints:
        fingerprint_kinds.append(
            ("protected_fingerprints", "protected_fingerprint_shards", "discarded_reference_shards", "discarded_reference_fingerprints"),
        )
    for kind, key, discard_names, discard_count in fingerprint_kinds:
        kept = kept_fp if kind == "segment_fingerprints" else None
        bucket = []
        for shard in state[key]:
            file_path = root / kind / shard["name"]
            try:
                if kind == "protected_fingerprints":
                    rows = _load_verified_protected_rows(
                        context, shard, file_path, keep_arrays=load_protected_fingerprints,
                    )
                    if rows is None:
                        raise RuntimeError("hash")
                else:
                    if hashlib.sha256(file_path.read_bytes()).hexdigest() != shard["sha256"]:
                        raise RuntimeError("hash")
                    rows = _load_fingerprint_shard(file_path, kind)
                if kind == "segment_fingerprints":
                    allowed = {str(uid) for uid in (shard.get("uids") or [])}
                    for row in rows:
                        if allowed and str(row["uid"]) not in allowed:
                            continue
                        loaded["segment_fingerprints_by_uid"][row["uid"]] = row["fingerprint"]
                        loaded["segment_fingerprint_audio_sha256"][row["uid"]] = row.get("audio_sha256") or ""
                    kept_fp.append(shard)
                else:
                    expected = {
                        entry.reference_uid: str(entry.sha256_pcm or entry.source_sha256 or "").strip().lower()
                        for entry in context.protected_entries
                    }
                    matched = []
                    min_items = int(context.config.overlap.min_overlap_items)
                    for row in rows:
                        fingerprint = row.get("fingerprint")
                        if is_valid_fingerprint(fingerprint):
                            n_items = len(fingerprint)
                        else:
                            n_items = int(row.get("n_items") or 0)
                        if n_items < min_items:
                            loaded["discarded_reference_fingerprints"] += 1
                            continue
                        stored = str(row.get("source_sha256") or "").strip().lower()
                        wanted = expected.get(row["uid"]) if expected else None
                        if not _is_valid_sha256(stored) or (wanted and wanted != stored):
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


# fpcalc 1.5.1 cannot emit min_overlap_items=20 below this duration.
# These are NB11 eligibility rules. They are not segmentation or overlap parameters.
_FPCALC_MIN_OVERLAP_DURATION_SECONDS = 5.1
PROTECTED_PERCEPTUAL_MIN_DURATION_SECONDS = _FPCALC_MIN_OVERLAP_DURATION_SECONDS
U_PERCEPTUAL_MIN_DURATION_SECONDS = _FPCALC_MIN_OVERLAP_DURATION_SECONDS
PROTECTED_PERCEPTUAL_POLICY = "protected_perceptual_min_duration_v1"
U_PERCEPTUAL_POLICY = "u_perceptual_min_duration_v1"
PROTECTED_PERCEPTUAL_FPCALC_VERSION = "1.5.1"
SHORT_NOT_PERCEPTUALLY_ELIGIBLE = "SHORT_NOT_PERCEPTUALLY_ELIGIBLE"
PROTECTED_EXACT_POLICY = "protected_exact_pcm_v1"
PROTECTED_EXACT_MATCH = "protected_exact"
_SHORT_ELIGIBILITY_FILENAME = "protected_short_eligibility.jsonl"
_EXACT_IDENTITY_FILENAME = "protected_exact_identity.jsonl"
_MATERIALIZED_PCM_FILENAME = "protected_materialized_pcm.json"
_MATERIALIZED_PCM_SCHEMA = "rq2-protected-materialized-pcm-1"
_SHORT_WRITE_BATCH = 32


def _duration_reaches_perceptual_minimum(n_samples: int, sample_rate: int) -> bool:
    """True when duration >= 5.1s. Compared in integers so the 5.1 boundary is exact."""
    return int(n_samples) * 10 >= int(sample_rate) * 51


def _actual_wav_dimensions(path) -> dict:
    """Read frames and rate from a reconstructed WAV. Zero metadata is not used."""
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            sample_rate = int(handle.getframerate() or 0)
            n_samples = int(handle.getnframes() or 0)
    except Exception as exc:
        raise RuntimeError(f"invalid reconstructed protected audio: {path}") from exc
    if channels != 1 or width != 2 or sample_rate <= 0 or n_samples <= 0:
        raise RuntimeError(
            f"invalid reconstructed protected audio: rate={sample_rate} samples={n_samples} "
            f"channels={channels} width={width}"
        )
    return {
        "actual_n_samples": n_samples,
        "actual_sample_rate": sample_rate,
        "actual_duration_seconds": round(n_samples / float(sample_rate), 6),
    }


def _verify_pinned_wav_dimensions(entry: ProtectedReferenceEntry, dims: dict) -> None:
    """Nonzero pinned metadata must match the WAV. Zero means the metadata is unavailable."""
    if int(entry.n_samples or 0) and int(entry.n_samples) != int(dims["actual_n_samples"]):
        raise RuntimeError(
            f"n_samples mismatch for {entry.reference_uid}: "
            f"wav {dims['actual_n_samples']} != pinned {entry.n_samples}"
        )
    if int(entry.sample_rate or 0) and int(entry.sample_rate) != int(dims["actual_sample_rate"]):
        raise RuntimeError(
            f"sample_rate mismatch for {entry.reference_uid}: "
            f"wav {dims['actual_sample_rate']} != pinned {entry.sample_rate}"
        )


def _short_eligibility_path(context: UCleanContext) -> Path:
    return checkpoint_root(context) / _SHORT_ELIGIBILITY_FILENAME


def _load_short_eligibility_records(context: UCleanContext) -> List[dict]:
    """Load audited short references. A policy or hash mismatch drops only this file."""
    state = _read_checkpoint_state(context)
    meta = dict(state.get("protected_short_eligibility") or {})
    path = _short_eligibility_path(context)
    if not path.is_file() or not meta.get("sha256"):
        return []
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != str(meta.get("sha256") or ""):
        return []
    if str(meta.get("policy") or "") != PROTECTED_PERCEPTUAL_POLICY:
        return []
    if float(meta.get("protected_perceptual_min_duration_seconds") or 0) != PROTECTED_PERCEPTUAL_MIN_DURATION_SECONDS:
        return []
    if int(meta.get("min_overlap_items") or 0) != int(context.config.overlap.min_overlap_items):
        return []
    rows = []
    seen = set()
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = str(row.get("reference_uid") or "")
        if not uid or uid in seen:
            raise RuntimeError(f"duplicate protected short-eligibility uid: {uid}")
        if str(row.get("status") or "") != SHORT_NOT_PERCEPTUALLY_ELIGIBLE:
            raise RuntimeError(f"unexpected protected eligibility status for {uid}")
        if float(row.get("protected_perceptual_min_duration_seconds") or 0) != PROTECTED_PERCEPTUAL_MIN_DURATION_SECONDS:
            return []
        if str(row.get("fpcalc_version") or "") != PROTECTED_PERCEPTUAL_FPCALC_VERSION:
            raise RuntimeError(
                f"fpcalc version {row.get('fpcalc_version')!r} is outside {PROTECTED_PERCEPTUAL_POLICY}; "
                f"the 5.1s boundary was calibrated on fpcalc {PROTECTED_PERCEPTUAL_FPCALC_VERSION}"
            )
        if not _is_valid_sha256(str(row.get("sha256_pcm") or "")):
            return []
        seen.add(uid)
        rows.append(row)
    if str(meta.get("fpcalc_version") or "") not in ("", PROTECTED_PERCEPTUAL_FPCALC_VERSION):
        raise RuntimeError(
            f"fpcalc version {meta.get('fpcalc_version')!r} is outside {PROTECTED_PERCEPTUAL_POLICY}"
        )
    return rows


def _write_short_eligibility_records(context: UCleanContext, rows: Sequence[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (str(row.get("split") or ""), str(row.get("reference_uid") or "")))
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
    path = _short_eligibility_path(context)
    atomic_write_text(path, payload)
    state = _read_checkpoint_state(context)
    state["protected_short_eligibility"] = {
        "path": _SHORT_ELIGIBILITY_FILENAME,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "policy": PROTECTED_PERCEPTUAL_POLICY,
        "protected_perceptual_min_duration_seconds": PROTECTED_PERCEPTUAL_MIN_DURATION_SECONDS,
        "min_overlap_items": int(context.config.overlap.min_overlap_items),
        "fpcalc_version": PROTECTED_PERCEPTUAL_FPCALC_VERSION,
        "n_records": len(ordered),
        "uids": [str(row["reference_uid"]) for row in ordered],
    }
    _write_checkpoint_state(context, state)


def _protected_identity_key(split, uid) -> Tuple[str, str]:
    return (str(split or ""), str(uid or ""))


def _bind_protected_identity(expected: Dict[Tuple[str, str], str], kind: str, split, uid) -> Tuple[str, str]:
    """Fail closed unless (split, uid) is a current protected entry.

    A valid key is a direct mapping lookup. The full expected set is scanned
    only to explain an unknown UID or a split mismatch.
    """
    key = _protected_identity_key(split, uid)
    if not key[1]:
        raise RuntimeError(f"unknown protected {kind} uid")
    if key in expected:
        return key
    owners = [item for item in expected if item[1] == key[1]]
    if not owners:
        raise RuntimeError(f"unknown protected {kind} uid {key[1]}")
    raise RuntimeError(
        f"protected {kind} split mismatch for {key[1]}: "
        f"stored {key[0]} != entry {owners[0][0]}"
    )


def _protected_uid_owner_index(expected: Dict[Tuple[str, str], ProtectedReferenceEntry]) -> Dict[str, List[Tuple[str, str]]]:
    """Map each UID to its (split, uid) keys. Built once per identity pass."""
    owners: Dict[str, List[Tuple[str, str]]] = {}
    for key in expected:
        owners.setdefault(key[1], []).append(key)
    return owners


def _expected_protected_index(context: UCleanContext) -> Dict[Tuple[str, str], ProtectedReferenceEntry]:
    expected: Dict[Tuple[str, str], ProtectedReferenceEntry] = {}
    for entry in context.protected_entries:
        key = _protected_identity_key(entry.split, entry.reference_uid)
        if not key[1] or key in expected:
            raise RuntimeError(f"duplicate protected identity {key[0]}:{key[1]}")
        expected[key] = entry
    return expected


def _protected_short_hashes(context: UCleanContext, expected: Dict[Tuple[str, str], ProtectedReferenceEntry]) -> Dict[Tuple[str, str], str]:
    shorts: Dict[Tuple[str, str], str] = {}
    for row in _load_short_eligibility_records(context):
        key = _bind_protected_identity(expected, "short-eligibility", row.get("split"), row.get("reference_uid"))
        if key in shorts:
            raise RuntimeError(f"duplicate protected accounting for {key[0]}:{key[1]}")
        digest = str(row.get("sha256_pcm") or "").strip().lower()
        if not _is_valid_sha256(digest):
            raise RuntimeError(f"short protected reference {key[1]} has no materialized sha256_pcm")
        shorts[key] = digest
    return shorts


def _protected_pcm_index(
    context: UCleanContext,
    fingerprinted: Dict[Tuple[str, str], str],
    shorts: Dict[Tuple[str, str], str],
    expected: Dict[Tuple[str, str], ProtectedReferenceEntry],
) -> dict:
    overlap = [key for key in fingerprinted if key in shorts]
    if overlap:
        key = overlap[0]
        raise RuntimeError(f"duplicate protected accounting for {key[0]}:{key[1]}")
    min_items = int(context.config.overlap.min_overlap_items)
    unaccounted = [key for key in expected if key not in fingerprinted and key not in shorts]
    return {
        "expected": expected,
        "fingerprinted": fingerprinted,
        "shorts": shorts,
        "unaccounted": unaccounted,
        "min_overlap_items": min_items,
    }


def _indexed_protected_pcm(context: UCleanContext) -> dict:
    """PCM hashes from verified fingerprint rows. Wrong identities fail closed."""
    expected = _expected_protected_index(context)
    min_items = int(context.config.overlap.min_overlap_items)
    fingerprinted: Dict[Tuple[str, str], str] = {}
    for row in _iter_protected_accounting_rows(context):
        uid = str(row.get("uid") or "")
        n_items = int(row.get("n_items") or 0)
        if n_items <= 0:
            raise RuntimeError(f"invalid protected fingerprint uid: {uid}")
        if n_items < min_items:
            raise RuntimeError(
                f"protected fingerprint shorter than min_overlap_items for {uid}: "
                f"items={n_items} required={min_items}"
            )
        key = _bind_protected_identity(expected, "fingerprint", row.get("split"), uid)
        if key in fingerprinted:
            raise RuntimeError(f"duplicate protected fingerprint identity {key[0]}:{key[1]}")
        digest = str(row.get("source_sha256") or "").strip().lower()
        if not _is_valid_sha256(digest):
            raise RuntimeError(f"protected fingerprint {uid} has no materialized sha256_pcm")
        fingerprinted[key] = digest
    shorts = _protected_short_hashes(context, expected)
    return _protected_pcm_index(context, fingerprinted, shorts, expected)


def _materialized_pcm_path(context: UCleanContext) -> Path:
    return checkpoint_root(context) / _MATERIALIZED_PCM_FILENAME


def _load_materialized_pcm_index(context: UCleanContext) -> Dict[Tuple[str, str], str]:
    """PCM hashes recovered from an earlier shard fallback.

    The ledger is trusted only when its file hash, schema, and checkpoint
    compatibility match. A missing or stale ledger is ignored so the shard
    fallback can rebuild it.
    """
    state = _read_checkpoint_state(context)
    meta = dict(state.get("protected_materialized_pcm") or {})
    path = _materialized_pcm_path(context)
    expected_sha = str(meta.get("sha256") or "")
    if not path.is_file() or not _is_valid_sha256(expected_sha):
        return {}
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        return {}
    if str(meta.get("schema_version") or "") != _MATERIALIZED_PCM_SCHEMA:
        return {}
    try:
        document = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    if str(document.get("schema_version") or "") != _MATERIALIZED_PCM_SCHEMA:
        return {}
    if document.get("compatibility_key") != compatibility_key(context):
        return {}
    audio_identity = str(context.reference_provenance.get("protected_audio_identity_sha256") or "")
    if str(document.get("protected_audio_identity_sha256") or "") != audio_identity:
        return {}
    found: Dict[Tuple[str, str], str] = {}
    for row in document.get("records") or []:
        if not isinstance(row, dict):
            return {}
        key = _protected_identity_key(row.get("split"), row.get("uid"))
        digest = str(row.get("sha256_pcm") or "").strip().lower()
        if not key[1] or key in found or not _is_valid_sha256(digest):
            return {}
        found[key] = digest
    return found


def _write_materialized_pcm_index(context: UCleanContext, hashes: Dict[Tuple[str, str], str]) -> None:
    """Atomically store one recovery batch. state.json is rewritten once."""
    document = {
        "schema_version": _MATERIALIZED_PCM_SCHEMA,
        "compatibility_key": compatibility_key(context),
        "protected_audio_identity_sha256": str(
            context.reference_provenance.get("protected_audio_identity_sha256") or ""
        ),
        "records": [
            {"split": split, "uid": uid, "sha256_pcm": digest}
            for (split, uid), digest in sorted(hashes.items())
        ],
    }
    path = _materialized_pcm_path(context)
    atomic_write_text(path, json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    state = _read_checkpoint_state(context)
    state["protected_materialized_pcm"] = {
        "path": _MATERIALIZED_PCM_FILENAME,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_version": _MATERIALIZED_PCM_SCHEMA,
        "n_records": len(hashes),
    }
    _write_checkpoint_state(context, state)


def _pinned_fingerprint_digest(entry: ProtectedReferenceEntry) -> str:
    return str(entry.sha256_pcm or entry.source_sha256 or "").strip().lower()


def _recover_pinned_pcm_from_shards(context: UCleanContext, groups: Dict[int, dict]) -> Dict[Tuple[str, str], str]:
    """Read source_audio_sha256 from each affected shard once."""
    root = checkpoint_root(context) / "protected_fingerprints"
    recovered: Dict[Tuple[str, str], str] = {}
    for ordinal in sorted(groups):
        group = groups[ordinal]
        shard = group["shard"]
        path = root / str(shard.get("name") or "")
        rows, reason = _match_shard_rows(context, shard, path, keep_arrays=False)
        if reason or rows is None:
            _fail_protected_match_shard(ordinal, shard, reason or "invalid logical batch")
        listed = {str(uid) for uid in (shard.get("uids") or [])}
        by_uid: Dict[str, dict] = {}
        for row in rows:
            uid = str(row.get("uid") or "")
            if uid in by_uid:
                raise RuntimeError(
                    f"duplicate protected fingerprint identity in shard {shard.get('name')}: {uid}"
                )
            by_uid[uid] = row
        for key, entry, uid in group["pending"]:
            if uid not in listed or uid not in by_uid or str(by_uid[uid].get("uid") or "") != uid:
                raise RuntimeError(f"protected fingerprint {uid} missing from shard {shard.get('name')}")
            row = by_uid[uid]
            stored_split = str(row.get("split") or "")
            if stored_split != str(entry.split):
                raise RuntimeError(
                    f"protected fingerprint split mismatch for {uid}: "
                    f"stored {stored_split} != entry {entry.split}"
                )
            digest = str(row.get("source_sha256") or "").strip().lower()
            if not _is_valid_sha256(digest):
                raise RuntimeError(f"protected fingerprint {uid} has invalid source_audio_sha256")
            recovered[key] = digest
    return recovered


def _pinned_protected_pcm(context: UCleanContext) -> dict:
    """Exact-PCM hashes from pinned identity, with a shard fallback for blank rows.

    A valid ``sha256_pcm`` or ``source_sha256`` is used directly and does not
    open parquet. A blank derived row whose UID is already fingerprinted
    recovers ``source_audio_sha256`` from that checkpoint shard. Shards are
    grouped, so each affected shard is verified once. Recovered hashes are
    written once to ``protected_materialized_pcm.json``. On a valid checkpoint
    the hash set matches ``_indexed_protected_pcm``.
    """
    expected = _expected_protected_index(context)
    owners_by_uid = _protected_uid_owner_index(expected)
    state = _read_checkpoint_state(context)
    ledger = _load_materialized_pcm_index(context)
    fingerprinted: Dict[Tuple[str, str], str] = {}
    groups: Dict[int, dict] = {}
    seen_uids = set()
    for ordinal, shard in enumerate(list(state["protected_fingerprint_shards"])):
        for raw_uid in shard.get("uids") or []:
            uid = str(raw_uid or "")
            owners = owners_by_uid.get(uid, [])
            if not owners:
                raise RuntimeError(f"unknown protected fingerprint uid {uid}")
            if len(owners) != 1:
                raise RuntimeError(f"ambiguous protected fingerprint uid {uid}")
            entry = expected[owners[0]]
            key = _protected_identity_key(entry.split, entry.reference_uid)
            if uid in seen_uids or key in fingerprinted:
                raise RuntimeError(f"duplicate protected fingerprint identity {key[0]}:{key[1]}")
            seen_uids.add(uid)
            digest = _pinned_fingerprint_digest(entry)
            if _is_valid_sha256(digest):
                fingerprinted[key] = digest
                continue
            recovered = ledger.get(key)
            if recovered and _is_valid_sha256(recovered):
                fingerprinted[key] = recovered
                continue
            group = groups.get(ordinal)
            if group is None:
                group = {"shard": shard, "pending": []}
                groups[ordinal] = group
            group["pending"].append((key, entry, uid))
    if groups:
        for key, digest in _recover_pinned_pcm_from_shards(context, groups).items():
            if key in fingerprinted:
                raise RuntimeError(f"duplicate protected fingerprint identity {key[0]}:{key[1]}")
            fingerprinted[key] = digest
            ledger[key] = digest
        _write_materialized_pcm_index(context, ledger)
    shorts = _protected_short_hashes(context, expected)
    return _protected_pcm_index(context, fingerprinted, shorts, expected)


def _protected_accounting_stats(indexed: dict) -> dict:
    total = len(indexed["expected"])
    fingerprinted = indexed["fingerprinted"]
    shorts = indexed["shorts"]
    unaccounted = indexed["unaccounted"]
    if len(fingerprinted) + len(shorts) + len(unaccounted) != total:
        raise RuntimeError("protected accounting identities are not a partition of the reference set")
    return {
        "policy": PROTECTED_PERCEPTUAL_POLICY,
        "protected_perceptual_min_duration_seconds": PROTECTED_PERCEPTUAL_MIN_DURATION_SECONDS,
        "fpcalc_version": PROTECTED_PERCEPTUAL_FPCALC_VERSION,
        "min_overlap_items": indexed["min_overlap_items"],
        "n_protected_total": total,
        "n_protected_perceptual_eligible": len(fingerprinted),
        "n_protected_fingerprinted": len(fingerprinted),
        "n_protected_short_not_perceptually_eligible": len(shorts),
        "n_protected_unaccounted": len(unaccounted),
    }


def protected_reference_accounting(context: UCleanContext) -> dict:
    """Count fingerprints and audited shorts by (split, reference_uid)."""
    return _protected_accounting_stats(_indexed_protected_pcm(context))


def protected_fingerprint_coverage_complete(context: UCleanContext) -> bool:
    """True when every protected UID is accounted for.

    Accounted means exactly one of: a persisted fingerprint with at least
    ``min_overlap_items`` values, or an audited SHORT_NOT_PERCEPTUALLY_ELIGIBLE
    record from the materialized WAV. The historical name is kept for the
    downstream success gate. Short rows are not fingerprints.
    """
    stats = protected_reference_accounting(context)
    provenance = dict(context.reference_provenance)
    provenance["protected_perceptual_eligibility"] = stats
    context.reference_provenance = provenance
    if not references_cover_splits(context.protected_entries):
        return False
    return (
        stats["n_protected_unaccounted"] == 0
        and stats["n_protected_fingerprinted"] + stats["n_protected_short_not_perceptually_eligible"] == stats["n_protected_total"]
    )


def protected_reference_uids(context: UCleanContext) -> set:
    """UIDs already stored in verified protected shards, without loading fingerprint arrays."""
    state = _read_checkpoint_state(context)
    return _shard_uids(state["protected_fingerprint_shards"])


def _remember_protected_eligibility(context: UCleanContext) -> dict:
    """Record eligibility from checkpoint metadata. Fingerprint parquet stays closed."""
    stats = _protected_accounting_stats(_pinned_protected_pcm(context))
    provenance = dict(context.reference_provenance)
    provenance["protected_perceptual_eligibility"] = stats
    context.reference_provenance = provenance
    return stats


def _short_eligibility_record(entry: ProtectedReferenceEntry, dims: dict, sha256_pcm: str, min_overlap_items: int) -> dict:
    return {
        "reference_uid": entry.reference_uid,
        "split": entry.split,
        "sha256_pcm": sha256_pcm,
        "actual_n_samples": int(dims["actual_n_samples"]),
        "actual_sample_rate": int(dims["actual_sample_rate"]),
        "actual_duration_seconds": float(dims["actual_duration_seconds"]),
        "status": SHORT_NOT_PERCEPTUALLY_ELIGIBLE,
        "reason": SHORT_NOT_PERCEPTUALLY_ELIGIBLE,
        "protected_perceptual_min_duration_seconds": PROTECTED_PERCEPTUAL_MIN_DURATION_SECONDS,
        "min_overlap_items": int(min_overlap_items),
        "fpcalc_version": PROTECTED_PERCEPTUAL_FPCALC_VERSION,
        "policy": PROTECTED_PERCEPTUAL_POLICY,
    }


def _require_compatible_fpcalc(context: UCleanContext) -> str:
    """The 5.1s boundary is valid for fpcalc 1.5.1. Another version fails closed."""
    from src.rq2_audio_fingerprint import fpcalc_version

    version = fpcalc_version(context.config.overlap.fpcalc_binary)
    if version != PROTECTED_PERCEPTUAL_FPCALC_VERSION:
        raise RuntimeError(
            f"fpcalc {version} is outside {PROTECTED_PERCEPTUAL_POLICY}; "
            f"protected perceptual eligibility was calibrated on fpcalc {PROTECTED_PERCEPTUAL_FPCALC_VERSION}"
        )
    return version


def generate_protected_fingerprints(context: UCleanContext, resolver, *, fingerprint_fn=None) -> int:
    """Fingerprint protected refs that are long enough for the frozen matcher.

    A materialized WAV shorter than 5.1 seconds is audited as
    SHORT_NOT_PERCEPTUALLY_ELIGIBLE and is not sent to fpcalc. Completed
    fingerprint shards and the short-eligibility file are reused. Segmentation
    and U fingerprint shards are not rewritten.
    """
    short_rows = _load_short_eligibility_records(context)
    short_uids = {str(row["reference_uid"]) for row in short_rows}
    fingerprinted = protected_reference_uids(context)
    overlap = sorted(short_uids & fingerprinted)
    if overlap:
        raise RuntimeError(f"duplicate protected accounting for {overlap[:5]}")
    pending = [
        entry for entry in context.protected_entries
        if entry.reference_uid not in fingerprinted and entry.reference_uid not in short_uids
    ]
    written = 0
    if pending:
        if fingerprint_fn is None:
            _require_compatible_fpcalc(context)
            from src.rq2_audio_fingerprint import compute_fingerprint

            def fingerprint_fn(path):  # type: ignore
                return compute_fingerprint(path, context.config.overlap)
        min_items = int(context.config.overlap.min_overlap_items)
        records = list(short_rows)
        pending_shorts: List[dict] = []
        batch: List[dict] = []

        def flush_shorts(force: bool = False) -> None:
            if not pending_shorts:
                return
            if not force and len(pending_shorts) < _SHORT_WRITE_BATCH:
                return
            records.extend(pending_shorts)
            pending_shorts.clear()
            _write_short_eligibility_records(context, records)

        for item in resolver.iter_materialized_audio(pending):
            entry = item.entry
            try:
                dims = _actual_wav_dimensions(item.path)
            except RuntimeError as exc:
                flush_shorts(force=True)
                raise RuntimeError(
                    f"invalid reconstructed protected audio for {entry.reference_uid} "
                    f"split={entry.split}: {exc}"
                ) from exc
            _verify_pinned_wav_dimensions(entry, dims)
            pcm_hash = str(item.sha256_pcm or "").strip().lower()
            if not _is_valid_sha256(pcm_hash):
                flush_shorts(force=True)
                raise RuntimeError(f"materialized protected audio has no sha256_pcm for {entry.reference_uid}")
            if not _duration_reaches_perceptual_minimum(dims["actual_n_samples"], dims["actual_sample_rate"]):
                pending_shorts.append(_short_eligibility_record(entry, dims, pcm_hash, min_items))
                flush_shorts()
                continue
            flush_shorts(force=True)
            try:
                fingerprint = fingerprint_fn(item.path)
            except Exception as exc:
                raise RuntimeError(
                    f"fpcalc failed for protected reference {entry.reference_uid} "
                    f"split={entry.split} duration={dims['actual_duration_seconds']}: {exc}"
                ) from exc
            n_items = len(fingerprint) if is_valid_fingerprint(fingerprint) else 0
            if n_items < min_items:
                raise RuntimeError(
                    f"protected fingerprint below min_overlap_items for {entry.reference_uid} "
                    f"split={entry.split} duration={dims['actual_duration_seconds']} "
                    f"items={n_items} required={min_items}"
                )
            batch.append({
                "uid": entry.reference_uid,
                "reference_uid": entry.reference_uid,
                "split": entry.split,
                "source_sha256": pcm_hash,
                "fingerprint": fingerprint,
            })
            if len(batch) >= 32:
                append_fingerprint_shard(context, "protected_fingerprints", batch)
                written += len(batch)
                batch = []
        flush_shorts(force=True)
        if batch:
            append_fingerprint_shard(context, "protected_fingerprints", batch)
            written += len(batch)
    _remember_protected_eligibility(context)
    return written


def _exact_identity_path(context: UCleanContext) -> Path:
    return checkpoint_root(context) / _EXACT_IDENTITY_FILENAME


def _load_exact_identity_records(context: UCleanContext) -> List[dict]:
    state = _read_checkpoint_state(context)
    meta = dict(state.get("protected_exact_identity") or {})
    path = _exact_identity_path(context)
    if not path.is_file() or not meta.get("sha256"):
        return []
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != str(meta.get("sha256") or ""):
        return []
    if str(meta.get("policy") or "") != PROTECTED_EXACT_POLICY:
        return []
    rows = []
    seen = set()
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = _protected_identity_key(row.get("split"), row.get("reference_uid"))
        if not key[1] or key in seen:
            raise RuntimeError(f"duplicate protected exact identity {key[0]}:{key[1]}")
        if str(row.get("policy") or "") != PROTECTED_EXACT_POLICY:
            return []
        if not _is_valid_sha256(str(row.get("sha256_pcm") or "")):
            return []
        seen.add(key)
        rows.append(row)
    return rows


def _write_exact_identity_records(context: UCleanContext, rows: Sequence[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (str(row.get("split") or ""), str(row.get("reference_uid") or "")))
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
    path = _exact_identity_path(context)
    atomic_write_text(path, payload)
    state = _read_checkpoint_state(context)
    state["protected_exact_identity"] = {
        "path": _EXACT_IDENTITY_FILENAME,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "policy": PROTECTED_EXACT_POLICY,
        "n_records": len(ordered),
        "identities": [f"{row['split']}:{row['reference_uid']}" for row in ordered],
    }
    _write_checkpoint_state(context, state)


def protected_exact_identity_complete(context: UCleanContext) -> bool:
    """True when every current protected identity has a matching exact-PCM ledger row."""
    indexed = _indexed_protected_pcm(context)
    if indexed["unaccounted"]:
        return False
    rows = _load_exact_identity_records(context)
    found: Dict[Tuple[str, str], str] = {}
    for row in rows:
        key = _protected_identity_key(row.get("split"), row.get("reference_uid"))
        if key in found:
            raise RuntimeError(f"duplicate protected exact identity {key[0]}:{key[1]}")
        found[key] = str(row.get("sha256_pcm") or "").strip().lower()
    hashes = dict(indexed["fingerprinted"])
    hashes.update(indexed["shorts"])
    if set(found) != set(hashes):
        return False
    return all(found[key] == digest for key, digest in hashes.items())


def apply_protected_exact_identity(segments: List[dict], context: UCleanContext) -> List[MatchEvidence]:
    """Exclude retained U segments whose PCM hash equals a protected reference.

    Every materialized protected reference is checked, including clips that are
    too short to fingerprint. The comparison always runs against the segments
    in memory, so a resume cannot keep a matching segment just because a ledger
    file already exists. No fingerprint is invented for a short reference.
    """
    indexed = _pinned_protected_pcm(context)
    if indexed["unaccounted"]:
        sample = [f"{split}:{uid}" for split, uid in indexed["unaccounted"][:5]]
        raise RuntimeError(
            "exact protected identity requires every protected reference to be materialized first; "
            f"unaccounted={sample}"
        )
    hashes = dict(indexed["fingerprinted"])
    hashes.update(indexed["shorts"])
    return _commit_exact_protected_identity(segments, context, hashes)


def _commit_exact_protected_identity(
    segments: List[dict],
    context: UCleanContext,
    hashes: Dict[Tuple[str, str], str],
) -> List[MatchEvidence]:
    """Apply one exact-PCM hash set. The set decides which segments are excluded."""
    by_pcm: Dict[str, List[dict]] = {}
    for row in segments:
        if row.get("u_clean_status") != RETAINED_STATUS:
            continue
        digest = str(row.get("segment_pcm16_sha256") or "").strip().lower()
        if _is_valid_sha256(digest):
            by_pcm.setdefault(digest, []).append(row)
    evidence: List[MatchEvidence] = []
    records = []
    for split, uid in sorted(hashes):
        digest = hashes[(split, uid)]
        matched = list(by_pcm.get(digest, []))
        for row in matched:
            flag = _SPLIT_TO_FLAG.get(split)
            if flag:
                row[flag] = True
            _exclude(row, EXCLUDED_PROTECTED_OVERLAP)
            evidence.append(MatchEvidence(
                candidate_uid=str(row.get("segment_uid") or ""),
                reference_uid=uid,
                reference_split=split,
                matched_duration_seconds=float(row.get("duration_seconds") or 0.0),
                alignment_offset=0,
                similarity=1.0,
                match_type=PROTECTED_EXACT_MATCH,
            ))
        records.append({
            "reference_uid": uid,
            "split": split,
            "sha256_pcm": digest,
            "n_exact_matches": len(matched),
            "match_type": PROTECTED_EXACT_MATCH,
            "policy": PROTECTED_EXACT_POLICY,
        })
    _write_exact_identity_records(context, records)
    provenance = dict(context.reference_provenance)
    provenance["protected_exact_identity"] = {
        "policy": PROTECTED_EXACT_POLICY,
        "n_checked": len(records),
        "n_exact_matches": sum(int(row["n_exact_matches"]) for row in records),
    }
    context.reference_provenance = provenance
    return evidence


_PROTECTED_SHARD_CACHE: Dict[int, Dict[str, dict]] = {}


def _protected_shard_cache(context: UCleanContext) -> Dict[str, dict]:
    """Process-local verified shard metadata. It is not checkpoint trust."""
    return _PROTECTED_SHARD_CACHE.setdefault(id(context), {})


def _read_expected_file_bytes(path: Path, expected_sha: str) -> Optional[bytes]:
    """Read a shard once and accept it only when the bytes match the checkpoint sha."""
    if not path.is_file():
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != str(expected_sha or ""):
        return None
    return data


def _protected_rows_from_bytes(data: bytes, *, strict_fingerprint_hash: bool = False) -> List[dict]:
    import io
    import pandas as pd

    frame = pd.read_parquet(io.BytesIO(data))
    rows = []
    for _, row in frame.iterrows():
        fingerprint = [int(item) for item in list(row["fingerprint"])]
        if fingerprint_sha256(fingerprint) != str(row["fingerprint_sha256"]):
            if strict_fingerprint_hash:
                raise RuntimeError("invalid fingerprint_sha256")
            continue
        rows.append({
            "uid": str(row["reference_uid"]),
            "split": str(row["split"]),
            "source_sha256": str(row["source_audio_sha256"]),
            "fingerprint_sha256": str(row["fingerprint_sha256"]),
            "n_items": len(fingerprint),
            "fingerprint": fingerprint,
        })
    return rows


def _light_protected_rows(rows: Sequence[dict]) -> List[dict]:
    return [
        {
            "uid": row["uid"],
            "split": row["split"],
            "source_sha256": row["source_sha256"],
            "fingerprint_sha256": row["fingerprint_sha256"],
            "n_items": int(row["n_items"]),
        }
        for row in rows
    ]


def _load_verified_protected_rows(
    context: UCleanContext,
    shard: dict,
    path: Path,
    *,
    keep_arrays: bool,
) -> Optional[List[dict]]:
    """Return verified protected rows. A warm cache does not reread the parquet."""
    cache = _protected_shard_cache(context)
    name = str(shard.get("name") or "")
    expected = str(shard.get("sha256") or "")
    cached = cache.get(name)
    current = path.stat() if path.is_file() else None
    if (
        cached is not None
        and current is not None
        and cached.get("sha256") == expected
        and cached.get("size") == current.st_size
        and cached.get("mtime_ns") == current.st_mtime_ns
        and not keep_arrays
    ):
        return list(cached["rows"])
    data = _read_expected_file_bytes(path, expected)
    if data is None or current is None:
        cache.pop(name, None)
        return None
    rows = _protected_rows_from_bytes(data)
    cache[name] = {
        "sha256": expected,
        "size": current.st_size,
        "mtime_ns": current.st_mtime_ns,
        "rows": _light_protected_rows(rows),
    }
    if keep_arrays:
        return rows
    return list(cache[name]["rows"])


class _MatchShardRejected(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _load_protected_match_arrays(context: UCleanContext, shard: dict, path: Path) -> List[dict]:
    """Load fingerprint arrays for a shard that must be compared, not reused."""
    rows, reason = _match_shard_rows(context, shard, path, keep_arrays=True)
    if reason or rows is None:
        raise _MatchShardRejected(reason or "invalid logical batch")
    return rows


def _fail_protected_match_shard(ordinal: int, shard: dict, reason: str) -> None:
    raise RuntimeError(
        "protected fingerprint shard failed verification: ordinal=%s shard=%s reason=%s"
        % (int(ordinal), shard.get("name"), reason)
    )


def _match_shard_rows(context: UCleanContext, shard: dict, path: Path, *, keep_arrays: bool):
    """Verify one checkpoint-listed shard. A failure is a reason, not a skip."""
    name = str(shard.get("name") or "")
    expected_sha = str(shard.get("sha256") or "")
    if not path.is_file():
        return None, "missing file"
    current = path.stat()
    cache = _protected_shard_cache(context)
    cached = cache.get(name)
    fresh = (
        cached is not None
        and cached.get("sha256") == expected_sha
        and cached.get("size") == current.st_size
        and cached.get("mtime_ns") == current.st_mtime_ns
        and cached.get("fingerprint_hashes_valid")
    )
    if fresh and not keep_arrays:
        return list(cached["rows"]), None
    data = _read_expected_file_bytes(path, expected_sha)
    if data is None:
        cache.pop(name, None)
        return None, "file sha256 mismatch"
    try:
        rows = _protected_rows_from_bytes(data, strict_fingerprint_hash=True)
    except RuntimeError as exc:
        cache.pop(name, None)
        if str(exc) == "invalid fingerprint_sha256":
            return None, "invalid fingerprint_sha256"
        raise
    cache[name] = {
        "sha256": expected_sha,
        "size": current.st_size,
        "mtime_ns": current.st_mtime_ns,
        "rows": _light_protected_rows(rows),
        "fingerprint_hashes_valid": True,
    }
    if keep_arrays:
        return rows, None
    return list(cache[name]["rows"]), None


def _require_match_batch(
    context: UCleanContext,
    ordinal: int,
    shard: dict,
    path: Path,
    allowed: set,
    expected: Dict[str, str],
    min_items: int,
    *,
    keep_arrays: bool,
) -> List[dict]:
    if keep_arrays:
        try:
            rows = _load_protected_match_arrays(context, shard, path)
        except _MatchShardRejected as exc:
            _fail_protected_match_shard(ordinal, shard, exc.reason)
    else:
        rows, reason = _match_shard_rows(context, shard, path, keep_arrays=False)
        if reason or rows is None:
            _fail_protected_match_shard(ordinal, shard, reason or "invalid logical batch")
    batch = _filter_protected_batch(rows, allowed, expected, min_items, include_fingerprint=keep_arrays)
    if not batch:
        _fail_protected_match_shard(ordinal, shard, "invalid logical batch")
    return batch


def _protected_source_index(context: UCleanContext) -> Dict[str, str]:
    return {
        entry.reference_uid: str(entry.sha256_pcm or entry.source_sha256 or "").strip().lower()
        for entry in context.protected_entries
    }


def _filter_protected_batch(
    rows: Sequence[dict],
    allowed: set,
    expected: Dict[str, str],
    min_items: int,
    *,
    include_fingerprint: bool,
) -> List[dict]:
    batch = []
    for row in rows:
        if allowed and row["uid"] not in allowed:
            continue
        stored = str(row.get("source_sha256") or "").strip().lower()
        wanted = expected.get(row["uid"]) if expected else None
        if not _is_valid_sha256(stored) or (wanted and wanted != stored):
            continue
        n_items = int(row.get("n_items") or 0)
        if include_fingerprint:
            fingerprint = row.get("fingerprint")
            n_items = len(fingerprint) if is_valid_fingerprint(fingerprint) else 0
        if n_items < min_items:
            raise RuntimeError(
                f"protected fingerprint below min_overlap_items for {row['uid']} "
                f"split={row.get('split')} items={n_items} required={min_items}"
            )
        item = {
            "uid": row["uid"],
            "split": row["split"],
            "source_sha256": row["source_sha256"],
            "fingerprint_sha256": row["fingerprint_sha256"],
            "n_items": n_items,
        }
        if include_fingerprint:
            item["fingerprint"] = list(row["fingerprint"])
        batch.append(item)
    return batch


def _iter_protected_fingerprint_jobs(context: UCleanContext):
    """Verified protected shards in checkpoint order. Hash mismatches are skipped."""
    state = _read_checkpoint_state(context)
    expected = _protected_source_index(context)
    root = checkpoint_root(context) / "protected_fingerprints"
    min_items = int(context.config.overlap.min_overlap_items)
    for ordinal, shard in enumerate(list(state["protected_fingerprint_shards"])):
        allowed = {str(uid) for uid in (shard.get("uids") or [])}
        file_path = root / shard["name"]
        rows = _load_verified_protected_rows(context, shard, file_path, keep_arrays=True)
        if rows is None:
            continue
        batch = _filter_protected_batch(rows, allowed, expected, min_items, include_fingerprint=True)
        if batch:
            yield {
                "ordinal": ordinal,
                "input_shard_name": shard["name"],
                "input_shard_sha256": shard["sha256"],
                "batch": batch,
            }


def _match_compute_job(ordinal: int, shard: dict, batch: Sequence[dict], batch_sha: str) -> dict:
    return {
        "action": "compute",
        "job": {
            "ordinal": ordinal,
            "input_shard_name": shard["name"],
            "input_shard_sha256": shard["sha256"],
            "input_batch_sha256": batch_sha,
            "batch": list(batch),
        },
    }


def _iter_protected_match_units(context: UCleanContext, base: dict):
    """Yield one reuse-or-compute decision at a time. Reuse does not keep arrays."""
    state = _read_checkpoint_state(context)
    expected = _protected_source_index(context)
    root = checkpoint_root(context) / "protected_fingerprints"
    min_items = int(context.config.overlap.min_overlap_items)
    for ordinal, shard in enumerate(list(state["protected_fingerprint_shards"])):
        allowed = {str(uid) for uid in (shard.get("uids") or [])}
        file_path = root / shard["name"]
        if not _protected_match_path(context, ordinal).is_file():
            batch = _require_match_batch(
                context, ordinal, shard, file_path, allowed, expected, min_items, keep_arrays=True,
            )
            yield _match_compute_job(ordinal, shard, batch, _protected_input_batch_sha256(batch))
            continue
        meta = _require_match_batch(
            context, ordinal, shard, file_path, allowed, expected, min_items, keep_arrays=False,
        )
        batch_sha = _protected_input_batch_sha256(meta)
        compatibility = _shard_match_compatibility(base, shard["sha256"], batch_sha)
        loaded = _read_protected_match_shard(context, ordinal, compatibility, len(meta))
        if loaded is not None:
            yield {
                "action": "reuse",
                "ordinal": ordinal,
                "document": loaded,
                "batch_length": len(meta),
            }
            continue
        batch = _require_match_batch(
            context, ordinal, shard, file_path, allowed, expected, min_items, keep_arrays=True,
        )
        yield _match_compute_job(ordinal, shard, batch, _protected_input_batch_sha256(batch))


def _iter_protected_accounting_rows(context: UCleanContext):
    """Verified protected identity rows. Fingerprint arrays are not retained."""
    state = _read_checkpoint_state(context)
    expected = _protected_source_index(context)
    root = checkpoint_root(context) / "protected_fingerprints"
    min_items = int(context.config.overlap.min_overlap_items)
    for shard in list(state["protected_fingerprint_shards"]):
        file_path = root / shard["name"]
        light = _load_verified_protected_rows(context, shard, file_path, keep_arrays=False)
        if light is None:
            continue
        allowed = {str(uid) for uid in (shard.get("uids") or [])}
        for row in _filter_protected_batch(light, allowed, expected, min_items, include_fingerprint=False):
            yield row


def iter_protected_fingerprint_shards(context: UCleanContext):
    """Yield one verified protected shard at a time."""
    for job in _iter_protected_fingerprint_jobs(context):
        yield job["batch"]


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
