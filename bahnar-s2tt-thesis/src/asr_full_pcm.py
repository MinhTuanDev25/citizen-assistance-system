"""
Deterministic PCM16 pipeline + Hugging Face Parquet reference helpers.

Split out of ``asr_full_data`` so the byte-exact audio contract lives in one
place: every hydrate/verify path must go through :func:`canonical_pcm16_bytes`.

Determinism contract (``AUDIO_PCM_PIPELINE_VERSION``):
  1. decode with soundfile at float64 (``always_2d=False``)
  2. mono via arithmetic mean across channels
  3. resample only when ``sr != target_sr`` (kept for abnormal data)
  4. clip to [-1, 1]
  5. ``numpy.round(x * 32767)`` then cast to int16
  6. serialise little-endian ("<i2")

Changing ANY step above requires bumping AUDIO_PCM_PIPELINE_VERSION, otherwise
previously stored ``sha256_pcm`` values become silently wrong.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import numpy as np

AUDIO_PCM_PIPELINE_VERSION = "full_pcm16_le_v1"

PCM16_DTYPE = "<i2"
PCM16_SCALE = 32767
WAV_HEADER_BYTES = 44


class AudioQaError(ValueError):
    """Hard audio QA failure; `reason` matches NB02 check_waveform reasons."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Hugging Face parquet references
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HfParquetRef:
    """Normalized pointer to one parquet shard inside a HF dataset repo."""

    repo_id: str
    filename: str
    revision: str
    repo_type: str = "dataset"

    @property
    def shard_key(self) -> str:
        """Stable identity used for state/sidecar keys (revision-independent)."""
        return self.filename

    def as_dict(self) -> Dict[str, str]:
        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "revision": self.revision,
            "repo_type": self.repo_type,
        }


def parse_hf_parquet_url(url: Any) -> HfParquetRef:
    """
    Parse a ``huggingface.co`` resolve/blob URL into repo_id/filename/revision.

    Manifests store fully-qualified URLs such as::

        https://huggingface.co/datasets/<owner>/<name>/resolve/refs%2Fconvert%2Fparquet/default/train/0033.parquet

    ``hf_hub_download`` needs ``filename="default/train/0033.parquet"`` and
    ``revision="refs/convert/parquet"`` — passing the URL verbatim 404s.
    """
    raw = str(url or "").strip()
    if not raw:
        raise ValueError("empty parquet reference")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"not an http(s) parquet URL: {raw!r}")
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(f"not a huggingface.co URL: {raw!r}")

    parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
    if not parts:
        raise ValueError(f"parquet URL has no path: {raw!r}")

    repo_type = "model"
    if parts[0] in {"datasets", "spaces"}:
        repo_type = "dataset" if parts[0] == "datasets" else "space"
        parts = parts[1:]
    if len(parts) < 4:
        raise ValueError(f"parquet URL too short to contain repo/resolve/file: {raw!r}")

    owner, name = parts[0], parts[1]
    marker = parts[2]
    if marker not in {"resolve", "blob", "raw"}:
        raise ValueError(f"parquet URL missing resolve/blob segment: {raw!r}")

    rest = parts[3:]
    if not rest:
        raise ValueError(f"parquet URL missing revision + filename: {raw!r}")

    # "refs/convert/parquet" survives as three segments once unquoted.
    if len(rest) >= 3 and rest[0] == "refs":
        revision = "/".join(rest[:3])
        filename_parts = rest[3:]
    else:
        revision = rest[0]
        filename_parts = rest[1:]
    if not filename_parts:
        raise ValueError(f"parquet URL missing filename after revision: {raw!r}")

    return HfParquetRef(
        repo_id=f"{owner}/{name}",
        filename="/".join(filename_parts),
        revision=revision,
        repo_type=repo_type,
    )


def normalize_parquet_ref(
    parquet_file: Any,
    *,
    expected_repo_id: str,
    parquet_revision: str,
) -> HfParquetRef:
    """
    Normalize a manifest ``parquet_file`` value and pin it to an immutable revision.

    ``parquet_revision`` must be the resolved snapshot commit SHA (not a moving
    alias such as ``refs/convert/parquet``), so audio bytes stay bound even if
    Hugging Face re-converts the dataset.
    """
    if not parquet_revision or not str(parquet_revision).strip():
        raise ValueError("parquet_revision is required (immutable snapshot sha)")
    rev = str(parquet_revision).strip()
    if rev.startswith("refs/"):
        raise ValueError(
            f"parquet_revision must be an immutable commit sha, got alias {rev!r}"
        )

    raw = str(parquet_file or "").strip()
    if not raw:
        raise ValueError("empty parquet_file")
    if raw.lower().startswith("http://") or raw.lower().startswith("https://"):
        ref = parse_hf_parquet_url(raw)
        if ref.repo_id != str(expected_repo_id):
            raise ValueError(
                f"parquet URL repo {ref.repo_id!r} != expected {expected_repo_id!r}"
            )
        filename = ref.filename
        repo_type = ref.repo_type
    else:
        filename = raw.lstrip("/")
        repo_type = "dataset"
    return HfParquetRef(
        repo_id=str(expected_repo_id),
        filename=filename,
        revision=rev,
        repo_type=repo_type,
    )


DYNAMIC_PARQUET_REF = "refs/convert/parquet"
ALLOWED_PARQUET_PREFIX = "default/train/"


def assert_parquet_files_are_train_only(
    filenames: Iterable[str],
    *,
    allowed_prefix: str = ALLOWED_PARQUET_PREFIX,
) -> List[str]:
    """
    Reject manifests pointing at the frozen ``test``/``validation`` parquet dirs.

    NB01 carves both training splits out of the upstream ``train`` split, so any
    other prefix means frozen data leaked into a trainable manifest.
    """
    seen: List[str] = []
    bad: List[str] = []
    for name in filenames:
        token = str(name or "").strip()
        if not token or token in seen:
            continue
        seen.append(token)
        if not token.startswith(allowed_prefix):
            bad.append(token)
    if bad:
        raise RuntimeError(
            f"Manifest references parquet outside {allowed_prefix!r} "
            f"({len(bad)}): {sorted(bad)[:5]}"
        )
    return seen


def verify_pinned_parquet_snapshot(
    *,
    repo_id: str,
    pinned_revision: str,
    required_filenames: Iterable[str],
    api: Optional[Any] = None,
    dynamic_ref: str = DYNAMIC_PARQUET_REF,
) -> Dict[str, Any]:
    """
    Read-only preflight for the immutable parquet snapshot (no bulk download).

    Fails only on things that actually break us: the pinned commit not being
    resolvable, or a required shard missing from it. The auto-conversion ref
    moving is reported as a warning — the whole point of pinning a commit SHA is
    that we no longer care where the mutable ref points.
    """
    if api is None:
        from huggingface_hub import HfApi  # imported lazily so tests can inject

        api = HfApi()
    pinned = str(pinned_revision).strip()
    if len(pinned) != 40 or not all(c in "0123456789abcdef" for c in pinned.lower()):
        raise RuntimeError(
            f"EXPECTED_PARQUET_REVISION must be a 40-hex immutable commit SHA, got {pinned!r}"
        )
    required = sorted({str(f) for f in required_filenames if str(f).strip()})
    try:
        present = set(api.list_repo_files(repo_id, revision=pinned, repo_type="dataset"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Pinned parquet revision {pinned} does not resolve for {repo_id!r} "
            f"({type(exc).__name__}: {exc}); refusing to start prepare"
        ) from exc
    missing = [name for name in required if name not in present]
    if missing:
        raise RuntimeError(
            f"Pinned parquet snapshot {pinned} is missing {len(missing)} required shard(s) "
            f"for {repo_id!r}: {missing[:5]}"
        )
    warnings: List[str] = []
    dynamic_sha = None
    try:
        dynamic_sha = str(api.repo_info(repo_id, revision=dynamic_ref, repo_type="dataset").sha)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"could not resolve {dynamic_ref}: {type(exc).__name__}")
    else:
        if dynamic_sha != pinned:
            warnings.append(
                f"{dynamic_ref} now points at {dynamic_sha}, not the pinned {pinned}; "
                "still using the pinned snapshot"
            )
    return {
        "repo_id": str(repo_id),
        "pinned_revision": pinned,
        "dynamic_ref": dynamic_ref,
        "dynamic_revision": dynamic_sha,
        "dynamic_matches_pinned": dynamic_sha == pinned if dynamic_sha else None,
        "required_count": len(required),
        "warnings": warnings,
    }


def fetch_parquet_shard_sizes(
    *,
    repo_id: str,
    revision: str,
    filenames: Iterable[str],
    api: Optional[Any] = None,
) -> Dict[str, int]:
    """
    Real byte size of every required shard at a pinned revision.

    A disk preflight built on a guessed shard size is not a preflight, so a
    missing or zero size is an error here rather than something to paper over
    with a default.
    """
    if api is None:
        from huggingface_hub import HfApi  # imported lazily so tests can inject

        api = HfApi()
    wanted = sorted({str(f) for f in filenames if str(f).strip()})
    try:
        info = api.repo_info(
            repo_id, revision=str(revision), repo_type="dataset", files_metadata=True
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not read parquet file metadata for {repo_id!r} at {revision}: "
            f"{type(exc).__name__}: {exc}; refusing to start prepare"
        ) from exc
    sizes: Dict[str, int] = {}
    for sibling in getattr(info, "siblings", None) or []:
        name = str(getattr(sibling, "rfilename", "") or "")
        size = getattr(sibling, "size", None)
        if name in wanted and size:
            sizes[name] = int(size)
    missing = [name for name in wanted if name not in sizes]
    if missing:
        raise RuntimeError(
            f"Hugging Face reported no size for {len(missing)} required shard(s) at "
            f"{revision}: {missing[:5]}; refusing to budget disk from a guess"
        )
    return sizes


def safe_shard_key(shard_key: Any) -> str:
    """Filesystem-safe token for a shard key (used for sidecar filenames)."""
    raw = str(shard_key or "").strip()
    if not raw:
        raise ValueError("empty shard key")
    keep = []
    for ch in raw:
        keep.append(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_")
    token = "".join(keep).strip("._") or "shard"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{token[-80:]}__{digest}"


# ---------------------------------------------------------------------------
# Deterministic PCM16
# ---------------------------------------------------------------------------

def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def to_mono_float64(array: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (mono float64 1-D, n_channels_in). Mean-mix is part of the contract."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=False), 1
    if arr.ndim != 2:
        raise ValueError(f"unsupported waveform ndim={arr.ndim}")
    # soundfile returns (frames, channels)
    channels = int(arr.shape[1])
    mono = arr.astype(np.float64, copy=False).mean(axis=1)
    return mono, channels


def float_to_pcm16_bytes(mono: np.ndarray) -> bytes:
    """Clip → round → int16 → little-endian bytes (pinned quantisation rule)."""
    arr = np.asarray(mono, dtype=np.float64)
    clipped = np.clip(arr, -1.0, 1.0)
    quantised = np.round(clipped * PCM16_SCALE).astype(np.int16)
    return quantised.astype(PCM16_DTYPE, copy=False).tobytes()


def canonical_pcm16_from_array(
    array: np.ndarray,
    sample_rate: int,
    *,
    target_sr: int = 16000,
    resample_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Canonicalise an already-decoded waveform into the pinned PCM16 form."""
    mono, channels_in = to_mono_float64(array)
    if mono.size == 0:
        raise AudioQaError("empty")
    if not np.isfinite(mono).all():
        # Quantising NaN/inf would silently yield plausible-looking silence or
        # full-scale constants, so reject before resample/quantisation.
        raise AudioQaError("non_finite_values")
    sr_in = int(sample_rate)
    resampled = False
    if sr_in != int(target_sr):
        if resample_fn is None:
            from src.data_utils import resample_audio as resample_fn  # type: ignore
        mono = np.asarray(
            resample_fn(mono.astype(np.float32, copy=False), sr_in, int(target_sr)),
            dtype=np.float64,
        )
        resampled = True
    pcm = float_to_pcm16_bytes(mono)
    n_samples = len(pcm) // 2
    return {
        "pcm": pcm,
        "sha256_pcm": sha256_hex(pcm),
        "n_samples": int(n_samples),
        "sample_rate": int(target_sr),
        "source_sample_rate": sr_in,
        "channels_in": int(channels_in),
        "resampled": bool(resampled),
        "duration_seconds": float(n_samples / float(target_sr)) if target_sr else 0.0,
        "audio_pcm_pipeline_version": AUDIO_PCM_PIPELINE_VERSION,
    }


def canonical_pcm16_from_bytes(
    raw: bytes,
    *,
    target_sr: int = 16000,
    resample_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Decode encoded audio bytes (FLAC/WAV/...) and canonicalise to PCM16.

    Also returns ``sha256_source`` so a later mismatch can distinguish
    "upstream data changed" from "our pipeline changed".
    """
    import soundfile as sf

    if not raw:
        raise ValueError("empty audio payload")
    with sf.SoundFile(io.BytesIO(raw)) as handle:
        subtype = str(handle.subtype)
        fmt = str(handle.format)
        sr_in = int(handle.samplerate)
        channels = int(handle.channels)
        array = handle.read(dtype="float64", always_2d=False)
    out = canonical_pcm16_from_array(
        array, sr_in, target_sr=target_sr, resample_fn=resample_fn
    )
    out.update({
        "sha256_source": sha256_hex(raw),
        "source_subtype": subtype,
        "source_format": fmt,
        "source_channels": channels,
        "source_bytes": int(len(raw)),
    })
    out["channels_in"] = channels
    return out


def build_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Minimal deterministic 16-bit mono PCM WAV container (44-byte header)."""
    n_bytes = len(pcm)
    header = b"RIFF"
    header += struct.pack("<I", 36 + n_bytes)
    header += b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, int(sample_rate), int(sample_rate) * 2, 2, 16)
    header += b"data"
    header += struct.pack("<I", n_bytes)
    return header + pcm


def write_wav_from_pcm16(path: Union[str, Path], pcm: bytes, sample_rate: int) -> Path:
    """Atomically write a canonical WAV; the file is a pure function of (pcm, sr)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(build_wav_bytes(pcm, sample_rate))
    os.replace(str(tmp), str(out))
    return out


def read_pcm16_payload(path: Union[str, Path]) -> Dict[str, Any]:
    """Read back a canonical WAV and recompute the PCM SHA (header excluded)."""
    import soundfile as sf

    p = Path(path)
    with sf.SoundFile(str(p)) as handle:
        sr = int(handle.samplerate)
        channels = int(handle.channels)
        subtype = str(handle.subtype)
        # Read int16 directly: float decoding divides by 32768 while the writer
        # scales by 32767, so a float round-trip would shift samples and break
        # the PCM hash.
        array = handle.read(dtype="int16", always_2d=False)
    arr = np.asarray(array)
    if arr.ndim != 1:
        raise ValueError(f"canonical WAV must be mono, got shape {arr.shape} for {p}")
    pcm = arr.astype(PCM16_DTYPE, copy=False).tobytes()
    return {
        "pcm": pcm,
        "sha256_pcm": sha256_hex(pcm),
        "n_samples": len(pcm) // 2,
        "sample_rate": sr,
        "channels": channels,
        "subtype": subtype,
    }


# ---------------------------------------------------------------------------
# Local disk guards + Hugging Face cache reclamation
# ---------------------------------------------------------------------------

def disk_free_bytes(path: Union[str, Path]) -> int:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return int(shutil.disk_usage(str(target)).free)


def wav_bytes_for_samples(n_samples: Any) -> int:
    """Exact size of a canonical mono PCM16 WAV holding ``n_samples`` samples."""
    return int(n_samples) * 2 + WAV_HEADER_BYTES


def assert_local_disk_budget(
    path: Union[str, Path],
    *,
    needs: Dict[str, int],
    reserve_bytes: int,
    label: str,
) -> Dict[str, Any]:
    """
    Fail-closed disk preflight built from an itemised budget.

    ``needs`` is reported back in full so an out-of-space failure names the
    component that blew the budget instead of a bare total.
    """
    items = {str(k): int(v) for k, v in (needs or {}).items() if int(v) > 0}
    required = sum(items.values()) + int(reserve_bytes)
    free = disk_free_bytes(path)
    report = {
        "path": str(path),
        "label": label,
        "free_bytes": free,
        "reserve_bytes": int(reserve_bytes),
        "required_bytes": required,
        "needs": items,
        "ok": free >= required,
    }
    if not report["ok"]:
        breakdown = ", ".join(f"{k}={v / 1e9:.1f}GB" for k, v in sorted(items.items()))
        raise RuntimeError(
            f"Insufficient local disk for {label}: free={free / 1e9:.1f}GB "
            f"required={required / 1e9:.1f}GB "
            f"(reserve={int(reserve_bytes) / 1e9:.1f}GB, {breakdown}) at {path}"
        )
    return report


def assert_disk_headroom(
    path: Union[str, Path],
    *,
    need_bytes: int,
    label: str = "shard",
    safety_bytes: int = 2 * 1024 ** 3,
) -> Dict[str, Any]:
    """Fail-closed before writing: never start a shard we cannot finish."""
    free = disk_free_bytes(path)
    required = int(need_bytes) + int(safety_bytes)
    if free < required:
        raise RuntimeError(
            f"Insufficient disk headroom for {label}: free={free / 1e9:.1f}GB "
            f"required={required / 1e9:.1f}GB at {path}"
        )
    return {"free_bytes": free, "required_bytes": required, "path": str(path)}


def delete_hf_cache_blob(local_path: Union[str, Path]) -> int:
    """
    Actually reclaim bytes for a downloaded HF file.

    ``hf_hub_download`` returns ``snapshots/<rev>/<file>`` which is a *symlink*
    into ``blobs/``; unlinking only the symlink frees nothing. Returns bytes freed.
    """
    p = Path(local_path)
    freed = 0
    try:
        if p.is_symlink():
            blob = Path(os.path.realpath(str(p)))
            if blob.is_file():
                freed += blob.stat().st_size
                blob.unlink()
            p.unlink()
            return freed
        if p.is_file():
            freed += p.stat().st_size
            p.unlink()
    except FileNotFoundError:
        return freed
    return freed


def cleanup_shard_download(
    local_path: Union[str, Path, None],
    *,
    extra_dirs: Optional[Any] = None,
) -> int:
    """Delete a downloaded shard (blob included) plus optional temp dirs."""
    freed = 0
    if local_path:
        freed += delete_hf_cache_blob(local_path)
    for extra in list(extra_dirs or []):
        d = Path(extra)
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file():
                    try:
                        freed += f.stat().st_size
                    except OSError:
                        pass
            shutil.rmtree(d, ignore_errors=True)
    return freed
