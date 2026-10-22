"""Standalone C0 cascaded inference (ASR then MT).

Decoding matches Notebook 06 helpers in ``src.rq1_inference`` after the
waveform exists: 16 kHz mono float32, CTC argmax, ``normalize_bahnar_ctc_v1``,
then BARTPho generate with the configured source length, max length, and
beams, then ``normalize_mt_text_v1``.

ASR serving consumes the validated waveform in memory. It does not re-open
the audio path. RQ1 ``run_asr_predictions`` is left unchanged.

MT serving loads the Hub tokenizer through ``load_mt_tokenizer``, the same
entry as RQ1 restore. Fine-tuned weights still come from the configured
checkpoint directory. This module does not train, does not scan for a newer
checkpoint step, and does not read the Notebook 04 runtime root.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from src.c0_tokenizer import (
    LOCKED_MODEL_ID,
    LOCKED_MODEL_REVISION,
    PARITY_NOT_VERIFIED,
    TOKENIZER_SOURCE_HUB_PINNED_RQ1,
    C0TokenizerError,
    load_serving_mt_tokenizer,
)
from src.data_utils import normalize_bahnar_ctc_v1
from src.mt_normalize import normalize_mt_text_v1
from src.rq1_contract import sha256_file
from src.rq1_inference import run_mt_from_asr

MODEL_ID = "c0_cascaded"
SAMPLE_RATE = 16000
# Serving bound for a citizen utterance. Not an RQ1 evaluation contract.
DEFAULT_MAX_AUDIO_DURATION_SECONDS = 30.0
# 16 kHz PCM16 mono for 30 s is about 1 MB; 10 MiB leaves headroom for WAV headers.
DEFAULT_MAX_AUDIO_BYTES = 10 * 1024 * 1024
VERIFIED_ASR_CHECKPOINT_STEP = 12500
VERIFIED_MT_CHECKPOINT_STEP = 17500
VERIFIED_ASR_EXPERIMENT_ID = "full_xlsr300m_v1"
VERIFIED_MT_EXPERIMENT_ID = "mt_bartpho_syllable_v1"
ASR_PRETRAINED_ID = "facebook/wav2vec2-xls-r-300m"
ASR_PRETRAINED_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
RQ1_MAX_SOURCE_LENGTH = 256
RQ1_GENERATION_MAX_LENGTH = 256
RQ1_NUM_BEAMS = 4
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ASR_PROCESSOR_KWARGS = {
    "sampling_rate": SAMPLE_RATE,
    "padding": True,
    "return_tensors": "pt",
}

ASR_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "vocab.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
)
MT_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "dict.txt",
    "generation_config.json",
)
TRAINING_STATE_FILES = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
)


class C0Error(RuntimeError):
    """Inference failed before a complete Bahnar and Vietnamese pair exists."""

    error_code = "c0_error"


class C0ConfigError(C0Error):
    """Checkpoint or generation config is not usable."""

    error_code = "config_error"


class C0AudioError(C0Error):
    """Audio cannot enter the evaluation decode path."""

    error_code = "audio_invalid"


class C0EmptyOutputError(C0Error):
    """Serving rejects empty ASR or MT text. RQ1 evaluation does not use this gate."""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


class C0ModelLoadError(C0Error):
    """Weights or ML dependencies could not be loaded."""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


class C0ChecksumError(C0Error):
    """Configured or research checksum does not match the weights on disk."""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


def load_c0_config(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise C0ConfigError(f"Config encoding error: {config_path}") from exc
    except OSError as exc:
        raise C0ConfigError(f"Cannot read config: {config_path}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise C0ConfigError(f"Invalid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise C0ConfigError(f"Config must be a mapping: {config_path}")
    return raw


def configured_path(value: object) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text)


def missing_checkpoint_files(checkpoint: Path, required: tuple[str, ...]) -> list[str]:
    root = Path(checkpoint)
    if not root.is_dir():
        return [f"missing directory {root}"]
    return [name for name in required if not (root / name).is_file()]


def assert_checkpoint_ready(checkpoint: Path, required: tuple[str, ...], *, label: str) -> Path:
    root = configured_path(checkpoint)
    if root is None:
        raise C0ConfigError(f"{label} checkpoint path is empty")
    missing = missing_checkpoint_files(root, required)
    if missing:
        raise C0ConfigError(f"{label} checkpoint {root} is incomplete: {', '.join(missing)}")
    return root


def checkpoint_step(path: Path) -> Optional[int]:
    name = Path(path).name
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix) :])
    except ValueError:
        return None


def optional_sha256(config: Mapping[str, Any], key: str) -> Optional[str]:
    text = str(config.get(key) or "").strip().lower()
    if not text:
        return None
    if not _SHA256_RE.fullmatch(text):
        raise C0ConfigError(f"{key} must be a 64-character hex SHA-256 or empty")
    return text


DEFAULT_RESEARCH_PINS = Path(__file__).resolve().parents[1] / "configs" / "c0_research_pins.yaml"


def load_research_pins(
    config: Mapping[str, Any],
    *,
    default_path: Optional[Path] = DEFAULT_RESEARCH_PINS,
) -> dict[str, Optional[str]]:
    """Configured research-reference hashes.

    Matching this file is integrity against a configured pointer, not independent
    research attestation. Never read operator ``asr_weights_sha256`` YAML as provenance.
    """
    empty = {"asr": None, "mt": None, "source": None}
    if "research_pins" in config and not str(config.get("research_pins") or "").strip():
        return empty
    path = configured_path(config.get("research_pins"))
    if path is None:
        path = default_path if default_path is not None and default_path.is_file() else None
    if path is None:
        return empty
    if not path.is_file():
        raise C0ConfigError(f"research_pins file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise C0ConfigError(f"Cannot read research_pins: {path}") from exc
    if not isinstance(raw, dict):
        raise C0ConfigError(f"research_pins must be a mapping: {path}")
    return {
        "asr": optional_sha256(raw, "asr_weights_sha256"),
        "mt": optional_sha256(raw, "mt_weights_sha256"),
        "source": str(path),
    }


def assert_operator_checksum(
    root: Path,
    pin: Optional[str],
    *,
    label: str,
    hasher: Optional[Callable[[Path], str]] = None,
) -> None:
    """Fail closed when an operator pin is set and the file does not match. Runs before model load."""
    if pin is None:
        return
    hash_fn = hasher or sha256_file
    weights = Path(root) / "model.safetensors"
    if not weights.is_file():
        raise C0ChecksumError("checkpoint_weights_missing", f"{label} weights missing for checksum")
    try:
        got = hash_fn(weights)
    except OSError as exc:
        raise C0ChecksumError("checkpoint_weights_unreadable", f"{label} weights are not readable") from exc
    if str(got).lower() != pin:
        raise C0ChecksumError(
            "checkpoint_checksum_mismatch",
            f"{label} weights SHA-256 does not match the configured pin",
        )


def assess_checkpoint(
    path: Optional[Path],
    required: tuple[str, ...],
    *,
    expected_step: int,
    configured_sha256: Optional[str] = None,
    research_sha256: Optional[str] = None,
    hasher: Optional[Callable[[Path], str]] = None,
) -> dict[str, Any]:
    """Describe one checkpoint.

    Operator YAML hash match is ``configured_hash_match``.
    Research-pins YAML match is ``research_reference_hash_match``.
    Neither label is independent research attestation.
    """
    if path is None:
        return {
            "checkpoint": None,
            "checkpoint_step": None,
            "expected_checkpoint_step": expected_step,
            "step_name_match": False,
            "artifacts_present": False,
            "identity": "missing",
            "missing_files": list(required),
            "weights_sha256": None,
            "weights_sha256_status": "NOT_COMPUTED",
            "configured_hash_match": None,
            "research_hash_match": None,
            "research_provenance": "NOT_VERIFIED",
        }
    root = Path(path)
    step = checkpoint_step(root)
    step_name_match = step == expected_step
    missing = missing_checkpoint_files(root, required)
    artifacts_present = root.is_dir() and not missing
    weights = root / "model.safetensors"
    sha = None
    sha_status = "NOT_COMPUTED"
    hash_fn = hasher or sha256_file
    if (configured_sha256 or research_sha256) and weights.is_file():
        sha = hash_fn(weights)
        sha_status = "COMPUTED"
    elif (configured_sha256 or research_sha256) and not weights.is_file():
        sha_status = "MISSING_FILE"
    configured_match: Optional[bool] = None if not configured_sha256 else bool(sha and sha.lower() == configured_sha256.lower())
    research_match: Optional[bool] = None if not research_sha256 else bool(sha and sha.lower() == research_sha256.lower())
    if research_match is True and artifacts_present and step_name_match:
        identity = "research_reference_hash_match"
        research_provenance = "REFERENCE_HASH_MATCH"
    elif configured_match is True and artifacts_present and step_name_match:
        identity = "configured_hash_match"
        research_provenance = "NOT_VERIFIED"
    elif step_name_match:
        identity = "expected_step_match"
        research_provenance = "NOT_VERIFIED"
    else:
        identity = "custom_unverified"
        research_provenance = "NOT_VERIFIED"
    if research_sha256 and research_match is False:
        research_provenance = "NOT_VERIFIED"
        if identity == "research_reference_hash_match":
            identity = "custom_unverified"
    return {
        "checkpoint": str(root),
        "checkpoint_step": step,
        "expected_checkpoint_step": expected_step,
        "step_name_match": step_name_match,
        "artifacts_present": artifacts_present,
        "identity": identity,
        "missing_files": missing,
        "weights_sha256": sha,
        "weights_sha256_status": sha_status,
        "configured_hash_match": configured_match,
        "research_hash_match": research_match,
        "research_provenance": research_provenance,
    }


def pair_checkpoint_identity(asr: Mapping[str, Any], mt: Mapping[str, Any]) -> str:
    if asr.get("identity") == "missing" and mt.get("identity") == "missing":
        return "missing"
    if asr.get("identity") == "research_reference_hash_match" and mt.get("identity") == "research_reference_hash_match":
        return "research_reference_hash_match"
    if asr.get("identity") == "custom_unverified" or mt.get("identity") == "custom_unverified":
        return "custom_unverified"
    if asr.get("identity") == "configured_hash_match" and mt.get("identity") == "configured_hash_match":
        return "configured_hash_match"
    if asr.get("step_name_match") and mt.get("step_name_match"):
        return "expected_step_match"
    return "custom_unverified"


def pair_research_provenance(asr: Mapping[str, Any], mt: Mapping[str, Any]) -> str:
    """Pair-level provenance. Configured reference match is never independent VERIFIED."""
    if asr.get("research_provenance") == "REFERENCE_HASH_MATCH" and mt.get("research_provenance") == "REFERENCE_HASH_MATCH":
        return "REFERENCE_HASH_MATCH"
    return "NOT_VERIFIED"


def checkpoint_identity(
    asr_path: Path,
    mt_path: Path,
    *,
    asr_sha256: Optional[str] = None,
    mt_sha256: Optional[str] = None,
    asr_research_sha256: Optional[str] = None,
    mt_research_sha256: Optional[str] = None,
) -> str:
    asr = assess_checkpoint(
        Path(asr_path) if asr_path is not None else None,
        ASR_REQUIRED_FILES,
        expected_step=VERIFIED_ASR_CHECKPOINT_STEP,
        configured_sha256=asr_sha256,
        research_sha256=asr_research_sha256,
    )
    mt = assess_checkpoint(
        Path(mt_path) if mt_path is not None else None,
        MT_REQUIRED_FILES,
        expected_step=VERIFIED_MT_CHECKPOINT_STEP,
        configured_sha256=mt_sha256,
        research_sha256=mt_research_sha256,
    )
    return pair_checkpoint_identity(asr, mt)


def generation_matches_rq1(max_source_length: int, generation_max_length: int, num_beams: int) -> bool:
    return (
        int(max_source_length) == RQ1_MAX_SOURCE_LENGTH
        and int(generation_max_length) == RQ1_GENERATION_MAX_LENGTH
        and int(num_beams) == RQ1_NUM_BEAMS
    )


def resolve_device(name: str) -> str:
    choice = str(name or "").strip().lower()
    if choice not in {"cpu", "cuda", "auto"}:
        raise C0ConfigError(f"device must be cpu, cuda, or auto (got {name!r})")
    if choice == "cpu":
        return "cpu"
    import torch

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise C0ConfigError("device=cuda but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_mono_float32(wav: np.ndarray) -> np.ndarray:
    arr = np.asarray(wav)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return np.asarray(arr, dtype=np.float32)


def prepare_asr_waveforms(wavs: list[np.ndarray]) -> list[np.ndarray]:
    """Stereo mean + float32, matching ``run_asr_predictions`` after ``soundfile.read``."""
    return [_as_mono_float32(wav) for wav in wavs]


def duration_ms(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> int:
    return int(round((len(wav) / float(sample_rate)) * 1000.0))


def _positive_number(config: Mapping[str, Any], key: str, default: float, *, integer: bool = False) -> float:
    raw = config[key] if key in config and config.get(key) is not None else default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise C0ConfigError(f"{key} must be a finite number > 0") from exc
    if not math.isfinite(value) or value <= 0:
        raise C0ConfigError(f"{key} must be a finite number > 0")
    if integer:
        if not float(value).is_integer():
            raise C0ConfigError(f"{key} must be an integer > 0")
        return int(value)
    return value


def read_c0_audio(
    path: Path,
    *,
    reader: Optional[Callable[..., Any]] = None,
    max_duration_seconds: float = DEFAULT_MAX_AUDIO_DURATION_SECONDS,
    max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
) -> tuple[np.ndarray, int]:
    """Read one utterance once. The returned waveform is what ASR must consume."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise C0AudioError(f"Audio file not found: {audio_path}")
    try:
        size = audio_path.stat().st_size
    except OSError as exc:
        raise C0AudioError(f"Audio file is not readable: {audio_path}") from exc
    if size <= 0:
        raise C0AudioError(f"Audio file is empty: {audio_path}")
    if size > int(max_audio_bytes):
        raise C0AudioError(
            f"Audio file exceeds max_audio_bytes={int(max_audio_bytes)}: {audio_path}"
        )
    if reader is None:
        import soundfile as sf

        reader = sf.read
    try:
        wav, sr = reader(str(audio_path), dtype="float32", always_2d=False)
    except C0Error:
        raise
    except Exception as exc:
        raise C0AudioError(f"Audio cannot be decoded: {audio_path}") from exc
    if int(sr) != SAMPLE_RATE:
        raise C0AudioError(f"ASR audio sample rate {sr} != {SAMPLE_RATE} for {audio_path}")
    mono = _as_mono_float32(wav)
    if mono.size == 0:
        raise C0AudioError(f"Audio is empty: {audio_path}")
    if not np.isfinite(mono).all():
        raise C0AudioError(f"Audio has non-finite samples: {audio_path}")
    seconds = float(len(mono)) / float(SAMPLE_RATE)
    if not math.isfinite(float(max_duration_seconds)) or float(max_duration_seconds) <= 0:
        raise C0AudioError("max_audio_duration_seconds must be a finite number > 0")
    if seconds > float(max_duration_seconds):
        raise C0AudioError(
            f"Audio duration {seconds:.3f}s exceeds max_audio_duration_seconds={max_duration_seconds}"
        )
    return mono, SAMPLE_RATE


def transcribe_waveform(model: Any, processor: Any, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """CTC decode one in-memory waveform.

    Call pattern matches ``run_asr_predictions`` after waveforms are already
    float32 mono: processor kwargs, attention mask, argmax, batch_decode,
    ``normalize_bahnar_ctc_v1``. This function is the C0 serving adapter.
    It does not replace the RQ1 evaluation entry point.
    """
    import torch

    if int(sample_rate) != SAMPLE_RATE:
        raise C0AudioError(f"ASR audio sample rate {sample_rate} != {SAMPLE_RATE}")
    wavs = prepare_asr_waveforms([np.asarray(wav)])
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        batch = processor(wavs, **ASR_PROCESSOR_KWARGS)
        iv = batch.input_values.to(device)
        am = getattr(batch, "attention_mask", None)
        kwargs = {"input_values": iv}
        if am is not None:
            kwargs["attention_mask"] = am.to(device)
        logits = model(**kwargs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        texts = processor.batch_decode(pred_ids)
    return normalize_bahnar_ctc_v1(texts[0])


def translate_asr_text(
    model: Any,
    tokenizer: Any,
    asr_text: str,
    *,
    max_source_length: int,
    generation_max_length: int,
    num_beams: int,
) -> str:
    frame = pd.DataFrame([{"record_uid": "utt", "asr_pred_bahnar": asr_text}])
    pred = run_mt_from_asr(
        model=model,
        tokenizer=tokenizer,
        asr_predictions=frame,
        max_source_length=int(max_source_length),
        generation_max_length=int(generation_max_length),
        num_beams=int(num_beams),
        batch_size=1,
    )
    return normalize_mt_text_v1(pred.iloc[0]["c0_pred_vi"])


def _wrap_model_load(exc: BaseException) -> C0ModelLoadError:
    if isinstance(exc, ImportError):
        return C0ModelLoadError(
            "dependency_missing",
            f"{type(exc).__name__}: required ML dependency is missing",
        )
    return C0ModelLoadError(
        "model_load_failed",
        f"{type(exc).__name__}: model could not be loaded",
    )


def load_asr_bundle(checkpoint: Path, device: str) -> dict[str, Any]:
    root = assert_checkpoint_ready(checkpoint, ASR_REQUIRED_FILES, label="ASR")
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(str(root))
        model = Wav2Vec2ForCTC.from_pretrained(str(root)).to(device)
        model.eval()
    except C0Error:
        raise
    except Exception as exc:
        raise _wrap_model_load(exc) from exc
    return {"model": model, "processor": processor, "checkpoint": str(root)}


def load_mt_bundle(
    checkpoint: Path,
    device: str,
    *,
    config: Mapping[str, Any],
    tokenizer_loader: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    root = assert_checkpoint_ready(checkpoint, MT_REQUIRED_FILES, label="MT")
    loader = tokenizer_loader or load_serving_mt_tokenizer
    try:
        tok_bundle = loader(config)
    except C0TokenizerError as exc:
        raise C0ConfigError(str(exc)) from exc
    try:
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(str(root)).to(device)
        model.eval()
    except C0Error:
        raise
    except Exception as exc:
        raise _wrap_model_load(exc) from exc
    return {
        "model": model,
        "tokenizer": tok_bundle["tokenizer"],
        "checkpoint": str(root),
        "tokenizer_model_id": tok_bundle.get("tokenizer_model_id", LOCKED_MODEL_ID),
        "tokenizer_revision": tok_bundle.get("tokenizer_revision", LOCKED_MODEL_REVISION),
        "tokenizer_source": tok_bundle.get("tokenizer_source", TOKENIZER_SOURCE_HUB_PINNED_RQ1),
        "tokenizer_parity": tok_bundle.get("tokenizer_parity", PARITY_NOT_VERIFIED),
    }


def _generation_ints(config: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        max_source = int(config["max_source_length"])
        max_length = int(config["generation_max_length"])
        beams = int(config["num_beams"])
    except (KeyError, TypeError, ValueError) as exc:
        raise C0ConfigError("generation config requires integer max_source_length, generation_max_length, num_beams") from exc
    if min(max_source, max_length, beams) < 1:
        raise C0ConfigError("generation lengths and num_beams must be >= 1")
    return max_source, max_length, beams


def _public_error(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, C0Error):
        return getattr(exc, "error_code", "c0_error"), str(exc)
    if isinstance(exc, ImportError):
        return "dependency_missing", f"{type(exc).__name__}: required ML dependency is missing"
    if isinstance(exc, (yaml.YAMLError, OSError, UnicodeError, ValueError, TypeError, KeyError)):
        return "config_error", f"{type(exc).__name__}: configuration could not be used"
    return "unexpected_error", f"{type(exc).__name__}: inference could not be completed"


class C0Cascade:
    """Loaded once. ``run`` accepts one Bahnar audio file."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        asr_bundle: Optional[Mapping[str, Any]] = None,
        mt_bundle: Optional[Mapping[str, Any]] = None,
        transcribe: Callable[..., str] = transcribe_waveform,
        translate: Callable[..., str] = translate_asr_text,
        audio_reader: Optional[Callable[..., Any]] = None,
        tokenizer_loader: Optional[Callable[..., dict[str, Any]]] = None,
    ) -> None:
        asr_path = configured_path(config.get("asr_checkpoint")) or Path()
        mt_path = configured_path(config.get("mt_checkpoint")) or Path()
        if asr_bundle is None:
            asr_path = assert_checkpoint_ready(config.get("asr_checkpoint"), ASR_REQUIRED_FILES, label="ASR")
        if mt_bundle is None:
            mt_path = assert_checkpoint_ready(config.get("mt_checkpoint"), MT_REQUIRED_FILES, label="MT")
        self.max_source_length, self.generation_max_length, self.num_beams = _generation_ints(config)
        self.max_audio_duration_seconds = float(
            _positive_number(config, "max_audio_duration_seconds", DEFAULT_MAX_AUDIO_DURATION_SECONDS)
        )
        self.max_audio_bytes = int(
            _positive_number(config, "max_audio_bytes", DEFAULT_MAX_AUDIO_BYTES, integer=True)
        )
        checksum_asr = asr_path if asr_bundle is None else configured_path((asr_bundle or {}).get("checkpoint") or config.get("asr_checkpoint")) or asr_path
        checksum_mt = mt_path if mt_bundle is None else configured_path((mt_bundle or {}).get("checkpoint") or config.get("mt_checkpoint")) or mt_path
        assert_operator_checksum(checksum_asr, optional_sha256(config, "asr_weights_sha256"), label="ASR")
        assert_operator_checksum(checksum_mt, optional_sha256(config, "mt_weights_sha256"), label="MT")
        research = load_research_pins(config)
        self.device = resolve_device(str(config.get("device") or "cpu")) if asr_bundle is None or mt_bundle is None else str(config.get("device") or "cpu")
        self.asr = dict(asr_bundle) if asr_bundle is not None else load_asr_bundle(asr_path, self.device)
        self.mt = dict(mt_bundle) if mt_bundle is not None else load_mt_bundle(
            mt_path, self.device, config=config, tokenizer_loader=tokenizer_loader
        )
        self.transcribe = transcribe
        self.translate = translate
        self.audio_reader = audio_reader
        self.asr_checkpoint = str(self.asr.get("checkpoint") or asr_path)
        self.mt_checkpoint = str(self.mt.get("checkpoint") or mt_path)
        self.asr_assessment = assess_checkpoint(
            configured_path(self.asr_checkpoint),
            ASR_REQUIRED_FILES,
            expected_step=VERIFIED_ASR_CHECKPOINT_STEP,
            configured_sha256=optional_sha256(config, "asr_weights_sha256"),
            research_sha256=research.get("asr"),
        )
        self.mt_assessment = assess_checkpoint(
            configured_path(self.mt_checkpoint),
            MT_REQUIRED_FILES,
            expected_step=VERIFIED_MT_CHECKPOINT_STEP,
            configured_sha256=optional_sha256(config, "mt_weights_sha256"),
            research_sha256=research.get("mt"),
        )
        self.identity = pair_checkpoint_identity(self.asr_assessment, self.mt_assessment)
        self.artifacts_present = bool(
            self.asr_assessment["artifacts_present"] and self.mt_assessment["artifacts_present"]
        )
        self.research_provenance = pair_research_provenance(self.asr_assessment, self.mt_assessment)
        self.generation_matches_rq1 = generation_matches_rq1(
            self.max_source_length, self.generation_max_length, self.num_beams
        )
        self.tokenizer_parity = str(self.mt.get("tokenizer_parity") or PARITY_NOT_VERIFIED)
        self.tokenizer_source = str(self.mt.get("tokenizer_source") or TOKENIZER_SOURCE_HUB_PINNED_RQ1)
        self.tokenizer_model_id = str(self.mt.get("tokenizer_model_id") or LOCKED_MODEL_ID)
        self.tokenizer_revision = str(self.mt.get("tokenizer_revision") or LOCKED_MODEL_REVISION)
        self.mt_training_contract = str(config.get("mt_training_contract") or "").strip()

    def _base_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_id": MODEL_ID,
            "asr_checkpoint": self.asr_checkpoint,
            "mt_checkpoint": self.mt_checkpoint,
            "checkpoint_identity": self.identity,
            "artifacts_present": self.artifacts_present,
            "research_provenance": self.research_provenance,
            "generation_matches_rq1": self.generation_matches_rq1,
            "tokenizer_source": self.tokenizer_source,
            "tokenizer_parity": self.tokenizer_parity,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
        }
        asr_step = self.asr_assessment.get("checkpoint_step")
        mt_step = self.mt_assessment.get("checkpoint_step")
        if asr_step is not None:
            payload["asr_checkpoint_step"] = asr_step
        if mt_step is not None:
            payload["mt_checkpoint_step"] = mt_step
        return payload

    def infer_waveform(self, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
        """Run ASR then MT on an already-decoded waveform. Does not read a file.

        ``duration_ms`` is audio length at ``sample_rate``. ``latency_ms`` is
        wall time of this call (transcribe + translate), not HTTP overhead.
        """
        started = time.perf_counter()
        return self._complete_from_wav(self._base_payload(), wav, sample_rate, started)

    def run(self, audio_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        payload = self._base_payload()
        try:
            wav, sr = read_c0_audio(
                audio_path,
                reader=self.audio_reader,
                max_duration_seconds=self.max_audio_duration_seconds,
                max_audio_bytes=self.max_audio_bytes,
            )
        except C0Error as exc:
            payload["error"] = str(exc)
            payload["error_code"] = getattr(exc, "error_code", "c0_error")
            payload["latency_ms"] = int(round((time.perf_counter() - started) * 1000.0))
            return payload
        except Exception as exc:
            code, message = _public_error(exc)
            payload["error"] = message
            payload["error_code"] = code
            payload["latency_ms"] = int(round((time.perf_counter() - started) * 1000.0))
            return payload
        return self._complete_from_wav(payload, wav, sr, started)

    def _complete_from_wav(
        self,
        payload: dict[str, Any],
        wav: np.ndarray,
        sample_rate: int,
        started: float,
    ) -> dict[str, Any]:
        try:
            payload["duration_ms"] = duration_ms(wav, sample_rate)
            asr_text = self.transcribe(self.asr["model"], self.asr["processor"], wav, sample_rate=sample_rate)
            asr_text = str(asr_text).strip()
            if not asr_text:
                raise C0EmptyOutputError("empty_asr", "ASR produced empty Bahnar transcript")
            payload["asr_text_bahnar"] = asr_text
            vi = self.translate(
                self.mt["model"],
                self.mt["tokenizer"],
                asr_text,
                max_source_length=self.max_source_length,
                generation_max_length=self.generation_max_length,
                num_beams=self.num_beams,
            )
            vi = str(vi).strip()
            if not vi:
                raise C0EmptyOutputError("empty_mt", "MT produced empty Vietnamese translation")
            payload["translated_text_vi"] = vi
        except C0Error as exc:
            payload["error"] = str(exc)
            payload["error_code"] = getattr(exc, "error_code", "c0_error")
        except Exception as exc:
            code, message = _public_error(exc)
            payload["error"] = message
            payload["error_code"] = code
        payload["latency_ms"] = int(round((time.perf_counter() - started) * 1000.0))
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C0 cascaded Bahnar ASR + Vietnamese MT inference")
    parser.add_argument("--config", required=True, type=Path, help="YAML with checkpoint paths and generation settings")
    parser.add_argument("--audio", required=True, type=Path, help="Bahnar audio file")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_c0_config(args.config)
        cascade = C0Cascade(config)
        payload = cascade.run(args.audio)
    except C0Error as exc:
        payload = {
            "model_id": MODEL_ID,
            "error": str(exc),
            "error_code": getattr(exc, "error_code", "config_error"),
        }
    except Exception as exc:
        code, message = _public_error(exc)
        payload = {"model_id": MODEL_ID, "error": message, "error_code": code}
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 1 if payload.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
