"""Persistent C0 model runtime. Loaded once per worker, reused per request."""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from src.c0_cascaded import (
    C0Cascade,
    C0Error,
    RQ1_GENERATION_MAX_LENGTH,
    RQ1_MAX_SOURCE_LENGTH,
    RQ1_NUM_BEAMS,
    generation_matches_rq1,
    load_c0_config,
)
from src.c0_speech.errors import SpeechAPIError, SpeechConfigError, map_c0_error_code
from src.c0_speech.settings import SpeechSettings
from src.c0_tokenizer import PARITY_VERIFIED

PUBLIC_SAFE = "Inference could not be completed"

HTTP_PUBLIC_FIELDS = (
    "model_id",
    "asr_text_bahnar",
    "translated_text_vi",
    "duration_ms",
    "latency_ms",
    "tokenizer_parity",
    "research_provenance",
    "checkpoint_identity",
    "generation_matches_rq1",
    "tokenizer_source",
    "tokenizer_model_id",
    "tokenizer_revision",
    "asr_checkpoint_step",
    "mt_checkpoint_step",
    "artifacts_present",
)


class InferGate:
    """Single-flight GPU occupancy. Non-blocking acquire.

    Step 7.4 admits exactly one inference. ``max_inflight`` must be 1 so
    the advertised limit matches actual execution (no hidden queue).
    """

    def __init__(self, max_inflight: int) -> None:
        if max_inflight != 1:
            raise ValueError("max_inflight must be 1")
        self.max_inflight = max_inflight
        self._lock = threading.Lock()
        self._inflight = 0
        self._accepting = True

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting

    def shutdown(self) -> None:
        with self._lock:
            self._accepting = False

    def try_acquire(self) -> str:
        """Return empty string if acquired, otherwise an error code."""
        with self._lock:
            if not self._accepting:
                return "shutting_down"
            if self._inflight >= self.max_inflight:
                return "overloaded"
            self._inflight += 1
            return ""

    def release(self) -> None:
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1


class SpeechRuntime:
    """Owns one C0Cascade. ``infer`` is thread-safe only through InferGate."""

    def __init__(
        self,
        settings: SpeechSettings,
        *,
        cascade: Optional[C0Cascade] = None,
        cascade_factory: Optional[Callable[[Mapping[str, Any]], C0Cascade]] = None,
        config_loader: Optional[Callable[..., dict[str, Any]]] = None,
    ) -> None:
        self.settings = settings
        self.gate = InferGate(settings.max_inflight)
        self.load_count = 0
        self.infer_count = 0
        self.init_ms = 0
        self.ready = False
        self.init_error: Optional[SpeechAPIError] = None
        self.cascade: Optional[C0Cascade] = cascade
        self._cascade_factory = cascade_factory or (lambda cfg: C0Cascade(cfg))
        self._config_loader = config_loader or load_c0_config
        self._infer_lock = threading.Lock()

    def start(self) -> None:
        started = time.perf_counter()
        loaded_config: Optional[dict[str, Any]] = None
        try:
            if self.cascade is None:
                if self.settings.config_path is None:
                    self.init_error = SpeechConfigError("config_error", "C0_INFERENCE_CONFIG is required")
                    self.ready = False
                    if self.settings.fail_fast:
                        raise self.init_error
                    return
                config = dict(self._config_loader(self.settings.config_path))
                if self.settings.device_override:
                    config["device"] = self.settings.device_override
                config["max_audio_duration_seconds"] = self.settings.max_audio_seconds
                config["max_audio_bytes"] = self.settings.max_upload_bytes
                loaded_config = config
                self.cascade = self._cascade_factory(config)
            contract = serving_contract_error(self.cascade, config=loaded_config)
            if contract is not None:
                self.init_error = contract
                self.ready = False
                if self.settings.fail_fast:
                    raise contract
                return
            self.load_count = max(self.load_count, 1)
            self.ready = True
            self.init_error = None
        except SpeechAPIError as exc:
            self.init_error = exc
            self.ready = False
            if self.settings.fail_fast:
                raise
        except C0Error as exc:
            status, code = map_c0_error_code(getattr(exc, "error_code", "config_error"))
            self.init_error = SpeechAPIError(code, str(exc), status)
            self.ready = False
            if self.settings.fail_fast:
                raise self.init_error from exc
        except Exception as exc:
            self.init_error = SpeechAPIError("model_load_failed", "Model runtime could not be loaded", 503)
            self.ready = False
            if self.settings.fail_fast:
                raise self.init_error from exc
        finally:
            self.init_ms = int(round((time.perf_counter() - started) * 1000.0))

    def close(self) -> None:
        self.gate.shutdown()
        self.ready = False

    async def drain(self, timeout_s: Optional[float] = None) -> None:
        """Stop admission, then wait for the in-flight GPU slot to empty.

        Does not cancel the worker thread. Raises if CUDA/CPU work is still
        occupying the slot when the grace period expires.
        """
        self.close()
        limit = float(self.settings.shutdown_timeout_s if timeout_s is None else timeout_s)
        deadline = time.monotonic() + max(limit, 0.0)
        while self.gate.inflight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if self.gate.inflight > 0:
            raise SpeechAPIError(
                "shutdown_timeout",
                "In-flight inference did not finish before the shutdown grace period",
                503,
            )

    def infer(self, wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
        if not self.ready or self.cascade is None:
            raise SpeechAPIError("not_ready", "Model runtime is not ready", 503)
        with self._infer_lock:
            self.infer_count += 1
            payload = self.cascade.infer_waveform(wav, sample_rate)
        if payload.get("error"):
            status, code = map_c0_error_code(str(payload.get("error_code") or ""))
            raise SpeechAPIError(code, PUBLIC_SAFE, status)
        return payload


def serving_contract_error(
    cascade: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Optional[SpeechAPIError]:
    """Speech-serving gates. Independent of research_provenance."""
    parity = str(getattr(cascade, "tokenizer_parity", "") or "")
    if parity != PARITY_VERIFIED:
        return SpeechAPIError("tokenizer_not_verified", "Tokenizer parity is not VERIFIED", 503)
    if not _generation_contract_ok(cascade):
        return SpeechAPIError("generation_not_rq1", "Generation settings do not match the RQ1 contract", 503)
    if not _mt_training_contract_ok(cascade, config):
        return SpeechAPIError("mt_contract_missing", "MT training contract path is missing or not a file", 503)
    return None


def _generation_contract_ok(cascade: Any) -> bool:
    if not bool(getattr(cascade, "generation_matches_rq1", False)):
        return False
    try:
        max_source_length = int(getattr(cascade, "max_source_length"))
        generation_max_length = int(getattr(cascade, "generation_max_length"))
        num_beams = int(getattr(cascade, "num_beams"))
    except (TypeError, ValueError, AttributeError):
        return False
    return generation_matches_rq1(max_source_length, generation_max_length, num_beams) and (
        max_source_length == RQ1_MAX_SOURCE_LENGTH
        and generation_max_length == RQ1_GENERATION_MAX_LENGTH
        and num_beams == RQ1_NUM_BEAMS
    )


def _mt_training_contract_ok(cascade: Any, config: Optional[Mapping[str, Any]]) -> bool:
    if config is not None:
        raw = str(config.get("mt_training_contract") or "").strip()
    else:
        raw = str(getattr(cascade, "mt_training_contract", "") or "").strip()
    if not raw:
        return False
    try:
        return Path(raw).is_file()
    except OSError:
        return False


def public_inference_body(payload: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {"request_id": request_id}
    for key in HTTP_PUBLIC_FIELDS:
        if key in payload:
            body[key] = payload[key]
    return body
