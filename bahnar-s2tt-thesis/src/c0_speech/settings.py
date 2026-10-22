"""Environment-driven Speech API settings. No secrets in source files."""
from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path

from src.c0_cascaded import DEFAULT_MAX_AUDIO_BYTES, DEFAULT_MAX_AUDIO_DURATION_SECONDS
from src.c0_speech.errors import SpeechConfigError

MIN_TOKEN_LEN = 16
MAX_TOKEN_LEN = 256
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 30.0
DEFAULT_MAX_INFLIGHT = 1
LOOPBACK_HOSTNAMES = frozenset({"localhost", "localhost.", "ip6-localhost", "ip6-loopback"})


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SpeechConfigError("config_error", f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SpeechConfigError("config_error", f"{name} must be a finite number") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise SpeechConfigError("config_error", f"{name} must be a finite number")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw == "":
        return default
    token = raw.lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise SpeechConfigError("config_error", f"{name} must be true or false")


@dataclass(frozen=True)
class SpeechSettings:
    config_path: Path | None
    host: str
    port: int
    device_override: str
    max_upload_bytes: int
    max_audio_seconds: float
    max_inflight: int
    request_timeout_s: float
    shutdown_timeout_s: float
    log_level: str
    service_token: str
    auth_required: bool
    fail_fast: bool
    workers: int

    def validate(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise SpeechConfigError("config_error", "SPEECH_PORT must be between 1 and 65535")
        if self.max_upload_bytes < 1:
            raise SpeechConfigError("config_error", "SPEECH_MAX_UPLOAD_BYTES must be > 0")
        if self.max_audio_seconds <= 0:
            raise SpeechConfigError("config_error", "SPEECH_MAX_AUDIO_SECONDS must be > 0")
        if self.max_inflight != 1:
            raise SpeechConfigError("config_error", "SPEECH_MAX_INFLIGHT must be 1")
        if self.request_timeout_s <= 0:
            raise SpeechConfigError("config_error", "SPEECH_REQUEST_TIMEOUT_S must be > 0")
        if self.shutdown_timeout_s <= 0:
            raise SpeechConfigError("config_error", "SPEECH_SHUTDOWN_TIMEOUT_S must be > 0")
        if self.workers != 1:
            raise SpeechConfigError("config_error", "SPEECH_WORKERS must be 1 (one model-owning process)")
        if self.device_override and self.device_override not in {"cpu", "cuda", "auto"}:
            raise SpeechConfigError("config_error", "SPEECH_DEVICE must be cpu, cuda, auto, or empty")
        if self.auth_required:
            if len(self.service_token) < MIN_TOKEN_LEN or len(self.service_token) > MAX_TOKEN_LEN:
                raise SpeechConfigError("config_error", "SPEECH_SERVICE_TOKEN is missing or invalid")
        if not self.auth_required and not bind_is_loopback(self.host):
            raise SpeechConfigError(
                "config_error",
                "SPEECH_AUTH=off is only allowed when SPEECH_HOST is a loopback address",
            )


def bind_is_loopback(host: str) -> bool:
    """True only for explicit IPv4/IPv6 loopback binds, not LAN/public/wildcard."""
    raw = (host or "").strip().lower()
    if not raw:
        return False
    if raw.startswith("[") and raw.endswith("]") and len(raw) > 2:
        raw = raw[1:-1]
    if "%" in raw:
        raw = raw.split("%", 1)[0]
    if raw in LOOPBACK_HOSTNAMES:
        return True
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return False
    if addr.is_loopback:
        return True
    mapped = getattr(addr, "ipv4_mapped", None)
    return mapped is not None and mapped.is_loopback


def load_settings(environ: dict[str, str] | None = None) -> SpeechSettings:
    if environ is not None:
        old = {k: os.environ.get(k) for k in environ}
        os.environ.update(environ)
        try:
            return _load()
        finally:
            for key, prev in old.items():
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev
    return _load()


def _load() -> SpeechSettings:
    auth_mode = _env("SPEECH_AUTH", "on").lower()
    if auth_mode not in {"on", "off"}:
        raise SpeechConfigError("config_error", "SPEECH_AUTH must be on or off")
    config_raw = _env("C0_INFERENCE_CONFIG")
    settings = SpeechSettings(
        config_path=Path(config_raw) if config_raw else None,
        host=_env("SPEECH_HOST", DEFAULT_HOST) or DEFAULT_HOST,
        port=_env_int("SPEECH_PORT", DEFAULT_PORT),
        device_override=_env("SPEECH_DEVICE"),
        max_upload_bytes=_env_int("SPEECH_MAX_UPLOAD_BYTES", DEFAULT_MAX_AUDIO_BYTES),
        max_audio_seconds=_env_float("SPEECH_MAX_AUDIO_SECONDS", DEFAULT_MAX_AUDIO_DURATION_SECONDS),
        max_inflight=_env_int("SPEECH_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT),
        request_timeout_s=_env_float("SPEECH_REQUEST_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        shutdown_timeout_s=_env_float("SPEECH_SHUTDOWN_TIMEOUT_S", DEFAULT_SHUTDOWN_TIMEOUT_S),
        log_level=(_env("SPEECH_LOG_LEVEL", "INFO") or "INFO").upper(),
        service_token=_env("SPEECH_SERVICE_TOKEN"),
        auth_required=auth_mode != "off",
        fail_fast=_env_bool("SPEECH_FAIL_FAST", True),
        workers=_env_int("SPEECH_WORKERS", 1),
    )
    settings.validate()
    return settings
