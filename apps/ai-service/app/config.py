"""Runtime configuration for the AI service — env-driven, no secrets in source.

LLM_PROVIDER selects the extraction backend:
  mock    — deterministic, no network, safe default for local/test.
  openai  — requires OPENAI_API_KEY (+ optional LLM_MODEL / OPENAI_BASE_URL).
  gemini  — requires GEMINI_API_KEY (+ optional LLM_MODEL / GEMINI_BASE_URL).

Readiness (`GET /ready`) fails closed when a real provider is selected but its
required key/model is missing — never silently falls back to mock in that case.
An unrecognized LLM_PROVIDER is a startup error, not a silent switch to mock.
AI_SERVICE_TOKEN is required so POST /v1/extract can reject unauthorized callers.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from app.models.extract import MAX_MODEL_LEN, MODEL_RE


class ConfigError(ValueError):
    """Invalid process configuration. Safe to log: it never includes secret values."""

VALID_PROVIDERS = ("mock", "openai", "gemini")

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_MOCK_MODEL = "mock-extract-v1"
MIN_SERVICE_TOKEN_LEN = 16
MAX_SERVICE_TOKEN_LEN = 256


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} is not an integer") from exc


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    openai_api_key: str | None
    openai_base_url: str
    gemini_api_key: str | None
    gemini_base_url: str
    request_timeout_s: float
    service_token: str = ""
    rate_per_minute: int = 60
    max_inflight: int = 4
    retry_after_s: int = 1
    index_mode: str = "mock"
    object_bucket: str = "cas-documents"
    qdrant_url: str = ""
    qdrant_collection: str = "knowledge_chunks"
    database_url: str = ""
    embedding_model_dir: str = ""
    ocr_model_dir: str = ""
    embedding_provider: str = "fake"
    ocr_provider: str = "fake"
    index_max_inflight: int = 1
    pipeline_timeout_s: float = 120
    max_pdf_pages: int = 50
    max_ocr_pages: int = 20
    embed_batch_size: int = 8

    def provider_ready(self) -> tuple[bool, str | None]:
        """Returns (ready, reason_if_not). Used by /ready — fail closed."""
        if not self.service_token:
            return False, "AI_SERVICE_TOKEN is required"
        if self.provider == "mock":
            return True, None
        if self.provider == "openai":
            if not self.openai_api_key:
                return False, "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            if not self.model:
                return False, "LLM_MODEL is required when LLM_PROVIDER=openai"
            return True, None
        if self.provider == "gemini":
            if not self.gemini_api_key:
                return False, "GEMINI_API_KEY is required when LLM_PROVIDER=gemini"
            if not self.model:
                return False, "LLM_MODEL is required when LLM_PROVIDER=gemini"
            return True, None
        return False, "LLM_PROVIDER is not one of mock, openai, gemini"

    def status_dict(self) -> dict:
        """Non-secret status payload for GET /v1/status. Never includes API keys."""
        ready, reason = self.provider_ready()
        if ready and self.index_mode == "pipeline":
            ready, reason = self.index_ready()
        return {
            "provider": self.provider,
            "model": self.model or None,
            "ready": ready,
            "reason": reason,
            "rate_per_minute": self.rate_per_minute,
            "max_inflight": self.max_inflight,
            "index_mode": self.index_mode,
            "embedding_provider": self.embedding_provider,
            "ocr_provider": self.ocr_provider,
            "models_loaded": False,
            "postgres_configured": bool(self.database_url),
            "qdrant_configured": bool(self.qdrant_url),
            "object_storage_configured": bool(self.object_bucket),
        }

    def index_ready(self) -> tuple[bool, str | None]:
        if self.index_mode != "pipeline":
            return True, None
        if not self.database_url or not self.qdrant_url or not self.object_bucket:
            return False, "pipeline dependencies are not configured"
        if self.embedding_provider == "onnx" and not self.embedding_model_dir:
            return False, "embedding model is not ready"
        if self.ocr_provider == "paddle" and not self.ocr_model_dir:
            return False, "ocr model is not ready"
        return True, None


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower() or "mock"
    if provider not in VALID_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER must be one of {', '.join(VALID_PROVIDERS)}"
        )

    token = os.getenv("AI_SERVICE_TOKEN", "").strip()
    if len(token) < MIN_SERVICE_TOKEN_LEN or len(token) > MAX_SERVICE_TOKEN_LEN:
        raise ConfigError(
            f"AI_SERVICE_TOKEN must be {MIN_SERVICE_TOKEN_LEN} to {MAX_SERVICE_TOKEN_LEN} characters"
        )

    default_model = DEFAULT_MOCK_MODEL
    if provider == "openai":
        default_model = DEFAULT_OPENAI_MODEL
    elif provider == "gemini":
        default_model = DEFAULT_GEMINI_MODEL
    model = os.getenv("LLM_MODEL", "").strip() or default_model
    if len(model) > MAX_MODEL_LEN or MODEL_RE.fullmatch(model) is None:
        raise ConfigError("LLM_MODEL must match the response model-id pattern")

    timeout_s = _env_float("LLM_REQUEST_TIMEOUT_S", 8.0)
    if timeout_s <= 0 or timeout_s > 8:
        raise ConfigError("LLM_REQUEST_TIMEOUT_S must be in (0, 8]")

    rate_per_minute = _bounded_int("EXTRACT_RATE_PER_MINUTE", 60, 1, 10000)
    max_inflight = _bounded_int("EXTRACT_MAX_INFLIGHT", 4, 1, 128)
    retry_after_s = _bounded_int("EXTRACT_RETRY_AFTER_SECONDS", 1, 1, 60)
    index_mode, embedding_provider, ocr_provider = _pipeline_settings()

    return Settings(
        provider=provider,
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_base_url=os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ).strip(),
        request_timeout_s=timeout_s,
        service_token=token,
        rate_per_minute=rate_per_minute,
        max_inflight=max_inflight,
        retry_after_s=retry_after_s,
        index_mode=index_mode,
        object_bucket=os.getenv("OBJECT_STORAGE_BUCKET", "cas-documents").strip() or "cas-documents",
        qdrant_url=os.getenv("QDRANT_URL", "").strip(),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "knowledge_chunks").strip() or "knowledge_chunks",
        database_url=os.getenv("DATABASE_URL", "").strip(),
        embedding_model_dir=os.getenv("EMBEDDING_MODEL_DIR", "").strip(),
        ocr_model_dir=os.getenv("OCR_MODEL_DIR", "").strip(),
        embedding_provider=embedding_provider,
        ocr_provider=ocr_provider,
        index_max_inflight=_bounded_int("INDEX_MAX_INFLIGHT", 1, 1, 4),
        pipeline_timeout_s=float(_bounded_int("INDEX_PIPELINE_TIMEOUT_SECONDS", 120, 1, 900)),
        max_pdf_pages=_bounded_int("INDEX_MAX_PDF_PAGES", 50, 1, 500),
        max_ocr_pages=_bounded_int("INDEX_MAX_OCR_PAGES", 20, 1, 100),
        embed_batch_size=_bounded_int("EMBED_BATCH_SIZE", 8, 1, 32),
    )


def _pipeline_settings() -> tuple[str, str, str]:
    mode = os.getenv("INDEX_MODE", "mock").strip().lower() or "mock"
    if mode not in ("mock", "pipeline"):
        raise ConfigError("INDEX_MODE must be mock or pipeline")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    ocr_provider = os.getenv("OCR_PROVIDER", "").strip().lower()
    if embedding_provider == "":
        embedding_provider = "onnx" if mode == "pipeline" else "fake"
    if ocr_provider == "":
        ocr_provider = "paddle" if mode == "pipeline" else "fake"
    if mode != "pipeline":
        return mode, embedding_provider or "fake", ocr_provider or "fake"
    if embedding_provider not in ("onnx", "fake"):
        raise ConfigError("EMBEDDING_PROVIDER must be onnx or fake")
    if ocr_provider not in ("paddle", "fake"):
        raise ConfigError("OCR_PROVIDER must be paddle or fake")
    if embedding_provider == "onnx" and not os.getenv("EMBEDDING_MODEL_DIR", "").strip():
        raise ConfigError("EMBEDDING_MODEL_DIR is required when EMBEDDING_PROVIDER=onnx")
    if ocr_provider == "paddle" and not os.getenv("OCR_MODEL_DIR", "").strip():
        raise ConfigError("OCR_MODEL_DIR is required when OCR_PROVIDER=paddle")
    if not os.getenv("QDRANT_URL", "").strip() or not os.getenv("DATABASE_URL", "").strip():
        raise ConfigError("QDRANT_URL and DATABASE_URL are required when INDEX_MODE=pipeline")
    return mode, embedding_provider, ocr_provider


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    value = _env_int(name, default)
    if value < low or value > high:
        raise ConfigError(f"{name} is outside the allowed range")
    return value
