"""Provider interface + structured error for the extraction backends."""

from __future__ import annotations

from typing import Protocol

from app.models.extract import ExtractRequest, ExtractResponse

# Stable reason codes surfaced to the Go client's fallback-reason metadata.
REASON_TIMEOUT = "timeout"
REASON_HTTP_ERROR = "http_error"
REASON_MALFORMED = "malformed_json"
REASON_SCHEMA_VIOLATION = "schema_violation"
REASON_CONFIG_ERROR = "config_error"


class ProviderError(Exception):
    """Raised by any provider on failure. Never partially fills a response —
    callers must treat this as "no extraction happened", not a degraded one.
    """

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ExtractProvider(Protocol):
    name: str
    model: str

    def extract(self, request: ExtractRequest) -> ExtractResponse:
        """Return a schema-valid ExtractResponse or raise ProviderError."""
        ...
