"""Provider-selection + logging wrapper around POST /v1/extract.

Logging here MUST NEVER include the citizen message or any raw provider
request/response payload (which may embed the message and, for real
providers, an Authorization header) — only non-PII shape metadata.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.models.extract import ExtractRequest, ExtractResponse
from app.providers.base import ExtractProvider, ProviderError
from app.providers.mock import MockProvider

logger = logging.getLogger("ai_service.extract")


class ExtractServiceError(Exception):
    """Reason is one of the app.providers.base REASON_* constants."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def build_provider(settings: Settings) -> ExtractProvider:
    if settings.provider == "openai":
        from app.providers.openai import OpenAIProvider

        return OpenAIProvider(settings)
    if settings.provider == "gemini":
        from app.providers.gemini import GeminiProvider

        return GeminiProvider(settings)
    return MockProvider()


class ExtractService:
    def __init__(self, settings: Settings, provider: ExtractProvider | None = None) -> None:
        self._settings = settings
        self._provider = provider or build_provider(settings)

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def extract(self, request: ExtractRequest) -> ExtractResponse:
        logger.info(
            "extract_request request_id=%s candidates=%d pinned=%s provider=%s",
            request.request_id,
            len(request.candidates),
            bool(request.pinned_context),
            self._settings.provider,
        )
        try:
            response = self._provider.extract(request)
        except ProviderError as exc:
            logger.warning(
                "extract_provider_error request_id=%s reason=%s provider=%s",
                request.request_id,
                exc.reason,
                self._settings.provider,
            )
            raise ExtractServiceError(exc.reason, str(exc)) from exc
        logger.info(
            "extract_response request_id=%s procedure_selected=%s slot_count=%d",
            request.request_id,
            bool(response.intent and response.intent.procedure_code),
            len(response.slots),
        )
        return response
