"""Gemini-backed provider. Not exercised by any automated test with a live
network call — tests inject a fake `httpx.Client` (via `transport=`) so no
API key is ever required to run the suite.
"""

from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.models.extract import ExtractRequest, ExtractResponse
from app.providers.base import (
    REASON_CONFIG_ERROR,
    REASON_HTTP_ERROR,
    REASON_MALFORMED,
    REASON_SCHEMA_VIOLATION,
    REASON_TIMEOUT,
    ProviderError,
)
from app.providers.prompt import build_system_prompt, build_user_payload


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.gemini_api_key:
            raise ProviderError(REASON_CONFIG_ERROR, "GEMINI_API_KEY is not configured")
        self._settings = settings
        self.model = settings.model
        self._client = client or httpx.Client(
            base_url=settings.gemini_base_url, timeout=settings.request_timeout_s
        )

    def extract(self, request: ExtractRequest) -> ExtractResponse:
        prompt = build_system_prompt() + "\n\n" + json.dumps(
            build_user_payload(request), ensure_ascii=False
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        path = f"/models/{self._settings.model}:generateContent"
        try:
            resp = self._client.post(
                path, json=body, params={"key": self._settings.gemini_api_key}
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(REASON_TIMEOUT, f"gemini request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(REASON_HTTP_ERROR, f"gemini request failed: {exc}") from exc

        if resp.status_code != 200:
            raise ProviderError(
                REASON_HTTP_ERROR, f"gemini returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        try:
            payload = resp.json()
            content = payload["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content)
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(REASON_MALFORMED, f"gemini returned malformed content: {exc}") from exc

        # The model must not choose these. Overwrite even if the JSON set them.
        parsed["provider"] = self.name
        parsed["model"] = self._settings.model
        try:
            return ExtractResponse.model_validate(parsed)
        except ValidationError as exc:
            raise ProviderError(REASON_SCHEMA_VIOLATION, f"gemini output failed schema: {exc}") from exc
