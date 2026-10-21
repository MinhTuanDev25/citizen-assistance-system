"""Gemini-backed provider. Not exercised by any automated test with a live
network call — tests inject a fake `httpx.Client` (via `transport=`) so no
API key is ever required to run the suite.
"""

from __future__ import annotations

import json
import time

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
from app.providers.gemini_schema import (
    EXTRACT_RESPONSE_JSON_SCHEMA,
    parse_usage_metadata,
    redact_gemini_error,
)
from app.providers.prompt import build_system_prompt, build_user_payload

# One bounded retry for transient Gemini HTTP statuses. Tests may set this to 0.
TRANSIENT_RETRY_SLEEP_S = 1.0


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
        max_tokens = getattr(self._settings, "max_output_tokens", 1024) or 1024
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": EXTRACT_RESPONSE_JSON_SCHEMA,
            },
        }
        path = f"/models/{self._settings.model}:generateContent"
        try:
            resp = self._post_generate(path, body)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                REASON_TIMEOUT, f"gemini request timed out: {redact_gemini_error(str(exc))}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                REASON_HTTP_ERROR, f"gemini request failed: {redact_gemini_error(str(exc))}"
            ) from exc

        http_status = resp.status_code
        if http_status != 200:
            raise ProviderError(
                REASON_HTTP_ERROR,
                f"gemini returned HTTP {http_status}: {redact_gemini_error(resp.text)}",
            )

        usage = parse_usage_metadata({})
        try:
            payload = resp.json()
            usage = parse_usage_metadata(payload)
            content = payload["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                REASON_MALFORMED, f"gemini returned malformed content: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ProviderError(
                REASON_MALFORMED,
                f"gemini returned JSON {type(parsed).__name__}, expected object",
            )

        # The model must not choose these. Overwrite even if the JSON set them.
        parsed["provider"] = self.name
        parsed["model"] = self._settings.model
        try:
            result = ExtractResponse.model_validate(parsed)
        except ValidationError as exc:
            raise ProviderError(
                REASON_SCHEMA_VIOLATION, f"gemini output failed schema: {exc}"
            ) from exc
        return result.model_copy(update={"usage": usage, "http_status": http_status})

    def _post_generate(self, path: str, body: dict) -> httpx.Response:
        """Single generateContent call, with one retry only on HTTP 429/503.

        Auth uses the x-goog-api-key header so the key is never placed in the
        request URL (httpx access logs print URLs).
        """
        headers = {"x-goog-api-key": self._settings.gemini_api_key}
        resp = self._client.post(path, json=body, headers=headers)
        if resp.status_code in (429, 503):
            time.sleep(max(0.0, TRANSIENT_RETRY_SLEEP_S))
            resp = self._client.post(path, json=body, headers=headers)
        return resp
