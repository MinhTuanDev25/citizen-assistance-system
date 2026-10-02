"""OpenAI-backed provider. Not exercised by any automated test with a live
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

_JSON_INSTRUCTION = (
    "Respond with a single JSON object only, matching the schema described. "
    "No markdown, no prose, no extra keys."
)


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.openai_api_key:
            raise ProviderError(REASON_CONFIG_ERROR, "OPENAI_API_KEY is not configured")
        self._settings = settings
        self.model = settings.model
        self._client = client or httpx.Client(
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_s,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        )

    def extract(self, request: ExtractRequest) -> ExtractResponse:
        body = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": build_system_prompt() + " " + _JSON_INSTRUCTION},
                {"role": "user", "content": json.dumps(build_user_payload(request), ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        try:
            resp = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(REASON_TIMEOUT, f"openai request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(REASON_HTTP_ERROR, f"openai request failed: {exc}") from exc

        if resp.status_code != 200:
            raise ProviderError(
                REASON_HTTP_ERROR, f"openai returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        try:
            payload = resp.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(REASON_MALFORMED, f"openai returned malformed content: {exc}") from exc

        # The model must not choose these. Overwrite even if the JSON set them.
        parsed["provider"] = self.name
        parsed["model"] = self._settings.model
        try:
            return ExtractResponse.model_validate(parsed)
        except ValidationError as exc:
            raise ProviderError(REASON_SCHEMA_VIOLATION, f"openai output failed schema: {exc}") from exc
