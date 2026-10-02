"""OpenAI/Gemini provider tests using httpx.MockTransport — no real network
call is ever made, and no API key is required to run this suite. This is the
only place these adapters are exercised.
"""

import json

import httpx
import pytest

from app.config import Settings
from app.models.extract import Candidate, ExtractRequest, SlotSpec
from app.providers.base import (
    REASON_HTTP_ERROR,
    REASON_MALFORMED,
    REASON_SCHEMA_VIOLATION,
    REASON_TIMEOUT,
    ProviderError,
)
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider

REQUEST = ExtractRequest(
    request_id="11111111-1111-1111-1111-111111111111",
    message="ba",
    candidates=[
        Candidate(
            procedure_code="dk_khai_sinh",
            name="Đăng ký khai sinh",
            intent_examples=[],
            slots={
                "nguoi_di_dang_ky": SlotSpec(
                    type="enum", question="?", enum_values=["cha", "me"]
                )
            },
        )
    ],
    pinned_context=None,
)

VALID_MODEL_OUTPUT = {
    "schema_version": "extract.v1",
    "intent": {"procedure_code": "dk_khai_sinh", "confidence": 0.9, "alternatives": []},
    "slots_for_procedure_code": "dk_khai_sinh",
    "slots": [
        {
            "key": "nguoi_di_dang_ky",
            "value": "cha",
            "confidence": 0.9,
            "evidence": "ba",
            "operation": "set",
        }
    ],
    "abstain_reason": None,
    "provider": "attacker",
    "model": "PII-TOKEN-7788 Nguyen Van A",
}


def _openai_settings() -> Settings:
    return Settings(
        provider="openai",
        model="gpt-4o-mini",
        openai_api_key="fake-not-real",
        openai_base_url="https://api.openai.com/v1",
        gemini_api_key=None,
        gemini_base_url="",
        request_timeout_s=1.0,
    )


def _client_with_handler(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")


def test_openai_happy_path():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(VALID_MODEL_OUTPUT)}}
                ]
            },
        )

    provider = OpenAIProvider(_openai_settings(), client=_client_with_handler(handler))
    resp = provider.extract(REQUEST)
    assert resp.intent.procedure_code == "dk_khai_sinh"
    assert resp.provider == "openai"
    assert resp.model == "gpt-4o-mini"
    assert "PII-TOKEN-7788" not in resp.model


def test_openai_timeout_maps_to_reason_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("boom", request=request)

    provider = OpenAIProvider(_openai_settings(), client=_client_with_handler(handler))
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_TIMEOUT


def test_openai_http_error_status_maps_to_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    provider = OpenAIProvider(_openai_settings(), client=_client_with_handler(handler))
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_HTTP_ERROR


def test_openai_malformed_json_content_maps_to_malformed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    provider = OpenAIProvider(_openai_settings(), client=_client_with_handler(handler))
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_MALFORMED


def test_openai_schema_violating_output_maps_to_schema_violation():
    def handler(request: httpx.Request) -> httpx.Response:
        bad = dict(VALID_MODEL_OUTPUT)
        bad["intent"] = {"procedure_code": "dk_khai_sinh", "confidence": 5.0, "alternatives": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(bad)}}]})

    provider = OpenAIProvider(_openai_settings(), client=_client_with_handler(handler))
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_SCHEMA_VIOLATION


def _gemini_settings() -> Settings:
    return Settings(
        provider="gemini",
        model="gemini-1.5-flash",
        openai_api_key=None,
        openai_base_url="",
        gemini_api_key="fake-not-real",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        request_timeout_s=1.0,
    )


def test_gemini_happy_path():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(VALID_MODEL_OUTPUT)}]}}
                ]
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    provider = GeminiProvider(_gemini_settings(), client=client)
    resp = provider.extract(REQUEST)
    assert resp.intent.procedure_code == "dk_khai_sinh"
    assert resp.provider == "gemini"
    assert resp.model == "gemini-1.5-flash"
    assert "PII-TOKEN-7788" not in resp.model


def test_gemini_timeout_maps_to_reason_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("boom", request=request)

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    provider = GeminiProvider(_gemini_settings(), client=client)
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_TIMEOUT


def test_openai_provider_requires_api_key():
    with pytest.raises(ProviderError):
        OpenAIProvider(
            Settings(
                provider="openai",
                model="gpt-4o-mini",
                openai_api_key=None,
                openai_base_url="https://api.openai.com/v1",
                gemini_api_key=None,
                gemini_base_url="",
                request_timeout_s=1.0,
            )
        )
