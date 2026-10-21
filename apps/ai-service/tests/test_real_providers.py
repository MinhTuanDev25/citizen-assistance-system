"""OpenAI/Gemini provider tests using httpx.MockTransport — no real network
call is ever made, and no API key is required to run this suite. This is the
only place these adapters are exercised.
"""

import json
import threading

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
from app.providers.gemini_schema import EXTRACT_RESPONSE_JSON_SCHEMA, parse_usage_metadata, redact_gemini_error
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


@pytest.mark.parametrize(
    "body",
    [
        json.dumps(["not", "an", "object"]),
        json.dumps("string-not-object"),
        json.dumps(42),
        json.dumps(True),
        "null",
    ],
)
def test_openai_non_object_json_maps_to_malformed(body: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})

    provider = OpenAIProvider(_openai_settings(), client=_client_with_handler(handler))
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_MALFORMED
    assert "expected object" in str(exc_info.value)


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
        model="gemini-3.5-flash-lite",
        openai_api_key=None,
        openai_base_url="",
        gemini_api_key="fake-not-real",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        request_timeout_s=1.0,
        max_output_tokens=1024,
    )


def test_gemini_happy_path():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["url"] = str(request.url)
        captured["api_key_header"] = request.headers.get("x-goog-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(VALID_MODEL_OUTPUT)}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 7,
                    "totalTokenCount": 18,
                },
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
    assert resp.model == "gemini-3.5-flash-lite"
    assert "PII-TOKEN-7788" not in resp.model
    assert captured["path"].endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert "key=" not in captured["url"]
    assert captured["api_key_header"] == "fake-not-real"
    gen = captured["body"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseJsonSchema"] == EXTRACT_RESPONSE_JSON_SCHEMA
    assert gen["maxOutputTokens"] == 1024
    assert "provider" not in gen["responseJsonSchema"]["properties"]
    assert "model" not in gen["responseJsonSchema"]["properties"]
    assert resp.usage["prompt_token_count"] == 11
    assert resp.usage["candidates_token_count"] == 7
    assert resp.usage["total_token_count"] == 18
    assert resp.http_status == 200
    dumped = resp.model_dump(mode="json")
    assert "usage" not in dumped
    assert "http_status" not in dumped


def _gemini_text_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def _gemini_provider_for(handler) -> GeminiProvider:
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    return GeminiProvider(_gemini_settings(), client=client)


@pytest.mark.parametrize(
    "body",
    [
        json.dumps(["not", "an", "object"]),
        json.dumps("string-not-object"),
        json.dumps(42),
        json.dumps(True),
        "null",
    ],
)
def test_gemini_non_object_json_maps_to_malformed(body: str):
    provider = _gemini_provider_for(lambda _request: _gemini_text_response(body))
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_MALFORMED
    assert "expected object" in str(exc_info.value)


def test_gemini_retries_once_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("app.providers.gemini.TRANSIENT_RETRY_SLEEP_S", 0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(VALID_MODEL_OUTPUT)}]}}
                ]
            },
        )

    provider = _gemini_provider_for(handler)
    resp = provider.extract(REQUEST)
    assert calls["n"] == 2
    assert resp.provider == "gemini"
    assert resp.http_status == 200


def test_gemini_does_not_retry_on_403():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="denied")

    provider = _gemini_provider_for(handler)
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_HTTP_ERROR
    assert calls["n"] == 1


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


def test_gemini_http_error_redacts_key_material():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="models/gemini-1.5-flash is not found key=AIzaSyDummyTestKeyMaterial")

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    provider = GeminiProvider(_gemini_settings(), client=client)
    with pytest.raises(ProviderError) as exc_info:
        provider.extract(REQUEST)
    assert exc_info.value.reason == REASON_HTTP_ERROR
    assert "AIza" not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)


def test_parse_usage_metadata_and_redact():
    assert parse_usage_metadata({})["total_token_count"] is None
    assert redact_gemini_error("https://example/v1beta/models/x:generateContent?key=secret-token-value") == (
        "https://example/v1beta/models/x:generateContent?key=<redacted>"
    )
    redacted_aq = redact_gemini_error("denied key=AQ.AbcdEFGHijklMNOP")
    assert "AQ." not in redacted_aq
    assert "<redacted>" in redacted_aq


def test_gemini_concurrent_requests_do_not_swap_usage():
    barrier = threading.Barrier(2)
    req_a = REQUEST.model_copy(update={"message": "marker-aaa ba"})
    req_b = REQUEST.model_copy(update={"message": "marker-bbb ba"})

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8")
        barrier.wait(timeout=5)
        if "marker-aaa" in body:
            usage = {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18}
        else:
            usage = {"promptTokenCount": 99, "candidatesTokenCount": 3, "totalTokenCount": 102}
        return httpx.Response(
            200,
            json={
                "usageMetadata": usage,
                "candidates": [{"content": {"parts": [{"text": json.dumps(VALID_MODEL_OUTPUT)}]}}],
            },
        )

    provider = _gemini_provider_for(handler)
    out: dict[str, object] = {}

    def run(name: str, req: ExtractRequest) -> None:
        resp = provider.extract(req)
        out[name] = (resp.usage["prompt_token_count"], resp.usage["candidates_token_count"], resp.http_status)

    t1 = threading.Thread(target=run, args=("a", req_a))
    t2 = threading.Thread(target=run, args=("b", req_b))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert out["a"] == (11, 7, 200)
    assert out["b"] == (99, 3, 200)


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
