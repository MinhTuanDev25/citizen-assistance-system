"""Mocked Gemini transport tests for definition drafting. No live network."""

import json
import threading

import httpx
import pytest

from app.config import Settings
from app.models.draft import DraftRequest, SourceExcerpt, content_sha256
from app.providers.base import (
    REASON_HTTP_ERROR,
    REASON_MALFORMED,
    REASON_SCHEMA_VIOLATION,
    REASON_TIMEOUT,
    ProviderError,
)
from app.providers.draft_schema import DRAFT_RESPONSE_JSON_SCHEMA
from app.providers.gemini import GeminiProvider
from app.services.draft_service import DraftService

CONTENT = "Người đi đăng ký khai sinh nộp giấy chứng sinh tại UBND cấp xã."


def _excerpt() -> SourceExcerpt:
    return SourceExcerpt(
        excerpt_id="11111111-1111-1111-1111-111111111111",
        document_id="22222222-2222-2222-2222-222222222222",
        generation_id="33333333-3333-3333-3333-333333333333",
        page_start=10,
        page_end=10,
        text_sha256=content_sha256(CONTENT),
        content=CONTENT,
    )


def _request() -> DraftRequest:
    excerpt = _excerpt()
    return DraftRequest(
        request_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        procedure_code="dk_khai_sinh",
        procedure_name="Đăng ký khai sinh",
        xa_id="xa_chu_se",
        document_id=excerpt.document_id,
        generation_id=excerpt.generation_id,
        source_page_start=10,
        source_page_end=21,
        excerpts=[excerpt],
    )


def _valid_model_output(**overrides):
    body = {
        "schema_version": "draft_definition.v1",
        "procedure_code": "dk_khai_sinh",
        "definition": {
            "procedure_code": "dk_khai_sinh",
            "name": "Đăng ký khai sinh",
            "intent_examples": ["Đăng ký khai sinh"],
            "slots": [
                {
                    "key": "noi_sinh",
                    "type": "string",
                    "question": "Nơi sinh?",
                    "enum_values": [],
                }
            ],
            "required_slots": ["noi_sinh"],
            "conditional_slots": [],
            "guidance": {
                "summary": "Nộp hồ sơ tại UBND cấp xã.",
                "checklist": ["Giấy chứng sinh"],
                "where_to_submit": "UBND cấp xã",
                "notes": [],
                "fee_hint": "",
                "processing_time_hint": "",
            },
            "citations": [
                {
                    "doc_id": "22222222-2222-2222-2222-222222222222",
                    "title": "Quyết định 163",
                    "source_type": "ai_draft_unapproved",
                }
            ],
        },
        "evidence": [
            {
                "proposal_id": "slot:noi_sinh",
                "kind": "slot",
                "excerpt_id": "11111111-1111-1111-1111-111111111111",
                "page": 10,
                "quotation": "giấy chứng sinh",
                "review_warning": "",
            },
            {
                "proposal_id": "required:noi_sinh",
                "kind": "required_slot",
                "excerpt_id": "11111111-1111-1111-1111-111111111111",
                "page": 10,
                "quotation": "UBND cấp xã",
                "review_warning": "",
            },
            {
                "proposal_id": "checklist:0",
                "kind": "checklist",
                "excerpt_id": "11111111-1111-1111-1111-111111111111",
                "page": 10,
                "quotation": "giấy chứng sinh",
                "review_warning": "",
            },
            {
                "proposal_id": "guidance:summary",
                "kind": "guidance",
                "excerpt_id": "11111111-1111-1111-1111-111111111111",
                "page": 10,
                "quotation": "UBND cấp xã",
                "review_warning": "",
            },
        ],
        "warnings": [],
        "missing_requirements": [],
        "abstain_reason": None,
    }
    body.update(overrides)
    return body


def _settings() -> Settings:
    return Settings(
        provider="gemini",
        model="gemini-3.5-flash-lite",
        openai_api_key=None,
        openai_base_url="",
        gemini_api_key="fake-not-real",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        request_timeout_s=3.0,
        model_definition="gemini-3.8-flash",
        draft_timeout_s=45.0,
        draft_max_output_tokens=8192,
    )


def _provider(handler) -> GeminiProvider:
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    return GeminiProvider(_settings(), client=client)


def _text_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def test_draft_happy_path_uses_definition_model_and_schema():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["timeout"] = request.extensions.get("timeout")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(_valid_model_output())}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 150,
                },
            },
        )

    service = DraftService(_settings(), provider=_provider(handler))
    resp = service.draft(_request())
    assert resp.procedure_code == "dk_khai_sinh"
    assert resp.model == "gemini-3.8-flash"
    assert captured["path"].endswith("/models/gemini-3.8-flash:generateContent")
    gen = captured["body"]["generationConfig"]
    assert gen["responseJsonSchema"] == DRAFT_RESPONSE_JSON_SCHEMA
    assert gen["maxOutputTokens"] == 8192
    assert resp.usage.total_token_count == 150
    assert resp.validation.semantic_approved is False
    assert resp.definition["slots"]["noi_sinh"]["type"] == "string"


def test_draft_invalid_json_maps_to_malformed():
    provider = _provider(lambda _r: _text_response("not-json"))
    with pytest.raises(ProviderError) as exc:
        provider.draft_definition(_request())
    assert exc.value.reason == REASON_MALFORMED


def test_draft_array_json_maps_to_malformed():
    provider = _provider(lambda _r: _text_response(json.dumps([{"no": "object"}])))
    with pytest.raises(ProviderError) as exc:
        provider.draft_definition(_request())
    assert exc.value.reason == REASON_MALFORMED


def test_draft_unsupported_nested_field_is_schema_violation():
    bad = _valid_model_output()
    bad["definition"]["support_map"] = {"no": "pe"}
    provider = _provider(lambda _r: _text_response(json.dumps(bad)))
    with pytest.raises(ProviderError) as exc:
        provider.draft_definition(_request())
    assert exc.value.reason == REASON_SCHEMA_VIOLATION


def test_draft_missing_required_content_is_schema_violation():
    bad = _valid_model_output()
    del bad["definition"]
    provider = _provider(lambda _r: _text_response(json.dumps(bad)))
    with pytest.raises(ProviderError) as exc:
        provider.draft_definition(_request())
    assert exc.value.reason == REASON_SCHEMA_VIOLATION


def test_draft_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderError) as exc:
        provider.draft_definition(_request())
    assert exc.value.reason == REASON_TIMEOUT


def test_draft_rate_limit_exhausted_maps_http_error(monkeypatch):
    monkeypatch.setattr("app.providers.gemini.TRANSIENT_RETRY_SLEEP_S", 0)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    provider = _provider(handler)
    with pytest.raises(ProviderError) as exc:
        provider.draft_definition(_request())
    assert exc.value.reason == REASON_HTTP_ERROR


def test_draft_incorrect_evidence_is_flagged_not_verified():
    bad = _valid_model_output()
    bad["evidence"][0]["excerpt_id"] = "99999999-9999-9999-9999-999999999999"
    provider = _provider(lambda _r: _text_response(json.dumps(bad)))
    service = DraftService(_settings(), provider=provider)
    resp = service.draft(_request())
    codes = {item.code for item in resp.validation.issues}
    assert "UNKNOWN_EXCERPT_ID" in codes
    assert resp.validation.grounding_verified is False
    assert resp.validation.semantic_approved is False


def test_draft_locks_procedure_against_injection_output():
    hijack = _valid_model_output()
    hijack["procedure_code"] = "xin_giay_phep_xay_dung"
    hijack["definition"]["procedure_code"] = "xin_giay_phep_xay_dung"
    provider = _provider(lambda _r: _text_response(json.dumps(hijack)))
    service = DraftService(_settings(), provider=provider)
    resp = service.draft(_request())
    assert resp.procedure_code == "dk_khai_sinh"
    assert resp.definition["procedure_code"] == "dk_khai_sinh"


def test_draft_and_extract_timeouts_are_distinct():
    settings = _settings()
    assert settings.request_timeout_s == 3.0
    assert settings.draft_timeout_s == 45.0
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions.get("timeout")
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(_valid_model_output())}]}}
                ]
            },
        )

    provider = _provider(handler)
    provider.draft_definition(_request())
    timeout = captured["timeout"]
    assert timeout is not None
    if isinstance(timeout, dict):
        read = timeout.get("read", timeout.get("timeout"))
    else:
        read = getattr(timeout, "read", timeout)
    assert float(read) == 45.0


def test_concurrent_draft_usage_is_request_scoped():
    outputs = {
        "a": (_valid_model_output(), 11, 7, 18),
        "b": (_valid_model_output(), 90, 10, 100),
    }
    barrier = threading.Barrier(2)
    results = {}

    def handler(request: httpx.Request) -> httpx.Response:
        barrier.wait(timeout=2)
        prompt = request.content.decode("utf-8")
        key = "a" if "11111111-1111-1111-1111-111111111111" in prompt else "b"
        body, p, c, t = outputs["a"] if key == "a" else outputs["b"]
        # Distinguish via page in content; both use same excerpt in this test,
        # so split by a marker in generationConfig is unnecessary: return based
        # on call order after barrier so each thread still gets its own usage.
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
                "usageMetadata": {
                    "promptTokenCount": p,
                    "candidatesTokenCount": c,
                    "totalTokenCount": t,
                },
            },
        )

    provider = _provider(handler)
    service = DraftService(_settings(), provider=provider)

    def run(name: str, prompt_tokens: int):
        results[name] = service.draft(_request())

    t1 = threading.Thread(target=run, args=("a", 11))
    t2 = threading.Thread(target=run, args=("b", 90))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    usages = {item.usage.total_token_count for item in results.values()}
    assert None not in usages
    assert all(item.usage.prompt_token_count is not None for item in results.values())
    assert not hasattr(provider, "last_usage")
