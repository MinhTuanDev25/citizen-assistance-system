"""HTTP-level contract tests for POST /v1/extract.

Runs against the default LLM_PROVIDER=mock — no network, no API keys.
"""

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import AUTH_HEADER, TEST_SERVICE_TOKEN

client = TestClient(app)


def post(body):
    return client.post("/v1/extract", json=body, headers=AUTH_HEADER)

VALID_BODY = {
    "schema_version": "extract.v1",
    "request_id": "11111111-1111-1111-1111-111111111111",
    "message": "ba",
    "candidates": [
        {
            "procedure_code": "dk_khai_sinh",
            "name": "Đăng ký khai sinh",
            "intent_examples": ["đăng ký khai sinh"],
            "slots": {
                "nguoi_di_dang_ky": {
                    "type": "enum",
                    "question": "Ai đi đăng ký?",
                    "enum_values": ["cha", "me"],
                }
            },
        }
    ],
    "pinned_context": {
        "procedure_code": "dk_khai_sinh",
        "allowed_slot_keys": ["nguoi_di_dang_ky"],
        "slot_state": {},
    },
}


def test_extract_happy_path():
    r = post(VALID_BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "extract.v1"
    assert body["provider"] == "mock"
    keyed = {s["key"]: s["value"] for s in body["slots"]}
    assert keyed.get("nguoi_di_dang_ky") == "cha"


def test_extract_rejects_unknown_top_level_field():
    bad = dict(VALID_BODY, unexpected_field="nope")
    r = post(bad)
    assert r.status_code == 422


def test_extract_rejects_wrong_schema_version():
    bad = dict(VALID_BODY, schema_version="extract.v2")
    r = post(bad)
    assert r.status_code == 422


def test_extract_rejects_empty_message():
    bad = dict(VALID_BODY, message="")
    r = post(bad)
    assert r.status_code == 422


def test_extract_rejects_oversized_message():
    bad = dict(VALID_BODY, message="a" * 5000)
    r = post(bad)
    assert r.status_code == 422


def test_extract_rejects_non_uuid_request_id():
    bad = dict(VALID_BODY, request_id="not-a-uuid")
    r = post(bad)
    assert r.status_code == 422


def test_extract_rejects_duplicate_candidate_codes():
    bad = dict(VALID_BODY, candidates=VALID_BODY["candidates"] * 2)
    r = post(bad)
    assert r.status_code == 422


def test_extract_rejects_pinned_context_not_in_candidates():
    bad = dict(
        VALID_BODY,
        pinned_context={
            "procedure_code": "does_not_exist",
            "allowed_slot_keys": [],
            "slot_state": {},
        },
    )
    r = post(bad)
    assert r.status_code == 422


def test_extract_rejects_unknown_slot_type():
    bad_body = {
        **VALID_BODY,
        "candidates": [
            {
                **VALID_BODY["candidates"][0],
                "slots": {
                    "x": {"type": "free_text", "question": "?", "enum_values": []},
                },
            }
        ],
    }
    r = post(bad_body)
    assert r.status_code == 422


def test_extract_rejects_out_of_range_confidence_in_response_would_be_caught_upstream():
    # This exercises that the response model itself would reject an
    # out-of-range confidence, guarding provider bugs before they ever leave
    # the process. Constructed directly against the model, not the HTTP path.
    from pydantic import ValidationError

    from app.models.extract import IntentResult

    try:
        IntentResult(procedure_code="x", confidence=1.5)
        raised = False
    except ValidationError:
        raised = True
    assert raised


def test_extract_rejects_missing_token():
    r = client.post("/v1/extract", json=VALID_BODY)
    assert r.status_code == 401
    assert TEST_SERVICE_TOKEN not in r.text


def test_extract_rejects_wrong_token():
    r = client.post(
        "/v1/extract",
        json=VALID_BODY,
        headers={"Authorization": "Bearer not-the-token"},
    )
    assert r.status_code == 401
    assert TEST_SERVICE_TOKEN not in r.text


def test_extract_rejects_oversized_intent_example():
    bad = {
        **VALID_BODY,
        "candidates": [
            {
                **VALID_BODY["candidates"][0],
                "intent_examples": ["x" * 301],
            }
        ],
    }
    r = post(bad)
    assert r.status_code == 422


def test_extract_rejects_body_over_limit():
    payload = b'{"message":"' + (b"a" * (256 * 1024)) + b'"}'
    r = client.post(
        "/v1/extract",
        content=payload,
        headers={**AUTH_HEADER, "Content-Type": "application/json"},
    )
    assert r.status_code == 413
    assert TEST_SERVICE_TOKEN not in r.text
