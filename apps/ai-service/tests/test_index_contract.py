import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.models.index import FORCE_FAIL_CHECKSUM
from tests.conftest import AUTH_HEADER


def _body(**overrides):
    payload = {
        "schema_version": "index.v1",
        "document_id": str(uuid.uuid4()),
        "xa_id": "xa_chu_se",
        "domain_id": "ho_tich_chung_thuc",
        "checksum": "a" * 64,
        "procedure_version_id": str(uuid.uuid4()),
        "relationship_type": "SOURCE",
        "page_range": "1-2",
    }
    payload.update(overrides)
    return payload


def test_mock_index_ready_and_failed_without_chunk_fields():
    client = TestClient(app)
    ready = client.post("/v1/index", json=_body(), headers=AUTH_HEADER)
    assert ready.status_code == 200
    body = ready.json()
    assert body["outcome"] == "READY"
    assert body["error_code"] is None
    assert "embedding" not in body
    assert "content" not in body
    failed = client.post("/v1/index", json=_body(checksum=FORCE_FAIL_CHECKSUM), headers=AUTH_HEADER)
    assert failed.status_code == 200
    assert failed.json()["outcome"] == "FAILED"
    assert failed.json()["error_code"] == "mock_failed"


def test_index_rejects_reversed_page_range_trailing_json_and_oversize():
    client = TestClient(app)
    reversed_range = client.post("/v1/index", json=_body(page_range="2-1"), headers=AUTH_HEADER)
    assert reversed_range.status_code == 422
    trailing = client.post(
        "/v1/index",
        content=_body_raw() + b'{"extra":true}',
        headers={**AUTH_HEADER, "Content-Type": "application/json"},
    )
    assert trailing.status_code == 422
    huge = b'{"schema_version":"index.v1","pad":"' + (b"a" * 9000) + b'"}'
    over = client.post(
        "/v1/index",
        content=huge,
        headers={**AUTH_HEADER, "Content-Type": "application/json"},
    )
    assert over.status_code == 413


def _body_raw():
    import json

    return json.dumps(_body()).encode()


def test_extract_and_index_body_limits_stay_independent_when_concurrent():
    import asyncio
    import json

    import httpx

    extract_body = json.dumps(
        {
            "schema_version": "extract.v1",
            "request_id": str(uuid.uuid4()),
            "message": "x" * 20000,
            "candidates": [],
        }
    ).encode()
    index_body = b'{"schema_version":"index.v1","pad":"' + (b"a" * 9000) + b'"}'
    headers = {**AUTH_HEADER, "Content-Type": "application/json"}

    async def once():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            extract_res, index_res = await asyncio.gather(
                client.post("/v1/extract", content=extract_body, headers=headers),
                client.post("/v1/index", content=index_body, headers=headers),
            )
            assert extract_res.status_code != 413
            assert index_res.status_code == 413

    async def many():
        await asyncio.gather(*(once() for _ in range(12)))

    asyncio.run(many())


def test_index_rejects_unknown_fields_and_missing_token():
    client = TestClient(app)
    extra = client.post("/v1/index", json=_body(pdf_bytes="nope"), headers=AUTH_HEADER)
    assert extra.status_code == 422
    missing = client.post("/v1/index", json=_body())
    assert missing.status_code == 401
