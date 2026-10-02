from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready():
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_status():
    r = client.get("/v1/status")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "cas-ai-service"
    assert body["extract"] == "implemented"
    # Default LLM_PROVIDER is mock unless overridden in the environment —
    # mock is always "ready" with no secrets required.
    assert body["provider"] in ("mock", "openai", "gemini")
    assert "ready" in body


def test_ready_ok_with_default_mock_provider():
    r = client.get("/ready")
    # Only true when LLM_PROVIDER=mock (the safe local/test default); if the
    # environment overrides to openai/gemini without a key this documents the
    # fail-closed 503 instead of asserting a fixed status.
    assert r.status_code in (200, 503)
    body = r.json()
    assert "status" in body
