import pytest

from app.config import ConfigError, Settings, load_settings


def test_mock_provider_always_ready():
    s = Settings(
        provider="mock",
        model="",
        openai_api_key=None,
        openai_base_url="",
        gemini_api_key=None,
        gemini_base_url="",
        request_timeout_s=1.0,
        service_token="test-service-token",
    )
    ok, reason = s.provider_ready()
    assert ok is True
    assert reason is None
    status = s.status_dict()
    assert status["provider"] == "mock"
    assert status["ready"] is True


def test_openai_provider_not_ready_without_key():
    s = Settings(
        provider="openai",
        model="gpt-4o-mini",
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        gemini_api_key=None,
        gemini_base_url="",
        request_timeout_s=1.0,
        service_token="test-service-token",
    )
    ok, reason = s.provider_ready()
    assert ok is False
    assert "OPENAI_API_KEY" in reason
    # Status dict must never include a secret, only the boolean/reason.
    status = s.status_dict()
    assert "api_key" not in status
    assert "openai_api_key" not in status


def test_gemini_provider_ready_with_key_and_model():
    s = Settings(
        provider="gemini",
        model="gemini-1.5-flash",
        openai_api_key=None,
        openai_base_url="",
        gemini_api_key="fake-key-not-real",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        request_timeout_s=1.0,
        service_token="test-service-token",
    )
    ok, reason = s.provider_ready()
    assert ok is True
    assert reason is None


def test_unknown_provider_is_not_ready_and_not_rewritten_to_mock():
    s = Settings(
        provider="opneai",
        model="",
        openai_api_key=None,
        openai_base_url="",
        gemini_api_key=None,
        gemini_base_url="",
        request_timeout_s=1.0,
        service_token="test-service-token",
    )
    ok, reason = s.provider_ready()
    assert ok is False
    assert s.provider == "opneai"
    assert "mock, openai, gemini" in (reason or "")
    assert "opneai" not in (reason or "")


def test_load_settings_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "opneai")
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    with pytest.raises(ConfigError):
        load_settings()


def test_load_settings_rejects_bad_timeout(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_S", "nope")
    with pytest.raises(ConfigError):
        load_settings()


def test_load_settings_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("AI_SERVICE_TOKEN", raising=False)
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert "AI_SERVICE_TOKEN" in str(exc.value)


def test_load_settings_rejects_short_token_without_echoing_it(monkeypatch):
    secret = "short-secret"
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("AI_SERVICE_TOKEN", secret)
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert secret not in str(exc.value)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "+inf"])
def test_load_settings_rejects_non_finite_timeout(monkeypatch, raw):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_S", raw)
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert raw.lower() not in str(exc.value).lower()


def test_load_settings_rejects_invalid_model_without_echoing_it(monkeypatch):
    bad = "not a model 0901234567"
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setenv("LLM_MODEL", bad)
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert "0901234567" not in str(exc.value)
    assert bad not in str(exc.value)


def test_status_dict_does_not_include_token():
    s = Settings(
        provider="mock",
        model="",
        openai_api_key=None,
        openai_base_url="",
        gemini_api_key=None,
        gemini_base_url="",
        request_timeout_s=1.0,
        service_token="test-service-token",
    )
    blob = str(s.status_dict())
    assert "test-service-token" not in blob
