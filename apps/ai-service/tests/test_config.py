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


def _pipeline_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setenv("INDEX_MODE", "pipeline")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "onnx")
    monkeypatch.setenv("OCR_PROVIDER", "paddle")
    monkeypatch.setenv("EMBEDDING_MODEL_DIR", "/models/embedding")
    monkeypatch.setenv("OCR_MODEL_DIR", "/models/ocr")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("DATABASE_URL", "postgres://example")
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "https://s3api-eu-ro-1.runpod.io")
    monkeypatch.setenv("OBJECT_STORAGE_ACCESS_KEY", "prod-access")
    monkeypatch.setenv("OBJECT_STORAGE_SECRET_KEY", "prod-secret-value")
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "cas-documents")
    monkeypatch.setenv("OBJECT_STORAGE_REGION", "eu-ro-1")
    monkeypatch.setenv("OBJECT_STORAGE_USE_SSL", "true")
    monkeypatch.setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "false")


def test_pipeline_requires_storage_credentials(monkeypatch):
    from app.indexing.object_storage import storage_config

    _pipeline_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "prod")
    settings = load_settings()
    assert settings.index_mode == "pipeline"
    cfg = storage_config()
    assert cfg["access_key"] == "prod-access"
    assert cfg["secret_key"] == "prod-secret-value"
    assert cfg["secure"] is True
    assert cfg["auto_create"] is False
    monkeypatch.delenv("OBJECT_STORAGE_ACCESS_KEY")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_ACCESS_KEY"):
        load_settings()
    monkeypatch.setenv("OBJECT_STORAGE_ACCESS_KEY", "prod-access")
    monkeypatch.delenv("OBJECT_STORAGE_SECRET_KEY")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_SECRET_KEY"):
        load_settings()
    monkeypatch.setenv("OBJECT_STORAGE_SECRET_KEY", "prod-secret-value")
    monkeypatch.delenv("OBJECT_STORAGE_ENDPOINT")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_ENDPOINT"):
        load_settings()
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "http://storage.internal:9000")
    monkeypatch.setenv("OBJECT_STORAGE_USE_SSL", "false")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_USE_SSL"):
        load_settings()
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "https://s3api-eu-ro-1.runpod.io")
    monkeypatch.setenv("OBJECT_STORAGE_USE_SSL", "true")
    monkeypatch.setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "true")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_AUTO_CREATE_BUCKET"):
        load_settings()
    monkeypatch.setenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET", "false")
    monkeypatch.setenv("OBJECT_STORAGE_ACCESS_KEY", "minioadmin")
    with pytest.raises(ConfigError, match="development defaults"):
        load_settings()


def test_prod_pipeline_requires_an_explicit_bucket(monkeypatch):
    from app.indexing.object_storage import object_bucket_name, storage_config

    _pipeline_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.delenv("OBJECT_STORAGE_BUCKET", raising=False)
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_BUCKET"):
        load_settings()
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_BUCKET"):
        storage_config()
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_BUCKET"):
        object_bucket_name()
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_BUCKET"):
        load_settings()
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "   ")
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_BUCKET"):
        load_settings()
    with pytest.raises(ConfigError, match="OBJECT_STORAGE_BUCKET"):
        storage_config()
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "staging-documents")
    settings = load_settings()
    cfg = storage_config()
    assert settings.object_bucket == "staging-documents"
    assert cfg["bucket"] == "staging-documents"
    assert cfg["auto_create"] is False
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("OBJECT_STORAGE_BUCKET", raising=False)
    settings = load_settings()
    assert settings.object_bucket == "cas-documents"
    assert storage_config()["bucket"] == "cas-documents"


def test_prod_pipeline_does_not_fallback_or_autocreate_bucket(monkeypatch):
    from app.indexing.object_storage import ensure_bucket, storage_config

    _pipeline_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "staging-documents")
    cfg = storage_config()
    assert cfg["bucket"] != "cas-documents"
    assert cfg["auto_create"] is False

    class Probe:
        def bucket_exists(self, bucket):
            assert bucket == "staging-documents"
            return False

        def make_bucket(self, bucket):
            raise AssertionError("production must not auto-create " + bucket)

    assert ensure_bucket(Probe(), cfg["bucket"], cfg["auto_create"]) is False


def test_prod_refuses_mock_index_and_fake_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("INDEX_MODE", "mock")
    with pytest.raises(ConfigError, match="INDEX_MODE=mock"):
        load_settings()
    monkeypatch.setenv("INDEX_MODE", "pipeline")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("OCR_PROVIDER", "paddle")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("DATABASE_URL", "postgres://example")
    with pytest.raises(ConfigError, match="fake embedding"):
        load_settings()


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
