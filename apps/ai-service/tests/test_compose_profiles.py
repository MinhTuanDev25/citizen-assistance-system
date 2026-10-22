"""Compose profile checks. Keyword-only must not wait on ai-service."""

from pathlib import Path


def _service_block(text: str, name: str, nxt: str) -> str:
    start = text.index(f"\n  {name}:\n")
    end = text.index(f"\n  {nxt}:\n", start + 1)
    return text[start:end]


def test_keyword_profile_does_not_require_ai_service():
    text = Path(__file__).resolve().parents[3].joinpath("deploy/docker-compose.yml").read_text()
    api = _service_block(text, "api", "ai-service")
    ai = _service_block(text, "ai-service", "web")
    assert "\n    profiles:\n      - app\n" in api
    assert "required: false" in api
    assert "\n      - ai\n" in ai
    assert "- app" not in ai
    assert "- full" not in ai
    assert "ports:" not in ai
    start = text.index("\n  minio:\n")
    minio = text[start:text.index("\nvolumes:\n", start)]
    assert "bitnamilegacy/minio:2024.12.18" in minio
    assert "latest" not in minio
    assert "- storage" in minio
    assert "- app" not in minio
    assert "required: false" in api
    assert "minio:" in api
    assert "miniodata:/bitnami/minio/data" in minio
    assert "miniodata:/data\n" not in minio
    assert "VITE_ADMIN_INGESTION=true" in text
    assert "ADMIN_INGESTION_ENABLED=true" in text
    assert "ADMIN_INDEXING_ENABLED: ${ADMIN_INDEXING_ENABLED:-false}" in text
    assert "VITE_ADMIN_INDEXING: ${VITE_ADMIN_INDEXING:-false}" in text
