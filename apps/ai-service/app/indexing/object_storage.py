"""S3-compatible object storage settings shared by MinIO and RunPod."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from app.config import ConfigError, _is_runpod


def normalize_endpoint(raw: str, use_ssl: bool | None = None) -> tuple[str, bool]:
    """Return host:port and whether TLS is required.

    Accepts a hostname, host:port, or a URL with http/https.
    A RunPod hostname without a scheme uses port 443 and requires TLS.
    """
    text = (raw or "").strip()
    if not text:
        raise ConfigError("OBJECT_STORAGE_ENDPOINT is required")
    secure = bool(use_ssl)
    runpod = _is_runpod(text)
    if "://" in text:
        parsed = urlparse(text)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ConfigError("OBJECT_STORAGE_ENDPOINT is invalid")
        if runpod and parsed.scheme != "https":
            raise ConfigError("RunPod object storage requires TLS")
        secure = parsed.scheme == "https"
        host = parsed.hostname
        port = parsed.port
        if port is None:
            port = 443 if secure else 9000
        if runpod:
            if port != 443:
                raise ConfigError("RunPod object storage requires port 443")
            return f"{host}:443", True
        return f"{host}:{port}", secure
    if text.endswith("/"):
        text = text.rstrip("/")
    if runpod:
        if not secure:
            raise ConfigError("OBJECT_STORAGE_USE_SSL must be true for RunPod object storage")
        host, sep, port_text = text.rpartition(":")
        if not sep or not host:
            return f"{text}:443", True
        if port_text != "443":
            raise ConfigError("RunPod object storage requires port 443")
        return f"{host}:443", True
    if ":" not in text:
        port = 443 if secure else 9000
        return f"{text}:{port}", secure
    return text, secure


def storage_config() -> dict:
    endpoint_raw = os.getenv("OBJECT_STORAGE_ENDPOINT", "minio:9000").strip() or "minio:9000"
    secure_raw = os.getenv("OBJECT_STORAGE_USE_SSL", "false").strip().lower()
    secure = secure_raw in ("1", "true", "yes", "on")
    endpoint, secure = normalize_endpoint(endpoint_raw, secure)
    region = os.getenv("OBJECT_STORAGE_REGION", "").strip()
    auto_raw = os.getenv("OBJECT_STORAGE_AUTO_CREATE_BUCKET")
    if auto_raw is None or auto_raw.strip() == "":
        auto_create = not _is_runpod(endpoint_raw)
    else:
        auto_create = auto_raw.strip().lower() in ("1", "true", "yes", "on")
    if _is_runpod(endpoint_raw):
        if not region:
            raise ConfigError("OBJECT_STORAGE_REGION is required for RunPod object storage")
        if auto_create:
            raise ConfigError("OBJECT_STORAGE_AUTO_CREATE_BUCKET must be false for RunPod object storage")
        auto_create = False
    return {
        "endpoint": endpoint,
        "secure": secure,
        "region": region,
        "auto_create": auto_create,
        "access_key": os.getenv("OBJECT_STORAGE_ACCESS_KEY", ""),
        "secret_key": os.getenv("OBJECT_STORAGE_SECRET_KEY", ""),
        "bucket": os.getenv("OBJECT_STORAGE_BUCKET", "cas-documents").strip() or "cas-documents",
    }


def open_client(http_client=None, config: dict | None = None):
    """Build a MinIO client. Region is passed through so location lookup is skipped."""
    from minio import Minio

    cfg = config or storage_config()
    kwargs = {
        "access_key": cfg["access_key"],
        "secret_key": cfg["secret_key"],
        "secure": cfg["secure"],
        "region": cfg["region"] or None,
    }
    if http_client is not None:
        kwargs["http_client"] = http_client
    return Minio(cfg["endpoint"], **kwargs)


def ensure_bucket(client, bucket: str, auto_create: bool) -> bool:
    """Create the bucket only when auto-create is enabled and the bucket is missing."""
    exists = client.bucket_exists(bucket)
    if exists or not auto_create:
        return bool(exists)
    client.make_bucket(bucket)
    return True
