"""Citizen Assistance AI service — FastAPI app.

GET  /health       liveness
GET  /ready        readiness (fails closed if a configured real provider is
                    missing its required secret/model — never silently
                    downgrades to mock)
GET  /v1/status    non-secret service metadata (provider/model/ready)
POST /v1/extract   P2 LLM Extract — strict contract, see app/models/extract.py

No endpoint here ever logs the citizen `message` field or any provider
Authorization header/API key.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from fastapi.responses import JSONResponse

from app.config import ConfigError, Settings, load_settings
from app.models.extract import MAX_REQUEST_BODY_BYTES, ExtractRequest, ExtractResponse
from app.indexing.contract import IndexRequestV2, IndexResponseV2
from app.indexing.errors import IndexFailure
from app.indexing.pipeline import PIPELINE_VERSION, IndexJob, run_pipeline
from app.models.index import MAX_INDEX_BODY_BYTES, IndexRequest, IndexResponse, mock_index
from app.ratelimit import ExtractLimiter
from app.providers.base import (
    REASON_CONFIG_ERROR,
    REASON_HTTP_ERROR,
    REASON_MALFORMED,
    REASON_SCHEMA_VIOLATION,
    REASON_TIMEOUT,
)
from app.services.extract_service import ExtractService, ExtractServiceError

logging.basicConfig(level=logging.INFO)

try:
    _settings: Settings = load_settings()
except ConfigError as exc:
    logging.error("ai-service configuration error: %s", exc)
    raise

from app.indexing.runtime import IndexGate, build_models, current_health

_models = None
_gate = IndexGate(_settings.index_max_inflight)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _models
    if _settings.index_mode == "pipeline":
        _models = build_models(_settings)
    yield
    _gate.shutdown(timeout_s=5, models=_models)


app = FastAPI(title="cas-ai-service", version="0.2.0", lifespan=_lifespan)


class _ExtractBodyLimit:
    """Reject oversized extract and index bodies with 413 before they are parsed."""

    def __init__(self, app) -> None:
        self.app = app
        self.limits = {
            "/v1/extract": MAX_REQUEST_BODY_BYTES,
            "/v1/index": MAX_INDEX_BODY_BYTES,
        }

    async def __call__(self, scope, receive, send):
        limit = self.limits.get(scope.get("path") or "")
        if scope["type"] != "http" or scope.get("method") != "POST" or limit is None:
            await self.app(scope, receive, send)
            return
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body += message.get("body", b"")
            if len(body) > limit:
                payload = json.dumps({"detail": {"reason": "payload_too_large"}}).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": payload})
                return
            if not message.get("more_body", False):
                break

        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)


app.add_middleware(_ExtractBodyLimit)

_extract_service = ExtractService(_settings)
_limiter = ExtractLimiter(_settings.rate_per_minute, _settings.max_inflight, _settings.retry_after_s)


def get_settings() -> Settings:
    return _settings


def get_extract_service() -> ExtractService:
    return _extract_service


def get_limiter() -> ExtractLimiter:
    return _limiter


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    settings = get_settings()
    ok, reason = settings.provider_ready()
    health = current_health(settings, _models, force=True) if ok else {
        "ready": False,
        "reason": reason,
        "checked_at": None,
        "initialized": _models is not None,
        "configured": False,
        "embedding_status": None,
        "embedding_variant": None,
        "cpu_compatible": None,
        "ocr_status": None,
        "ocr_workers_expected": 0,
        "ocr_workers_alive": 0,
    }
    if ok and not _gate.serving:
        ok = False
        reason = "pipeline_busy"
    elif ok:
        ok = bool(health["ready"])
        reason = health["reason"]
    body = {
        "status": "ready" if ok else "not_ready",
        "reason": reason,
        "checked_at": health.get("checked_at"),
        "last_checked_at": health.get("checked_at"),
        "initialized": health.get("initialized"),
        "configured": health.get("configured"),
        "embedding_status": health.get("embedding_status"),
        "embedding_variant": health.get("embedding_variant"),
        "cpu_compatible": health.get("cpu_compatible"),
        "ocr_status": health.get("ocr_status"),
        "ocr_workers_expected": health.get("ocr_workers_expected"),
        "ocr_workers_alive": health.get("ocr_workers_alive"),
        "index_mode": settings.index_mode,
        "embedding_provider": settings.embedding_provider,
        "ocr_provider": settings.ocr_provider,
    }
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/v1/status")
def status() -> dict:
    settings = get_settings()
    pipeline = settings.index_mode == "pipeline"
    body = {
        "service": "cas-ai-service",
        "phase": "4b-content-pipeline" if pipeline else "4a-index-foundation",
        "extract": "implemented",
        "index": "pipeline" if pipeline else "mock",
        "rag": "not_implemented",
    }
    body.update(settings.status_dict())
    provider_ok, provider_reason = settings.provider_ready()
    health = current_health(settings, _models)
    if provider_ok and not _gate.serving:
        body["ready"] = False
        body["reason"] = "pipeline_busy"
    elif provider_ok:
        body["ready"] = bool(health["ready"])
        body["reason"] = health["reason"]
    else:
        body["ready"] = False
        body["reason"] = provider_reason
    body["models_loaded"] = _models is not None
    body["initialized"] = health.get("initialized")
    body["configured"] = health.get("configured")
    body["health_checked_at"] = health.get("checked_at")
    body["last_checked_at"] = health.get("checked_at")
    body["embedding_status"] = health.get("embedding_status")
    body["embedding_variant"] = health.get("embedding_variant")
    body["cpu_compatible"] = health.get("cpu_compatible")
    body["ocr_status"] = health.get("ocr_status")
    body["ocr_workers_expected"] = health.get("ocr_workers_expected")
    body["ocr_workers_alive"] = health.get("ocr_workers_alive")
    return body


_REASON_HTTP_STATUS = {
    REASON_TIMEOUT: 504,
    REASON_HTTP_ERROR: 502,
    REASON_MALFORMED: 502,
    REASON_SCHEMA_VIOLATION: 502,
    REASON_CONFIG_ERROR: 503,
}


def _token_matches(request: Request) -> bool:
    expected = get_settings().service_token
    header = request.headers.get("authorization", "")
    prefix = "Bearer "
    if not expected or not header.startswith(prefix):
        return False
    got = header[len(prefix) :]
    if len(got) != len(expected):
        return False
    return hmac.compare_digest(got, expected)


@app.post("/v1/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest, http_request: Request) -> ExtractResponse:
    """Strict-contract extraction. FastAPI/Pydantic reject malformed bodies
    with 422 automatically (extra="forbid" on every model). Provider-level
    failures map to 502/503/504 below; the Go client treats any non-200 as a
    fallback signal and never trusts a partial body.

    Authorization is a service token. The header value is never logged.
    """
    if not _token_matches(http_request):
        raise HTTPException(status_code=401, detail={"reason": "unauthorized"})
    limiter = get_limiter()
    allowed, limit_reason = limiter.try_acquire()
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"reason": limit_reason},
            headers={"Retry-After": str(limiter.retry_after_s)},
        )
    try:
        return get_extract_service().extract(request)
    except ExtractServiceError as exc:
        status_code = _REASON_HTTP_STATUS.get(exc.reason, 502)
        raise HTTPException(status_code=status_code, detail={"reason": exc.reason}) from exc
    finally:
        limiter.release()


@app.post("/v1/index")
async def index_document(http_request: Request):
    """Mock index.v1, or the offline index.v2 pipeline. Pipeline mode never calls the mock."""
    if not _token_matches(http_request):
        raise HTTPException(status_code=401, detail={"reason": "unauthorized"})
    raw = await http_request.body()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail={"reason": "invalid_json"}) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail={"reason": "invalid_json"})
    version = payload.get("schema_version")
    settings = get_settings()
    if version == "index.v1":
        if settings.index_mode != "mock":
            raise HTTPException(status_code=422, detail={"reason": "mock_disabled"})
        try:
            body = IndexRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail={"reason": "validation"}) from exc
        return mock_index(body)
    if version == "index.v2":
        if settings.index_mode != "pipeline":
            raise HTTPException(status_code=422, detail={"reason": "pipeline_disabled"})
        try:
            body = IndexRequestV2.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail={"reason": "validation"}) from exc
        try:
            from app.indexing.deadline import Deadline

            clock = Deadline(settings.pipeline_timeout_s)
            future = _gate.submit(lambda: _run_v2(body, settings, clock))
            return await asyncio.wait_for(asyncio.wrap_future(future), clock.remaining())
        except IndexFailure as exc:
            return _failed(body, exc.code)
        except asyncio.TimeoutError:
            return _failed(body, "timeout")
    raise HTTPException(status_code=422, detail={"reason": "schema_version"})


def _run_v2(body: IndexRequestV2, settings: Settings, deadline=None) -> IndexResponseV2:
    from app.indexing.chunk import ChunkConfig
    from app.indexing.postgres import PostgresChunks
    from app.indexing.qdrant import QdrantWriter

    if _models is None:
        raise IndexFailure("embedding_model_missing")
    embedder, ocr, renderer = _models
    job = IndexJob(
        schema_version=body.schema_version,
        xa_id=body.xa_id,
        document_id=body.document_id,
        procedure_id=body.procedure_id,
        procedure_version_id=body.procedure_version_id,
        job_id=body.job_id,
        claim_token=body.claim_token,
        generation_id=body.generation_id,
        bucket=body.bucket,
        object_key=body.object_key,
        checksum=body.checksum,
        page_range=body.page_range,
        pipeline_version=body.pipeline_version or PIPELINE_VERSION,
    )
    try:
        deps = _Deps(
            objects=_minio_objects(settings),
            ocr=ocr,
            embedder=embedder,
            vectors=QdrantWriter(settings.qdrant_url, settings.qdrant_collection, batch_size=settings.embed_batch_size),
            chunks=PostgresChunks(settings.database_url),
            renderer=renderer,
        )
        result = run_pipeline(
            job,
            deps,
            max_bytes=20 * 1024 * 1024,
            chunk_config=ChunkConfig(128, 160, 16),
            bucket=settings.object_bucket,
            max_pages=settings.max_pdf_pages,
            max_ocr_pages=settings.max_ocr_pages,
            deadline=deadline,
        )
    except IndexFailure as exc:
        return _failed(body, exc.code)
    return IndexResponseV2(
        schema_version="index.v2",
        document_id=body.document_id,
        procedure_version_id=body.procedure_version_id,
        xa_id=body.xa_id,
        job_id=body.job_id,
        generation_id=result.generation_id,
        outcome="READY",
        error_code=None,
        source_sha256=result.source_sha256,
        content_sha256=result.content_sha256,
        pages_processed=result.pages_processed,
        native_pages=result.native_pages,
        ocr_pages=result.ocr_pages,
        chunk_count=result.chunk_count,
        vector_count=result.vector_count,
        manifest_hash=result.manifest_hash,
        pipeline_version=result.pipeline_version,
        extraction_version=result.extraction_version,
        ocr_version=result.ocr_version,
        embedding_model_id=result.embedding_model_id,
        embedding_revision=result.embedding_revision,
        embedding_checksum=result.embedding_checksum,
        vector_dimension=result.vector_dimension,
    )


def _failed(body: IndexRequestV2, code: str) -> IndexResponseV2:
    return IndexResponseV2(
        schema_version="index.v2",
        document_id=body.document_id,
        procedure_version_id=body.procedure_version_id,
        xa_id=body.xa_id,
        job_id=body.job_id,
        generation_id=body.generation_id,
        outcome="FAILED",
        error_code=code,
    )


class _Deps:
    def __init__(self, objects, ocr, embedder, vectors, chunks, renderer) -> None:
        self.objects = objects
        self.ocr = ocr
        self.embedder = embedder
        self.vectors = vectors
        self.chunks = chunks
        self.renderer = renderer


def _minio_objects(settings: Settings):
    from app.indexing.runtime import fetch_object

    class _Store:
        def get(self, bucket: str, key: str, max_bytes: int | None = None, timeout_s: float | None = None, deadline=None) -> bytes:
            if settings.object_bucket and bucket != settings.object_bucket:
                raise IndexFailure("source_not_found")
            return fetch_object(bucket, key, max_bytes, timeout_s, deadline)

    return _Store()
