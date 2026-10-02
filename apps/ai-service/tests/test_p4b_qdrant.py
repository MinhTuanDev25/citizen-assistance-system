"""Qdrant upsert must observe a completed operation, not only HTTP 200."""

from __future__ import annotations

import uuid

import httpx

from app.indexing.errors import IndexFailure
from app.indexing.pipeline import SavedChunk
from app.indexing.qdrant import QdrantWriter


def _chunk() -> SavedChunk:
    return SavedChunk(0, 1, 1, "xin chao", "a" * 64, 2, "native", uuid.uuid4())


def test_acknowledged_upsert_is_not_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path.endswith("/points"):
            return httpx.Response(200, json={"result": {"status": "acknowledged"}, "status": "ok"})
        if request.method == "PUT":
            return httpx.Response(200, json={"result": True, "status": "ok"})
        return httpx.Response(500, json={"status": "error"})

    writer = QdrantWriter("http://qdrant", "knowledge_chunks")
    transport = httpx.MockTransport(handler)
    original = httpx.Client

    class _Client(original):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.Client = _Client
    try:
        try:
            writer.upsert(uuid.uuid4(), [_chunk()], [[0.0] * 384], {
                "xa_id": "xa_chu_se",
                "document_id": str(uuid.uuid4()),
                "procedure_version_id": str(uuid.uuid4()),
                "vector_dimension": 384,
            })
            raised = None
        except IndexFailure as exc:
            raised = exc
    finally:
        httpx.Client = original
    assert raised is not None and raised.code == "qdrant_failed"
