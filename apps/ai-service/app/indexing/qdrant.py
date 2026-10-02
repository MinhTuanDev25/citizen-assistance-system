"""Qdrant writes wait until the operation is completed, then the points are read back."""

from __future__ import annotations

import math
import uuid

import httpx

from app.indexing.errors import IndexFailure

POINT_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
DEFAULT_BATCH = 32


def point_id(generation_id: uuid.UUID, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{generation_id}:{chunk_index}"))


def _filter(payload_base: dict, generation_id: uuid.UUID) -> dict:
    must = [
        {"key": "xa_id", "match": {"value": payload_base["xa_id"]}},
        {"key": "document_id", "match": {"value": payload_base["document_id"]}},
        {"key": "procedure_version_id", "match": {"value": payload_base["procedure_version_id"]}},
        {"key": "generation_id", "match": {"value": str(generation_id)}},
    ]
    return {"must": must}


class QdrantWriter:
    def __init__(self, base_url: str, collection: str, timeout_s: float = 5.0, batch_size: int = DEFAULT_BATCH) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.timeout_s = timeout_s
        self.batch_size = max(1, batch_size)

    def upsert(self, generation_id: uuid.UUID, chunks: list, vectors: list[list[float]], payload_base: dict, deadline=None) -> int:
        if len(chunks) != len(vectors) or not chunks:
            raise IndexFailure("qdrant_failed")
        dimension = payload_base["vector_dimension"]
        points = []
        expected_ids = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != dimension or any(not math.isfinite(item) for item in vector):
                raise IndexFailure("embedding_dimension")
            pid = point_id(generation_id, chunk.index)
            expected_ids.append(pid)
            points.append(
                {
                    "id": pid,
                    "vector": vector,
                    "payload": _payload(payload_base, generation_id, chunk),
                }
            )
        self._ensure(dimension, deadline)
        for start in range(0, len(points), self.batch_size):
            self._put_batch(points[start : start + self.batch_size], deadline)
        return self._verify(generation_id, expected_ids, points, payload_base, deadline)

    def delete_generation(self, generation_id: uuid.UUID, deadline=None) -> None:
        body = {"filter": {"must": [{"key": "generation_id", "match": {"value": str(generation_id)}}]}}
        response = self._request(
            "POST",
            f"{self.base_url}/collections/{self.collection}/points/delete",
            deadline,
            params={"wait": "true"},
            json=body,
        )
        if response.status_code != 200 or _operation_status(response) != "completed":
            raise IndexFailure("qdrant_failed")
        counted = self._request(
            "POST",
            f"{self.base_url}/collections/{self.collection}/points/count",
            deadline,
            json={"filter": {"must": [{"key": "generation_id", "match": {"value": str(generation_id)}}]}, "exact": True},
        )
        if counted.status_code != 200 or _exact_count(counted) != 0:
            raise IndexFailure("qdrant_failed")

    def _request(self, method: str, url: str, deadline, **kwargs):
        timeout = self.timeout_s if deadline is None else deadline.timeout_for_io()
        try:
            with httpx.Client() as client:
                response = client.request(method, url, timeout=timeout, **kwargs)
        except IndexFailure:
            raise
        except httpx.TimeoutException as exc:
            if deadline is not None:
                deadline.check()
            raise IndexFailure("qdrant_failed") from exc
        except httpx.HTTPError as exc:
            raise IndexFailure("qdrant_failed") from exc
        if deadline is not None:
            deadline.check()
        return response

    def _ensure(self, dimension: int, deadline) -> None:
        ensure = self._request(
            "PUT",
            f"{self.base_url}/collections/{self.collection}",
            deadline,
            json={"vectors": {"size": dimension, "distance": "Cosine"}},
        )
        if ensure.status_code not in (200, 409):
            raise IndexFailure("qdrant_failed")

    def _put_batch(self, points: list[dict], deadline) -> None:
        written = self._request(
            "PUT",
            f"{self.base_url}/collections/{self.collection}/points",
            deadline,
            params={"wait": "true"},
            json={"points": points},
        )
        if written.status_code != 200 or _operation_status(written) != "completed":
            raise IndexFailure("qdrant_failed")

    def _verify(self, generation_id: uuid.UUID, expected_ids: list[str], points: list[dict], payload_base: dict, deadline) -> int:
        counted = self._request(
            "POST",
            f"{self.base_url}/collections/{self.collection}/points/count",
            deadline,
            json={"filter": _filter(payload_base, generation_id), "exact": True},
        )
        count = _exact_count(counted) if counted.status_code == 200 else None
        if count != len(expected_ids):
            raise IndexFailure("qdrant_failed")
        fetched = self._request(
            "POST",
            f"{self.base_url}/collections/{self.collection}/points",
            deadline,
            json={"ids": expected_ids, "with_payload": True, "with_vector": True},
        )
        if fetched.status_code != 200:
            raise IndexFailure("qdrant_failed")
        rows = fetched.json().get("result", [])
        by_id = {row.get("id"): row for row in rows}
        if set(by_id) != set(expected_ids):
            raise IndexFailure("qdrant_failed")
        expected_payload = {item["id"]: item["payload"] for item in points}
        dimension = payload_base["vector_dimension"]
        for pid in expected_ids:
            row = by_id[pid]
            vector = row.get("vector") or []
            if len(vector) != dimension or any(not math.isfinite(float(item)) for item in vector):
                raise IndexFailure("qdrant_failed")
            if row.get("payload") != expected_payload[pid]:
                raise IndexFailure("qdrant_failed")
        return count


def _exact_count(response: httpx.Response):
    try:
        body = response.json()
    except ValueError as exc:
        raise IndexFailure("qdrant_failed") from exc
    result = body.get("result")
    if not isinstance(result, dict) or "count" not in result:
        raise IndexFailure("qdrant_failed")
    return result.get("count")


def _operation_status(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError as exc:
        raise IndexFailure("qdrant_failed") from exc
    result = body.get("result")
    if isinstance(result, dict):
        return str(result.get("status") or "")
    return ""


def _payload(payload_base: dict, generation_id: uuid.UUID, chunk) -> dict:
    return {
        "xa_id": payload_base["xa_id"],
        "document_id": payload_base["document_id"],
        "procedure_version_id": payload_base["procedure_version_id"],
        "generation_id": str(generation_id),
        "chunk_id": str(chunk.chunk_id),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
    }


class MemoryVectors:
    def __init__(self, fail: str | bool | None = None) -> None:
        self.fail = fail
        self.points: list[dict] = []
        self.batches = 0

    def upsert(self, generation_id: uuid.UUID, chunks: list, vectors: list[list[float]], payload_base: dict, deadline=None) -> int:
        if deadline is not None:
            deadline.check()
        if self.fail in (True, "partial", "acknowledged", "restart"):
            if self.fail == "acknowledged":
                raise IndexFailure("qdrant_failed")
            if self.fail in ("partial", "restart") and chunks:
                self.points.append({"id": point_id(generation_id, chunks[0].index), "payload": {"generation_id": str(generation_id)}})
                self.batches += 1
            raise IndexFailure("qdrant_failed")
        if self.fail == "dimension":
            raise IndexFailure("embedding_dimension")
        if len(chunks) != len(vectors):
            raise IndexFailure("qdrant_failed")
        fresh = [item for item in self.points if item["payload"].get("generation_id") != str(generation_id)]
        for chunk, vector in zip(chunks, vectors):
            fresh.append(
                {
                    "id": point_id(generation_id, chunk.index),
                    "vector": vector,
                    "payload": _payload(payload_base, generation_id, chunk),
                }
            )
        self.points = fresh
        if self.fail == "missing":
            self.points = self.points[:-1]
            raise IndexFailure("qdrant_failed")
        return len(chunks)

    def delete_generation(self, generation_id: uuid.UUID, deadline=None) -> None:
        if deadline is not None:
            deadline.check()
        self.points = [item for item in self.points if item["payload"].get("generation_id") != str(generation_id)]
