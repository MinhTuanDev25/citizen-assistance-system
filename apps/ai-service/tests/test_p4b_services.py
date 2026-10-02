"""Live MinIO, PostgreSQL, and Qdrant. Skipped unless CAS_P4B_INTEGRATION=1."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

from app.indexing.chunk import ChunkConfig
from app.indexing.pipeline import PIPELINE_VERSION, FakeEmbedder, IndexJob, run_pipeline
from app.indexing.postgres import PostgresChunks
from app.indexing.qdrant import QdrantWriter, point_id
from app.indexing.ocr import FakeOCR

pytestmark = pytest.mark.skipif(os.getenv("CAS_P4B_INTEGRATION") != "1", reason="set CAS_P4B_INTEGRATION=1")


def test_pipeline_writes_postgres_and_qdrant_from_minio():
    import psycopg
    from minio import Minio

    dsn = os.environ["DATABASE_URL"]
    endpoint = os.environ.get("OBJECT_STORAGE_ENDPOINT", "127.0.0.1:9000")
    qdrant = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    data = (Path(__file__).parent / "fixtures" / "native.pdf").read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    doc_id = uuid.uuid4()
    job_id = uuid.uuid4()
    generation_id = uuid.uuid4()
    request_id = uuid.uuid4()
    claim = uuid.uuid4()
    key = f"xa_chu_se/documents/{doc_id}/{checksum}.pdf"
    client = Minio(
        endpoint,
        access_key=os.environ.get("OBJECT_STORAGE_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("OBJECT_STORAGE_SECRET_KEY", "minioadmin"),
        secure=False,
    )
    if not client.bucket_exists("cas-documents"):
        client.make_bucket("cas-documents")
    client.put_object("cas-documents", key, __import__("io").BytesIO(data), len(data), content_type="application/pdf")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT v.id, p.id, p.domain_id FROM procedure_versions v
                JOIN procedures p ON p.id = v.procedure_id
                WHERE p.xa_id = 'xa_chu_se' AND p.procedure_code = 'dk_khai_sinh' AND v.version = '1.0.0'
                """
            )
            version_id, procedure_id, domain_id = cur.fetchone()
            cur.execute("SELECT id FROM users WHERE email = 'admin@chuse.vn'")
            user_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO documents (
                    id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type,
                    file_size_bytes, processing_status, validity_status, uploaded_by
                ) VALUES (
                    %s,'xa_chu_se',%s,'p4b','a.pdf',%s,%s,'application/pdf',
                    %s,'UPLOADED','PENDING',%s
                )
                """,
                (doc_id, domain_id, f"s3://cas-documents/{key}", checksum, len(data), user_id),
            )
            cur.execute(
                """
                INSERT INTO procedure_version_documents (
                    procedure_version_id, document_id, procedure_id, xa_id, domain_id, relationship_type
                ) VALUES (%s,%s,%s,'xa_chu_se',%s,'SOURCE')
                """,
                (version_id, doc_id, procedure_id, domain_id),
            )
            cur.execute(
                """
                INSERT INTO document_index_jobs (
                    id, document_id, procedure_version_id, xa_id, request_id, payload_hash,
                    status, claim_token, claimed_at, claim_expires_at
                ) VALUES (%s,%s,%s,'xa_chu_se',%s,'abc','CLAIMED',%s,now(),now() + interval '1 minute')
                """,
                (job_id, doc_id, version_id, request_id, claim),
            )
            cur.execute(
                """
                INSERT INTO document_index_generations (
                    id, xa_id, document_id, procedure_version_id, job_id, status,
                    pipeline_version, extraction_version, ocr_version, chunk_config_hash,
                    embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
                    source_sha256, content_sha256, manifest_hash
                ) VALUES (
                    %s,'xa_chu_se',%s,%s,%s,'STAGING',
                    'pending','pending','pending','pending','pending','pending','',0,
                    %s,'',''
                )
                """,
                (generation_id, doc_id, version_id, job_id, checksum),
            )
        conn.commit()
    try:
        job = IndexJob(
            schema_version="index.v2",
            xa_id="xa_chu_se",
            document_id=doc_id,
            procedure_id=procedure_id,
            procedure_version_id=version_id,
            job_id=job_id,
            claim_token=claim,
            generation_id=generation_id,
            bucket="cas-documents",
            object_key=key,
            checksum=checksum,
            page_range=None,
            pipeline_version=PIPELINE_VERSION,
        )

        class Deps:
            objects = _Minio(client)
            ocr = FakeOCR("khong goi")
            embedder = FakeEmbedder()
            vectors = QdrantWriter(qdrant, "p4b_test")
            chunks = PostgresChunks(dsn)

        result = run_pipeline(job, Deps(), max_bytes=200_000, chunk_config=ChunkConfig(32, 40, 4), bucket="cas-documents")
        assert result.chunk_count == result.vector_count >= 1
        assert result.ocr_pages == 0
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM knowledge_chunks WHERE generation_id = %s AND xa_id = 'xa_chu_se'",
                    (generation_id,),
                )
                assert cur.fetchone()[0] == result.chunk_count
                cur.execute("SELECT status FROM document_index_generations WHERE id = %s", (generation_id,))
                assert cur.fetchone()[0] == "STAGING"
        import httpx

        point = httpx.get(f"{qdrant}/collections/p4b_test/points/{point_id(generation_id, 0)}", timeout=5).json()
        payload = point["result"]["payload"]
        assert payload["xa_id"] == "xa_chu_se"
        assert payload["generation_id"] == str(generation_id)
        assert payload["document_id"] == str(doc_id)
    finally:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM knowledge_chunks WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM document_index_generations WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM document_index_jobs WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM procedure_version_documents WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            conn.commit()
        client.remove_object("cas-documents", key)


def test_qdrant_faults_against_a_real_server():
    """Faults are injected in front of a real Qdrant, not a fake vector store."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from app.indexing.errors import IndexFailure
    from app.indexing.pipeline import SavedChunk
    from app.indexing.qdrant import QdrantWriter

    upstream = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    state = {"fault": "ok", "puts": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._forward()

        def do_PUT(self):
            self._forward()

        def do_POST(self):
            self._forward()

        def log_message(self, *_args):
            return

        def _forward(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            path = self.path
            if state["fault"] == "partial" and self.command == "PUT" and path.split("?", 1)[0].endswith("/points"):
                self._send(500, b'{"status":"error"}')
                return
            if state["fault"] == "restart" and self.command == "PUT" and path.split("?", 1)[0].endswith("/points"):
                state["puts"] += 1
                if state["puts"] > 1:
                    self._send(500, b'{"status":"error"}')
                    return
            if state["fault"] == "dimension" and self.command == "PUT" and path.split("?", 1)[0].endswith("/points"):
                payload = json.loads(body)
                for point in payload.get("points", []):
                    point["vector"] = [0.0, 0.0]
                body = json.dumps(payload).encode()
            request = urllib.request.Request(upstream + path, data=body if self.command != "GET" else None, method=self.command)
            request.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    raw, code = response.read(), response.status
            except urllib.error.HTTPError as exc:
                raw, code = exc.read(), exc.code
            if state["fault"] == "acknowledged" and self.command == "PUT" and path.split("?", 1)[0].endswith("/points"):
                payload = json.loads(raw)
                payload.setdefault("result", {})["status"] = "acknowledged"
                raw = json.dumps(payload).encode()
            if state["fault"] == "missing" and path.split("?", 1)[0].endswith("/points/count"):
                raw, code = b'{"result":{"count":0},"status":"ok"}', 200
            self._send(code, raw)

        def _send(self, code: int, raw: bytes):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def payload(dimension=384):
        return {
            "xa_id": "xa_chu_se",
            "document_id": str(uuid.uuid4()),
            "procedure_version_id": str(uuid.uuid4()),
            "vector_dimension": dimension,
        }

    def chunks(n):
        return [SavedChunk(i, 1, 1, "xin chao", "a" * 64, 2, "native", uuid.uuid4()) for i in range(n)]

    def vectors(n):
        return [[0.1] * 384 for _ in range(n)]

    try:
        for fault in ("partial", "acknowledged", "dimension", "missing", "restart"):
            state["fault"] = fault
            state["puts"] = 0
            writer = QdrantWriter(base, f"p4b_{fault}", batch_size=1)
            with pytest.raises(IndexFailure) as raised:
                writer.upsert(uuid.uuid4(), chunks(2), vectors(2), payload())
            assert raised.value.code == "qdrant_failed"
        state["fault"] = "ok"
        generation = uuid.uuid4()
        writer = QdrantWriter(base, "p4b_duplicate", batch_size=1)
        body = payload()
        first = writer.upsert(generation, chunks(2), vectors(2), body)
        second = writer.upsert(generation, chunks(2), vectors(2), body)
        assert first == second == 2
    finally:
        server.shutdown()


class _Minio:
    def __init__(self, client) -> None:
        self.client = client

    def get(self, bucket: str, key: str, max_bytes: int | None = None) -> bytes:
        response = self.client.get_object(bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
