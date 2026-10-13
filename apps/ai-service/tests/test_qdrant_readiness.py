"""Qdrant readiness is a read. Collection creation is an explicit bootstrap."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import uuid

import pytest

from app.indexing.errors import IndexFailure
from app.indexing.runtime import _qdrant, provision_qdrant_collection


class Probe:
    def __init__(self, status=200, body=None, methods=None):
        self.status = status
        self.body = body or {"result": {"config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}}}}
        self.methods = methods if methods is not None else []
        self.puts = 0
        self.gets = 0

    def handler(self):
        probe = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                probe.gets += 1
                probe.methods.append("GET")
                payload = json.dumps(probe.body).encode()
                self.send_response(probe.status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_PUT(self):
                probe.puts += 1
                probe.methods.append("PUT")
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                payload = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                return

        return Handler


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_existing_compatible_collection_is_ready_without_writes():
    probe = Probe()
    server, url = _serve(probe.handler())
    try:
        assert _qdrant(url, "knowledge_chunks") is None
        assert probe.gets == 1
        assert probe.puts == 0
    finally:
        server.shutdown()


def test_missing_collection_is_not_ready_and_does_not_create():
    probe = Probe(status=404, body={"status": {"error": "Not found"}})
    server, url = _serve(probe.handler())
    try:
        assert _qdrant(url, "knowledge_chunks") == "qdrant_collection_missing"
        assert probe.gets == 1
        assert probe.puts == 0
    finally:
        server.shutdown()


def test_wrong_dimension_is_not_ready():
    probe = Probe(body={"result": {"config": {"params": {"vectors": {"size": 768, "distance": "Cosine"}}}}})
    server, url = _serve(probe.handler())
    try:
        assert _qdrant(url, "knowledge_chunks") == "qdrant_collection_mismatch"
        assert probe.puts == 0
    finally:
        server.shutdown()


def test_unreachable_qdrant_is_not_ready():
    assert _qdrant("http://127.0.0.1:1", "knowledge_chunks") == "qdrant_unreachable"


def test_provision_creates_once_and_leaves_an_existing_collection():
    created = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if created["n"] == 0:
                self.send_response(404)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            payload = json.dumps({"result": {"config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}}}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_PUT(self):
            created["n"] += 1
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            payload = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        provision_qdrant_collection(url, "knowledge_chunks")
        provision_qdrant_collection(url, "knowledge_chunks")
        assert created["n"] == 1
    finally:
        server.shutdown()


def test_provision_refuses_an_incompatible_collection():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({"result": {"config": {"params": {"vectors": {"size": 768, "distance": "Cosine"}}}}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_PUT(self):
            raise AssertionError("existing collection must not be recreated")

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(IndexFailure):
            provision_qdrant_collection(url, "knowledge_chunks")
    finally:
        server.shutdown()


def test_generation_delete_body_is_unrelated_to_readiness():
    from p4b_cleanup import generation_delete_body

    owned = uuid.uuid4()
    assert str(owned) in str(generation_delete_body(owned))
