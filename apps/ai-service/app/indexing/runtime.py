"""One model load per process, and a slot that stays held until the worker ends."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

from app.config import ConfigError
from app.indexing.errors import IndexFailure
from app.indexing.supervisor import IndexGate

log = logging.getLogger("cas.index")


def ocr_pool_size(_settings) -> int:
    """One pipeline process serves one job, so it owns one OCR worker."""
    return 1


def build_models(settings):
    """Load each real adapter once. Fake adapters are also created once."""
    started = time.perf_counter()
    from app.indexing.ocr import FakeOCR, FakeRenderer, PdfiumRenderer
    from app.indexing.pipeline import FakeEmbedder

    if settings.embedding_provider == "fake":
        embedder = FakeEmbedder()
    elif settings.embedding_provider == "onnx":
        from app.indexing.embed import OnnxProcess, require_supported_variant

        require_supported_variant()
        embedder = OnnxProcess(settings.embedding_model_dir, batch_size=settings.embed_batch_size)
        embedder.start(settings.pipeline_timeout_s)
    else:
        raise ConfigError("EMBEDDING_PROVIDER must be onnx or fake")
    if settings.ocr_provider == "fake":
        ocr = FakeOCR()
        renderer = FakeRenderer()
    elif settings.ocr_provider == "paddle":
        from app.indexing.ocr_supervisor import OcrSupervisor

        ocr = OcrSupervisor(
            ocr_pool_size(settings),
            {"kind": "paddle", "model_dir": settings.ocr_model_dir},
            replace_timeout_s=settings.pipeline_timeout_s,
        )
        ocr.start(settings.pipeline_timeout_s)
        renderer = PdfiumRenderer()
    else:
        raise ConfigError("OCR_PROVIDER must be paddle or fake")
    if time.perf_counter() - started > settings.pipeline_timeout_s:
        for obj in (embedder, ocr):
            shutdown = getattr(obj, "shutdown", None)
            if shutdown is not None:
                shutdown()
        raise ConfigError("pipeline model initialization exceeded INDEX_PIPELINE_TIMEOUT_SECONDS")
    return embedder, ocr, renderer


def _split_models(models):
    if isinstance(models, tuple):
        embedder = models[0] if models else None
        ocr = models[1] if len(models) > 1 else None
        return embedder is not None or ocr is not None, embedder, ocr
    return bool(models), None, None


def _worker_problem(settings, models) -> str | None:
    loaded, embedder, ocr = _split_models(models)
    if settings.embedding_provider == "onnx":
        if not loaded or embedder is None:
            return "embedding_model_missing"
        status = getattr(embedder, "status", "")
        if status != "ready":
            return "embedding_model_missing"
        alive = getattr(embedder, "is_alive", None)
        if callable(alive) and not alive():
            return "embedding_model_missing"
        if getattr(embedder, "cpu_compatible", None) is False:
            return "embedding_cpu_unsupported"
    if settings.ocr_provider == "paddle":
        if not loaded or ocr is None:
            return "ocr_model_missing"
        if getattr(ocr, "status", "") == "recovering":
            return "worker_recovering"
        if getattr(ocr, "status", "") != "ready":
            return "ocr_model_missing"
        expected = int(getattr(ocr, "slots", 0) or 0)
        alive_count = ocr.alive_workers() if hasattr(ocr, "alive_workers") else 0
        if expected < 1 or alive_count < expected:
            return "ocr_model_missing"
    return None


def probe_pipeline(settings, models, gate=None) -> tuple[bool, str | None]:
    if gate is not None and getattr(gate, "started", False):
        worker = gate.health_problem(settings)
    else:
        worker = _worker_problem(settings, models)
    if worker is not None:
        return False, worker
    if not _postgres(settings.database_url):
        return False, "postgres_unreachable"
    qdrant = _qdrant(settings.qdrant_url, settings.qdrant_collection)
    if qdrant is not None:
        return False, qdrant
    if not _minio(settings):
        return False, "object_storage_unreachable"
    return True, None


def _postgres(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        log.warning("ready_probe code=postgres_unreachable")
        return False


def _qdrant(base_url: str, collection: str) -> str | None:
    try:
        import httpx

        url = base_url.rstrip("/") + "/collections/" + collection
        response = httpx.get(url, timeout=2.0)
        if response.status_code == 404:
            created = httpx.put(
                url,
                json={"vectors": {"size": 384, "distance": "Cosine"}},
                timeout=2.0,
            )
            if created.status_code not in (200, 409):
                log.warning("ready_probe code=qdrant_unreachable")
                return "qdrant_unreachable"
            response = httpx.get(url, timeout=2.0)
        if response.status_code != 200:
            log.warning("ready_probe code=qdrant_unreachable")
            return "qdrant_unreachable"
        vectors = response.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
        if vectors.get("size") != 384 or vectors.get("distance") != "Cosine":
            log.warning("ready_probe code=qdrant_collection_mismatch")
            return "qdrant_collection_mismatch"
        return None
    except Exception:
        log.warning("ready_probe code=qdrant_unreachable")
        return "qdrant_unreachable"


_MINIO_CLIENT = None
_DATA_MINIO: dict = {}
_DATA_MINIO_LOCK = threading.Lock()
_HEALTH: dict = {"ready": None, "reason": None, "checked_at": 0.0, "monotonic": 0.0}
_HEALTH_TTL_S = 5.0


def minio_http_client():
    """Short connect/read deadlines and no retry storm for readiness."""
    import urllib3

    return urllib3.PoolManager(
        timeout=urllib3.util.Timeout(connect=1.0, read=1.0),
        retries=urllib3.util.Retry(total=0, connect=0, read=0, redirect=0),
    )


def minio_client():
    global _MINIO_CLIENT
    if _MINIO_CLIENT is None:
        from app.indexing.object_storage import open_client, storage_config

        _MINIO_CLIENT = open_client(minio_http_client(), storage_config())
    return _MINIO_CLIENT


def _minio(settings) -> bool:
    try:
        from app.indexing.object_storage import ensure_bucket, storage_config

        cfg = storage_config()
        return bool(ensure_bucket(minio_client(), settings.object_bucket, cfg["auto_create"]))
    except Exception:
        log.warning("ready_probe code=object_storage_unreachable")
        return False


def data_minio_client(connect_s: float = 2.0, read_s: float = 5.0):
    """Shared data-path client. Connect and read deadlines stay bounded; retries do not multiply them."""
    import urllib3
    from app.indexing.object_storage import open_client, storage_config

    cfg = storage_config()
    key = (cfg["endpoint"], cfg["region"], cfg["secure"], round(max(0.2, connect_s), 1), round(max(0.2, read_s), 1))
    with _DATA_MINIO_LOCK:
        client = _DATA_MINIO.get(key)
        if client is None:
            http = urllib3.PoolManager(
                timeout=urllib3.util.Timeout(connect=key[3], read=key[4]),
                retries=urllib3.util.Retry(total=1, connect=1, read=0, redirect=0, status=0),
            )
            client = open_client(http, cfg)
            _DATA_MINIO[key] = client
        return client


def _bounded_object_client(connect_s: float, read_s: float, total_s: float | None):
    """A client whose total timeout covers the whole download, not each read block."""
    import urllib3
    from app.indexing.object_storage import open_client, storage_config

    timeout = urllib3.util.Timeout(connect=connect_s, read=read_s, total=total_s)
    http = urllib3.PoolManager(
        timeout=timeout,
        retries=urllib3.util.Retry(total=0, connect=0, read=0, redirect=0, status=0),
    )
    return open_client(http, storage_config())


def _tighten_read_timeout(response, seconds: float) -> None:
    raw = getattr(response, "_fp", None)
    inner = getattr(raw, "raw", None) if raw is not None else None
    sock = getattr(inner, "_sock", None) if inner is not None else None
    if sock is None:
        return
    try:
        sock.settimeout(seconds)
    except OSError:
        pass


def fetch_object(bucket: str, key: str, max_bytes: int | None = None, timeout_s: float | None = None, deadline=None) -> bytes:
    """Download one object and always release the connection."""
    from app.indexing.deadline import Deadline
    from app.indexing.errors import IndexFailure
    from app.indexing.pdf import read_bounded

    clock = deadline
    if clock is None and timeout_s is not None:
        clock = Deadline(float(timeout_s))
    if clock is not None:
        clock.check()
        left = clock.remaining()
        client = _bounded_object_client(left, left, left)
    else:
        client = data_minio_client(2.0, 5.0)
    response = None
    closer = None
    try:
        response = client.get_object(bucket, key)
        if clock is not None:
            def closer():
                raw = getattr(getattr(response, "_fp", None), "raw", None)
                sock = getattr(getattr(raw, "_fp", None), "raw", None)
                sock = getattr(sock, "_sock", sock)
                if sock is not None and hasattr(sock, "shutdown"):
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                response.close()
                response.release_conn()

            clock.bind(closer)
        chunks = []
        while True:
            if clock is not None:
                left = clock.remaining()
                _tighten_read_timeout(response, left)
            block = response.read(64 * 1024)
            if clock is not None:
                clock.check()
            if not block:
                break
            chunks.append(block)
            if max_bytes is not None and sum(len(item) for item in chunks) > max_bytes:
                raise IndexFailure("pdf_oversized")
        return read_bounded(chunks, max_bytes if max_bytes is not None else 20 * 1024 * 1024)
    except IndexFailure:
        raise
    except Exception as exc:
        if clock is not None and clock.cancelled:
            raise IndexFailure("timeout") from exc
        marker = getattr(exc, "code", "") or ""
        if marker in ("NoSuchKey", "NoSuchBucket") or "NoSuchKey" in type(exc).__name__ or "NoSuchBucket" in type(exc).__name__:
            raise IndexFailure("source_not_found") from exc
        message = str(exc)
        if "NoSuchKey" in message or "NoSuchBucket" in message:
            raise IndexFailure("source_not_found") from exc
        raise IndexFailure("source_timeout") from exc
    finally:
        if clock is not None and closer is not None:
            clock.unbind(closer)
        if response is not None:
            response.close()
            response.release_conn()


def _health_view(settings, models, gate=None) -> dict:
    loaded, embedder, ocr = _split_models(models)
    if gate is not None and getattr(gate, "started", False):
        loaded = gate.status == "ready"
    if settings.index_mode != "pipeline":
        ok, reason = True, None
    else:
        ok, reason = probe_pipeline(settings, models, gate)
    view = {
        "ready": bool(ok),
        "reason": reason,
        "initialized": bool(loaded),
        "configured": bool(settings.database_url and settings.qdrant_url and settings.object_bucket),
        "embedding_status": getattr(embedder, "status", None) if embedder is not None else None,
        "embedding_variant": getattr(embedder, "variant", None) if embedder is not None else None,
        "cpu_compatible": getattr(embedder, "cpu_compatible", None) if embedder is not None else None,
        "ocr_status": getattr(ocr, "status", None) if ocr is not None else None,
        "ocr_workers_expected": int(getattr(ocr, "slots", 0) or 0) if ocr is not None else 0,
        "ocr_workers_alive": int(ocr.alive_workers()) if ocr is not None and hasattr(ocr, "alive_workers") else 0,
        "ocr_workers_recovering": 0,
        "ocr_workers_failed": 0,
    }
    if gate is not None and getattr(gate, "started", False) and settings.ocr_provider == "paddle":
        census = gate.ocr_census()
        view["ocr_workers_expected"] = census["expected"]
        view["ocr_workers_alive"] = census["alive"]
        view["ocr_workers_recovering"] = census["recovering"]
        view["ocr_workers_failed"] = census["failed"]
        if census["recovering"]:
            view["ocr_status"] = "recovering"
        elif census["failed"] or census["missing"] or census["alive"] < census["expected"]:
            view["ocr_status"] = "failed"
    return view


def _worker_snapshot(models) -> dict:
    _loaded, embedder, ocr = _split_models(models)
    return {
        "embedding_status": getattr(embedder, "status", None) if embedder is not None else None,
        "cpu_compatible": getattr(embedder, "cpu_compatible", None) if embedder is not None else None,
        "ocr_status": getattr(ocr, "status", None) if ocr is not None else None,
        "ocr_workers_alive": int(ocr.alive_workers()) if ocr is not None and hasattr(ocr, "alive_workers") else 0,
    }


def current_health(settings, models=None, force: bool = False, gate=None) -> dict:
    """One bounded health result shared by /ready and /v1/status.

    A fresh cache entry is discarded as soon as a worker status or live count changes.
    """
    now = time.monotonic()
    workers = _worker_snapshot(models)
    if gate is not None:
        workers["gate_status"] = getattr(gate, "status", None)
        if hasattr(gate, "ocr_census"):
            workers["ocr_census"] = tuple(sorted(gate.ocr_census().items()))
    cached = _HEALTH.get("ready") is not None and now - _HEALTH.get("monotonic", 0.0) < _HEALTH_TTL_S
    if not force and cached and all(_HEALTH.get(key) == value for key, value in workers.items()):
        return dict(_HEALTH)
    live = _health_view(settings, models, gate)
    live["checked_at"] = time.time()
    live["monotonic"] = now
    _HEALTH.clear()
    _HEALTH.update(live)
    return dict(_HEALTH)
