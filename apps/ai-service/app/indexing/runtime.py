"""One model load per process, and a slot that stays held until the worker ends."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from app.config import ConfigError
from app.indexing.errors import IndexFailure

log = logging.getLogger("cas.index")


class IndexGate:
    def __init__(self, limit: int) -> None:
        self._sem = threading.BoundedSemaphore(limit)
        self._executor = ThreadPoolExecutor(max_workers=limit, thread_name_prefix="index-pipeline")
        self._lock = threading.Lock()
        self._pending = 0
        self._idle = threading.Event()
        self._idle.set()
        self._closed = False

    @property
    def serving(self) -> bool:
        return not self._closed

    def submit(self, fn):
        if self._closed:
            raise IndexFailure("pipeline_busy")
        if not self._sem.acquire(blocking=False):
            raise IndexFailure("pipeline_busy")
        with self._lock:
            self._pending += 1
            self._idle.clear()

        def work():
            try:
                return fn()
            finally:
                with self._lock:
                    self._pending -= 1
                    pending = self._pending
                self._sem.release()
                if pending == 0:
                    self._idle.set()

        return self._executor.submit(work)

    def pending(self) -> int:
        with self._lock:
            return self._pending

    def run(self, fn, timeout_s: float):
        future = self.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            raise IndexFailure("timeout") from exc

    def shutdown(self, timeout_s: float = 5.0, models=None) -> int:
        """Reject new work, stop model subprocesses, then wait until in-flight calls leave."""
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        for item in models or ():
            if item is None:
                continue
            stop = getattr(item, "shutdown", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    log.warning("index_gate_shutdown_worker_failed")
        self._idle.wait(timeout_s)
        pending = self.pending()
        if pending:
            log.warning("index_gate_shutdown_pending count=%s", pending)
        return pending


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
            settings.index_max_inflight,
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
        if getattr(ocr, "status", "") != "ready":
            return "ocr_model_missing"
        expected = int(getattr(ocr, "slots", 0) or 0)
        alive_count = ocr.alive_workers() if hasattr(ocr, "alive_workers") else 0
        if expected < 1 or alive_count < expected:
            return "ocr_model_missing"
    return None


def probe_pipeline(settings, models) -> tuple[bool, str | None]:
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
        from minio import Minio

        endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT", "minio:9000")
        secure = os.getenv("OBJECT_STORAGE_USE_SSL", "false").lower() in ("1", "true", "yes")
        _MINIO_CLIENT = Minio(
            endpoint,
            access_key=os.getenv("OBJECT_STORAGE_ACCESS_KEY", ""),
            secret_key=os.getenv("OBJECT_STORAGE_SECRET_KEY", ""),
            secure=secure,
            http_client=minio_http_client(),
        )
    return _MINIO_CLIENT


def _minio(settings) -> bool:
    try:
        return bool(minio_client().bucket_exists(settings.object_bucket))
    except Exception:
        log.warning("ready_probe code=object_storage_unreachable")
        return False


def data_minio_client(connect_s: float = 2.0, read_s: float = 5.0):
    """Shared data-path client. Connect and read deadlines stay bounded; retries do not multiply them."""
    import urllib3
    from minio import Minio

    key = (round(max(0.2, connect_s), 1), round(max(0.2, read_s), 1))
    with _DATA_MINIO_LOCK:
        client = _DATA_MINIO.get(key)
        if client is None:
            endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT", "minio:9000")
            secure = os.getenv("OBJECT_STORAGE_USE_SSL", "false").lower() in ("1", "true", "yes")
            http = urllib3.PoolManager(
                timeout=urllib3.util.Timeout(connect=key[0], read=key[1]),
                retries=urllib3.util.Retry(total=1, connect=1, read=0, redirect=0, status=0),
            )
            client = Minio(
                endpoint,
                access_key=os.getenv("OBJECT_STORAGE_ACCESS_KEY", ""),
                secret_key=os.getenv("OBJECT_STORAGE_SECRET_KEY", ""),
                secure=secure,
                http_client=http,
            )
            _DATA_MINIO[key] = client
        return client


def _bounded_object_client(connect_s: float, read_s: float, total_s: float | None):
    """A client whose total timeout covers the whole download, not each read block."""
    import urllib3
    from minio import Minio

    timeout = urllib3.util.Timeout(connect=connect_s, read=read_s, total=total_s)
    http = urllib3.PoolManager(
        timeout=timeout,
        retries=urllib3.util.Retry(total=0, connect=0, read=0, redirect=0, status=0),
    )
    endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT", "minio:9000")
    secure = os.getenv("OBJECT_STORAGE_USE_SSL", "false").lower() in ("1", "true", "yes")
    return Minio(
        endpoint,
        access_key=os.getenv("OBJECT_STORAGE_ACCESS_KEY", ""),
        secret_key=os.getenv("OBJECT_STORAGE_SECRET_KEY", ""),
        secure=secure,
        http_client=http,
    )


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
    try:
        response = client.get_object(bucket, key)
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
        marker = getattr(exc, "code", "") or ""
        if marker in ("NoSuchKey", "NoSuchBucket") or "NoSuchKey" in type(exc).__name__ or "NoSuchBucket" in type(exc).__name__:
            raise IndexFailure("source_not_found") from exc
        message = str(exc)
        if "NoSuchKey" in message or "NoSuchBucket" in message:
            raise IndexFailure("source_not_found") from exc
        raise IndexFailure("source_timeout") from exc
    finally:
        if response is not None:
            response.close()
            response.release_conn()


def _health_view(settings, models) -> dict:
    loaded, embedder, ocr = _split_models(models)
    if settings.index_mode != "pipeline":
        ok, reason = True, None
    else:
        ok, reason = probe_pipeline(settings, models)
    return {
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
    }


def _worker_snapshot(models) -> dict:
    _loaded, embedder, ocr = _split_models(models)
    return {
        "embedding_status": getattr(embedder, "status", None) if embedder is not None else None,
        "cpu_compatible": getattr(embedder, "cpu_compatible", None) if embedder is not None else None,
        "ocr_status": getattr(ocr, "status", None) if ocr is not None else None,
        "ocr_workers_alive": int(ocr.alive_workers()) if ocr is not None and hasattr(ocr, "alive_workers") else 0,
    }


def current_health(settings, models=None, force: bool = False) -> dict:
    """One bounded health result shared by /ready and /v1/status.

    A fresh cache entry is discarded as soon as a worker status or live count changes.
    """
    now = time.monotonic()
    workers = _worker_snapshot(models)
    cached = _HEALTH.get("ready") is not None and now - _HEALTH.get("monotonic", 0.0) < _HEALTH_TTL_S
    if not force and cached and all(_HEALTH.get(key) == value for key, value in workers.items()):
        return dict(_HEALTH)
    live = _health_view(settings, models)
    live["checked_at"] = time.time()
    live["monotonic"] = now
    _HEALTH.clear()
    _HEALTH.update(live)
    return dict(_HEALTH)
