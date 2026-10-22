"""FastAPI Speech Inference API wrapping the persistent C0 runtime."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.c0_speech.audio import decode_wav_bytes
from src.c0_speech.errors import (
    PUBLIC_ERROR_MESSAGES,
    SpeechAPIError,
    SpeechAudioError,
    public_error_body,
)
from src.c0_speech.runtime import SpeechRuntime, public_inference_body
from src.c0_speech.settings import SpeechSettings, load_settings

log = logging.getLogger("c0_speech")

INFER_PATH = "/v1/speech/infer"


def _scope_header(scope: dict[str, Any], name: bytes) -> str:
    for key, value in scope.get("headers") or []:
        if key == name and value:
            try:
                return value.decode("ascii")
            except UnicodeError:
                return ""
    return ""


def _token_ok_value(header: str, settings: SpeechSettings) -> bool:
    if not settings.auth_required:
        return True
    expected = settings.service_token
    prefix = "Bearer "
    if not expected or not header.startswith(prefix):
        return False
    got = header[len(prefix) :]
    if len(got) != len(expected):
        return False
    return hmac.compare_digest(got, expected)


class _UploadLimit:
    """Auth first, then cap the entire HTTP request body while reading.

    ``SPEECH_MAX_UPLOAD_BYTES`` counts multipart boundaries and part headers
    as well as the WAV bytes. No quadratic byte copies.
    """

    def __init__(self, app, settings: SpeechSettings) -> None:
        self.app = app
        self.settings = settings
        self.limit = int(settings.max_upload_bytes)

    async def __call__(self, scope, receive, send):
        path = scope.get("path") or ""
        if scope["type"] != "http" or scope.get("method") != "POST" or path != INFER_PATH:
            await self.app(scope, receive, send)
            return
        request_id = _scope_header(scope, b"x-request-id") or str(uuid.uuid4())
        if not _token_ok_value(_scope_header(scope, b"authorization"), self.settings):
            payload = json.dumps(public_error_body("unauthorized", request_id=request_id)).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": payload})
            return
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            piece = message.get("body", b"") or b""
            total += len(piece)
            if total > self.limit:
                payload = json.dumps(public_error_body("payload_too_large", request_id=request_id)).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": payload})
                return
            if piece:
                chunks.append(piece)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay, send)


def _header_request_id(scope: dict[str, Any]) -> str:
    return _scope_header(scope, b"x-request-id") or str(uuid.uuid4())


def _request_id(request: Request) -> str:
    header = request.headers.get("x-request-id", "").strip()
    return header or str(uuid.uuid4())


def _token_ok(request: Request, settings: SpeechSettings) -> bool:
    return _token_ok_value(request.headers.get("authorization", ""), settings)


def _error_response(code: str, status: int, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=public_error_body(code, request_id=request_id))


def _finish_inference_task(task: "asyncio.Task[Any]", runtime: SpeechRuntime) -> None:
    """Release the GPU slot once and retrieve the worker result/exception."""
    try:
        runtime.gate.release()
    finally:
        _consume_task_outcome(task)


def _consume_task_outcome(task: "asyncio.Task[Any]") -> None:
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        return
    if exc is None:
        return
    log.warning(
        "inference_worker_finished_with_error type=%s code=%s",
        type(exc).__name__,
        getattr(exc, "code", None) or "none",
    )


async def run_inference_slot(
    runtime: SpeechRuntime,
    wav: Any,
    sample_rate: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Acquire one GPU slot. Release exactly once when the worker task finishes."""
    reason = runtime.gate.try_acquire()
    if reason:
        raise SpeechAPIError(reason, PUBLIC_ERROR_MESSAGES.get(reason, reason), 503)
    task = asyncio.create_task(asyncio.to_thread(runtime.infer, wav, sample_rate))
    task.add_done_callback(lambda done: _finish_inference_task(done, runtime))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise SpeechAPIError("inference_timeout", PUBLIC_ERROR_MESSAGES["inference_timeout"], 504) from exc
    except asyncio.CancelledError:
        raise


def create_app(
    settings: Optional[SpeechSettings] = None,
    runtime: Optional[SpeechRuntime] = None,
) -> FastAPI:
    settings = settings or load_settings()
    state: dict[str, Any] = {"settings": settings, "runtime": runtime, "load_calls": 0}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        current = state["runtime"]
        if current is None:
            current = SpeechRuntime(settings)
            state["runtime"] = current
        if not current.ready:
            state["load_calls"] = int(state["load_calls"]) + 1
            current.start()
        yield
        rt = state.get("runtime")
        if isinstance(rt, SpeechRuntime):
            await rt.drain(timeout_s=settings.shutdown_timeout_s)

    app = FastAPI(
        title="cas-c0-speech",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(_UploadLimit, settings=settings)
    app.state.speech = state

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        rt: SpeechRuntime | None = state.get("runtime")
        if rt is None or not rt.ready:
            code = "not_ready"
            if rt is not None and rt.init_error is not None:
                code = rt.init_error.code
            return _error_response(code, 503, "readyz")
        return JSONResponse(
            {
                "status": "ready",
                "model_id": "c0_cascaded",
                "load_count": rt.load_count,
                "init_ms": rt.init_ms,
                "inflight": rt.gate.inflight,
                "tokenizer_parity": getattr(rt.cascade, "tokenizer_parity", None),
                "generation_matches_rq1": getattr(rt.cascade, "generation_matches_rq1", None),
            }
        )

    @app.post(INFER_PATH)
    async def infer(request: Request) -> JSONResponse:
        started = time.perf_counter()
        request_id = _request_id(request)
        rt: SpeechRuntime | None = state.get("runtime")
        if not _token_ok(request, settings):
            _log(request_id, 401, started, "unauthorized", ready=bool(rt and rt.ready))
            return _error_response("unauthorized", 401, request_id)
        if rt is None or not rt.ready:
            _log(request_id, 503, started, "not_ready", ready=False)
            return _error_response("not_ready", 503, request_id)
        form = None
        upload = None
        try:
            try:
                form = await request.form()
            except Exception:
                _log(request_id, 400, started, "audio_invalid", ready=True)
                return _error_response("audio_invalid", 400, request_id)
            upload = form.get("audio")
            if upload is None:
                _log(request_id, 400, started, "audio_missing", ready=True)
                return _error_response("audio_missing", 400, request_id)
            if isinstance(upload, str):
                _log(request_id, 400, started, "audio_invalid", ready=True)
                return _error_response("audio_invalid", 400, request_id)
            try:
                data = await upload.read()
            except Exception:
                _log(request_id, 400, started, "audio_invalid", ready=True)
                return _error_response("audio_invalid", 400, request_id)
            try:
                wav, sr = decode_wav_bytes(
                    data,
                    max_bytes=settings.max_upload_bytes,
                    max_duration_seconds=settings.max_audio_seconds,
                )
            except SpeechAudioError as exc:
                _log(request_id, exc.status, started, exc.code, ready=True)
                return _error_response(exc.code, exc.status, request_id)
            del data
            try:
                payload = await run_inference_slot(rt, wav, sr, settings.request_timeout_s)
            except SpeechAPIError as exc:
                _log(request_id, exc.status, started, exc.code, ready=True)
                return _error_response(exc.code, exc.status, request_id)
            except asyncio.CancelledError:
                _log(request_id, 499, started, "cancelled", ready=True)
                raise
            except Exception:
                _log(request_id, 500, started, "inference_failed", ready=True)
                return _error_response("inference_failed", 500, request_id)
            body = public_inference_body(payload, request_id=request_id)
            _log(
                request_id,
                200,
                started,
                "ok",
                ready=True,
                duration_ms=body.get("duration_ms"),
                inference_ms=body.get("latency_ms"),
            )
            return JSONResponse(body)
        finally:
            if upload is not None and not isinstance(upload, str):
                close = getattr(upload, "close", None)
                if callable(close):
                    with suppress(Exception):
                        result = close()
                        if asyncio.iscoroutine(result):
                            await result
            if form is not None:
                close = getattr(form, "close", None)
                if callable(close):
                    with suppress(Exception):
                        result = close()
                        if asyncio.iscoroutine(result):
                            await result

    return app


def _log(
    request_id: str,
    status: int,
    started: float,
    category: str,
    *,
    ready: bool,
    duration_ms: Any = None,
    inference_ms: Any = None,
) -> None:
    http_ms = int(round((time.perf_counter() - started) * 1000.0))
    log.info(
        "speech_request request_id=%s status=%s http_ms=%s ready=%s category=%s duration_ms=%s inference_ms=%s",
        request_id,
        status,
        http_ms,
        ready,
        category,
        duration_ms,
        inference_ms,
    )


def run() -> None:
    import uvicorn

    settings = load_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=settings.log_level.lower())
