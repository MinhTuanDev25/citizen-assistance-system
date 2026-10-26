"""Speech API tests. Fake C0 runtime only — not GPU or checkpoint evidence."""
from __future__ import annotations

import asyncio
import io
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.c0_cascaded import C0ChecksumError, C0ConfigError, SAMPLE_RATE
from src.c0_speech.audio import decode_wav_bytes
from src.c0_speech.errors import SpeechAPIError, SpeechAudioError, SpeechConfigError
from src.c0_speech.app import create_app, run_inference_slot
from src.c0_speech.runtime import InferGate, SpeechRuntime, public_inference_body
from src.c0_speech.settings import SpeechSettings, bind_is_loopback, load_settings

TOKEN = "test-speech-token-1"
ROOT = Path(__file__).resolve().parents[1]
_FAKE_MT_CONTRACT = Path(tempfile.gettempdir()) / "cas_c0_speech_fake_mt_training_contract.json"
if not _FAKE_MT_CONTRACT.is_file():
    _FAKE_MT_CONTRACT.write_text("{}", encoding="utf-8")


def _settings(**kwargs) -> SpeechSettings:
    values = dict(
        config_path=None,
        host="127.0.0.1",
        port=8090,
        device_override="",
        max_upload_bytes=10 * 1024 * 1024,
        max_audio_seconds=30.0,
        max_inflight=1,
        request_timeout_s=2.0,
        shutdown_timeout_s=1.0,
        log_level="ERROR",
        service_token=TOKEN,
        auth_required=True,
        fail_fast=False,
        workers=1,
    )
    values.update(kwargs)
    return SpeechSettings(**values)


def pcm16_wav_bytes(samples: np.ndarray, *, sr: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    pcm = np.clip(np.asarray(samples) * 32767.0, -32768, 32767).astype("<i2")
    if channels == 2:
        if pcm.ndim == 1:
            pcm = np.stack([pcm, pcm], axis=1)
        pcm = pcm.reshape(-1)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class FakeCascade:
    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        error_payload: dict | None = None,
        tokenizer_parity: str = "VERIFIED",
        generation_matches_rq1: bool = True,
        mt_training_contract: str | None = None,
        max_source_length: int = 256,
        generation_max_length: int = 256,
        num_beams: int = 4,
        raise_after: BaseException | None = None,
    ) -> None:
        self.delay_s = delay_s
        self.error_payload = error_payload
        self.raise_after = raise_after
        self.tokenizer_parity = tokenizer_parity
        self.generation_matches_rq1 = generation_matches_rq1
        self.mt_training_contract = (
            str(_FAKE_MT_CONTRACT) if mt_training_contract is None else mt_training_contract
        )
        self.max_source_length = max_source_length
        self.generation_max_length = generation_max_length
        self.num_beams = num_beams
        self.calls = 0

    def infer_waveform(self, wav, sample_rate=SAMPLE_RATE):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raise_after is not None:
            exc = self.raise_after
            self.raise_after = None
            self.delay_s = 0.0
            raise exc
        duration = int(round((len(wav) / float(sample_rate)) * 1000.0))
        payload = {
            "model_id": "c0_cascaded",
            "asr_text_bahnar": "xin",
            "translated_text_vi": "tôi",
            "duration_ms": duration,
            "latency_ms": 7,
            "tokenizer_parity": "VERIFIED",
            "research_provenance": "NOT_VERIFIED",
            "checkpoint_identity": "expected_step_match",
            "generation_matches_rq1": True,
            "tokenizer_source": "hub_pinned_rq1",
            "tokenizer_model_id": "vinai/bartpho-syllable",
            "tokenizer_revision": "36eee8b4d648dd99da56462edcda3c5c97f7f3de",
            "asr_checkpoint_step": 12500,
            "mt_checkpoint_step": 17500,
            "artifacts_present": True,
            "asr_checkpoint": "/secret/asr/checkpoint-12500",
            "mt_checkpoint": "/secret/mt/checkpoint-17500",
        }
        if self.error_payload:
            payload.update(self.error_payload)
        return payload


def _app(cascade=None, **setting_kw):
    cascade = cascade or FakeCascade()
    settings = _settings(**setting_kw)
    runtime = SpeechRuntime(settings, cascade=cascade)
    return create_app(settings, runtime), runtime, cascade


def _client(cascade=None, **setting_kw):
    app, runtime, cascade = _app(cascade, **setting_kw)
    return TestClient(app), runtime, cascade


def _headers(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Request-ID": "11111111-1111-1111-1111-111111111111"}


def _post_audio(client: TestClient, data: bytes, filename: str = "utt.wav", token: str = TOKEN):
    return client.post(
        "/v1/speech/infer",
        files={"audio": (filename, data, "audio/wav")},
        headers=_headers(token),
    )


# --- settings ---


def test_settings_reject_bad_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPEECH_AUTH", "off")
    monkeypatch.setenv("SPEECH_WORKERS", "2")
    with pytest.raises(SpeechConfigError, match="SPEECH_WORKERS"):
        load_settings()
    monkeypatch.setenv("SPEECH_WORKERS", "1")
    monkeypatch.setenv("SPEECH_MAX_INFLIGHT", "2")
    with pytest.raises(SpeechConfigError, match="SPEECH_MAX_INFLIGHT"):
        load_settings()
    monkeypatch.setenv("SPEECH_MAX_INFLIGHT", "1")
    monkeypatch.setenv("SPEECH_AUTH", "on")
    monkeypatch.delenv("SPEECH_SERVICE_TOKEN", raising=False)
    with pytest.raises(SpeechConfigError, match="SPEECH_SERVICE_TOKEN"):
        load_settings()


def test_settings_auth_off_allows_empty_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPEECH_AUTH", "off")
    monkeypatch.setenv("SPEECH_HOST", "127.0.0.1")
    monkeypatch.delenv("SPEECH_SERVICE_TOKEN", raising=False)
    loaded = load_settings()
    assert loaded.auth_required is False
    assert loaded.workers == 1
    assert loaded.max_inflight == 1


def test_settings_reject_wildcard_without_auth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPEECH_AUTH", "off")
    monkeypatch.setenv("SPEECH_HOST", "0.0.0.0")
    with pytest.raises(SpeechConfigError, match="loopback"):
        load_settings()


def test_settings_auth_off_loopback_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPEECH_AUTH", "off")
    monkeypatch.delenv("SPEECH_SERVICE_TOKEN", raising=False)
    for host in ("127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost"):
        monkeypatch.setenv("SPEECH_HOST", host)
        loaded = load_settings()
        assert loaded.auth_required is False
        assert bind_is_loopback(host) is True
    for host in (
        "0.0.0.0",
        "::",
        "[::]",
        "::0",
        "192.0.2.44",
        "10.1.2.3",
        "192.168.0.8",
        "172.16.5.9",
        "8.8.8.8",
        "example.com",
        "gpu-host",
    ):
        monkeypatch.setenv("SPEECH_HOST", host)
        with pytest.raises(SpeechConfigError, match="loopback"):
            load_settings()
        assert bind_is_loopback(host) is False


def test_settings_auth_on_allows_non_loopback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPEECH_AUTH", "on")
    monkeypatch.setenv("SPEECH_SERVICE_TOKEN", TOKEN)
    monkeypatch.setenv("SPEECH_HOST", "192.168.1.10")
    loaded = load_settings()
    assert loaded.auth_required is True
    assert loaded.host == "192.168.1.10"
    monkeypatch.setenv("SPEECH_HOST", "0.0.0.0")
    wildcard = load_settings()
    assert wildcard.auth_required is True
    assert wildcard.host == "0.0.0.0"


# --- audio validation ---


def test_decode_rejects_empty_unsupported_rate_stereo_and_duration():
    with pytest.raises(SpeechAudioError) as empty:
        decode_wav_bytes(b"", max_bytes=100, max_duration_seconds=30)
    assert empty.value.code == "audio_empty"
    with pytest.raises(SpeechAudioError) as media:
        decode_wav_bytes(b"ID3mp3", max_bytes=100, max_duration_seconds=30)
    assert media.value.code == "audio_unsupported"
    too_big = b"RIFF" + b"xxxxWAVE" + b"x" * 20
    with pytest.raises(SpeechAudioError) as size:
        decode_wav_bytes(too_big, max_bytes=8, max_duration_seconds=30)
    assert size.value.status == 413
    wav_8k = pcm16_wav_bytes(np.zeros(160, dtype=np.float32), sr=8000)
    with pytest.raises(SpeechAudioError) as rate:
        decode_wav_bytes(wav_8k, max_bytes=10_000_000, max_duration_seconds=30)
    assert rate.value.code == "audio_sample_rate"
    stereo = pcm16_wav_bytes(np.zeros(160, dtype=np.float32), channels=2)
    with pytest.raises(SpeechAudioError) as ch:
        decode_wav_bytes(stereo, max_bytes=10_000_000, max_duration_seconds=30)
    assert ch.value.code == "audio_channels"
    long = pcm16_wav_bytes(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    with pytest.raises(SpeechAudioError) as dur:
        decode_wav_bytes(long, max_bytes=10_000_000, max_duration_seconds=1)
    assert dur.value.code == "audio_too_long"


def test_decode_accepts_mono_16k():
    wav_bytes = pcm16_wav_bytes(np.zeros(320, dtype=np.float32))
    wav, sr = decode_wav_bytes(wav_bytes, max_bytes=10_000_000, max_duration_seconds=30)
    assert sr == SAMPLE_RATE
    assert wav.ndim == 1
    assert wav.dtype == np.float32


# --- runtime lifecycle ---


def test_runtime_injected_cascade_loads_once():
    cascade = FakeCascade()
    rt = SpeechRuntime(_settings(), cascade=cascade)
    rt.start()
    rt.start()
    assert rt.ready is True
    assert rt.load_count == 1
    wav = np.ones(160, dtype=np.float32)
    rt.infer(wav, SAMPLE_RATE)
    rt.infer(wav, SAMPLE_RATE)
    assert cascade.calls == 2
    assert rt.cascade is cascade


def test_runtime_init_failures(tmp_path: Path):
    cfg = tmp_path / "c0.yaml"
    cfg.write_text("device: cpu\n", encoding="utf-8")

    def boom(_config):
        raise C0ConfigError("tokenizer failed")

    tok = SpeechRuntime(_settings(config_path=cfg, fail_fast=False), cascade_factory=boom)
    tok.start()
    assert tok.ready is False
    assert tok.init_error is not None
    assert tok.init_error.code in {"config_error", "tokenizer_mismatch"}

    class TokFail(C0ConfigError):
        error_code = "tokenizer_mismatch"

    tok2 = SpeechRuntime(_settings(config_path=cfg, fail_fast=False), cascade_factory=lambda _c: (_ for _ in ()).throw(TokFail("fingerprint")))
    tok2.start()
    assert tok2.ready is False
    assert tok2.init_error.code == "tokenizer_mismatch"

    def checksum(_config):
        raise C0ChecksumError("checkpoint_checksum_mismatch", "weights differ")

    chk = SpeechRuntime(_settings(config_path=cfg, fail_fast=False), cascade_factory=checksum)
    chk.start()
    assert chk.ready is False
    assert chk.init_error.code == "checkpoint_checksum_mismatch"

    with pytest.raises(SpeechConfigError):
        SpeechRuntime(_settings(fail_fast=True)).start()


def test_infer_gate_overload_and_shutdown():
    with pytest.raises(ValueError, match="must be 1"):
        InferGate(2)
    gate = InferGate(1)
    assert gate.try_acquire() == ""
    assert gate.try_acquire() == "overloaded"
    gate.release()
    assert gate.try_acquire() == ""
    gate.shutdown()
    gate.release()
    assert gate.try_acquire() == "shutting_down"


def test_public_body_strips_paths():
    payload = FakeCascade().infer_waveform(np.ones(8), SAMPLE_RATE)
    body = public_inference_body(payload, request_id="rid")
    blob = str(body)
    assert "/secret/" not in blob
    assert "asr_checkpoint" not in body
    assert body["research_provenance"] == "NOT_VERIFIED"
    assert body["tokenizer_parity"] == "VERIFIED"


# --- HTTP ---


def test_healthz_does_not_infer_or_reload():
    client, runtime, cascade = _client()
    with client:
        first = client.get("/healthz")
        second = client.get("/healthz")
    assert first.status_code == 200
    assert second.json()["status"] == "ok"
    assert cascade.calls == 0
    assert runtime.load_count == 1
    assert runtime.infer_count == 0


def test_readyz_ok_and_failed_startup():
    client, runtime, _ = _client()
    with client:
        ok = client.get("/readyz")
    assert ok.status_code == 200
    assert ok.json()["status"] == "ready"
    assert ok.json()["load_count"] == 1

    cfg = Path("unused")
    rt = SpeechRuntime(_settings(config_path=cfg, fail_fast=False), cascade_factory=lambda _c: (_ for _ in ()).throw(C0ConfigError("nope")))
    app = create_app(_settings(fail_fast=False), rt)
    with TestClient(app) as failed:
        ready = failed.get("/readyz")
        health = failed.get("/healthz")
    assert health.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] in {"config_error", "not_ready"}
    with TestClient(app) as again:
        infer = again.post(
            "/v1/speech/infer",
            files={"audio": ("a.wav", pcm16_wav_bytes(np.zeros(160)), "audio/wav")},
            headers=_headers(),
        )
        assert infer.status_code == 503


def test_infer_success_reuses_runtime_and_hides_paths(tmp_path: Path):
    client, runtime, cascade = _client()
    wav_bytes = pcm16_wav_bytes(np.zeros(SAMPLE_RATE, dtype=np.float32))
    before = list(tmp_path.iterdir())
    with client:
        first = _post_audio(client, wav_bytes)
        second = _post_audio(client, wav_bytes)
    assert first.status_code == 200
    assert second.status_code == 200
    body = first.json()
    assert body["model_id"] == "c0_cascaded"
    assert body["asr_text_bahnar"] == "xin"
    assert body["translated_text_vi"] == "tôi"
    assert body["duration_ms"] == 1000
    assert body["latency_ms"] == 7
    assert body["tokenizer_parity"] == "VERIFIED"
    assert body["research_provenance"] == "NOT_VERIFIED"
    assert body["checkpoint_identity"] == "expected_step_match"
    assert cascade.calls == 2
    assert runtime.cascade is cascade
    assert runtime.load_count == 1
    dumped = str(body)
    assert "/secret/" not in dumped
    assert "asr_checkpoint" not in body
    assert list(tmp_path.iterdir()) == before


def test_http_validation_errors():
    client, _, _ = _client()
    with client:
        missing = client.post("/v1/speech/infer", headers=_headers())
        assert missing.status_code == 400
        assert missing.json()["error"]["code"] == "audio_missing"
        empty = _post_audio(client, b"")
        assert empty.status_code == 400
        assert empty.json()["error"]["code"] in {"audio_empty", "audio_invalid", "audio_unsupported"}
        mp3 = _post_audio(client, b"ID3\x04junk", filename="a.mp3")
        assert mp3.status_code == 400
        assert mp3.json()["error"]["code"] == "audio_unsupported"
        rate = _post_audio(client, pcm16_wav_bytes(np.zeros(160), sr=8000))
        assert rate.status_code == 400
        assert rate.json()["error"]["code"] == "audio_sample_rate"
        stereo = _post_audio(client, pcm16_wav_bytes(np.zeros(160), channels=2))
        assert stereo.status_code == 400
        assert stereo.json()["error"]["code"] == "audio_channels"
    client2, _, _ = _client(max_audio_seconds=0.5)
    with client2:
        too_long = _post_audio(client2, pcm16_wav_bytes(np.zeros(SAMPLE_RATE)))
        assert too_long.status_code == 400
        assert too_long.json()["error"]["code"] == "audio_too_long"
    client3, _, _ = _client(max_upload_bytes=64)
    with client3:
        oversized = _post_audio(client3, pcm16_wav_bytes(np.zeros(4000)))
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "payload_too_large"
        assert "traceback" not in oversized.text.lower()
        assert "/secret" not in oversized.text


def test_unauthorized_and_sanitized_runtime_error():
    client, _, _ = _client()
    wav_bytes = pcm16_wav_bytes(np.zeros(160))
    with client:
        no_token = client.post("/v1/speech/infer", files={"audio": ("a.wav", wav_bytes, "audio/wav")})
        assert no_token.status_code == 401
        bad = client.post(
            "/v1/speech/infer",
            files={"audio": ("a.wav", wav_bytes, "audio/wav")},
            headers=_headers("xxxxxxxxxxxxxxxx"),
        )
        assert bad.status_code == 401
        assert bad.json()["error"]["code"] == "unauthorized"

    boom = FakeCascade(error_payload={"error": "ASR produced empty Bahnar transcript at /secret/asr", "error_code": "empty_asr"})
    client_err, _, _ = _client(boom)
    with client_err:
        failed = _post_audio(client_err, wav_bytes)
        assert failed.status_code == 422
        assert failed.json()["error"]["code"] == "empty_asr"
        assert "/secret" not in failed.text
        assert "Traceback" not in failed.text


def test_overload_and_timeout():
    wav_bytes = pcm16_wav_bytes(np.zeros(160))
    slow = FakeCascade(delay_s=0.4)
    client_timeout, _, _ = _client(slow, request_timeout_s=0.05)
    with client_timeout:
        timed = _post_audio(client_timeout, wav_bytes)
        assert timed.status_code == 504
        assert timed.json()["error"]["code"] == "inference_timeout"

        blocker = FakeCascade(delay_s=0.4)
        settings = _settings(request_timeout_s=5, max_inflight=1)
        runtime = SpeechRuntime(settings, cascade=blocker)
        runtime.start()
        app2 = create_app(settings, runtime)

        async def _overload():
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app2)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                t1 = asyncio.create_task(
                    ac.post("/v1/speech/infer", files={"audio": ("a.wav", wav_bytes, "audio/wav")}, headers=_headers())
                )
                await asyncio.sleep(0.05)
                t2 = asyncio.create_task(
                    ac.post("/v1/speech/infer", files={"audio": ("b.wav", wav_bytes, "audio/wav")}, headers=_headers())
                )
                r1, r2 = await asyncio.gather(t1, t2)
                codes = sorted([r1.status_code, r2.status_code])
                assert codes == [200, 503]
                bodies = [r1.json(), r2.json()]
                assert any(b.get("error", {}).get("code") == "overloaded" for b in bodies if "error" in b)

        asyncio.run(_overload())


def test_timeout_consumes_worker_exception_without_leaking_slot(caplog):
    secret = RuntimeError("boom at /secret/mt/checkpoint-17500 transcript=xin")
    cascade = FakeCascade(delay_s=0.3, raise_after=secret)
    runtime = SpeechRuntime(_settings(), cascade=cascade)
    runtime.start()
    wav = np.ones(160, dtype=np.float32)

    async def _run():
        loop = asyncio.get_running_loop()
        unhandled = []
        loop.set_exception_handler(lambda _loop, ctx: unhandled.append(ctx))
        with pytest.raises(SpeechAPIError) as timed:
            await run_inference_slot(runtime, wav, SAMPLE_RATE, timeout_s=0.05)
        assert timed.value.code == "inference_timeout"
        assert runtime.gate.inflight == 1
        deadline = time.monotonic() + 2.0
        while runtime.gate.inflight != 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert runtime.gate.inflight == 0
        await asyncio.sleep(0.05)
        assert unhandled == []
        second = await run_inference_slot(runtime, wav, SAMPLE_RATE, timeout_s=5)
        assert second["asr_text_bahnar"] == "xin"

    with caplog.at_level("WARNING", logger="c0_speech"):
        asyncio.run(_run())
    blob = caplog.text
    assert "RuntimeError" in blob
    assert "/secret" not in blob
    assert "checkpoint-17500" not in blob
    assert "transcript=xin" not in blob


def test_http_timeout_then_worker_exception_allows_next_request():
    cascade = FakeCascade(
        delay_s=0.3,
        raise_after=RuntimeError("secret /workspace/bahnar-s2tt-thesis/checkpoint"),
    )
    client, runtime, _ = _client(cascade, request_timeout_s=0.05)
    wav_bytes = pcm16_wav_bytes(np.zeros(160))
    with client:
        timed = _post_audio(client, wav_bytes)
        assert timed.status_code == 504
        assert timed.json()["error"]["code"] == "inference_timeout"
        assert "/workspace" not in timed.text
        deadline = time.monotonic() + 2.0
        while runtime.gate.inflight != 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert runtime.gate.inflight == 0
        ok = _post_audio(client, wav_bytes)
        assert ok.status_code == 200
        assert ok.json()["asr_text_bahnar"] == "xin"


def test_cancelled_request_releases_gpu_slot():
    cascade = FakeCascade(delay_s=0.25)
    runtime = SpeechRuntime(_settings(), cascade=cascade)
    runtime.start()
    wav = np.ones(160, dtype=np.float32)

    async def _run():
        task = asyncio.create_task(run_inference_slot(runtime, wav, SAMPLE_RATE, timeout_s=5))
        await asyncio.sleep(0.05)
        assert runtime.gate.inflight == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.gate.inflight == 1
        deadline = time.monotonic() + 2.0
        while runtime.gate.inflight != 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert runtime.gate.inflight == 0
        second = await run_inference_slot(runtime, wav, SAMPLE_RATE, timeout_s=5)
        assert second["asr_text_bahnar"] == "xin"
        assert runtime.gate.inflight == 0

    asyncio.run(_run())


def test_http_cancel_then_second_infer_succeeds():
    cascade = FakeCascade(delay_s=0.3)
    settings = _settings()
    runtime = SpeechRuntime(settings, cascade=cascade)
    runtime.start()
    app = create_app(settings, runtime)
    wav_bytes = pcm16_wav_bytes(np.zeros(160))

    async def _run():
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            first = asyncio.create_task(
                ac.post(
                    "/v1/speech/infer",
                    files={"audio": ("a.wav", wav_bytes, "audio/wav")},
                    headers=_headers(),
                )
            )
            deadline = time.monotonic() + 1.0
            while runtime.gate.inflight == 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert runtime.gate.inflight == 1
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            wait_done = time.monotonic() + 2.0
            while runtime.gate.inflight != 0 and time.monotonic() < wait_done:
                await asyncio.sleep(0.02)
            assert runtime.gate.inflight == 0
            second = await ac.post(
                "/v1/speech/infer",
                files={"audio": ("b.wav", wav_bytes, "audio/wav")},
                headers=_headers(),
            )
            assert second.status_code == 200
            body = second.json()
            assert body["asr_text_bahnar"] == "xin"
            assert "asr_checkpoint" not in body
            assert "error" not in body

    asyncio.run(_run())


def test_serving_contract_blocks_unverified_tokenizer_and_non_rq1_generation():
    bad_tok = FakeCascade(tokenizer_parity="NOT_VERIFIED", generation_matches_rq1=True)
    rt = SpeechRuntime(_settings(fail_fast=False), cascade=bad_tok)
    rt.start()
    assert rt.ready is False
    assert rt.init_error.code == "tokenizer_not_verified"
    app = create_app(_settings(fail_fast=False), rt)
    with TestClient(app) as client:
        ready = client.get("/readyz")
        infer = _post_audio(client, pcm16_wav_bytes(np.zeros(160)))
    assert ready.status_code == 503
    assert ready.status_code != 200
    assert ready.json()["error"]["code"] == "tokenizer_not_verified"
    assert infer.status_code == 503

    bad_gen = FakeCascade(tokenizer_parity="VERIFIED", generation_matches_rq1=False)
    rt2 = SpeechRuntime(_settings(fail_fast=False), cascade=bad_gen)
    rt2.start()
    assert rt2.ready is False
    assert rt2.init_error.code == "generation_not_rq1"
    app2 = create_app(_settings(fail_fast=False), rt2)
    with TestClient(app2) as client:
        ready2 = client.get("/readyz")
    assert ready2.status_code == 503
    assert ready2.json()["error"]["code"] == "generation_not_rq1"

    mismatched = FakeCascade(
        tokenizer_parity="VERIFIED",
        generation_matches_rq1=True,
        max_source_length=128,
        generation_max_length=256,
        num_beams=4,
    )
    rt_mismatch = SpeechRuntime(_settings(fail_fast=False), cascade=mismatched)
    rt_mismatch.start()
    assert rt_mismatch.ready is False
    assert rt_mismatch.init_error.code == "generation_not_rq1"
    app_mismatch = create_app(_settings(fail_fast=False), rt_mismatch)
    with TestClient(app_mismatch) as client:
        ready_mismatch = client.get("/readyz")
    assert ready_mismatch.status_code == 503
    assert ready_mismatch.status_code != 200

    ok = FakeCascade(tokenizer_parity="VERIFIED", generation_matches_rq1=True)
    ok.research_provenance = "NOT_VERIFIED"
    rt3 = SpeechRuntime(_settings(fail_fast=False), cascade=ok)
    rt3.start()
    assert rt3.ready is True
    assert getattr(rt3.cascade, "research_provenance", "NOT_VERIFIED") == "NOT_VERIFIED"
    app_ok = create_app(_settings(fail_fast=False), rt3)
    with TestClient(app_ok) as client:
        assert client.get("/readyz").status_code == 200


def test_serving_contract_requires_mt_training_contract_file(tmp_path):
    missing = FakeCascade(mt_training_contract=str(tmp_path / "missing.json"))
    rt = SpeechRuntime(_settings(fail_fast=False), cascade=missing)
    rt.start()
    assert rt.ready is False
    assert rt.init_error.code == "mt_contract_missing"
    app = create_app(_settings(fail_fast=False), rt)
    with TestClient(app) as client:
        ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.status_code != 200
    assert ready.json()["error"]["code"] == "mt_contract_missing"
    assert str(tmp_path) not in ready.text

    contract = tmp_path / "mt_training_contract.json"
    contract.write_text("{}", encoding="utf-8")

    def factory(_config):
        return FakeCascade()

    missing_cfg = SpeechRuntime(
        _settings(config_path=tmp_path / "c0.yaml", fail_fast=False),
        cascade=None,
        cascade_factory=factory,
        config_loader=lambda _path: {"device": "cpu"},
    )
    missing_cfg.start()
    assert missing_cfg.ready is False
    assert missing_cfg.init_error.code == "mt_contract_missing"

    ok_cfg = SpeechRuntime(
        _settings(config_path=tmp_path / "c0.yaml", fail_fast=False),
        cascade=None,
        cascade_factory=factory,
        config_loader=lambda _path: {"mt_training_contract": str(contract)},
    )
    ok_cfg.start()
    assert ok_cfg.ready is True
    app_ok = create_app(_settings(fail_fast=False), ok_cfg)
    with TestClient(app_ok) as client:
        assert client.get("/readyz").status_code == 200


def test_shutdown_rejects_new_requests_and_drains_inflight():
    cascade = FakeCascade(delay_s=0.2)
    runtime = SpeechRuntime(_settings(shutdown_timeout_s=2.0), cascade=cascade)
    runtime.start()
    wav = np.ones(160, dtype=np.float32)

    async def _run():
        task = asyncio.create_task(run_inference_slot(runtime, wav, SAMPLE_RATE, timeout_s=5))
        await asyncio.sleep(0.05)
        assert runtime.gate.inflight == 1
        drain = asyncio.create_task(runtime.drain(timeout_s=2.0))
        await asyncio.sleep(0.02)
        with pytest.raises(SpeechAPIError) as blocked:
            await run_inference_slot(runtime, wav, SAMPLE_RATE, timeout_s=5)
        assert blocked.value.code == "shutting_down"
        await task
        await drain
        assert runtime.gate.inflight == 0
        assert runtime.ready is False
        assert runtime.gate.accepting is False

    asyncio.run(_run())


def test_drain_does_not_claim_success_while_gpu_busy():
    cascade = FakeCascade(delay_s=0.4)
    runtime = SpeechRuntime(_settings(shutdown_timeout_s=0.05), cascade=cascade)
    runtime.start()
    wav = np.ones(160, dtype=np.float32)

    async def _run():
        task = asyncio.create_task(run_inference_slot(runtime, wav, SAMPLE_RATE, timeout_s=5))
        deadline = time.monotonic() + 1.0
        while runtime.gate.inflight == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert runtime.gate.inflight == 1
        with pytest.raises(SpeechAPIError) as timed:
            await runtime.drain(timeout_s=0.05)
        assert timed.value.code == "shutdown_timeout"
        assert runtime.gate.inflight == 1
        assert runtime.gate.accepting is False
        await task
        assert runtime.gate.inflight == 0

    asyncio.run(_run())


def test_lifespan_stops_admission_on_shutdown():
    client, runtime, _ = _client()
    with client:
        assert client.get("/healthz").status_code == 200
        assert runtime.gate.accepting is True
        assert runtime.ready is True
    assert runtime.gate.accepting is False
    assert runtime.ready is False
    assert runtime.gate.inflight == 0


def test_c0_source_has_no_runtime_hardcode():
    speech_root = ROOT / "src" / "c0_speech"
    for path in speech_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "/tmp/bahnar-runtime" not in text
        assert "/workspace/bahnar-s2tt-thesis" not in text
