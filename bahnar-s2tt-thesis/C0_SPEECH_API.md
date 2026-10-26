# C0 Speech Inference API (Step 7.4)

Internal FastAPI wrapper around the existing C0 cascaded Bahnar ASR →
Vietnamese MT pipeline. Serving software, not a new research experiment.

Local tests on this machine used a fake runtime. **RunPod GPU API and real
audio HTTP inference are UNVERIFIED until the operator runs
`docs/RUNPOD.md`.**

## Architecture

```
Citizen Web / Mobile
        |
        v
Go Backend  (not in Step 7.4; authenticates citizens)
        |
        v
Internal Speech API  (this process, 127.0.0.1 by default)
  GET  /healthz
  GET  /readyz
  POST /v1/speech/infer
        |
        v
SpeechRuntime — one C0Cascade per worker
        |
        +--> XLS-R ASR  checkpoint-12500
        |       --> Bahnar transcript
        +--> BARTpho MT checkpoint-17500 + Hub tokenizer pin
                --> Vietnamese translation
```

Not hosted inside `apps/ai-service`. One uvicorn process, `SPEECH_WORKERS=1`,
`SPEECH_MAX_INFLIGHT=1`. Service token (`SPEECH_SERVICE_TOKEN`) for Go →
Speech. TLS is assumed at the private network / reverse-proxy layer.

## How existing C0 inference is reused

- CLI `python -m src.c0_cascaded --config … --audio …` is unchanged.
- `C0Cascade.infer_waveform` runs the same ASR then MT path in memory.
- Tokenizer pin, checksum gates, RQ1 generation 256/256/4, and provenance
  labels are unchanged. Successful HTTP inference does **not** mint
  `research_provenance=VERIFIED`. `expected_step_match` is still basename
  identity, not verified checkpoint identity.

## API contract

`POST /v1/speech/infer` — `multipart/form-data` field `audio`.

- WAV, 16 kHz, mono
- Max 30 s
- `SPEECH_MAX_UPLOAD_BYTES` (default 10 MiB) caps the **entire HTTP
  request body**, including multipart boundaries and part headers, not
  just the WAV octets. A WAV whose file size equals the limit can still
  return 413 because of that overhead. After the body is accepted,
  `decode_wav_bytes` still enforces the same byte cap on the extracted
  WAV field, and C0 `read_c0_audio` still applies `max_audio_bytes` to
  the waveform file on the CLI path.
- Stereo is **rejected** (CLI still mean-mixes, matching RQ1 evaluation)

Success JSON includes C0 fields plus `request_id`. Paths
(`asr_checkpoint`, `mt_checkpoint`) are omitted from HTTP.

| Field | Meaning |
|---|---|
| `duration_ms` | Audio length at 16 kHz |
| `latency_ms` | ASR + MT wall time for that request (not startup `init_ms`) |
| `tokenizer_parity` | Hub fingerprint vs MT training contract |
| `research_provenance` | Copied from C0 (`NOT_VERIFIED` or `REFERENCE_HASH_MATCH`) |

`GET /healthz` — liveness; no inference; no extra model load.
`GET /readyz` — 200 only after load **and** Speech serving gates:
existing MT training contract file, `tokenizer_parity=VERIFIED`,
`generation_matches_rq1=true`, and generation values 256/256/4.
`research_provenance=NOT_VERIFIED` remains a valid ready state. Checkpoint
and configured checksum checks still run inside `C0Cascade`.

GPU slot release is tied to the inference task finishing (success, error,
timeout, cancellation, shutdown). The done-callback also retrieves the
worker result/exception so a timeout cannot leave an unhandled task
exception. `SPEECH_MAX_INFLIGHT` must be 1.
`SPEECH_AUTH=off` is allowed only on loopback binds (`127.0.0.1`, `::1`,
`localhost`). Wildcard, LAN, public IP, and other hostnames require
`SPEECH_AUTH=on`.

Errors: 4xx validation, 401 missing/wrong token, 413 too large, 503 not
ready / overloaded / shutting down, 504 timeout, 5xx runtime. Response
bodies never include traces or filesystem paths.

## Files

| Path | Purpose |
|---|---|
| `src/c0_cascaded.py` | `infer_waveform` on a loaded cascade |
| `src/c0_speech/` | FastAPI app, settings, audio validation, runtime/gate |
| `tests/test_c0_speech.py` | Unit + HTTP tests with a fake cascade |
| `requirements-c0-speech-api.txt` | FastAPI extras on top of C0 inference pins |
| `configs/c0_speech_api.example.env` | Env template (no real paths/secrets) |
| `scripts/runpod/run_speech_api.sh` | One-worker startup |
| `docs/RUNPOD.md` | Operator GPU smoke commands |

## Local verification (this machine)

```text
python -m compileall -q src/c0_speech src/c0_cascaded.py
PYTHONPATH=. python -m pytest -q tests/test_c0_cascaded.py tests/test_c0_tokenizer.py tests/test_c0_speech.py
75 passed
```

Breakdown: 37 C0 cascaded, 11 tokenizer, 27 Speech API.

No real checkpoints, no GPU, no Hub download in these tests.

## Gates

```
SOURCE_REVIEWED                 PASS
CLI_REGRESSION_PASS             PASS (test_c0_cascaded + test_c0_tokenizer)
API_UNIT_TESTS_PASS             PASS
API_INTEGRATION_TESTS_PASS      PASS (fake runtime)
AUDIO_VALIDATION_PASS           PASS
MODEL_LIFECYCLE_PASS            PASS (fake runtime)
CONFIG_VALIDATION_PASS          PASS
TOKENIZER_CHECKS_PRESERVED      PASS
CHECKPOINT_CHECKS_PRESERVED     PASS
RESEARCH_ARTIFACTS_UNCHANGED    PASS
RUNPOD_GPU_API_VERIFIED         UNVERIFIED
REAL_AUDIO_API_VERIFIED         UNVERIFIED
STEP74_READY_FOR_RUNPOD         true
STEP74_COMPLETE                 false
```

## Limitations

- Concurrent second inference returns 503 `overloaded`. `SPEECH_MAX_INFLIGHT`
  must be 1; the advertised limit is the actual GPU occupancy.
- HTTP timeout / client cancel cannot abort CUDA; the slot stays occupied
  until the worker thread finishes, then it is released exactly once.
- Shutdown stops admission and waits `SPEECH_SHUTDOWN_TIMEOUT_S` (default 30).
  If GPU work is still running when the grace period ends, drain raises
  `shutdown_timeout` instead of claiming a clean stop.
- No Go/frontend integration in this step.
- No Dockerfile GPU claim.
