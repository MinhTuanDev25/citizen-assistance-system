# Phase 7 Step 7.4 — RunPod Speech API smoke (operator)

These commands are **prepared, not executed from Cursor**. Do not mark
`RUNPOD_GPU_API_VERIFIED` or `REAL_AUDIO_API_VERIFIED` until the operator
pastes real output.

Do not train, do not overwrite `configs/c0_inference.runpod.yaml`, do not
copy checkpoints into Git, and do not deploy the Citizen Assistance stack.

## Layout

Do **not** create a separate Git worktree. `run_speech_api.sh` derives
`CAS_ROOT` from its own path (`../..` from `scripts/phase7`). Override with
environment variables if a path differs.

| Role | Path / variable |
|---|---|
| CAS source | `CAS_ROOT=/workspace/citizen-assistance-system` |
| Thesis source | `THESIS_SOURCE_ROOT=$CAS_ROOT/bahnar-s2tt-thesis` |
| Artifact root | `ARTIFACT_ROOT=/workspace/bahnar-s2tt-thesis` |
| C0 venv | `C0_INFERENCE_VENV=/workspace/.venvs/c0-inference` |
| Operator C0 YAML | existing `configs/c0_inference.runpod.yaml` (do not commit) |

## 0. Install HTTP extras once (same venv, no ML pin changes)

```bash
export CAS_ROOT=/workspace/citizen-assistance-system
export THESIS_SOURCE_ROOT="${CAS_ROOT}/bahnar-s2tt-thesis"
export ARTIFACT_ROOT=/workspace/bahnar-s2tt-thesis
export C0_INFERENCE_VENV=/workspace/.venvs/c0-inference
source "${C0_INFERENCE_VENV}/bin/activate"
python -m pip install -r "${THESIS_SOURCE_ROOT}/requirements-c0-speech-api.txt"
```

## 1. Start the Speech API (one worker)

```bash
export CAS_ROOT=/workspace/citizen-assistance-system
export THESIS_SOURCE_ROOT="${CAS_ROOT}/bahnar-s2tt-thesis"
export ARTIFACT_ROOT=/workspace/bahnar-s2tt-thesis
export C0_INFERENCE_VENV=/workspace/.venvs/c0-inference
export C0_INFERENCE_CONFIG="${THESIS_SOURCE_ROOT}/configs/c0_inference.runpod.yaml"
source "${C0_INFERENCE_VENV}/bin/activate"
export PYTHONPATH="${THESIS_SOURCE_ROOT}"
export SPEECH_HOST=127.0.0.1
export SPEECH_PORT=8090
export SPEECH_DEVICE=cuda
export SPEECH_MAX_INFLIGHT=1
export SPEECH_WORKERS=1
export SPEECH_AUTH=on
export SPEECH_SERVICE_TOKEN='replace-with-operator-token'
export SPEECH_FAIL_FAST=true
cd "${THESIS_SOURCE_ROOT}"
python -m src.c0_speech
```

Or:

```bash
export CAS_ROOT=/workspace/citizen-assistance-system
export THESIS_SOURCE_ROOT="${CAS_ROOT}/bahnar-s2tt-thesis"
export C0_INFERENCE_CONFIG="${THESIS_SOURCE_ROOT}/configs/c0_inference.runpod.yaml"
export SPEECH_SERVICE_TOKEN='replace-with-operator-token'
export C0_INFERENCE_VENV=/workspace/.venvs/c0-inference
bash "${CAS_ROOT}/scripts/phase7/run_speech_api.sh"
```

Keep `SPEECH_AUTH=on` for this smoke. `SPEECH_AUTH=off` is rejected unless
`SPEECH_HOST` is loopback (`127.0.0.1` / `::1` / `localhost`).

Wait until logs show readiness (or `GET /readyz` returns 200). Record
`init_ms` from `/readyz`. That is model initialization time, not request
latency.

## 2. Health and readiness

```bash
curl -sS http://127.0.0.1:8090/healthz
curl -sS http://127.0.0.1:8090/readyz
```

`/healthz` must stay 200 even if you later stop accepting inference.
`/readyz` must be 200 only after ASR+MT load, a valid MT training
contract file, `tokenizer_parity=VERIFIED`, `generation_matches_rq1=true`,
generation 256/256/4, and the existing checkpoint/checksum gates.
`research_provenance=NOT_VERIFIED` is still a valid ready state.

## 3–5. Real Bahnar WAV, twice

```bash
TOKEN='replace-with-operator-token'
WAV=/path/to/bahnar.wav

curl -sS -D - -o /tmp/speech1.json \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-Request-ID: 11111111-1111-1111-1111-111111111111" \
  -F "audio=@${WAV};type=audio/wav" \
  http://127.0.0.1:8090/v1/speech/infer

curl -sS -D - -o /tmp/speech2.json \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-Request-ID: 22222222-2222-2222-2222-222222222222" \
  -F "audio=@${WAV};type=audio/wav" \
  http://127.0.0.1:8090/v1/speech/infer

python - <<'PY'
import json
for name in ("/tmp/speech1.json", "/tmp/speech2.json"):
    p = json.load(open(name, encoding="utf-8"))
    print(name, {
        "model_id": p.get("model_id"),
        "asr_len": len(p.get("asr_text_bahnar") or ""),
        "vi_len": len(p.get("translated_text_vi") or ""),
        "duration_ms": p.get("duration_ms"),
        "latency_ms": p.get("latency_ms"),
        "tokenizer_parity": p.get("tokenizer_parity"),
        "research_provenance": p.get("research_provenance"),
        "checkpoint_identity": p.get("checkpoint_identity"),
    })
    assert p.get("research_provenance") != "VERIFIED", "do not mint VERIFIED"
PY
```

## 6. Confirm weights are not reloaded per request

`/readyz` `load_count` must remain `1` after both POSTs.

Process RSS / GPU memory should not jump by a full model copy on the second
request. Logs must not show a second `from_pretrained` of the same
checkpoints. `latency_ms` in the JSON is inference time (ASR+MT), not
startup.

## 7. Measure request latency

Use the JSON `latency_ms` field (model inference) and curl `-D` / `time`
for total HTTP time. Do not treat startup `init_ms` as request latency.

## 8. Stop safely

Send SIGINT/SIGTERM to the `python -m src.c0_speech` process. Admission
stops immediately. In-flight GPU work is drained for
`SPEECH_SHUTDOWN_TIMEOUT_S` (default 30). New POSTs should return
`shutting_down` or the connection closes. Do not kill CUDA by deleting
checkpoint files, and do not treat a timed-out drain as a clean GPU stop.

## Field meanings

- `duration_ms`: waveform length at 16 kHz, milliseconds.
- `latency_ms`: wall time of ASR then MT for that request.
- `tokenizer_parity=VERIFIED` is Hub fingerprint vs MT contract, not
  checkpoint provenance.
- `checkpoint_identity=expected_step_match` is basename step only.
- `research_provenance=NOT_VERIFIED` remains accurate until independent
  research attestation exists. Successful inference does not change it.
