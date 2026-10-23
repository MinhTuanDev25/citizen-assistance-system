# Phase 7 Step 7.5 — Go Backend ↔ C0 Speech API (operator)

Prepared, not executed from Cursor. Do not mark RunPod gates PASS until
real output is pasted.

Do not train, do not expose port 8090 on the public internet, and do not
commit `SPEECH_SERVICE_TOKEN` or SSH keys.

Go never implements SSH. The operator opens the tunnel.

Citizen route:

```text
POST /api/v1/sessions/:sessionId/speech/translate
```

`POST /api/v1/speech/translate` returns HTTP 410.

`SPEECH_SERVICE_URL` must be loopback HTTP only:

- `http://127.0.0.1:18090` — Mac → SSH tunnel → RunPod Speech
- `http://127.0.0.1:8090` — Go and Speech in the same network namespace
- `http://localhost:8090` and `http://[::1]:8090` — same loopback policy

HTTP to public hosts, RFC1918, Docker DNS names, URL paths, query strings,
fragments, userinfo, and arbitrary HTTPS destinations is rejected at startup.

Audit is **best-effort**. A successful translation may still return HTTP 200
if the audit INSERT fails. Failures are logged as `speech_audit_failed`.
Cancellation uses a 2s detached timeout so a hung-up client cannot skip the write.

## A. Local Mac development (Go on the laptop, Speech on RunPod)

Speech still binds `127.0.0.1:8090` **inside** the pod. The laptop cannot
dial that address without a tunnel.

1. Start the Speech API on RunPod (`scripts/phase7/RUNPOD_SPEECH_API.md`).
2. From the Mac, open an operator-managed local forward. Use the pod's real
   SSH host/port from the RunPod UI (do not copy invented values):

```bash
ssh -N -L 18090:127.0.0.1:8090 <runpod-ssh-user>@<runpod-ssh-host> -p <runpod-ssh-port>
```

3. Confirm the tunnel, not a public 8090:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18090/healthz
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18090/readyz
```

4. Configure the **local** Go process:

```bash
export SPEECH_ENABLED=true
export SPEECH_SERVICE_URL=http://127.0.0.1:18090
export SPEECH_SERVICE_TOKEN='same-token-as-python'
export SPEECH_TIMEOUT_MS=130000
export SPEECH_MAX_UPLOAD_BYTES=10485760
export SPEECH_MAX_INFLIGHT=1
```

`SPEECH_MAX_UPLOAD_BYTES` is the FastAPI whole-body cap. Go accepts a smaller
WAV so the re-encoded multipart stays under that cap.

5. Create a guest session, then translate. WAV is an engineering fixture,
   not an RQ2 evaluation sample:

```bash
SID='session-uuid-from-create'
TOKEN='guest-token-from-create'
WAV=/path/to/uid_30d975299b477ae0c6d1e0b0efb4589e.wav

curl -sS -D - -o /tmp/go-speech.json \
  -H "X-Guest-Token: ${TOKEN}" \
  -H "X-Request-ID: 33333333-3333-3333-3333-333333333333" \
  -F "audio=@${WAV};type=audio/wav" \
  "http://127.0.0.1:8080/api/v1/sessions/${SID}/speech/translate"
```

## B. Same-pod integration (Go and Speech share a network namespace)

This is valid only when the Go process and `python -m src.c0_speech` share
loopback. A Go process inside Docker on the same machine does **not** share
`127.0.0.1` with a host Speech API unless the container network is explicitly
the host namespace.

```bash
export SPEECH_ENABLED=true
export SPEECH_SERVICE_URL=http://127.0.0.1:8090
export SPEECH_SERVICE_TOKEN='same-token-as-python'
```

```bash
curl -sS http://127.0.0.1:8090/healthz
curl -sS http://127.0.0.1:8090/readyz
```

Then the same session + multipart call as section A, against the Go listen address.

Apply migration `000021_speech_translation_requests` before expecting audit rows.

```sql
SELECT request_id, session_id, xa_id, outcome, http_status, model_id,
       tokenizer_parity, research_provenance, duration_ms, latency_ms
FROM speech_translation_requests
ORDER BY created_at DESC
LIMIT 5;
```

`research_provenance` is copied, never rewritten to `VERIFIED`.

Text chat must still work on `POST /api/v1/sessions/:sessionId/turns`.

## C. Production Docker Compose — go-live blocker (Option C)

`deploy/docker-compose.production.yml` runs Go `api` on the Compose network
(`http://ai-service:8001`). That container's `127.0.0.1` is **not** the RunPod
host loopback where Speech binds `127.0.0.1:8090`.

Do **not**:

- set `SPEECH_SERVICE_URL=http://127.0.0.1:8090` inside the `api` container
- set `network_mode: host` on `api` (breaks DNS to `ai-service`)
- publish Speech port `8090`
- add an RFC1918 / Docker-DNS exception to the M2 loopback policy
- invent a Speech sidecar image that this repo does not build

Supported Speech topologies:

| Topology | Go process | `SPEECH_SERVICE_URL` | Status |
|---|---|---|---|
| Development SSH tunnel | Go on the Mac | `http://127.0.0.1:18090` | Supported |
| Same-namespace process | Go + `python -m src.c0_speech` on the RunPod host | `http://127.0.0.1:8090` | Supported |
| Production Compose `api` → host Speech | Docker `api` | none | **Not implemented** — go-live blocker |

Leave `SPEECH_ENABLED=false` in production Compose. Enabling Speech without a
loopback URL fails closed at API startup.

Reachability checks (operator; paste real output, do not invent PASS):

```bash
# Dev tunnel, from the Mac that runs Go:
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18090/healthz

# Same-namespace process, from the RunPod host that runs Go:
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/healthz

# Docker api container localhost is the wrong namespace (expected fail):
docker compose -f deploy/docker-compose.production.yml exec api \
  wget -qO- http://127.0.0.1:8090/healthz
```

Apply `000021` before expecting audit rows. Rollback of `000021` drops only
Step 7.5 columns and indexes; it does not `DROP TABLE speech_translation_requests`.
