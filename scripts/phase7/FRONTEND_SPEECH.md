# Phase 7 Step 7.6 — Citizen Bahnar speech (frontend)

## 1. What this step implements

Citizen Chat can send Bahnar audio to the existing Go route

`POST /api/v1/sessions/:sessionId/speech/translate`

then **review and confirm** the Vietnamese text before it enters
`POST /api/v1/sessions/:sessionId/turns` (same path as typed chat).

Two inputs, one API and one confirmation panel:

1. **Micro Bahnar** — `getUserMedia` + Web Audio capture, resampled to
   PCM16 mono 16 kHz WAV. Not the browser SpeechRecognition API.
2. **Tải file WAV** — user-selected 16 kHz PCM16 mono WAV.

Flow: **Bắt đầu ghi âm → Dừng & Dịch → xem Bahnar + Việt → sửa Việt nếu cần → Xác nhận & Hỏi trợ lý → Chat Turn.**

Stopping a recording does **not** post a chat turn. Cancel posts nothing.
Selecting a file does **not** start inference.

## 2. Microphone behavior before 7.6

The 🎙 button on the composer used `window.SpeechRecognition` /
`webkitSpeechRecognition` with `lang = 'vi-VN'`. It filled the text box.
It was Vietnamese browser dictation, not Bahnar ASR.

## 3. Microphone behavior after 7.6

That Vietnamese dictation control is **kept** and relabeled
(`Nói câu hỏi bằng tiếng Việt`). It is not Bahnar ASR.

**Micro Bahnar** is a separate panel under the composer:

1. Citizen clicks **Bắt đầu ghi âm** (browser permission if needed).
2. UI shows a live recording indicator and elapsed time. No network call.
3. **Dừng & Dịch** stops capture, encodes WAV, then posts once to Go.
4. **Hủy ghi âm** stops capture, drops the buffer, and sends no request.
5. Auto-stop at 30 s encodes the WAV but does **not** auto-translate;
   the citizen must click **Dịch âm thanh**. The finalized microphone
   WAV is stored with the same pending-audio contract as Upload
   (`bytes`, `durationMs`, `sampleCount`, `name`, `source`, `gen`) so
   Translate and Retry can send it.

`SpeechRecognition` is never used as a Bahnar substitute. There is no
chunked streaming and no WebSocket inference.

## 4. Upload Audio workflow

1. **Tải file WAV** → **Chọn file WAV**.
2. Frontend validates RIFF/WAVE, PCM 16-bit, mono, 16 kHz, duration, size.
3. UI shows filename, size, duration.
4. **Dịch âm thanh** sends the same Go multipart request as the microphone.
5. **Gỡ file** clears the selection without a request.

Unsupported containers (WebM, MP3, stereo, 8-bit, 8 kHz) are rejected
in the browser and never uploaded.

## 5. WAV capture and encoding

`apps/web/src/lib/wav.js` writes a real RIFF/WAVE header (not a renamed
WebM file). Capture (`apps/web/src/lib/micRecorder.js`) uses:

- `navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, ... } })`
- `AudioContext` / `webkitAudioContext`
- `ScriptProcessor` input buffers (copied per callback)
- mix to mono, then a Hamming **windowed-sinc** resampler (65 taps,
  cutoff 0.9 × destination Nyquist) down to 16 kHz. Typical capture
  rates are 44.1 kHz and 48 kHz. This is not boxcar averaging or linear
  interpolation. See §17 for measured passband/stopband RMS.
- PCM signed 16-bit little-endian
- stop tracks + `AudioContext.close()` on stop, cancel, unmount, or
  mode switch

Requires a secure context (HTTPS or localhost).

## 6. Go API contract

Frontend talks **only** to the Go Backend. It never calls RunPod `:8090`
or the SSH tunnel `:18090`.

Request:

```http
POST /api/v1/sessions/{sessionId}/speech/translate
Accept: application/json
X-Request-ID: <uuid>
X-Guest-Token: <guest-token>   # guest sessions
Authorization: Bearer <jwt>    # logged-in users
Content-Type: multipart/form-data   # browser sets boundary

audio=<WAV-file>
```

Verified success envelope (7.5):

```json
{
  "request_id": "<uuid>",
  "data": {
    "request_id": "<uuid>",
    "model_id": "c0_cascaded",
    "asr_text_bahnar": "<Bahnar>",
    "translated_text_vi": "<Vietnamese>",
    "duration_ms": 14432,
    "latency_ms": 1254
  }
}
```

The client requires `model_id === c0_cascaded` and nonempty ASR + Việt.
Malformed 200 bodies are errors. Empty transcripts are not invented.

Mapped HTTP/codes: 400 `VALIDATION`, 401, 403, 404, 409 `CONFLICT`,
413, 415, **422 `SPEECH_NO_SPEECH`** (upstream `empty_asr` only — a
bare 502 stays `SPEECH_UNAVAILABLE`), 429, 5xx, `SPEECH_DISABLED` /
`OVERLOADED` / `TIMEOUT` / `UNAVAILABLE` / `INVALID_UPSTREAM` /
`UPSTREAM_AUTH`, network, abort. Silence is not retried on the same
WAV. Confirm sends optional `speech_request_id` on the Chat Turn.

## 7. Guest / JWT authentication

Reuses `ensureSession` / `cas.chatSession`. Guest: `X-Guest-Token`.
Logged-in: existing Bearer JWT from `cas_auth`. The panel does not mint
a new session per recording. Speech service tokens stay on the server.

## 8. Chat Turn integration

`BahnarSpeechPanel` calls the same `submit()` as the composer
(`executeChatTurn` → `POST .../turns`). Confirmed **Vietnamese** text is
the Chat Turn `message`. Optional JSON `speech_request_id` is the Go
speech audit `request_id` from the translate response. Typed chat omits
it. Bahnar ASR text is shown read-only and is not posted. Speech audit
does not store transcripts.

`submit()` returns `true`/`false` so a failed turn does not look like
speech success. The composer, quick-procedure chips, and
`CONFIRM_INTENT` buttons still use `submit`.

## 9. Development environment

Default Vite proxy: `/api` → `http://127.0.0.1:8080` (`make api-run` /
`make web-run`).

Docker API on `:8082` may have `SPEECH_ENABLED=false`. The verified
Speech-enabled native API was `:8083`.

Override with existing env, not hardcoded hosts in browser code:

```bash
VITE_API_PROXY=http://127.0.0.1:8083
```

`apps/web/vite.config.js` reads `process.env.VITE_API_PROXY`.

## 10. Local frontend commands

```bash
cd apps/web
npm install
npm run dev          # http://localhost:5173 , proxy → :8080
npm test             # vitest, happy-dom
npm run build
npm run test:ci      # vitest + dual production builds + bundle check
```

There is no separate TypeScript / ESLint script in this package
(JavaScript + Vite).

## 11. Connect the frontend to native Go on :8083

```bash
# terminal 1 — Go with Speech + tunnel already on 18090
cd apps/api
SPEECH_ENABLED=true \
SPEECH_SERVICE_URL=http://127.0.0.1:18090 \
SPEECH_SERVICE_TOKEN='same-as-python' \
API_ADDR=:8083 \
APP_ENV=local go run ./cmd/api

# terminal 2
cd apps/web
VITE_API_PROXY=http://127.0.0.1:8083 npm run dev
```

The browser still calls relative `/api/...`. Restart Vite after changing
`VITE_API_PROXY`. Apply migration `000021` on the database that process
uses.

## 12. Automated tests

```bash
cd apps/web
npx vitest run
```

Coverage (mocked `fetch` / `getUserMedia` / `AudioContext`):

| File | What it proves |
|---|---|
| `src/lib/wav.test.js` | RIFF/WAVE, PCM16 mono 16 kHz, resample, reject bad files |
| `src/lib/micRecorder.test.js` | start/encode once, cancel, permission, unsupported |
| `src/api/speech.test.js` | path, guest/JWT, multipart `audio`, envelope, HTTP errors |
| `src/components/BahnarSpeechPanel.test.jsx` | no request while recording, cancel=0, stop once, confirm |
| `src/pages/CitizenChatPage.speech.test.jsx` | typed chat still works; confirm → one turn; failed turn |

Frontend ceiling (matches Go/C0 **defaults**; deployments may lower
`SPEECH_MAX_UPLOAD_BYTES`):

- `SPEECH_MAX_UPLOAD_BYTES = 10 MiB`
- `SPEECH_MAX_AUDIO_SECONDS = 30`
- `SPEECH_MIN_AUDIO_SECONDS = 0.2`
- Timeout 130 s

## 13. Manual browser checks (when Go + Speech are up)

1. Typed chat still sends a turn; conversation is kept when opening
   Micro Bahnar / Tải file WAV.
2. Upload a real 16 kHz Bahnar WAV → Dịch âm thanh → confirm Việt.
3. Micro Bahnar → allow mic → Bắt đầu ghi âm → Dừng & Dịch → confirm.
4. Hủy ghi âm: Network tab has no `/speech/translate`.
5. Edit Việt before Xác nhận; Chat Turn JSON `message` is the edited text.
6. 🎙 Vietnamese dictation still fills the composer only.
7. Logged-out guest: `X-Guest-Token` on both speech and turns.
8. Logged-in: Bearer on both. Same `sessionId`.
9. Deny microphone: Vietnamese error, no request.
10. Oversized / MP3 / empty file: rejected before fetch.
11. Record until the 30 s cap → auto-stop → **Dịch âm thanh** (exactly one
    `/speech/translate`; no Chat Turn yet).
12. Stop the Speech API or unplug the network → error copy → **Thử dịch
    lại** resubmits the same WAV.
13. **Hủy yêu cầu dịch** then **Dịch âm thanh** / **Thử dịch lại**: one
    new request; a late response from the cancelled call must not appear.

## 14. RunPod-off development limitations

Automated tests mock Go. They do **not** prove:

- a physical microphone or Safari/Chrome permission UX
- live XLS-R / BARTpho inference
- tokenizer parity or research provenance
- production Docker-to-Speech networking (Step 7.5 M3)

When RunPod is stopped, the UI must show the Go error
(`SPEECH_UNAVAILABLE` / timeout), never a fake transcript. If the
current WAV is still valid, **Thử dịch lại** resubmits that same file
without recording again. Invalid / oversized WAV errors do not offer
Retry; the citizen must choose another file. **Hủy yêu cầu dịch** only
aborts the browser HTTP request.

## 15. Live RunPod E2E still required

When the pod and tunnel are up:

1. Point Vite at native Go `:8083` as above.
2. Upload a known Bahnar WAV (the 7.5 fixture if available).
3. Confirm HTTP 200, `model_id=c0_cascaded`, nonempty ASR + Việt.
4. Confirm the Chat Turn uses the edited Vietnamese text.
5. Confirm PostgreSQL `speech_translation_requests` audit `SUCCESS`.
6. Repeat with microphone capture of spoken Bahnar.

Until that run happens:
`STEP76_BROWSER_MICROPHONE_VERIFIED=NOT_VERIFIED`
`STEP76_REAL_RUNPOD_E2E_VERIFIED=NOT_VERIFIED`

## 16. Known limitations / deferred

- Production Docker API → Speech service networking remains a 7.5 M3
  go-live blocker. Step 7.6 does not change it.
- `ScriptProcessor` is deprecated; it is used for broad-browser capture
  without extra dependencies. AudioWorklet can replace it later.
- Frontend WAV ceiling is `SPEECH_MAX_WAV_BYTES = 10 MiB − 64 KiB`
  (`MaxAudioBytes` default). Go `SPEECH_MAX_UPLOAD_BYTES` is the whole
  body; deployments may lower it. Go remains the authority.
- No streaming ASR. No TTS. No change to research provenance
  (`NOT_VERIFIED`).
- Browser hardware verification was not executed in this implementation
  pass.
- **Hủy yêu cầu dịch** aborts the browser HTTP request. It does not stop
  inference already running on RunPod unless the Go/Speech APIs cancel
  that work (they currently do not expose client-driven GPU cancel).

## Frontend policy constants

`apps/web/src/config/speech.js` — 16000 Hz, 1 channel, 16-bit, field
`audio`, model `c0_cascaded`, `SPEECH_MAX_WAV_BYTES = 10420224`.

## 17. Review remediation (HIGH-01 … MEDIUM-04)

| ID | Root cause | Fix | Tests |
|---|---|---|---|
| HIGH-01 | `parseWav` trusted header `dataSize` | RIFF bounds, chunk bounds, physical payload, blockAlign/byteRate | Truncated 32044→100 now rejected; extra LIST; overflow |
| HIGH-02 | `beginRecord` ignored `opRef` after `getUserMedia` | `signal` + `stillValid(op)` + immediate `cap.cancel()`; no setState after unmount | pending → unmount/cancel/mode switch → grant stops tracks |
| HIGH-03 | `createContext()` outside cleanup | Entire post-`getUserMedia` path in try/cleanup | AudioContext / resume / node / connect failures stop tracks |
| MEDIUM-01 | `loadFile` had no generation | `fileGenRef`; latest selection wins | A→B, A finishes last; Gỡ while reading |
| MEDIUM-02 | `arrayBuffer()` before `file.size` | `assertWavUploadSize` first | oversized/empty never call `arrayBuffer` |
| MEDIUM-03 | boxcar / linear resample | Hamming windowed-sinc, cutoff 0.9×Nyquist dest | 12 kHz RMS ~0.0008 (was 0.24/0.56) |
| MEDIUM-04 | no cancel while `translating` | **Hủy yêu cầu dịch** aborts AbortController | late response ignored; retry allowed |
| HIGH-04 | abort listener after `resume()` | abort handler registered as soon as MediaStream exists; `stopTracks` is synchronous | abort while `resume()` pending → stop before Promise settles |
| HIGH-05 | `pendingRef` kept A while reading B | clear pending + `reading_file`; `translateNow` checks gen | A ready → select B (unread) → Dịch sends nothing |
| MEDIUM-05 | dangling RIFF bytes / missing pad | `offset === riffEnd`; odd chunks require pad | 1–7 dangling bytes and unpadded LIST rejected |
| LOW-01 | §5 described boxcar/linear | §5 matches windowed-sinc 65 taps | docs vs `resampleTo` |
| HIGH-06 | auto-stop / mic pending lacked `gen` | `makeSpeechPending` stamps `name`/`source`/`gen` for mic and upload | auto-stop → Dịch = 1 request; cancel/retry; replace recording A→B |
| MEDIUM-06 | `translation_error` hid Translate | **Thử dịch lại** when pending gen is current and the error is retryable | 500 / timeout / network retry; no Retry for invalid WAV |
| MEDIUM-07 | `cancelRecord` reset state after `await` without `stillValid` | capture `op` after `bumpOp`; ignore resume if a newer op started | slow `cap.cancel` → switch to Upload → file kept |
| LOW-02 | Retry offered for 401/403/404 | `isRetryableSpeechError` false for auth/session codes including 409 | 401 hides **Thử dịch lại** |
| MEDIUM-08 | auto-stop timer / ScriptProcessor can exceed 30.05s | clamp capture to 30s before encode; drop extra `onaudioprocess` samples | 30.06s and late timer still produce a 30s WAV; upload still rejects long files |

Resampler: 65-tap Hamming windowed sinc in `resampleTo`. No extra npm
dependency. Deterministic in Node/Vitest. Typical capture 44.1/48 kHz.

Measured 1 s sine, amplitude 0.5 (expected passband RMS ≈ 0.3536):

| Input | f | Output RMS | Threshold |
|---|---|---|---|
| 48 kHz | 1 kHz | 0.354 | 0.28–0.42 passband |
| 44.1 kHz | 1 kHz | 0.353 | 0.28–0.42 |
| 48 kHz | 3 kHz | 0.354 | passband |
| 48 kHz | 12 kHz | 0.00075 | stopband < 0.05 |
| 44.1 kHz | 12 kHz | 0.00080 | stopband < 0.05 |

Duration error: 0 samples on 1 s fixtures.

## 18. Pending-audio contract (HIGH-06 / MEDIUM-06)

Microphone encode and WAV upload both store:

`{ bytes, durationMs, sampleCount, name, source, gen }`

`gen` is `fileGenRef` at the moment the input is accepted. Translate
runs only when `pending.gen === fileGenRef.current`. Starting a new
recording, replacing a file, switching mode, or Gỡ file increments
`gen` and drops the old buffer.

Auto-stop at 30 s does **not** call Go. The citizen clicks **Dịch âm
thanh**. After HTTP 5xx / timeout / network errors, **Thử dịch lại**
resubmits the same current WAV. Validation / 413 / 415 and auth/session
errors (401 / 403 / 404 / 409) do not show Retry. Cancelled requests
return to the ready state; a late HTTP response cannot overwrite a
newer request. **Hủy ghi âm** does not reset a newer Upload/mic
operation after `cap.cancel()` resolves. Chat Turn still requires
**Xác nhận & Hỏi trợ lý**.

Microphone capture clamps to `SPEECH_MAX_AUDIO_SECONDS` (30) both while
buffering ScriptProcessor blocks and again after resampling, so a late
`setTimeout` or one extra 4096-sample callback cannot fail encode.
Selected Upload files longer than 30.05s are still rejected.

## 19. Silence vs Chat Turn linkage (Step 7.6 close)

### Finding 1 — silent microphone must not be HTTP 502

Trace of request `5b4f9936-9a4c-431e-b414-507f2c1cfc0f` from the live
E2E: valid PCM16 16 kHz WAV with no recognised speech → C0
`empty_asr` → FastAPI used to return **502** → Go `mapStatus` collapsed
that to `ErrInferenceFailed` → public **502 SPEECH_UNAVAILABLE**.
Frontend then showed a server/RunPod outage. The WAV was valid; ASR
finished; MT never ran.

Fix:

- FastAPI maps `empty_asr` to **HTTP 422**, not 502. `empty_mt` stays 502
  (ASR succeeded, translation failed).
- Go maps upstream code `empty_asr` to `ErrNoSpeech` **regardless of
  502 or 422**, so a live Speech API image that still returns 502
  `empty_asr` is already citizen-facing 422 after Go restart.
- Public envelope: **422 SPEECH_NO_SPEECH**.
- Frontend copy asks the citizen to record again. Same WAV is not
  retried (`isRetryableSpeechError` is false).

No RMS/VAD gate was added. Quiet Bahnar can sit near the noise floor;
an energy threshold would drop it. Empty ASR after full C0 inference
is the source of truth. Frontend still accepts silent and near-silent
WAV as a valid container.

### Finding 2 — empty transcripts are privacy; linkage was missing

`transcript_bahnar` / `output_vi` / `audio_uri` stay NULL on new Go
writes. That is **privacy-by-design** (migration 000021 + `Store.Record`
comment), not a frontend bug. The citizen still sees ASR + Việt in the
browser; those strings are not copied into the audit row.

`chat_turn_request_id` existed on the table and was never written.
Confirm now sends optional `speech_request_id` on
`POST /api/v1/sessions/:sessionId/turns`. Go updates the SUCCESS speech
row for the **same session** and that exact `request_id`. No timestamp
join, no backfill, no overwrite, no transcript write. Typed chat omits
the field. A pending Chat Turn retry reuses the first
`speech_request_id` for that session+message; swapping Speech IDs on
retry does not rebind the turn. Chat Turn idempotency stays bound to
the normalized message (typed chat unchanged).

User-visible Chat E2E PASS (message appears in `conversation_messages`)
is **not** the same as database-level speech↔turn linkage. Linkage
needs a Go restart + a new confirm after this change; historical NULL
rows are left alone.
