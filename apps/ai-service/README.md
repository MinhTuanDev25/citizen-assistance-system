# Python AI Service (FastAPI) — P2 LLM Extract

Health/readiness/status plus `POST /v1/extract`: structured intent + slot
extraction for the Go API's Decision Engine. Does **not** implement PDF
upload, embeddings, RAG retrieval, or speech (still later phases).
`POST /v1/index` is the P4A mock contract (`index.v1`): identifiers in, `READY` or `FAILED` out. It does not read a PDF or write embeddings.

## Run

```bash
cd apps/ai-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
AI_SERVICE_TOKEN=local-dev-ai-service-token uvicorn app.main:app --host 0.0.0.0 --port 8001
```

No LLM API key is required for local dev or tests — the default
`LLM_PROVIDER=mock` is a deterministic, network-free provider.
`AI_SERVICE_TOKEN` is still required (16–256 characters). The process
refuses to start if it is missing, too short, or if `LLM_MODEL` does not
match the response model-id pattern. Error text does not echo the token or
the rejected model.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness — 503 if a configured real provider (`openai`/`gemini`) is missing its API key/model; never silently falls back to mock |
| GET | `/v1/status` | Non-secret service metadata: provider, model, ready |
| POST | `/v1/extract` | Strict-contract extraction. Requires `Authorization: Bearer $AI_SERVICE_TOKEN`. After auth, a request-rate and in-flight cap apply; over limit is 429 with `Retry-After` and the provider is not called. Unknown fields and oversized nested strings are 422; a body over 256 KiB is 413 |
| GET | `/docs` | FastAPI-generated interactive OpenAPI (Swagger UI) |
| GET | `/openapi.json` | FastAPI-generated OpenAPI schema |

## Configuration (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_PROVIDER` | `mock` | `mock` \| `openai` \| `gemini` |
| `LLM_MODEL` | provider default (`mock-extract-v1`, `gpt-4o-mini`, or `gemini-1.5-flash`) | Validated at startup with the same pattern as response `model`: a leading letter, then letters, digits, `.`, `_`, `:`, `/`, `-`, at most 80 characters |
| `AI_SERVICE_TOKEN` | — | Required. 16–256 characters. Never logged |
| `OPENAI_API_KEY` | — | Required only when `LLM_PROVIDER=openai` |
| `GEMINI_API_KEY` | — | Required only when `LLM_PROVIDER=gemini` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Override for self-hosted/proxy |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | Override |
| `LLM_REQUEST_TIMEOUT_S` | `8.0` | Finite timeout in `(0, 8]` |
| `EXTRACT_RATE_PER_MINUTE` | `60` | Authenticated extract requests per 60s. Excess is 429 |
| `EXTRACT_MAX_INFLIGHT` | `4` | Concurrent provider calls. Excess is 429 |
| `EXTRACT_RETRY_AFTER_SECONDS` | `1` | `Retry-After` value on 429 |

Payload ceilings (`message`, examples, questions, enum values, slot keys,
slot state) are the constants in `app/models/extract.py`. They are not
environment variables.

## `POST /v1/extract` contract (summary)

Request: `schema_version`, `request_id` (UUID), `message` (≤4000 code
points), `candidates[]` (`procedure_code`, `name`, `intent_examples`,
`slots{type,question,enum_values}`), optional `pinned_context`
(`procedure_code`, `allowed_slot_keys`, `slot_state`). Every model uses
Pydantic `extra="forbid"` — unknown fields are rejected with HTTP 422, not
silently dropped.

Response: `schema_version`, `intent{procedure_code, confidence, alternatives}`,
`slots_for_procedure_code`, `slots[]{key, value, confidence, evidence,
operation}`, `abstain_reason`, `provider`, `model`.

The model may only ever pick a `procedure_code` from `candidates` and only
fill slot keys declared on that candidate — it can never invent a procedure
or a slot. The **Go API independently re-validates every field** (see
`apps/api/internal/aiextract/validate.go`) before trusting any of it —
this service is not a trusted boundary on its own.

## Mock provider behavior (no network, used by default and in every test)

- `"ba"` → resolves the person-who-registers enum slot to `"cha"` (colloquial
  Southern Vietnamese for "father").
- `"tôi"` alone is **never** guessed into any slot (string, boolean, or enum).
- `"chưa đăng ký kết hôn"` → boolean slot resolves to `false`.
- Typo'd input (e.g. `"à quên chưa đưng kí kéth ôn"`) still resolves to
  `false` via typo-tolerant keyword matching.
- A message unrelated to any candidate → `procedure_code: null`, no slots,
  `abstain_reason` set.
- A slot key outside `allowed_slot_keys` (when a procedure is pinned) is
  never returned.

## Tests

```bash
pytest
```

No test ever makes a live network call: the mock provider is fully
deterministic, and the OpenAI/Gemini adapters are only exercised with an
injected `httpx.MockTransport` fake in `tests/test_real_providers.py`.

## Logging

Logs never include the citizen `message`, a raw provider request/response
body, or any API key — only request id, candidate/slot counts, and outcome
metadata.
