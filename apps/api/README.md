# Go Backend API (Gin)

## Run (local)

```bash
make db-up && make db-migrate
make api-run
```

Swagger UI: http://localhost:8080/swagger/index.html

Sau khi thêm/sửa API annotation:

```bash
make api-swagger   # regenerates apps/api/docs/
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (DB ping) |
| GET | `/swagger/*` | OpenAPI UI |
| POST | `/api/v1/auth/register` | Citizen register → JWT |
| POST | `/api/v1/auth/login` | Login → JWT |
| GET | `/api/v1/auth/me` | Current user (Bearer) |
| POST | `/api/v1/auth/logout` | Client discard token (Bearer) |
| GET | `/api/v1/communes` | List communes (`?active=true`) |
| GET | `/api/v1/communes/:id` | Get commune |
| GET | `/api/v1/domains` | List domains (`?active=true`) |
| GET | `/api/v1/domains/:id` | Get domain |
| POST | `/api/v1/domains` | Create domain (**ADMIN** JWT) |
| PUT | `/api/v1/domains/:id` | Update domain (**ADMIN** JWT) |
| DELETE | `/api/v1/domains/:id` | Soft-delete; `?hard=true` (**ADMIN** JWT) |
| GET | `/api/v1/procedures` | List ACTIVE procedures grouped by domain |
| GET | `/api/v1/procedures/by-code/:code` | Resolve by `(xa_id, procedure_code)` |
| GET | `/api/v1/procedures/:id/active-version` | Active version + definition JSON |
| POST | `/api/v1/sessions` | Create session (guest or Bearer JWT) |
| GET | `/api/v1/sessions/:sessionId/messages` | List messages |
| POST | `/api/v1/sessions/:sessionId/messages` | **410 Gone** — deprecated; use `POST .../turns` |
| POST | `/api/v1/sessions/:sessionId/turns` | One chat turn: detect procedure, slot state, decision |

### Auth users

Migrations do **not** ship default passwords. Create accounts explicitly:

- **Local demo** (`APP_ENV=local`): run `go run ./cmd/seed-demo` (optional `CAS_DEMO_ADMIN_PASSWORD`, `CAS_DEMO_CITIZEN_PASSWORD`; override guard with `CAS_ALLOW_DEMO_SEED=1`).
- **Production admin**: run `go run ./cmd/bootstrap-admin` with `BOOTSTRAP_ADMIN_EMAIL` and `BOOTSTRAP_ADMIN_PASSWORD` (≥ 12 chars, not demo passwords).

Header: `Authorization: Bearer <access_token>`. Prod: set `JWT_SECRET` (min 16 chars).

Guest chat: `X-Guest-Token`. Logged-in: Bearer → session `user_id` set. `POST /sessions/:id/turns` requires `X-Request-ID` (UUID) and runs the Decision Engine on the pinned ACTIVE definition.

## P2 — LLM Extract (optional, off by default)

When enabled, a turn's intent/slot detection is assisted by
`apps/ai-service` (`POST /v1/extract`) instead of relying only on the
keyword engine (`internal/decision`). The keyword engine remains the
fallback on any AI failure/timeout/invalid response, and it is the **only**
path used when this feature is disabled — behavior and all existing tests
are unchanged in that case.

Env vars (`internal/config/config.go`, all optional, safe defaults):

| Var | Default | Purpose |
|-----|---------|---------|
| `AI_EXTRACT_ENABLED` | `false` | Master switch. `false`/unset ⇒ pure keyword path, zero network calls |
| `AI_SERVICE_URL` | `""` | Base URL of `apps/ai-service`, e.g. `http://ai-service:8001` (Compose does not publish this port) |
| `AI_SERVICE_TOKEN` | `""` | Required when extraction is enabled. Sent as `Authorization: Bearer`. Never logged. Startup fails if enabled and this is empty |
| `AI_EXTRACT_TIMEOUT_MS` | `3000` | Per-call timeout, hard-capped at 8000ms regardless of this value |
| `AI_INTENT_SELECT_MIN` | `0.82` | AI confidence ≥ this ⇒ select immediately (same threshold family as the keyword matcher, see `internal/decision/policy.go`) |
| `AI_INTENT_CONFIRM_MIN` | `0.55` | AI confidence in `[ConfirmMin, SelectMin)` ⇒ ask the citizen to confirm |
| `AI_SLOT_CONFIDENCE_MIN` | `0.6` | Per-slot confidence floor; slots below this are silently dropped, not treated as a contract violation |

Design guarantees (see `internal/chat/ai_plan.go`, `internal/aiextract/`):

- **No network call ever happens while holding a database transaction/row
  lock.** A turn runs in three phases: a short read-only transaction that
  snapshots state and computes a fingerprint, the AI HTTP call strictly
  outside any transaction, then a write transaction that re-validates the
  fingerprint and falls back to keyword-only handling on any mismatch,
  timeout, or invalid response.
- Every AI response field is independently re-validated on the Go side
  (`aiextract.Validate`) — unknown procedure/slot keys, wrong types, enum
  values outside `enum_values`, non-finite numbers, and invalid
  `"correct"` operations are all rejected fail-closed (the entire AI
  response is discarded, not partially applied).
- The AI can only pre-fill or correct slot **values**. It never chooses the
  resulting action, never invents a procedure or a slot outside the
  definition, never edits guidance/checklist/citations, and never silently
  switches an already-pinned procedure — a differing AI-proposed procedure
  always goes through the existing `CONFIRM_INTENT` flow.
- A repeated `X-Request-ID` with the same payload calls the extractor at most
  once, including two in-flight requests. HTTP 429 is not retried; the turn
  falls back to keyword matching.
- Non-PII observability fields (`extract_source`, `extract_provider`,
  `extract_model`, `extract_latency_ms`, `extract_fallback_reason`,
  `accepted_slot_keys`, `rejected_slot_keys`) are attached to the assistant
  message's metadata — never the citizen message or a raw provider payload.

Keyword-only Compose (AI stays off; `ai-service` is not started and the API
does not wait for it):

```bash
cd deploy && docker compose --profile app up
```

AI-enabled Compose (mock provider, no LLM key). The API waits for
`ai-service` only because the `ai` profile is selected:

```bash
cd deploy && AI_EXTRACT_ENABLED=true docker compose --profile app --profile ai up
```

Local processes, same token on both sides:

```bash
cd apps/ai-service && AI_SERVICE_TOKEN=local-dev-ai-service-token uvicorn app.main:app --port 8001 &
cd apps/api && AI_EXTRACT_ENABLED=true AI_SERVICE_URL=http://127.0.0.1:8001 \
  AI_SERVICE_TOKEN=local-dev-ai-service-token make api-run
```

`AI_INTENT_*` and `AI_SLOT_CONFIDENCE_MIN` must be finite. `NaN` and `Inf`
fail startup and are not echoed. `AI_SERVICE_TOKEN` must be at least 16
characters whenever it is set.

## P3.1 admin PDF intake

Off unless `ADMIN_INGESTION_ENABLED=true`. Then the API requires object
storage settings and refuses to start if the private bucket cannot be
created. Keyword-only Compose does not start MinIO.

```bash
cd deploy && ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true \
  docker compose --profile app --profile storage up --build
```

`VITE_ADMIN_INGESTION` is a web build arg. Leaving it false keeps `/admin/documents` on the placeholder page even when the API routes are mounted.

P4A indexing stays off unless ingestion is on. Status is stored on each document–version link. The worker is mock `POST /v1/index`: it does not OCR, embed, write `knowledge_chunks`, or activate. One local command:

```bash
cd deploy && ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true ADMIN_INDEXING_ENABLED=true VITE_ADMIN_INDEXING=true \
  docker compose --profile app --profile storage --profile ai up --build
```

P4B keeps that mock path when `INDEX_MODE=mock`. The content pipeline, Qdrant, and generation publish are documented in `docs/p4b-runbook.md`.

```bash
cd deploy && ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true ADMIN_INDEXING_ENABLED=true VITE_ADMIN_INDEXING=true \
  INDEX_MODE=pipeline EMBEDDING_PROVIDER=fake OCR_PROVIDER=fake \
  docker compose --profile app --profile storage --profile ai --profile qdrant up --build
```

Admin JWT only:

- `POST /api/v1/admin/documents` multipart PDF (`file`, `title`, `domain_id`)
- `GET /api/v1/admin/documents` filters `domain_id`, `processing_status`, `validity_status`
- `GET /api/v1/admin/documents/:id`
- `GET /api/v1/admin/documents/:id/content` streams the PDF

`xa_id` comes from server config. A duplicate checksum in the same commune
is `409 DOCUMENT_DUPLICATE`. OCR, draft, and RAG are not part of this API.

## Logging

JSON stdout logs. `/health`, `/ready`, `/swagger` → debug when OK.

## Config

- Local: `configs/local/config.yaml`
- Prod: `APP_ENV=prod` + `DATABASE_URL` + `JWT_SECRET`
