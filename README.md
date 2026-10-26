# citizen-assistance-system

Monorepo for two tracks:

1. **Citizen Assistance System (Software V1)**: commune-level administrative
   procedure assistant for Chư Sê. Vietnamese chat with a Go Decision Engine,
   admin PDF ingestion and indexing, AI-drafted procedure definitions with human
   review, RAG citations, and Bahnar speech → Vietnamese via the thesis C0 model.
2. **Bahnar → Vietnamese speech translation thesis (RQ1/RQ2)** under
   [`bahnar-s2tt-thesis/`](bahnar-s2tt-thesis/README.md). The application only
   uses its `src/c0_speech` inference API.

## Layout

```text
apps/
  api/          Go (Gin): auth, sessions, Chat Turn, Decision Engine, admin, speech route
  ai-service/   Python (FastAPI): extract, draft definition, index pipeline, retrieve
  web/          React (Vite): Citizen Portal + Admin Portal
bahnar-s2tt-thesis/   research code, configs, tests; C0 Speech API (src/c0_speech)
deploy/
  docker-compose.yml             local stack (main workflow)
  docker-compose.p4b-test.yml    isolated stack for CI storage tests
  docker-compose.production.yml  VPS reference (CI-validated, Speech off)
  migrations/                    golang-migrate SQL (000001–000022)
  scripts/                       up.sh, migrate.sh
  speech-tunnel/                 SSH tunnel sidecar to the RunPod Speech API
docs/           STATUS, RUNPOD, SPEECH, p4b-runbook, phase-0 design + ADRs, archive/
scripts/        CI/ops helpers; runpod/ for scripts run on the GPU pod
```

## Run locally

Requires Docker. Configuration lives in `deploy/.env` (copy from
`deploy/.env.example`). Speech additionally needs `deploy/.env.speech.local`
(copy from `deploy/.env.speech.example`); without it everything else runs and
Speech is off.

```bash
make up      # build + start: postgres, migrate, api, web, ai-service, minio, qdrant, speech-tunnel
make logs
make down    # stops containers, keeps data volumes
```

Open the web URL printed by `make up` (`http://127.0.0.1:5174` with the current
`deploy/.env`). Admin screens need an ADMIN account.

Tests: `go test ./...` in `apps/api`, `npm run test:ci` in `apps/web`,
`pytest` in `apps/ai-service`. CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Documentation

| Topic | File |
|---|---|
| What is done, evidence level, what remains | [`docs/STATUS.md`](docs/STATUS.md) |
| Bahnar Speech citizen flow | [`docs/SPEECH.md`](docs/SPEECH.md) |
| RunPod: Speech API, tunnel, go-live plan | [`docs/RUNPOD.md`](docs/RUNPOD.md) |
| PDF indexing models and pipeline | [`docs/p4b-runbook.md`](docs/p4b-runbook.md) |
| Local database, Compose details | [`deploy/README.md`](deploy/README.md) |
| VPS production reference | [`deploy/README.production.md`](deploy/README.production.md) |
| Original design, ADRs, seed procedures | [`docs/phase-0/00-README.md`](docs/phase-0/00-README.md) |
| Historical acceptance reports | [`docs/archive/`](docs/archive/) |
| Per-app details | `apps/api/README.md`, `apps/ai-service/README.md`, `apps/web/README.md` |
