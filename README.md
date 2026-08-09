# citizen-assistance-system

AI-powered Citizen Assistance System for commune-level administrative procedures.

## Monorepo layout

```text
citizen-assistance-system/
├── apps/
│   ├── api/           # Go + Gin — auth, sessions, Decision Engine, admin
│   ├── ai-service/    # Python FastAPI — extract, RAG, embeddings, LLM
│   └── web/           # React + JS — Citizen Portal + Admin Portal
├── packages/
│   └── contracts/     # OpenAPI / shared schemas (optional)
├── deploy/            # Docker Compose, nginx, env examples
├── docs/              # Phase-0 design + ADRs
└── .github/workflows/ # CI/CD
```

## Stack (locked)

| Layer | Tech |
|-------|------|
| Frontend | React + JavaScript |
| Backend | Go (Gin) |
| AI | Python (FastAPI) |
| DB | PostgreSQL + pgvector |
| Files | MinIO (S3-compatible) |
| Deploy | Docker Compose + reverse proxy |

## Docs

- **A→Z plan (code → prod):** [`docs/IMPLEMENTATION-PLAN-A-TO-Z.md`](docs/IMPLEMENTATION-PLAN-A-TO-Z.md)
- Phase 0: [`docs/phase-0/00-README.md`](docs/phase-0/00-README.md)
- Tech lock: [`docs/phase-0/05-TECH-LOCK.md`](docs/phase-0/05-TECH-LOCK.md)
- **Local DB:** [`deploy/README.md`](deploy/README.md) — `make db-migrate`

## Status

- [x] Monorepo scaffold
- [x] Local Postgres migrations (schema + master seed)
- [x] Go API skeleton (Gin, config, DB pool, logger, request_id, health/ready)
- [ ] Phase 1: seed procedures + Decision Engine + chat turns
- [ ] Phase 2: AI extract (mock → LLM)
- [ ] Phase 3: React chat UI
- [ ] … → Production (see A→Z plan)
