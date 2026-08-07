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

- Phase 0: [`docs/phase-0/00-README.md`](docs/phase-0/00-README.md)
- Tech lock: [`docs/phase-0/05-TECH-LOCK.md`](docs/phase-0/05-TECH-LOCK.md)

## Status

- [x] Monorepo scaffold
- [ ] Phase 1: Go chat + Decision Engine
- [ ] Phase 1: AI extract (mock → LLM)
- [ ] Phase 1: React chat UI
