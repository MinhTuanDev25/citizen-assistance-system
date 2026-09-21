# citizen-assistance-system

Monorepo for two tracks that share this repository:

1. **Bahnar → Vietnamese speech-to-text translation (thesis)** — RQ1 cascaded vs direct S2TT, under [`bahnar-s2tt-thesis/`](bahnar-s2tt-thesis/).
2. **Citizen assistance chatbot (application)** — commune-level administrative procedures, under `apps/`.

The thesis is the research contribution. The chatbot is a separate demo/application layer and is not part of RQ1.

## Layout

```text
citizen-assistance-system/
├── bahnar-s2tt-thesis/   # RQ1 notebooks, src, configs, tests
├── apps/
│   ├── api/              # Go + Gin — auth, sessions, Decision Engine, admin
│   ├── ai-service/       # Python FastAPI — extract, RAG, embeddings, LLM
│   └── web/              # React + JS — Citizen Portal + Admin Portal
├── packages/contracts/   # OpenAPI / shared schemas
├── deploy/               # Docker Compose, nginx, env examples
├── docs/                 # Phase-0 design + ADRs (chatbot)
└── .github/workflows/
```

## Thesis (RQ1)

Compare a **cascaded ASR + MT** system (C0) with a **direct S2TT** system (D0) on `cuong06/Bahnar_Vietnamese`.

| Notebook | Role | Status |
|----------|------|--------|
| 01 | Audit + locked RQ1 splits | Implemented |
| 02 | ASR preflight / smoke | Implemented |
| 03 | Cascaded ASR (Bahnar speech → Bahnar text) | Implemented |
| 04 | Cascaded MT (Bahnar text → Vietnamese) | Implemented |
| 05 | Direct S2TT (Bahnar speech → Vietnamese) | Implemented |
| 06 | Frozen-test C0 vs D0 evaluation | Placeholder |

Start in [`bahnar-s2tt-thesis/README.md`](bahnar-s2tt-thesis/README.md). Do not jump to full train: `prepare → pilot → resume_test_a → restart kernel → resume_test_b → train → evaluate`.

## Chatbot (application)

| Layer | Tech |
|-------|------|
| Frontend | React + JavaScript |
| Backend | Go (Gin) |
| AI | Python (FastAPI) |
| DB | PostgreSQL + pgvector |
| Files | MinIO (S3-compatible) |
| Deploy | Docker Compose + reverse proxy |

- A→Z plan: [`docs/IMPLEMENTATION-PLAN-A-TO-Z.md`](docs/IMPLEMENTATION-PLAN-A-TO-Z.md)
- Phase 0: [`docs/phase-0/00-README.md`](docs/phase-0/00-README.md)
- Local DB: [`deploy/README.md`](deploy/README.md) — `make db-migrate`

## Status

**Thesis**

- [x] Notebooks 01–04 (split, ASR preflight, ASR, MT)
- [x] Notebook 05 Direct S2TT D0 (code + contracts; RunPod prepare still needs NB03 CSV pins)
- [ ] Notebook 06 frozen-test C0 vs D0

**Chatbot**

- [x] Monorepo scaffold
- [x] Local Postgres migrations (schema + master seed)
- [x] Go API skeleton
- [x] Citizen Web UI (Vite React — chat demo)
- [ ] Phase 1: seed procedures + Decision Engine + chat turns
- [ ] … → Production (see A→Z plan)
