# citizen-assistance-system

Monorepo for two tracks that share this repository:

1. **Bahnar → Vietnamese speech-to-text translation (thesis)** — RQ1 cascaded vs direct S2TT, under [`bahnar-s2tt-thesis/`](bahnar-s2tt-thesis/).
2. **Citizen assistance chatbot (application)** — commune-level administrative procedures, under `apps/`.

The thesis is the research contribution (RQ1/RQ2). The chatbot is the **application artifact** in proposal §4: hộ tịch, slot workflow, RAG citations, Bahnar speech → Vietnamese text. It does not replace the speech-translation experiments.

## Layout

```text
citizen-assistance-system/
├── bahnar-s2tt-thesis/   # RQ1 notebooks, src, configs, tests
├── apps/
│   ├── api/              # Go + Gin — auth, sessions, Decision Engine, admin
│   ├── ai-service/       # Python — extract, mock index (P4A). Embed and speech later
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

- [x] Monorepo + Postgres/pgvector + MinIO compose
- [x] Go chat turns + Decision Engine (seed hộ tịch; keyword extract = fallback)
- [x] Citizen web (text)
- [x] P2 `POST /v1/extract` LLM (optional; keyword fallback when `AI_EXTRACT_ENABLED` is off)
- [x] P3.1 admin PDF upload, list, detail, and download (`ADMIN_INGESTION_ENABLED` + `VITE_ADMIN_INGESTION`, both default off)
- [x] P4A indexing foundation: per-link status, mock index, retry. No embed, RAG, or activate. Run: `cd deploy && ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true ADMIN_INDEXING_ENABLED=true VITE_ADMIN_INDEXING=true docker compose --profile app --profile storage --profile ai up --build`
- [x] P4B content pipeline: native text, OCR adapter, chunks, Qdrant. No retrieval or activate. Run: `cd deploy && ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true ADMIN_INDEXING_ENABLED=true VITE_ADMIN_INDEXING=true INDEX_MODE=pipeline EMBEDDING_PROVIDER=fake OCR_PROVIDER=fake INDEX_PIPELINE_TIMEOUT_SECONDS=120 INDEX_CLAIM_TTL_SECONDS=180 docker compose --profile app --profile storage --profile ai --profile qdrant up --build`
- [ ] Admin embed → activate
- [ ] RAG citations + Bahnar speech in Python
- [ ] Docs Phase 0 aligned to proposal (2026-09-28)
