# Implementation Plan A→Z → Production

**Repo:** `citizen-assistance-system`  
**Goal:** Từ scaffold hiện tại đến hệ thống chạy trên server thật (1 xã), có DB, storage, CI/CD, monitoring.

**Stack đã chốt:** React+JS · Go Gin · Python FastAPI · PostgreSQL+pgvector · MinIO · Docker Compose · Nginx · GitHub Actions

---

## Bản đồ tổng thể

```text
P0 Scaffold ✅
  → P1 Core chat (Go + seed + Decision Engine)
  → P2 AI service (extract + mock/LLM)
  → P3 Web UI (Citizen)
  → P4 Admin pipeline (upload → review → publish → embed)
  → P5 RAG + citations
  → P6 Hardening (auth, audit, security)
  → P7 Infra local (Compose full stack)
  → P8 CI/CD
  → P9 Staging server
  → P10 UAT + metrics
  → P11 Production cutover
  → P12 Operate (backup, monitor, runbook)
```

---

## Phase 0 — Done (baseline)

- [x] Monorepo `apps/{api,ai-service,web}`, `deploy/`, `docs/phase-0/`
- [x] Seeds + schema + decision contract
- [x] Compose skeleton (Postgres+pgvector, MinIO)

**Exit:** repo trên `main`, docs Phase 0 trong repo.

---

## Phase 1 — Go Backend core (chat deterministic)

**Mục tiêu:** `POST /api/v1/chat/turns` chạy với seed JSON, không cần LLM thật.

| # | Task | Output |
|---|------|--------|
| 1.1 | `go mod` + Gin skeleton (`cmd/api`) | Health `GET /health` |
| 1.2 | Config từ env (`DATABASE_URL`, `XA_ID`, …) | `internal/config` |
| 1.3 | Migrate Postgres (users, sessions, messages, slot_state, domains, procedures, versions, audit) | SQL/goose/migrate |
| 1.4 | Seed loader từ `docs/phase-0/seeds/*.json` | Active procedures trong DB |
| 1.5 | Intent mock (keyword) trong Go *hoặc* gọi AI stub | Detect `dk_khai_sinh`… |
| 1.6 | Slot state + Decision Engine (`direct` / `ask_all_missing` / `final`) | Unit tests |
| 1.7 | Chat turn API + persist session/messages/audit | curl E2E khai sinh |

**Done when:** hỏi “làm khai sinh” → hỏi full missing từ JSON → trả lời đủ slot → guidance + citation stub.

**Không làm:** Admin UI, RAG thật, LLM cloud.

---

## Phase 2 — Python AI Service

**Mục tiêu:** Go gọi AI qua REST; bắt đầu mock, sau gắn LLM.

| # | Task | Output |
|---|------|--------|
| 2.1 | FastAPI skeleton + `/health` | Service port 8001 |
| 2.2 | `POST /v1/extract` — intent + slots (structured, allowed keys) | Contract JSON |
| 2.3 | Provider abstraction (`mock` → `openai`/`gemini`) | Env `LLM_PROVIDER` |
| 2.4 | Go client gọi AI + timeout/retry | Integration |
| 2.5 | (Optional) `POST /v1/generate` paraphrase reply | Soft NLG |

**Done when:** CI dùng mock; staging có thể bật LLM key.

---

## Phase 3 — Citizen Web

| # | Task | Output |
|---|------|--------|
| 3.1 | Vite + React JS app | `apps/web` |
| 3.2 | Chat UI (session_id, history, missing questions) | `/` |
| 3.3 | Call Go `/api/v1/chat/turns` | CORS + proxy |
| 3.4 | Guest session UX | Không bắt login V1 |

**Done when:** browser chat end-to-end với seed khai sinh / chứng thực.

---

## Phase 4 — Admin knowledge pipeline

| # | Task | Output |
|---|------|--------|
| 4.1 | MinIO bucket + Go upload API | PDF → object storage + `documents` row |
| 4.2 | AI `POST /v1/documents/extract` → draft | `procedure_drafts` |
| 4.3 | Admin Review UI (forms, không raw JSON) | `/admin` |
| 4.4 | Validate schema/business | Block approve if invalid |
| 4.5 | Publish: version + archive previous | `procedure_versions` |
| 4.6 | Chunk + embed (Python) → pgvector | Active only after index OK |
| 4.7 | Rollback API + UI | Audit reason |

**Done when:** upload PDF text → review → publish → citizen dùng version mới.

---

## Phase 5 — RAG + citations

| # | Task | Output |
|---|------|--------|
| 5.1 | Retrieve top-k chunks (filter `xa_id`, active version) | AI retrieve |
| 5.2 | Final guidance = JSON checklist + RAG explanation + citations | Knowledge orchestrator |
| 5.3 | Guardrail: RAG không override procedure JSON | Tests |
| 5.4 | Evaluation set (VN admin cases) | Metrics intent/slot/citation |

**Done when:** final answers luôn có citation; unsupported ≤ target (eval).

---

## Phase 6 — Security & hardening

| # | Task | Output |
|---|------|--------|
| 6.1 | JWT auth (citizen optional / admin required) | RBAC publish/rollback |
| 6.2 | Secrets chỉ env / secret manager | No keys in git |
| 6.3 | Rate limit chat/upload | Basic |
| 6.4 | PII redaction in logs | Policy |
| 6.5 | Dependency scan + OWASP checklist | Report |

---

## Phase 7 — Full local Docker stack

| # | Task | Output |
|---|------|--------|
| 7.1 | Dockerfile cho `api`, `ai-service`, `web` | Multi-stage builds |
| 7.2 | Compose: postgres, minio, api, ai, web, nginx | `deploy/docker-compose.yml` |
| 7.3 | Migrations on startup / init job | Idempotent |
| 7.4 | `.env.example` đầy đủ | Documented |
| 7.5 | One-command: `docker compose up --build` | Dev README |

**Done when:** máy mới clone → compose up → chat + admin chạy.

---

## Phase 8 — CI/CD

| # | Task | Output |
|---|------|--------|
| 8.1 | GitHub Actions: lint + unit Go/Python/Web | PR checks |
| 8.2 | Integration với mock LLM | CI green |
| 8.3 | Build & push images (GHCR/Docker Hub) | Tags `sha` / `semver` |
| 8.4 | Deploy job → staging (SSH hoặc compose pull) | Manual approve prod |

---

## Phase 9 — Staging server (pre-prod)

**Mục tiêu:** môi trường giống prod trên VPS.

| # | Task | Output |
|---|------|--------|
| 9.1 | Thuê VPS (2–4 vCPU, 4–8GB RAM) + domain | `staging.example.vn` |
| 9.2 | Cài Docker + Compose + firewall (22/80/443) | Hardened host |
| 9.3 | TLS (Caddy/Nginx + Let’s Encrypt) | HTTPS |
| 9.4 | Deploy stack từ CI hoặc script | Running staging |
| 9.5 | Postgres volume + daily backup cron | `.sql.gz` offsite |
| 9.6 | MinIO persistence + bucket policy | Private |
| 9.7 | Seed procedures + 1–2 PDF text thật (nếu có) | Ready for UAT |
| 9.8 | LLM key trên staging only | Secret file / env |

**Done when:** team + cán bộ thử trên HTTPS staging.

---

## Phase 10 — UAT & evaluation

| # | Task | Output |
|---|------|--------|
| 10.1 | Chạy test matrix (khai sinh, chứng thực, publish, rollback) | Pass critical |
| 10.2 | AI eval (intent ≥85%, slot F1 ≥80%, citation 100%) | Report |
| 10.3 | Perf smoke (20 concurrent, P95 non-AI) | Numbers |
| 10.4 | Security pass (no high/critical) | Checklist |
| 10.5 | UAT citizens + officers (satisfaction ≥4/5) | Fixes |

---

## Phase 11 — Production cutover

| # | Task | Output |
|---|------|--------|
| 11.1 | Prod VPS (hoặc promote staging) + domain chính | DNS |
| 11.2 | Fresh secrets (JWT, DB, MinIO, LLM) | Rotated |
| 11.3 | Restore/migrate schema + approved procedures only | Clean data |
| 11.4 | Blue/green hoặc downtime window ngắn | Compose pull + up |
| 11.5 | Smoke: health, chat, admin login, publish dry-run | Go-live |
| 11.6 | Monitoring alerts (uptime, 5xx, disk, AI fail) | On-call notes |

**Prod topology**

```text
Internet → Nginx/Caddy (TLS)
              ├── web
              └── /api → Go api → ai-service (internal)
                              ├── PostgreSQL+pgvector
                              └── MinIO
```

---

## Phase 12 — Operate (sau go-live)

| # | Task | Cadence |
|---|------|---------|
| 12.1 | Backup Postgres + MinIO | Daily; test restore monthly |
| 12.2 | Log retention + audit export | Weekly review |
| 12.3 | Patch images / deps | Monthly |
| 12.4 | Procedure content updates qua Admin | As regulations change |
| 12.5 | Incident runbook (rollback version / redeploy) | Documented |

---

## Thứ tự ưu tiên nếu thời gian hẹp (Must)

1. Phase 1 (chat deterministic)  
2. Phase 3 (Citizen UI)  
3. Phase 2 (AI extract — mock OK)  
4. Phase 4 (admin publish tối thiểu)  
5. Phase 7 + 9 (Compose → staging)  
6. Phase 5 RAG (có thể song song sau 4)  
7. Phase 8 CI + Phase 11 prod  

**Defer nếu thiếu thời gian:** voice, OCR, multi-xã, VNeID, payment.

---

## Checklist “Prod-ready”

- [ ] `docker compose` full stack documented  
- [ ] Migrations idempotent  
- [ ] HTTPS + firewall  
- [ ] Secrets not in git  
- [ ] Admin RBAC (publish/rollback)  
- [ ] Backup + restore tested  
- [ ] Health checks + basic metrics  
- [ ] Rollback procedure (app image + procedure version)  
- [ ] UAT signed off  
- [ ] Runbook + known limitations  

---

## Gợi ý tuần (16 tuần MSE — map nhanh)

| Weeks | Focus |
|-------|--------|
| 1–2 | Confirm procedures + env; Phase 1 start |
| 3–4 | Finish Phase 1–2 contracts |
| 5–7 | Phase 1 solid + Phase 2 LLM |
| 8–10 | Phase 4–5 AI/admin/RAG |
| 11–12 | Phase 3 UI + Phase 7 Compose |
| 13–14 | Phase 8–10 CI/staging/UAT |
| 15–16 | Phase 11–12 prod + docs |

---

## Next action (ngay)

**Bắt đầu Phase 1:** scaffold `apps/api` (Go module + Gin + migrate + Decision Engine + chat turns).

Khi sẵn sàng, bảo: **“làm Phase 1”**.
