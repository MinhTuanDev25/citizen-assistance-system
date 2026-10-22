# Phase 4 — Final acceptance report

Ngày: 2026-10-13. Máy local Apple Silicon (`go1.26.3`, Node v26.7.0). Baseline git: `571e5f92623bfc81d64e08c4f789b7f7326c0ba0` (Phase 1–3). **F2/F4/F5 và migration 000017 nằm trên working tree, chưa commit.** Không deploy staging/production. Không đụng `citizen_assistance` (port 5432 đóng). Không seed-demo trên DB development.

## 1. Source snapshot

Commit `571e5f9` **không** chứa F2/F4/F5. ZIP review là snapshot working tree trên baseline đó:

`/Users/minhtuan25/Desktop/citizen-assistance-phase4-f2f4f5-source.zip`

Gồm Go backend (`apps/api/internal/activation`, admin APIs, tests), React (`ProceduresPage.jsx` + tests), `deploy/migrations/000017_phase4_activation.{up,down}.sql`.

## A. Findings

| Finding | Root cause | Fix | Changed files | Tests | Status |
|---|---|---|---|---|---|
| F2 | DSN substring | `ValidateP4TestDSN` + `current_database()` | `isolation.go`, `isolation_test.go`, `integration_test.go` | Unit + live integration | PASS |
| F4 | Activate+refresh chung catch | Stage A/B, timeout unknown | `ProceduresPage.jsx`, test | Vitest | PASS |
| F5 | Parse JSON không khớp DB | `definitionIdentityReasons` | `eligibility.go`, tests | Unit + TestC fixture dùng đúng procedure_code | PASS |
| F1 PENDING | Giữ policy | Không bắt VALID | eligibility tests | PASS | Locked |
| F3 JSON editor | Ngoài phạm vi | Không xây | UI báo khi chưa ACTIVE | OUT OF SCOPE |

## B. Policy

- PENDING không chặn Activate nếu generation READY đã publish. EXPIRED/SUPERSEDED chặn.
- READY generation: chunks=vectors≥1, dim 384, published.
- Legacy definition: thiếu domain/xa_id/version thì bỏ qua; field lệch → `DEFINITION_IDENTITY_MISMATCH`.
- Candidate UI: copy từ ACTIVE.
- Phase 5 source-retention: xem runbook.

## C. Test evidence

| Test | Command / env | PASS/FAIL | Exit | Log |
|---|---|---|---|---|
| Isolation+unit | `go test ./internal/activation/` | PASS | 0 | `go_unit.txt` |
| Go `./...` + vet + race | earlier this session | PASS | 0 | `go_vet_race.txt` |
| Vitest + `npm run test:ci` | earlier | PASS 30 | 0 | `vitest.txt` `npm_test_ci.txt` |
| **P4 DB integration** | `CAS_P4_TEST=1 CAS_P4_TEST_PORT=5433 CAS_P4_TEST_DATABASE_URL=postgres://cas:cas@127.0.0.1:5433/cas_p4_test?sslmode=disable go test -tags=integration ./internal/activation/ -count=1 -v` | **PASS** (A–H, HTTP, concurrency, rollback, idempotency, audit, session pin) | 0 | `go_integration_cas_p4_test.txt` |
| **API real PDF E2E** | isolated pg:5433 + minio:9012 + qdrant:6336 + E5 onnx | **PASS** | 0 | `api_e2e.txt` |
| **Browser UI** | Vite 5173 + Cursor browser | **PASS** (hiển thị V2 ACTIVE / V1 ARCHIVED / PDF READY / audit) | 0 | `browser_e2e.txt` |
| PaddleOCR scanned PDF | paddleocr not in venv | **NOT_RUN** | — | used native extraction |

Isolated Postgres: container `cas-p4-test-pg` (`pgvector/pgvector:pg16`), DB `cas_p4_test`, migrate 000001–000017 dirty=false. Seed procedures from migrations + `seed-demo` **chỉ** trên DB này.

Real PDF path: `apps/ai-service/tests/fixtures/native.pdf` (622 bytes, 48 native chars). Index: `intfloat/multilingual-e5-small`, 1 chunk, 1 Qdrant point, dim 384. `OCR_PROVIDER=fake` vì thiếu package `paddleocr`; trang native đủ chữ nên OCR không chạy.

## D. Final status

```text
PHASE_4_DB_ISOLATION=PASS
PHASE_4_ACTIVATION_UI=PASS
PHASE_4_DEFINITION_IDENTITY=PASS
PHASE_4_DOCUMENT_VALIDITY_POLICY=PASS

PHASE_4_TRANSACTION=PASS
PHASE_4_CONCURRENCY=PASS
PHASE_4_IDEMPOTENCY=PASS
PHASE_4_AUDIT=PASS
PHASE_4_SESSION_PINNING=PASS

PHASE_4_GO_TESTS=PASS
PHASE_4_REACT_TESTS=PASS
PHASE_4_DB_INTEGRATION=PASS
PHASE_4_API_E2E=PASS
PHASE_4_BROWSER_E2E=PASS
PHASE_4_REAL_PDF_E2E=PASS

PHASE_4_CODE_COMPLETE=true
PHASE_4_STATUS=PASS
PHASE_5_CODE_READY=false

PRODUCTION_FOUNDATION_READY=false
FULL_APPLICATION_READY=false
STAGING_DEPLOYMENT=NOT_RUN
PRODUCTION_DEPLOYMENT=NOT_RUN
```

`PHASE_4_STATUS=PASS` = local application acceptance trên stack cô lập. Không phải production-ready. RAG Phase 5 chưa viết (`PHASE_5_CODE_READY=false`). Browser xác nhận kết quả Activate đã COMMIT, không bấm lại Activate vì V2 đã ACTIVE.

## Identifiers (non-secret)

- procedure `chung_thuc_ban_sao` `c20f493a-8724-4bd7-b7ad-7b679129acc3`
- V1 `2a2a810d-731a-45a1-be8f-2c1da05aff56` ARCHIVED
- V2 `b2485451-af33-45d8-bf05-24af578897a3` ACTIVE
- document `1d436d74-c257-458f-abe2-93ada4854d2a`
- old session `72e718c1-96e6-4172-9324-4f88722425a4` pin V1
- new session `55f56b1f-8dfb-45b1-9aa1-3e0344080f0b` pin V2
