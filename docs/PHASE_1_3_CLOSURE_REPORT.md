# Phase 1–3 closure

Ngày: 2026-10-13. Máy local Apple Silicon (`Darwin 25.6.0 arm64`, Python 3.9.6, go1.26.3 darwin/arm64, Docker 29.4.2 linux/arm64). Hai finding Medium đã xác nhận được sửa. Phase 4–8 không được viết. Không deploy staging hoặc production. Model smoke E5/PaddleOCR không chạy lại; evidence lịch sử `P4B_DONE` lúc `2026-10-13T03:15:05Z` được giữ nguyên.

Log lần chạy nằm ở `artifacts/phase_1_3_final_closure/` (gitignored). SHA-256 trong `MANIFEST.sha256`.

## A. Changed files

| File | Root cause | Fix | Regression tests |
|---|---|---|---|
| `apps/ai-service/app/indexing/object_storage.py` | `storage_config()` luôn default `cas-documents` khi thiếu `OBJECT_STORAGE_BUCKET` | `object_bucket_name()` chỉ default khi không phải `APP_ENV=prod`. Production thiếu/rỗng/whitespace raise `ConfigError` | `tests/test_config.py` |
| `apps/ai-service/app/config.py` | `_require_pipeline_storage()` đọc `cfg["bucket"]` sau default, nên production pipeline không bao giờ thấy bucket rỗng | Kiểm tra env thô trước `storage_config()`. `Settings.object_bucket` lấy từ `object_bucket_name()` | `test_prod_pipeline_requires_an_explicit_bucket`, `test_prod_pipeline_does_not_fallback_or_autocreate_bucket`, process độc lập |
| `apps/ai-service/tests/test_config.py` | Thiếu regression production bucket | Local thiếu bucket vẫn `cas-documents`. Prod thiếu/rỗng/whitespace FAIL. Prod bucket tường minh PASS. Không auto-create, không fallback | 18 passed |
| `apps/api/cmd/index-reaper/main.go` | `main()` gọi `run(context.Background())`. SIGTERM không được gắn. Cycle đang chạy dùng context cha nên bị hủy ngay | `signal.NotifyContext` SIGTERM/SIGINT. Cycle mới dừng khi parent done. In-flight dùng `context.Background()` + timeout riêng. AfterFunc chờ `INDEX_REAPER_SHUTDOWN_SECONDS` (1–120, default 30) rồi cancel work. Qdrant HTTP nhận work context. Log `reaper_shutdown_requested` / `reaper_draining` / `reaper_shutdown_completed` / `reaper_shutdown_timeout` / `reaper_stopped` | `cmd/index-reaper` tests |
| `apps/api/cmd/index-reaper/main_test.go` | Chưa có lifecycle tests | Idle skip, in-flight trong grace, timeout, không cycle mới, SIGTERM/SIGINT subprocess, cancel Qdrant, claim SQL không lấy READY/live claim, repeated cancel | 18 PASS, `-race` PASS |
| `deploy/docker-compose.production.yml` | Stop Docker mặc định 10s, ngắn hơn drain 30s | `stop_grace_period: 45s`, `INDEX_REAPER_SHUTDOWN_SECONDS` default 30 | `docker compose config` |
| `deploy/.env.production.example` | Thiếu biến shutdown | `INDEX_REAPER_SHUTDOWN_SECONDS=30` | Compose render |
| `deploy/README.production.md` | Contract chưa nói fail-fast bucket và drain | Ghi rõ Python tự kiểm tra bucket; reaper drain không atomic với Qdrant/Postgres | Không phải test |
| `.github/workflows/ci.yml` | Job `production-config` chưa khóa grace | Grep `INDEX_REAPER_SHUTDOWN_SECONDS` và `stop_grace_period: 45s` | YAML only; GitHub runner NOT_RUN |
| `docs/PHASE_1_3_CLOSURE_REPORT.md` | Báo cáo closure | File này | Không phải test |

`APP_ENV` chỉ nhận `prod` (so khớp `.strip().lower() == "prod"`). `INDEX_MODE` chỉ `mock` hoặc `pipeline`. Không thêm alias.

Qdrant delete và Postgres `cleanup_status` không cùng transaction. Claim SQL chỉ lấy `STAGING`/`FAILED`, bỏ generation đang link READY và job `CLAIMED` còn hạn. `SKIP LOCKED` tránh hai replica cùng claim. Nếu process chết sau khi Qdrant đã xóa và trước khi Postgres complete, lần sau claim lại hàng `RETRYABLE_FAILED` hoặc lease hết hạn; delete Qdrant là idempotent. Không có bảo đảm atomicity.

Ngoài scope (không sửa): Go API chỉ ghi `OBJECT_STORAGE_BUCKET` khi env khác rỗng, nên giá trị YAML còn có thể giữ nếu biến được set thành chuỗi rỗng. Production Compose vẫn bắt buộc interpolation. Python process độc lập là lỗ hổng đã xác nhận và đã đóng.

## B. Test results

| Test | Command | PASS/FAIL | Exit code | Evidence |
|---|---|---|---|---|
| Python compileall | `.venv/bin/python -m compileall -q app` | PASS | 0 | `python-compileall.txt` |
| Production/storage config | `.venv/bin/python -m pytest -v tests/test_config.py` | PASS 18 | 0 | `python-config.txt` |
| AI service độc lập thiếu bucket | `APP_ENV=prod INDEX_MODE=pipeline` không set `OBJECT_STORAGE_BUCKET`; `load_settings()` và `storage_config()` | PASS fail-fast | 0 | `python-independent-fail-fast.txt` |
| Full pytest | `.venv/bin/python -m pytest -q` | PASS 197, skip 2 | 0 | `python-pytest-full.txt` |
| gofmt | `gofmt -l .` | PASS | 0 | `go-test.txt` |
| go vet | `go vet ./...` | PASS | 0 | `go-test.txt` |
| Go tests | `go test ./... -count=1` | PASS | 0 | `go-test.txt` |
| Go race | `go test -race ./... -count=1` | PASS | 0 | `go-race.txt` |
| Index-reaper lifecycle | `go test ./cmd/index-reaper -count=1 -v` | PASS 18 | 0 | `go-index-reaper.txt` |
| Compose thiếu biến | `docker compose -f docker-compose.production.yml config` | PASS fail-closed | 1 | `docker-compose-missing.txt` |
| Compose render | cùng file với biến CI giả | PASS | 0 | `docker-compose-config.yml`, `docker-compose-checks.txt` |
| API image + reaper binary | `docker build -f apps/api/Dockerfile -t cas-api:phase13-closure .` | PASS linux/arm64 | 0 | `docker-api-build.txt` |
| Reaper thiếu config trong container | `docker run ... ./index-reaper` | PASS `config_missing` | 1 | `docker-reaper-stop.txt` |
| Docker stop reaper (TEST-NET-1, không phải DB dev) | `docker stop -t 8` | PASS, elapsed 2s, log drain/timeout/stopped | 0 | `docker-reaper-stop.txt` |
| E5/PaddleOCR model smoke | không chạy lại | HISTORICAL | — | `artifacts/phase_1_3_final_evidence/model-smoke.txt` |
| GitHub Actions `p4b-live` / `production-config` | không chạy | NOT_RUN | — | — |
| Staging / production deploy | không chạy | NOT_RUN | — | — |

## C. Final status

```text
AI_PRODUCTION_BUCKET_VALIDATION=PASS
AI_PRODUCTION_FAIL_FAST=PASS
INDEX_REAPER_SIGTERM=PASS
INDEX_REAPER_GRACEFUL_SHUTDOWN=PASS
INDEX_REAPER_RECOVERY=PASS

PYTHON_TESTS=PASS
GO_TESTS=PASS
DOCKER_VALIDATION=PASS

PHASE_1_LOCAL_BASELINE=PASS
PHASE_2_LOCAL_BASELINE=PASS
PHASE_3_LOCAL_BASELINE=PASS

PHASE_1_3_CLOSURE_READY=true

PRODUCTION_FOUNDATION_READY=false
FULL_APPLICATION_READY=false
STAGING_DEPLOYMENT=NOT_RUN
PRODUCTION_DEPLOYMENT=NOT_RUN
```

`PHASE_1_LOCAL_BASELINE`, `PHASE_2_LOCAL_BASELINE` và `PHASE_3_LOCAL_BASELINE` là baseline local đã có từ các vòng trước, không phải lần chạy lại upload/OCR/embedding. `DOCKER_VALIDATION=PASS` là Compose + image `linux/arm64` trên Mac; không phải chứng minh Linux x86_64 production. `PRODUCTION_FOUNDATION_READY` giữ false vì staging, CI runner, image AI trên máy đích, HTTPS/secrets runtime, backup restore và RunPod S3 live chưa được kiểm chứng.

## D. Staging readiness

| Mục | Trạng thái | Điều kiện để kiểm chứng |
|---|---|---|
| Linux target architecture + ONNX variant | NOT_RUN | Build trên máy đích. ARM64 dùng FP32. x86_64 cần AVX-512 VNNI cho INT8; thiếu VNNI phải fail closed |
| Docker image builds trên máy đích | PARTIAL | API `cas-api:phase13-closure` linux/arm64 đã build ở đây. AI image không rebuild vòng này (không đổi model). Web image không rebuild vòng này |
| Real E5 / PaddleOCR load | HISTORICAL | Smoke `2026-10-13T03:15:05Z` exit 0. Chạy lại trên staging với volume model đã pin |
| GitHub Actions `p4b-live` + `production-config` | NOT_RUN | Push/workflow trên runner GitHub; stack `citizen-assistance-p4b-test` |
| Dedicated PostgreSQL / Qdrant | NOT_RUN | Endpoint ngoài laptop, TLS Postgres, collection `knowledge_chunks` dim 384 cosine đã provision |
| RunPod S3 compatibility | NOT_RUN | Bucket id thật, region, TLS 443, `OBJECT_STORAGE_AUTO_CREATE_BUCKET=false`. Không dùng credential RunPod trong vòng này |
| HTTPS + runtime secrets | NOT_RUN | Reverse proxy TLS, secret store, không commit `.env` |
| PostgreSQL backup/restore | NOT_RUN | Dump và restore sang database không phải development |
| Staging PDF upload → indexing → READY | NOT_RUN | Admin ingestion/indexing bật, worker pipeline, generation READY |
| Staging security / integration | NOT_RUN | Auth, tenant isolation, `/ready` vs `/health`, không publish AI port |
| Phase 4 Activate Version | NOT_STARTED | Bắt đầu sau khi baseline này được khóa |
| Phase 5–8 | NOT_STARTED | RAG, LLM extract, Bahnar speech, E2E go-live |

## Recovery notes

Index-reaper không commit Postgres và Qdrant như một transaction. Shutdown hủy work context sau grace; claim/retry hiện có phục hồi hàng dở. READY và live claim không nằm trong claim SQL. Không xóa generation hợp lệ khi restart.

Phase 1–3 local baseline có thể khóa để sang Phase 4 — Activate Version. Ứng dụng chưa production-ready.
