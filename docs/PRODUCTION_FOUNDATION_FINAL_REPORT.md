# Production foundation final

Ngày: 2026-10-13. Máy local Apple Silicon. Sáu blocker của independent review được vá. Không deploy production. Phase 4–8 không được viết.

```text
AI_S3_PRODUCTION_CONFIG=PASS
AI_PRODUCTION_FAIL_FAST=PASS
AI_HEALTH_READINESS=PASS
CI_P4B_LIVE_INTEGRATION=PARTIAL
WEB_PRODUCTION_FEATURE_FLAGS=PASS
INDEX_REAPER_SCHEDULING=PASS
QDRANT_READINESS_READ_ONLY=PASS

GO_TESTS=PASS
PYTHON_TESTS=PASS
WEB_TESTS=PASS
DOCKER_PRODUCTION_BUILD=PARTIAL
DOCKER_PRODUCTION_CONFIG=PASS
STAGING_LIKE_SMOKE=NOT_RUN

PHASE_1_3_LOCAL_BASELINE=PASS
PRODUCTION_FOUNDATION_READY=false
FULL_APPLICATION_READY=false

RUNPOD_S3_LIVE_SMOKE=NOT_RUN
STAGING_DEPLOYMENT=NOT_RUN
PRODUCTION_DEPLOYMENT=NOT_RUN
GITHUB_ACTIONS=NOT_RUN
```

`PRODUCTION_FOUNDATION_READY` là false vì GitHub Actions chưa chạy job mới, image AI không được build lại trong vòng này, smoke staging với model thật không chạy, và restore chưa diễn tập. Sáu finding đã có regression trên máy local.

## Findings

| Severity | Finding | Root Cause | Fix | Tests | Status |
|---|---|---|---|---|---|
| High | AI không nhận S3 key | Compose chỉ truyền endpoint/bucket/region cho `ai-service` | Cùng bộ biến với API. `INDEX_MODE=pipeline` bắt key, bucket, endpoint. Prod từ chối `http://`, auto-create và `minioadmin` | `test_pipeline_requires_storage_credentials` | PASS |
| High | Healthcheck gọi `/health` | `/health` luôn `ok` | Compose gọi `/ready`. `/health` giữ liveness | `test_health_stays_ok_when_pipeline_is_not_ready` | PASS |
| High | CI `p4b-live` thiếu `CAS_P4B_TEST_*` | Job cũ trỏ `citizen_assistance:5432` | Job bật `docker-compose.p4b-test.yml` và truyền đúng biến guard | YAML + storage integration local | PARTIAL, GitHub chưa chạy |
| Medium | Web mặc định tắt admin | Vite arg `false` | Document rebuild args. `test:ci` vẫn build cả hai bundle | `npm run test:ci` | PASS |
| Medium | Index-reaper không chạy định kỳ | Binary chỉ one-shot, không có trong Compose | `INDEX_REAPER_INTERVAL_SECONDS`, vòng lặp, một replica, `SKIP LOCKED`. Binary nằm trong image API | `TestLoadConfigIntervalAndLoopStops` | PASS |
| Medium | `/ready` tạo collection Qdrant | Probe 404 thì `PUT` | Probe chỉ GET. `scripts/provision-qdrant.py` tạo một lần | `test_qdrant_readiness.py` | PASS |

## Tests

| Test | Command | Environment | Passed | Failed | Skipped | Exit Code | Evidence |
|---|---|---|---|---|---|---|---|
| Python full | `pytest -q` | local, không `CAS_P4B_INTEGRATION` | 195 | 0 | 2 | 0 | `artifacts/production_foundation_final/pytest.txt` |
| Storage integration | `pytest tests/test_p4b_services.py` | stack `citizen-assistance-p4b-test` | 2 | 0 | 0 | 0 | `storage-integration.txt` |
| Go | `go vet`, `go test ./... -count=1` | local go1.26 | ok | 0 | 0 | 0 | `go.txt` |
| Go race | `go test -race ./...` | local | ok | 0 | 0 | 0 | `go-race.txt` |
| Web | `npm run test:ci` | node 22 | 18 | 0 | 0 | 0 | `web-test.txt` |
| Production compose | `docker compose -f docker-compose.production.yml config` | local | fail-closed rồi render đủ key | 0 | 0 | 0/1 | `compose-production.txt` |
| API image | `docker build -f apps/api/Dockerfile` | linux/arm64, user `cas`, có `./index-reaper` | | | | 0 | `docker-api-build.txt` |
| AI image | không build | | | | | | NOT_RUN |
| GitHub Actions | job mới | | | | | | NOT_RUN |
| Staging smoke real model | | | | | | | NOT_RUN |

Hai test skip là storage integration trong full pytest. Chúng chạy riêng trên stack test.

## Limits

Image API và web đã build trên ARM64. Image AI không đổi ở vòng này nên không build lại. Linux x86_64 chưa có. RunPod S3 không được gọi. Collection `knowledge_chunks` trên Qdrant development không bị tạo bởi `/ready`.
