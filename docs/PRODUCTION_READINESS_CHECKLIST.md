# Production readiness checklist

Ngày đối chiếu: 2026-10-13. `PRODUCTION_FOUNDATION_READY=false`. `FULL_APPLICATION_READY=false`. `PRODUCTION_DEPLOYMENT=NOT_RUN`.

| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| Phase 1–3 local | PASS | Baseline trước đó và storage integration vòng này |
| DSN test `?` và `&` | PASS | Isolation tests |
| Cleanup khi client timeout | PASS | Fault injection |
| AI và API cùng S3 credentials | PASS | Compose render hai lần `OBJECT_STORAGE_ACCESS_KEY` |
| Prod fail-fast thiếu S3 key | PASS | `test_pipeline_requires_storage_credentials` |
| AI healthcheck `/ready` | PASS | `docker-compose.production.yml` |
| `/health` không phụ thuộc S3 | PASS | `test_health_stays_ok_when_pipeline_is_not_ready` |
| Qdrant `/ready` chỉ đọc | PASS | `test_qdrant_readiness.py` |
| Provision collection tách khỏi ready | PASS | `scripts/provision-qdrant.py` |
| CI `p4b-live` dùng stack test | YAML PASS, runner NOT_RUN | `.github/workflows/ci.yml` |
| Web flags documented, default off | PASS | Dockerfile + `test:ci` |
| Index-reaper interval | PASS | Compose + Go test vòng lặp |
| Production pipeline bắt buộc `OBJECT_STORAGE_BUCKET` | PASS | `test_config.py` + process độc lập |
| Index-reaper SIGTERM/SIGINT drain | PASS | Go tests + `docker stop` TEST-NET-1 |
| API image non-root + reaper binary | PASS ARM64 | `cas-api:phase13-closure` |
| AI image rebuild vòng này | NOT_RUN | |
| Linux x86_64 | NOT_RUN | |
| Staging E2E | NOT_RUN | |
| Backup restore | NOT_RUN | |
| RunPod S3 live | NOT_RUN | |
| Phase 4–8 | Chưa làm | |

`PRODUCTION_FOUNDATION_READY` chuyển true khi: runner GitHub chạy xanh job `p4b-live` và `production-config`, image AI được build trên kiến trúc máy đích, collection Qdrant được provision một lần trên staging, và một restore Postgres thành công trên database không phải development.
