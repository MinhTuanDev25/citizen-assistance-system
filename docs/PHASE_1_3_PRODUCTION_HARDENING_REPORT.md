# Phase 1–3 production hardening

Ngày: 2026-10-13. Máy local Apple Silicon. Hai lỗi đã xác nhận được sửa. Nền triển khai được tách khỏi Compose local. Không deploy production. Phase 4–8 không được viết.

```text
POSTGRES_DSN_FIX=PASS
AMBIGUOUS_SUCCESS_CLEANUP=PASS
INTEGRATION_ISOLATION=PASS

PHASE_1_3_LOCAL_BASELINE=PASS
PRODUCTION_CONFIG_VALIDATION=PASS
DOCKER_IMAGE_BUILD=PARTIAL
DATABASE_MIGRATION_SAFETY=PASS
SECRET_MANAGEMENT=PASS
SERVICE_HEALTH_READINESS=PARTIAL
SECURITY_CHECKS=PASS
OBSERVABILITY=PARTIAL
BACKUP_RESTORE_VERIFIED=NOT_RUN
STAGING_E2E=NOT_RUN
CI_PIPELINE=NOT_RUN

PRODUCTION_FOUNDATION_READY=false
FULL_APPLICATION_READY=false
PRODUCTION_DEPLOYMENT=NOT_RUN
```

`PHASE_1_3_LOCAL_BASELINE` gồm baseline đã có và lần chạy lại storage integration cùng pytest trong vòng này. `P4B_DONE` vẫn là log lịch sử, model smoke không chạy lại. `PRODUCTION_FOUNDATION_READY` là false vì image AI chưa build lại, CI GitHub chưa chạy, restore chưa diễn tập, và staging/production chưa có.

Vòng ngay sau đó vá S3 cho AI, `/ready`, CI isolation, web flags, index-reaper và probe Qdrant chỉ đọc. Kết quả mới nằm ở `docs/PRODUCTION_FOUNDATION_FINAL_REPORT.md`. Số liệu trong file này vẫn là evidence của vòng hardening trước.

## 1. Current fixes

### DSN separator

`lock_database_url()` ở `apps/ai-service/app/indexing/isolation.py` nối cố định `&hostaddr=127.0.0.1`. URL không có `?` khiến libpq coi đoạn đó là một phần của tên database, và `PGHOSTADDR` vẫn đổi được địa chỉ TCP.

Hàm vẫn chỉ dùng cho DSN test. URL đã có query dùng `&`. URL chưa có query dùng `?`. Production không đi qua hàm này. `APP_ENV=prod` yêu cầu `sslmode=require`, `verify-ca`, hoặc `verify-full`.

Regression, không mở database development: URL trần và URL có `?sslmode=disable` đều ra database `citizen_assistance_p4b_test`, port `5434`, `hostaddr=127.0.0.1` kể cả khi `PGHOSTADDR=127.0.0.2`. Query override, env override, host/port/database sai và endpoint dạng production vẫn bị từ chối.

### Cleanup trước lúc ghi

`tests/test_p4b_services.py` đăng ký xóa object, row và point theo ID của chính test trước `put_object`, trước `INSERT`, và trước pipeline. Xóa object bỏ qua `NoSuchKey`. `OwnedResources.finish` vẫn chạy một lần, không che exception chính, và không xóa ID có sẵn.

Test `test_ambiguous_success_is_cleaned_because_the_id_was_armed_first` giả lập server đã lưu rồi client ném `TimeoutError`. Cleanup của ba ID đã đăng ký vẫn chạy. Không có retry vòng lặp.

Hai lần storage integration trên stack `citizen-assistance-p4b-test`: mỗi lần 2 passed, exit 0. Sau đó test database còn 0 document. Số point `p4b_*` không tăng. Database development còn 163 document.

## 2. Production architecture

| Thành phần | Local | Production template |
|---|---|---|
| Web, API, AI | Compose `citizen-assistance`, cổng publish cho demo | `deploy/docker-compose.production.yml`, chỉ ba service ứng dụng |
| PostgreSQL | Container local, `sslmode=disable` | Endpoint ngoài, TLS bắt buộc khi `APP_ENV=prod` |
| Object storage | MinIO local, được phép tự tạo bucket | Endpoint cấu hình, không tự tạo bucket |
| Qdrant | Container local | Endpoint riêng, collection `knowledge_chunks`, 384 chiều |
| Model | FP32 trên ARM64 đã chạy local | ONNX + Paddle, không fallback sang fake khi `APP_ENV=prod` |
| Secrets | Default trong Compose local | Biến bắt buộc, file example để trống |
| HTTPS | Không | Reverse proxy phía trước `127.0.0.1:8088`, chưa triển khai |
| Backup | Volume Docker local | Quy trình ghi trong `deploy/README.production.md`, chưa restore |

`LLM_PROVIDER=mock` và `AI_EXTRACT_ENABLED=false` vẫn là mặc định production vì Phase 6 chưa có. Decision Engine Go không đổi.

## 3. Changed files

| File | Why changed | Implementation | Test |
|---|---|---|---|
| `apps/ai-service/app/indexing/isolation.py` | `&` trên URL không có query làm sai tên database | Chọn `?` hoặc `&` theo query hiện có | `test_lock_database_url_uses_the_right_query_separator` |
| `apps/ai-service/tests/test_p4b_services.py` | Cleanup đăng ký sau khi ghi trả về | Đăng ký theo ID trước lúc ghi; xóa object thiếu thì bỏ qua | Storage integration 2 lần |
| `apps/ai-service/tests/p4b_cleanup.py` | Contract cũ là đăng ký sau khi tạo | Docstring khớp thứ tự mới | Fault injection trong `test_p4b_isolation.py` |
| `apps/ai-service/tests/test_p4b_isolation.py` | Chưa phủ URL không có query và timeout sau khi server ghi | Hai test mới | 12 isolation tests trong full pytest |
| `apps/api/internal/config/config.go` | Prod có thể nhận DSN không TLS và secret local | `validateProduction` | `TestLoadProdRejectsLocalDatabaseAndIndexing` |
| `apps/ai-service/app/config.py` | Prod có thể chạy mock index hoặc fake model | Từ chối khi `APP_ENV=prod` | `test_prod_refuses_mock_index_and_fake_models` |
| `apps/api/Dockerfile` | Image API chạy root | User `cas` uid 10001 | Image `cas-api:phase13-hardening` |
| `apps/web/nginx.conf` | Thiếu security header | `nosniff`, `DENY`, `no-referrer` | Nằm trong image web đã build |
| `deploy/docker-compose.production.yml` | Local compose không dùng được để go-live | App tier, secret bắt buộc, không publish database | `docker compose config` |
| `deploy/.env.production.example` | Cần mẫu không chứa secret | Giá trị để trống | Không có secret trong file |
| `.github/workflows/ci.yml` | CI chưa kiểm tra file production | Job `production-config` | Chưa chạy trên GitHub |
| `docs/p4b-runbook.md` | Runbook chưa nói dấu `?` | Một câu về separator và phạm vi test | Không phải test |
| `deploy/README.production.md` | Chưa có hướng dẫn deploy tách khỏi local | Quy trình và các mục còn blocked | Không phải test |

## 4. Test evidence

| Test | Command | Result | Exit code | Evidence |
|---|---|---|---|---|
| Python full | `pytest -q` với `CAS_P4B_INTEGRATION` gỡ | 186 passed, 2 skipped | 0 | `artifacts/phase_1_3_production_hardening/pytest.txt` `2026-10-13T07:55:38Z` |
| Storage integration | `pytest -q tests/test_p4b_services.py` | 2 passed | 0 | Lần chạy trong session này |
| Storage integration lần 2 | cùng lệnh | 2 passed | 0 | Số point không tăng, test DB 0 document, dev DB 163 |
| Go | `go vet ./...`, `go test ./... -count=1` | ok | 0 | `artifacts/phase_1_3_production_hardening/go.txt` |
| API image | `docker build -f apps/api/Dockerfile` | `linux/arm64`, user `cas` | 0 | `cas-api:phase13-hardening` |
| API fail-fast | `docker run -e APP_ENV=prod` | `DATABASE_URL missing for APP_ENV=prod` | 2 | Process thoát trước khi listen |
| Web image | `docker build -f apps/web/Dockerfile` | `linux/arm64` | 0 | `cas-web:phase13-hardening` |
| Production compose | `docker compose -f docker-compose.production.yml config` | Thiếu biến thì exit 1; đủ biến thì không publish 5432/6333/9000 | 0 | Render local `/tmp/prod.yml` |
| AI image | không build | NOT_RUN | | Dockerfile không đổi ở vòng này |
| GitHub Actions | job mới chưa được runner chạy | NOT_RUN | | |

Hai test skip của full pytest là storage integration. Chúng được chạy riêng trên stack test.

## 5. Environment compatibility

| Environment | Build | Startup | Integration | Status |
|---|---|---|---|---|
| Mac M4 local | API và web image `linux/arm64` | API prod thiếu `DATABASE_URL` thoát | Storage integration PASS | PASS cho những mục đã chạy |
| Linux ARM64 | Cùng image API ở trên, chạy trong container | Fail-fast đã thấy | Full stack trong container NOT_RUN | PARTIAL |
| Linux x86_64 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Staging | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Production | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

E5 INT8 chỉ dành cho CPU có AVX-512 VNNI. Máy này dùng FP32. Không có bằng chứng INT8 trên Linux x86_64.

## 6. Remaining blockers

- Chưa có host staging, chứng chỉ TLS, credential RunPod, hoặc sự chấp thuận deploy.
- Chưa diễn tập restore PostgreSQL.
- Image AI chưa được build lại trong vòng này.
- Job `p4b-live` cũ trong CI vẫn trỏ database tên `citizen_assistance` của service CI và không đi qua guard stack test. Job đó không được chạy lại ở đây.
- Phase 4 Activate, Phase 5 RAG, Phase 6 LLM, Phase 7 Bahnar, Phase 8 nghiệm thu chưa có trong phần mềm.
