# Phase 1–3 isolation final

Ngày vá: 2026-10-13. Máy local Apple Silicon. Ba finding của review cùng ngày được vá trong test infrastructure. Pipeline indexing, model và Decision Engine không đổi. Model smoke không chạy lại. `P4B_DONE` giữ log lịch sử `artifacts/phase_1_3_final_evidence/model-smoke.txt` lúc `2026-10-13T03:15:05Z`, exit 0. Log của vòng vá này nằm ở `artifacts/phase_1_3_isolation_final/`.

```
PHASE_1_STATUS=PASS
PHASE_2_STATUS=PASS
PHASE_3_LOCAL_STATUS=PASS

P4B_MODEL_SMOKE=PASS
P4B_DONE=true
P4B_STORAGE_INTEGRATION=PASS
P4B_REAL_INDEXING=PASS

POSTGRES_DSN_GUARD=PASS
DOCKER_TEST_PORT_ISOLATION=PASS
QDRANT_TEST_CLEANUP=PASS

PHASE_1_3_BASELINE_READY=true

RUNPOD_S3_LIVE_SMOKE=NOT_RUN
FP32_INT8_EQUIVALENCE=NOT_RUN
PRODUCTION_DEPLOYMENT=NOT_RUN
```

`P4B_MODEL_SMOKE`, `P4B_DONE` và `P4B_REAL_INDEXING` là kết quả đã ghi trước vòng vá này. `P4B_STORAGE_INTEGRATION` là lần chạy mới sau khi guard và cổng đổi. Đây vẫn là FakeEmbedder và FakeOCR, không phải inference model thật. Không có xác nhận từ máy review độc lập.

## A. Findings

| Finding | Nguyên nhân | File | Test | Trạng thái |
|---|---|---|---|---|
| HIGH DSN query override | Guard dùng `urlparse`. libpq lấy `host`, `hostaddr`, `port`, `dbname`, `service` từ query string | `apps/ai-service/app/indexing/isolation.py` | `tests/test_p4b_isolation.py` | PASS |
| MEDIUM cổng test mở mọi interface | Compose publish `5434`, `9010`, `9011`, `6335` không gắn host | `deploy/docker-compose.p4b-test.yml` | `docker ps` chỉ còn `127.0.0.1` | PASS |
| MEDIUM point Qdrant còn lại | `finally` xóa PostgreSQL và MinIO, không xóa generation vừa tạo | `apps/ai-service/tests/test_p4b_services.py`, `tests/p4b_cleanup.py` | Hai lần chạy, số point không tăng | PASS |

Guard đọc DSN bằng `psycopg.conninfo.conninfo_to_dict`. Query string chỉ được có `sslmode=disable`. Tham số lặp, nhiều host, Unix socket, `service` và `servicefile` bị từ chối trước khi mở kết nối. `SELECT current_database()` vẫn chạy sau bước đó, trên cùng URL đã được chấp nhận.

`docker ps` sau khi tạo lại container, volume không bị xóa:

```
citizen-assistance-p4b-test-postgres-1 127.0.0.1:5434->5432/tcp
citizen-assistance-p4b-test-qdrant-1 127.0.0.1:6335->6333/tcp
citizen-assistance-p4b-test-minio-1 127.0.0.1:9010->9000/tcp, 127.0.0.1:9011->9001/tcp
```

Không có `0.0.0.0` hay `[::]`. Stack development không bị tạo lại.

Cleanup xóa theo `generation_id` của chính lần chạy, gọi lại lần thứ hai không lỗi. Point có sẵn trước lần chạy không nằm trong filter đó. Số point sau hai lần chạy bằng số point trước lần chạy đầu: `p4b_test` 2, `p4b_duplicate` 6, `p4b_missing` 6, `p4b_acknowledged` 3, `p4b_restart` 3, `p4b_dimension` 0, `p4b_partial` 0. Những point cũ là phần còn từ các lần chạy trước bản vá và không bị xóa.

## B. Regression

| Test | Command | Passed | Failed | Skipped | Exit | Evidence |
|---|---|---|---|---|---|---|
| Storage integration | `pytest -q tests/test_p4b_services.py` | 2 | 0 | 0 | 0 | `storage-integration.txt` `2026-10-13T04:12:04Z` |
| Storage integration lần 2 | cùng lệnh | 2 | 0 | 0 | 0 | `storage-integration-repeat.txt` `2026-10-13T04:12:40Z` |
| Python | `compileall` và `pytest -q` | 181 | 0 | 0 | 0 | `pytest.txt` `2026-10-13T04:12:42Z` |
| Go | `gofmt -l`, `go vet`, `go test ./... -count=1`, `go test -race ./...` | ok | 0 | 0 | 0 | `go.txt` `2026-10-13T04:12:43Z` |
| Web | `npm run test:ci` | 18 | 0 | 0 | 0 | `web-test.txt` `2026-10-13T04:12:45Z` |

Sau các lần chạy, database development vẫn 163 document và `knowledge_chunks` vẫn 2 point. Database test còn 0 document. SHA-256 từng file evidence nằm trong `MANIFEST.tsv`.

## C. Baseline

Phase 1–3 local giữ PASS theo evidence đã có. Ba finding an toàn của integration test đã được vá và chạy lại trên stack `citizen-assistance-p4b-test`.

`PHASE_1_3_BASELINE_READY=true` cho việc bắt đầu viết Phase 4. Tài khoản Postgres của stack test vẫn là user `cas` do image tạo, giới hạn bởi instance và DSN đã khóa, chưa phải role bị cắt quyền trong database. Production, RunPod S3 và so sánh FP32/INT8 vẫn `NOT_RUN`.

Review ngay sau vòng này chỉ ra hai lỗ còn lại: biến môi trường libpq (`PGHOSTADDR`, `PGSERVICE`, `PGSERVICEFILE`) và cleanup khi setup MinIO/PostgreSQL lỗi trước `try`. Hai mục đó được đóng trong `docs/PHASE_1_3_FINAL_PATCH_REPORT.md`. Số liệu trong file này vẫn là evidence của vòng DSN query, cổng loopback và cleanup theo generation.
