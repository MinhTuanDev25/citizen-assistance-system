# Phase 1–3 final patch

Ngày vá: 2026-10-13. Máy local Apple Silicon. Hai finding còn lại của isolation được vá trong test infrastructure. Pipeline indexing, model, Decision Engine và migration không đổi. Model smoke không chạy lại.

`P4B_DONE` giữ log lịch sử `artifacts/phase_1_3_final_evidence/model-smoke.txt` lúc `2026-10-13T03:15:05Z`, exit 0. Log của vòng này nằm ở `artifacts/phase_1_3_final_patch/`.

```text
POSTGRES_DSN_OVERRIDE_PROTECTION=PASS
POSTGRES_ENV_OVERRIDE_PROTECTION=PASS
DOCKER_TEST_PORT_ISOLATION=PASS
MINIO_EARLY_FAILURE_CLEANUP=PASS
POSTGRES_TEST_CLEANUP=PASS
QDRANT_TEST_CLEANUP=PASS
P4B_STORAGE_INTEGRATION=PASS

PHASE_1_STATUS=PASS
PHASE_2_STATUS=PASS
PHASE_3_LOCAL_STATUS=PASS
P4B_DONE=true

PHASE_1_3_BASELINE_READY=true
PRODUCTION_DEPLOYMENT=NOT_RUN
```

`PHASE_1_STATUS`, `PHASE_2_STATUS`, `PHASE_3_LOCAL_STATUS` và `P4B_DONE` là kết quả đã ghi trước vòng vá này. `P4B_STORAGE_INTEGRATION` là lần chạy mới trên stack `citizen-assistance-p4b-test`, vẫn dùng FakeEmbedder và FakeOCR.

## A. Changed files

| File | Root cause | Fix | Test |
|---|---|---|---|
| `apps/ai-service/app/indexing/isolation.py` | URL hợp lệ không chặn `PGHOSTADDR`, `PGSERVICE` và `PGSERVICEFILE`. libpq có thể đổi địa chỉ TCP trong khi host trên URL vẫn là `127.0.0.1` | Từ chối `PGHOST`, `PGHOSTADDR`, `PGPORT`, `PGDATABASE`, `PGSERVICE`, `PGSERVICEFILE`, `PGSYSCONFDIR` trong mapping và trong `os.environ`. URL được khóa thêm `hostaddr=127.0.0.1`. `connect_test_database` kiểm tra lại môi trường và endpoint libpq ngay trước `psycopg.connect`, rồi kiểm tra `hostaddr` và port của connection vừa mở | `tests/test_p4b_isolation.py` |
| `apps/ai-service/tests/p4b_cleanup.py` | Cleanup chỉ chạy khi đã vào `finally` của pipeline. Lỗi cleanup có thể bị nuốt khi test đã fail, hoặc chạy lại nhiều lần | `OwnedResources` chỉ đăng ký action sau khi tài nguyên được tạo. `finish` chạy một lần, theo thứ tự ngược. Có lỗi cleanup và không có exception chính thì ném lỗi đó. Có cả hai thì `RuntimeWarning` `CLEANUP_FAILED` và giữ exception chính | `test_early_failure_cleans_only_resources_that_were_created`, `test_cleanup_failure_stays_visible_beside_the_test_error` |
| `apps/ai-service/tests/test_p4b_services.py` | `put_object` và `INSERT` nằm trước `try` | Object được đăng ký ngay sau upload. Row và point được đăng ký ngay sau commit, trước `run_pipeline`. Mọi connection của test đi qua `connect_test_database` | Hai lần storage integration, số point không tăng |
| `apps/ai-service/tests/test_p4b_isolation.py` | Chưa có regression cho env override và fault injection | Chứng minh URL chưa khóa bị `PGHOSTADDR=127.0.0.2` và service file `hostaddr=127.0.0.3` đổi đích. Guard từ chối các biến đó. URL đã khóa vẫn ra `127.0.0.1:5434` | 10 passed, exit 0 |
| `docs/p4b-runbook.md` | Runbook chỉ mô tả chặn query string | Thêm contract chặn biến môi trường libpq và `hostaddr` đã khóa | Không phải test |
| `docs/PHASE_1_3_ISOLATION_FINAL_REPORT.md` | Báo cáo vòng trước chưa biết hai finding này | Thêm ghi chú trỏ sang báo cáo này. Số liệu vòng DSN query, cổng loopback và cleanup generation được giữ nguyên | Không phải test |

Query string cũ vẫn bị chặn: `host`, `hostaddr`, `port`, `dbname`, `service`, `servicefile`, `sslmode` lặp, nhiều host và Unix socket. Thiếu cấu hình và cấu hình production-like vẫn fail closed. Các unit test này không mở kết nối tới database development.

## B. Test results

| Test | Command | Passed | Failed | Skipped | Exit code |
|---|---|---|---|---|---|
| Isolation và fault injection | `pytest -v tests/test_p4b_isolation.py` | 10 | 0 | 0 | 0 |
| Storage integration | `pytest -q tests/test_p4b_services.py` | 2 | 0 | 0 | 0 |
| Storage integration lần 2 | cùng lệnh | 2 | 0 | 0 | 0 |
| Python, lần chạy song song với Go và web | `pytest -q` | 182 | 1 | 2 | 1 |
| Python, chạy lại một mình | `pytest -q` | 183 | 0 | 2 | 0 |
| Go | `gofmt -l .`, `go vet ./...`, `go test ./... -count=1`, `go test -race ./...` | ok | 0 | 0 | 0 |
| Web | `npm run test:ci` | 18 | 0 | 0 | 0 |

Hai test bị skip trong full pytest là `test_p4b_services.py`, vì `CAS_P4B_INTEGRATION` được gỡ. Chúng được chạy riêng trên stack test.

Lần pytest exit 1 fail ở `test_qdrant_cancel_stops_an_inflight_request`: `box.get("code")` còn `None` sau `thread.join(1)`. File test và `app/indexing/qdrant.py` không nằm trong bản vá. Lệnh chạy cùng lúc với `go test -race` và Vite. Chạy riêng `tests/test_p4b_deadline.py` được 16 passed, exit 0. Full pytest chạy lại một mình được 183 passed, exit 0.

## C. Evidence

`artifacts/phase_1_3_final_patch/MANIFEST.tsv`, SHA-256 `25ff8b869947098fdd59111c2429708e941f8a84720634ca4606cd4729427a72`.

| File | Started | SHA-256 |
|---|---|---|
| `isolation-env.txt` | `2026-10-13T07:24:39Z` | `a043223032b11ce9fc9c86b0b984d1322f2fdc3f75ae242ccf4e1afa0f01e6a9` |
| `storage-integration.txt` | `2026-10-13T07:24:23Z` | `6f55ed435b20534e28360d9b54efa519484356a765d49123db056218bd3c0935` |
| `storage-integration-repeat.txt` | `2026-10-13T07:24:24Z` | `7bc79d9eeb5e880c0df50d1323542e51d04b7575d033f91ebad32aa8917f4125` |
| `docker-ports-and-counts.txt` | `2026-10-13T07:24:26Z` | `4294827ebb0be364de15a62fe8ba7f7a481e4b4f16b78f632029ed7de1b54666` |
| `pytest.txt` | `2026-10-13T07:21:18Z` | `9be2e7fc58a93b9b7fa2741a86e6daedca13b032ef08b49e68941fb550f8517c` |
| `pytest-rerun.txt` | `2026-10-13T07:23:33Z` | `9c04d0d16d3ad1b2bf0e98c0e5f84b126cc725922dc1fa080ada3e281f01dc77` |
| `go.txt` | `2026-10-13T07:21:18Z` | `32e4f26bebebf9d677cc83eac710e4cd61db35aa40e908af84d5a42a42d88687` |
| `web-test.txt` | `2026-10-13T07:21:19Z` | `244f3c8588202b5fc51c566ba544b73f5925011470112874dafe6a459202df4b` |

Cổng đang chạy chỉ là `127.0.0.1:5434->5432`, `127.0.0.1:9010->9000`, `127.0.0.1:9011->9001`, `127.0.0.1:6335->6333`. Compose config có `host_ip: 127.0.0.1` cho cả bốn cổng. Volume không bị xóa.

Trước và sau các lần storage integration, kể cả hai lần ghi vào log:

| Nguồn | Trước | Sau |
|---|---|---|
| Test documents | 0 | 0 |
| `p4b_test` | 2 | 2 |
| `p4b_partial` | 0 | 0 |
| `p4b_acknowledged` | 3 | 3 |
| `p4b_dimension` | 0 | 0 |
| `p4b_missing` | 6 | 6 |
| `p4b_restart` | 3 | 3 |
| `p4b_duplicate` | 6 | 6 |
| Dev documents | 163 | 163 |
| Dev `knowledge_chunks` | 2 | 2 |

Collection `knowledge_chunks` không tồn tại trên Qdrant test. Các point `p4b_*` ở bảng trên có từ trước bản vá cleanup và không bị xóa.

## D. Baseline

`PHASE_1_3_BASELINE_READY=true`. Hai finding đã có regression test. Guard từ chối env override và query override đã biết. Storage integration chạy trên stack riêng và PASS hai lần. Cleanup không làm tăng tài nguyên. Go, web và pytest chạy lại không có regression của Phase 1–3.

Giới hạn còn lại, không thuộc hai finding này: user Postgres của stack test vẫn là `cas` do image tạo. Phạm vi của nó là instance và DSN đã khóa, chưa phải role bị cắt quyền. Production, RunPod S3 và so sánh FP32/INT8 vẫn `NOT_RUN`.

URL test không có query string bị `lock_database_url()` ghép `&hostaddr` sai. Lỗi đó được sửa ở vòng sau và ghi trong `docs/PHASE_1_3_PRODUCTION_HARDENING_REPORT.md`.
