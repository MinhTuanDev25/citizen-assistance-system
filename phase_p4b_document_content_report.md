# P4B Document Content Indexing — bundle (12)

Trạng thái tối đa của vòng này:

```text
P4B_CODE_READY_FOR_RUNPOD_AND_REAL_MODEL_SMOKE
P4B_MODEL_SMOKE_PENDING
RUNPOD_S3_LIVE_SMOKE_PENDING
WORKFLOW_NOT_RUN
```

Không ghi `P4B_DONE`, `PRODUCTION_READY`, `MODEL_SMOKE_PASS`, `RUNPOD_S3_LIVE_SMOKE_PASS`, hay `WORKFLOW_PASS`.

SHA-256 của ZIP chỉ nằm ở sidecar `phase_p4b_document_content_bundle(12).zip.sha256`, tính sau khi ZIP đã đóng. Digest không được nhúng vào file này vì file này nằm trong ZIP.

## 1. Shutdown có thể treo

`IndexGate.shutdown(timeout_s)` join recovery thread theo deadline, rồi gọi thêm `thread.join()` không timeout. Recovery bị kẹt làm `shutdown(timeout_s=5)` không trả về.

## 2. Join có hạn

Vòng `join()` không giới hạn đã được xóa. Tổng thời gian chờ các recovery thread nằm trong deadline `time.monotonic() + timeout_s`.

Recovery thread được tạo với `daemon=True`. Nếu thread còn sống khi deadline hết, shutdown không chờ tiếp, đánh dấu dirty, chạy lượt kill/reap cuối, và trả về với status `failed`.

`_closed` vẫn được công bố dưới lifecycle lock trước khi join. Thread đăng ký trước snapshot vẫn nằm trong snapshot. `proc.start()` vẫn nằm trong critical section với `_closed` và việc ghi process vào slot. Lock không được giữ trong lúc join. Thread còn sống sau deadline không start được process vì `_commit_spawn()` kiểm tra lại `_closed` dưới lock. Status không quay lại `ready` sau khi `_closed` là true. Exception của `proc.start()` vẫn đóng cả hai đầu pipe và không để slot ready.

## 3. Race test

Test mới không spawn process thật. Recovery đã đăng ký bị chặn bằng `Event` bên trong recovery. `shutdown(timeout_s=0.05)` trả về trong khoảng bounded, `_closed` là true, status là `failed`, không có `proc.start()` sau thời điểm trả về, và status không quay lại `ready`. Event chỉ được thả sau các assertion đó.

Các test bundle (11) vẫn chạy: chặn ngay trước `proc.start()`, chặn sau khi đăng ký và trước `thread.start()`, 50 lần kill leader rồi shutdown, và shutdown lần hai không spawn thêm.

Container supervision pin `pytest==8.3.4` và chạy các test này.

## 4. Lệnh đã chạy

| Lệnh | Exit | Kết quả |
|---|---|---|
| `pytest -q --tb=line` trong `apps/ai-service` | 0 | 171 passed, 2 skipped, 35.65s |
| `python -m compileall -q app tests` | 0 | PASS |
| `bash scripts/p4b-container-supervision-test.sh` | 0 | `pytest==8.3.4`, 26 passed in 25.50s, `ps=absent` |
| `gofmt -l .` trong `apps/api` | 0 | không có file lệch format |
| `go test ./...` | 0 | PASS |
| `go vet ./...` | 0 | PASS |
| `go test -race ./...` | 0 | PASS |
| Go integration reaper trên `citizen_assistance` | 0 | `ok .../internal/index 2.809s`. Không down database |
| `npm ci && npm run test:ci && npm run build` | 0 | Vitest 18 passed; cả hai bundle hygiene PASS; vite build PASS |
| compose config và build profile app, ai, storage, qdrant | 0 | api, web, ai-service built |
| `python3 scripts/p4b-model-smoke.py` | 2 | `P4B_MODEL_SMOKE_PENDING` |

## 5. Test bị skip

Hai test trong `tests/test_p4b_services.py` skip vì `CAS_P4B_INTEGRATION` không được đặt. Không tính là PASS:

- `test_pipeline_writes_postgres_and_qdrant_from_minio`
- `test_qdrant_faults_against_a_real_server`

## 6. Model, S3, workflow còn pending

`P4B_MODEL_SMOKE_PENDING`. Không có manifest embedding và OCR trên máy. Không tải weight.

`RUNPOD_S3_LIVE_SMOKE_PENDING`. Biến endpoint, access key, secret key, bucket và region đều unset.

`WORKFLOW_NOT_RUN`. Job `ai-supervision` có trong workflow, nhưng GitHub Actions chưa chạy workflow đó trong phiên này.
