# P4B — deadline tuyệt đối, bundle (5)

Trạng thái: `P4B_CODE_READY_FOR_REAL_MODEL_SMOKE`.

Không ghi `P4B_DONE`. Chưa có forward pass ONNX và PaddleOCR thật.

GitHub Actions: `WORKFLOW_NOT_RUN`.

Model smoke: `P4B_MODEL_SMOKE_PENDING`.

SHA-256 của ZIP nằm ở sidecar `phase_p4b_document_content_bundle.zip(5).sha256`, tính sau khi ZIP đã đóng. Hash không được ghi lại vào trong ZIP.

Không có retrieval, RAG, Activate, speech, hay model RQ1.

## File đã đổi so với bundle (4)

- `apps/ai-service/app/indexing/deadline.py` (mới)
- `apps/ai-service/app/indexing/qdrant.py`
- `apps/ai-service/app/indexing/embed.py`
- `apps/ai-service/app/indexing/ocr.py`
- `apps/ai-service/app/indexing/ocr_supervisor.py`
- `apps/ai-service/app/indexing/runtime.py`
- `apps/ai-service/app/indexing/pipeline.py`
- `apps/ai-service/app/indexing/postgres.py`
- `apps/ai-service/app/main.py`
- `apps/ai-service/tests/test_p4b_deadline.py` (mới)
- `apps/ai-service/tests/test_p4b_final.py`
- `apps/ai-service/tests/test_p4b_review.py`
- `apps/api/internal/index/migration_repair_test.go`
- `apps/api/internal/index/pipeline_integration_test.go`
- `docs/p4b-runbook.md`

Go reaper không đổi logic. Contract fail-closed của bundle (4) được giữ.

## Hành vi trước và sau

Trước đây mỗi stage nhận lại một lát của `INDEX_PIPELINE_TIMEOUT_SECONDS` đo từ đầu pipeline, và mỗi HTTP/SQL request dùng timeout đó như một ngân sách riêng. Một stage Qdrant bốn request có thể chạy lâu hơn tổng deadline. Nay mỗi request `index.v2` tạo đúng một `Deadline` bằng `time.monotonic() + INDEX_PIPELINE_TIMEOUT_SECONDS` trong handler, rồi truyền cùng object đó vào MinIO, PDF, OCR, đếm token, ONNX, Qdrant, PostgreSQL. `remaining()` hết giờ thì ném `IndexFailure("timeout")`. `asyncio.wait_for` chỉ chờ phần còn lại của cùng deadline đó.

`QdrantWriter` tính lại `timeout_for_io()` ngay trước từng request (ensure, từng batch upsert, count, fetch, delete, count sau delete). HTTP 404 khi delete không còn được coi là sạch. Thành công chỉ khi HTTP 200, JSON parse được, `result.status == completed`, và exact count bằng 0. Go reaper giữ cùng contract.

ONNX `_respawns` suốt đời process được thay bằng `_consecutive_failures`. Một operation được retry tối đa một lần, boot bằng budget còn lại. Request thành công thì counter về 0, nên sự cố độc lập sau đó vẫn recovery được. Crash liên tiếp không spawn vô hạn. `shutdown()` kill/join child mà không giữ pipe lock. Poll được cắt lát 50 ms để shutdown đánh thức caller đang chờ.

Sau timeout, pipeline không gọi thêm upsert hay insert. Cleanup sau upsert dở dùng một `Deadline` riêng 2 giây và vẫn phải chứng minh count 0. `IndexGate` từ chối request mới khi shutdown đã bắt đầu, và chỉ nhả slot khi worker kết thúc. Gate đóng thì `/ready` và `/v1/status` báo `pipeline_busy`.

## Deadline đi qua từng stage

Handler tạo `Deadline` và đưa vào `_run_v2` → `run_pipeline`. `objects.get` nhận cùng object. MinIO đặt connect, read, và total timeout bằng thời gian còn lại, rồi siết socket timeout trước mỗi block. OCR `read_page` lấy `timeout_for_io`. Worker thay thế chỉ được budget còn lại; hết budget thì supervisor sang `timed_out` và không spawn với timeout đầy. `count_tokens` và `embed_batch` nhận cùng deadline, kể cả nhiều lần đếm khi fit chunk. Thời gian chờ lock và IPC nằm trong deadline. Qdrant và PostgreSQL gọi `remaining()` trước từng request hoặc statement. `statement_timeout` được `set_config` lại trước UPDATE, DELETE, INSERT, và deadline được kiểm tra trước COMMIT. Hết giờ thì rollback.

## Dừng worker sau timeout

ONNX và OCR poll từng lát ngắn. Hết deadline thì đóng pipe, kill, join, rồi mới nhả slot. `IndexGate.shutdown` đánh dấu đóng, dừng executor, gọi `shutdown()` của model (kill/join, không lấy pipe lock), rồi chờ pending trong thời gian có trần. Request đã timeout không publish muộn vì claim hết hạn và cleanup claim vẫn chặn finish như bundle (4).

## Lệnh đã chạy

| Lệnh | Exit | Kết quả |
|---|---|---|
| `gofmt -w` trên file Go vừa sửa; `gofmt -l .` | 0 | không còn file lệch format |
| `go test ./...` trong `apps/api` | 0 | PASS |
| `go vet ./...` | 0 | PASS |
| `go test -race ./...` | 0 | PASS |
| `CAS_INTEGRATION=1 go test -tags integration -count=1 ./internal/index/` | 0 | PASS, gồm migration 16 và hai reaper |
| `pytest` trong `apps/ai-service` | 0 | 121 passed, 2 skipped |
| `python -m compileall app tests` | 0 | PASS |
| `npm ci` rồi `npm run test:ci` trong `apps/web` | 0 | 18 tests; bundle indexing OFF và ON PASS |
| `docker compose -f deploy/docker-compose.yml config` | 0 | PASS |
| `docker build -t cas-ai:p4b-review5 -f apps/ai-service/Dockerfile .` | 0 | PASS |
| `python3 scripts/p4b-model-smoke.py` | 2 | `manifest_verification=pending`, `runtime_compatibility=pending`, `forward_pass=pending`, `P4B_MODEL_SMOKE_PENDING` |

Hai test pytest bị skip vì chưa đặt `CAS_P4B_INTEGRATION=1`: live MinIO/PostgreSQL/Qdrant trong `tests/test_p4b_services.py`. Integration Go dùng PostgreSQL thật `citizen_assistance` và Qdrant giả trong test. Không down database. Không phải forward pass ONNX hay PaddleOCR.

## Limitations

Không có weight ONNX và PaddleOCR trên máy này. Máy là Apple Silicon, không có AVX512 VNNI, nên graph int8 không được load. GitHub Actions chưa chạy. Không có retrieval, RAG, Activate, speech, hay model RQ1. `LLM_PROVIDER=mock` cho local test.

P4B_CODE_READY_FOR_REAL_MODEL_SMOKE
