# Phase 1–3 completion report

Ngày chạy: 2026-10-12. Máy: macOS Apple Silicon ARM64, Docker Desktop linux/arm64. Stack local của lần chạy này là Compose project `citizen-assistance`, API `http://127.0.0.1:8082`, web `http://127.0.0.1:5174`. Process `make api-run` / Vite đang có trên 8080 và 5173 không bị tắt.

`deploy/.env` là override local, gitignored. Mặc định trong `deploy/.env.example` vẫn tắt ingestion và indexing. Không commit `.env`, weight, hay checkpoint. Không sửa migration 000001–000015. Không reset database `citizen_assistance` và không `down -v`.

Phase 4–8 chưa làm. Activate, RAG retrieve, và speech không được thêm. Domain chat công dân vẫn là `ho_tich_chung_thuc`.

## Kết quả

| Hạng mục | Kết quả | Bằng chứng |
|---|---|---|
| Phase 1 admin upload + index mock | PASS | UI upload trên 5174; PDF lên MinIO và Postgres; link rồi index mock về `READY` |
| Phase 2 tách câu tiếng Việt | PASS | Câu mẫu chỉ còn thiếu `nguoi_di_dang_ky`; `go test` và `go test -race` exit 0 |
| Phase 3 index model thật | PASS | PDF chữ và PDF scan ghi Postgres + Qdrant dim 384, checksum fp32 đã đối với file |
| Script `scripts/p4b-model-smoke.py` | PASS | Lần đầu exit 1 vì OCR ảnh vạch. Lần sau, fixture chữ, container in `P4B_DONE`, exit 0 |
| fp32 so với int8 | NOT_RUN | Đồ thị AVX-512 VNNI không chạy trên ARM64 |
| RunPod S3 | NOT_RUN | Không có credential, không gọi S3 thật |
| Deploy production | NOT_RUN | Chỉ stack local |
| GitHub Actions | NOT_RUN | Không chạy workflow |

Khối dưới đây là trạng thái sau lần sửa fixture. Lần chạy trước, smoke OCR trang vạch và in `P4B_MODEL_SMOKE_FAIL ocr_empty`. `P4B_DONE` chỉ là marker của script smoke, không phải production-ready.

```
PHASE_1_STATUS=PASS
PHASE_2_STATUS=PASS

PHASE_3_REAL_INDEXING=PASS
P4B_E5_MODEL_SMOKE=PASS
P4B_OCR_MODEL_SMOKE=PASS
P4B_MODEL_SMOKE=PASS
P4B_DONE=true

P4B_STORAGE_INTEGRATION=PASS
P4B_REAL_MODEL_INTEGRATION=PASS
P4B_OCR_QUALITY_EVALUATION=MEASURED

EMBEDDING_FP32_INT8_EQUIVALENCE=NOT_RUN
RUNPOD_S3_LIVE_SMOKE=NOT_RUN
GITHUB_ACTIONS=NOT_RUN
PRODUCTION_DEPLOYMENT=NOT_RUN
APPLE_SILICON_VARIANT=fp32
```

## Phase 1

Cờ local: `ADMIN_INGESTION_ENABLED`, `VITE_ADMIN_INGESTION`, `ADMIN_INDEXING_ENABLED`, `VITE_ADMIN_INDEXING` đều `true`. Web image build với hai arg Vite. Trang `http://127.0.0.1:5174/admin/documents` hiện form upload và danh sách, trong đó `Huong dan khai sinh phase1` ở trạng thái `READY`.

Index mock ban đầu 422 vì body Go gửi kèm UUID rỗng và `page_range` rỗng. Pydantic `extra=forbid` của `index.v1` từ chối. Lỗi bị ghi thành `worker_mismatch` vì response rỗng bị so ID. Sửa: marshal đúng field của từng contract, bỏ page range rỗng, và chỉ gán `worker_mismatch` khi worker trả 200 nhưng ID lệch.

Sau khi rebuild API, retry document `68931913-8648-45a0-a74c-c9737533608b` trả `link_status=READY`, job `9402ab52-dec1-4496-80fe-05e273f2fb66`.

| Case | HTTP | Code |
|---|---|---|
| File không phải PDF | 422 | `only .pdf files are accepted` |
| Công dân upload | 403 | `FORBIDDEN` |
| Document không tồn tại | 404 | `document not found` |
| Trùng checksum | 409 | `DOCUMENT_DUPLICATE` |
| PDF lớn hơn 20 MiB | 413 | `PAYLOAD_TOO_LARGE` |
| Retry khi đã READY | 422 | `document status does not allow this action` |
| Link sang domain khác | 422 | `document and procedure version do not match` |
| MinIO đang stop | 500 | `upload failed` |
| Upload sau khi start lại MinIO | 201 | lưu được |
| Tải content PDF đã upload | 200 | body là PDF, không phải JSON |

MinIO chỉ `docker stop` rồi `docker start`. Volume không bị xóa.

## Phase 2

`clauseSplit` không dùng `\b`. Dấu phẩy, chấm phẩy, xuống dòng, và `va`/`và` đứng riêng là chỗ tách. `vàng` và `advantage` không bị cắt vì hai bên phải là khoảng trắng.

Chat trên API 8082:

- `Đăng ký khai sinh` → `ASK_MISSING_SLOTS`, thiếu `noi_sinh`, `da_ket_hon`, `co_giay_chung_sinh`.
- `Ở Gia Lai, đã kết hôn rồi và không có giấy` → `ASK_MISSING_SLOTS`, chỉ thiếu `nguoi_di_dang_ky`.

`AI_EXTRACT_ENABLED` vẫn `false`. Contract `POST /v1/extract` không đổi. Keyword vẫn là đường trả lời khi chưa có provider LLM.

## Phase 3

Biến thể được chọn theo CPU, không fallback sang fake:

- Đủ cờ AVX-512 VNNI: `onnx/model_qint8_avx512_vnni.onnx`, SHA-256 `dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88`.
- `arm64` hoặc `aarch64`: `onnx/model.onnx` fp32 của cùng revision `6a0d452a575215f80b8f66276dd4ee5d504942c6`, SHA-256 `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`.
- CPU khác không đủ VNNI: `embedding_cpu_unsupported`.

Tokenizer `onnx/tokenizer.json` SHA-256 `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`. Dim 384. Prefix `passage: ` và `query: ` giữ nguyên. Inference trên host ARM64 trước khi nhận file: dim 384, norm 1.0, variant `fp32`.

Paddle trong image linux/arm64 import được: `paddlepaddle` 3.0.0, `paddleocr` 2.9.1. Weight lấy từ URL mà package đang cài dùng cho `lang=vi` (Latin PP-OCRv3 + det tiếng Anh PP-OCRv3 + cls mobile v2). File nằm trong `models/ocr/`, gitignored, có `manifest.json` SHA-256 từng file. Không tải model lúc process khởi động.

`INDEX_MODE=pipeline`, `EMBEDDING_PROVIDER=onnx`, `OCR_PROVIDER=paddle`. Claim TTL 180 giây, timeout index 120 giây. Qdrant `knowledge_chunks` tạo mới, size 384, cosine, 0 point trước khi index. Không trộn vector fake.

| PDF | Document | Kết quả ghi |
|---|---|---|
| Chữ tiếng Việt | `e8e7aae2-9efa-418d-8924-1fe156e8fbb1` | native 1, OCR 0, chunk 1, vector 1, checksum fp32 đủ 64 hex |
| Ảnh scan | `de95dc2d-7165-4547-81f5-1108a85a9ff8` | native 0, OCR 1, chunk 1, vector 1, cùng checksum fp32 |

Chunk chữ: `Hướng dẫn khai sinh tại Gia Lai. Công dân đã kết hôn và không có giấy chứng sinh thì người đi đăng ký là ông bà.` Cột `knowledge_chunks.embedding` là `vector(1536)` và để null. Vector thật nằm ở Qdrant. Sau hai lần index, collection có 2 point.

OCR trang scan đọc được `Giäy khai sinh s 123 Gia Lai`. Model Latin không giữ đủ dấu tiếng Việt. Đây là kết quả OCR thật, không phải fake.

`scripts/p4b-model-smoke.py` trong container, lần fixture vạch: embedding đi qua bước kiểm tra, rồi OCR trang 2 của `tests/fixtures/mixed.pdf` rỗng. Trang 2 render thành ảnh 167×417. Trang 1 của cùng file có 47 ký tự. Script exit 1. Lần đó không đổi script để ép `P4B_DONE`. Fixture chữ và exit 0 nằm ở mục sau.

Hai test `tests/test_p4b_services.py` vẫn skip. Test pipeline dùng `FakeEmbedder`, nên không chạy trên collection vừa ghi vector thật.

## Model artifact

| File | Nguồn | SHA-256 |
|---|---|---|
| `models/embedding/model.onnx` | Hugging Face `intfloat/multilingual-e5-small` @ `6a0d452a575215f80b8f66276dd4ee5d504942c6`, `onnx/model.onnx` | `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665` |
| `models/embedding/tokenizer.json` | cùng revision, `onnx/tokenizer.json` | `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39` |
| `models/ocr/det`, `rec`, `cls`, `ppocr_keys_v1.txt` | URL trong PaddleOCR 2.9.1 cho `lang=vi`; dict Latin của package | từng file trong `models/ocr/manifest.json` |

Đồ thị int8 VNNI không tải xuống vì máy này không chạy được. So sánh số với fp32: NOT_RUN. Không dùng chung một collection cho hai biến thể.

## File đã sửa

- `apps/api/internal/decision/extract.go`, `engine_test.go`
- `apps/api/internal/index/client.go`, `client_test.go`, `service.go`, `service_test.go`
- `apps/api/internal/repository/index.go`
- `apps/ai-service/app/indexing/embed.py`, `runtime.py`
- `apps/ai-service/tests/test_p4b_hardening.py`
- `scripts/p4b-model-smoke.py` (checksum theo variant đang chọn)
- `.gitignore` thêm `/models/`

`deploy/.env` và `models/` không vào git.

## Lệnh kiểm tra

| Lệnh | Exit |
|---|---|
| `gofmt -l apps/api` | 0, không file |
| `go vet ./...` trong `apps/api` | 0 |
| `go test ./... -count=1` trong `apps/api` | 0 |
| `go test -race ./... -count=1` trong `apps/api` | 0 |
| `python -m compileall -q app && pytest -q` trong `apps/ai-service` | 0, 172 passed, 2 skipped |
| `npm ci && npm run test:ci` trong `apps/web` | 0, 18 tests, bundle hygiene PASS cả hai chế độ cờ |
| `docker compose --env-file .env --profile app --profile storage --profile ai --profile qdrant config` | 0 |

## Chạy lại local

Từ `deploy/`:

```
docker compose --env-file .env --profile app --profile storage --profile ai --profile qdrant up -d --build
```

`.env` local hiện tại là `INDEX_MODE=pipeline`. Weight phải có ở `models/embedding` và `models/ocr` trước khi AI service `/ready`. Web phải build lại nếu đổi `VITE_ADMIN_INGESTION` hoặc `VITE_ADMIN_INDEXING`. Admin local dùng tài khoản đã có trong database, không chạy lại `seed-demo`.

## Smoke fixture và cô lập test, 2026-10-12

Lần smoke trước đọc trang 2 của `mixed.pdf`. Trang đó là ảnh vạch 64×16, không có chữ. PaddleOCR trả rỗng là đúng với ảnh đó. `mixed.pdf` giữ nguyên vai trò fixture âm: OCR rỗng không bị đổi thành chữ hành chính.

Fixture mới `apps/ai-service/tests/fixtures/ocr_text_scan.pdf` là một trang ảnh, không có text layer. Nội dung vẽ sẵn: `GIAY KHAI SINH`, `GIA LAI`, `SO 123`. Script smoke render trang 1, chạy E5 fp32 và PaddleOCR 2.9.1 trong container ARM64, rồi đối token. Không in nội dung OCR ra stdout.

Lệnh trong container `citizen-assistance-ai-service-1`, model dir `/models/embedding` và `/models/ocr`:

```
python /opt/cas/scripts/p4b-model-smoke.py
```

Stdout: `P4B_DONE embedding_dim=384 ocr_chars=29 image_shape=(634, 1317, 3)`. Exit 0. Log: `artifacts/phase_1_3_ocr/model-smoke.txt`.

Hai test `tests/test_p4b_services.py` vẫn dùng `FakeEmbedder` và `FakeOCR`. Chúng từ chối `citizen_assistance`, bucket `cas-documents`, MinIO `127.0.0.1:9000`, và Qdrant `127.0.0.1:6333` trước khi ghi. Chạy trên Compose project `citizen-assistance-p4b-test`: Postgres `127.0.0.1:5434` database `citizen_assistance_p4b_test`, MinIO `127.0.0.1:9010` bucket `cas-p4b-integration-test`, Qdrant `127.0.0.1:6335`.

```
CAS_P4B_INTEGRATION=1 pytest -q tests/test_p4b_services.py
```

2 passed, exit 0. Đây là `P4B_STORAGE_INTEGRATION`, không phải inference model thật. Sau khi chạy, database development vẫn 163 document và `knowledge_chunks` vẫn 2 point. Database test còn 0 document. Log: `artifacts/phase_1_3_ocr/storage-integration.txt`.

Index model thật của hai PDF `e8e7aae2-9efa-418d-8924-1fe156e8fbb1` và `de95dc2d-7165-4547-81f5-1108a85a9ff8` không chạy lại. `pipeline.py` không đổi trong lần này.

### Chất lượng OCR

Ground truth là chữ đã vẽ, không lấy từ output OCR. Model và dict không đổi. CER là edit distance chia số ký tự ground truth. WER là edit distance theo từ. Cột đã bỏ dấu chỉ để thấy phần lỗi không phải dấu.

| Fixture | Ground truth | OCR | CER | WER | CER đã bỏ dấu | Nhận xét |
|---|---|---|---|---|---|---|
| `ocr_text_scan.pdf` | `GIAY KHAI SINH` / `GIA LAI` / `SO 123` | Khớp từng dòng | 0.000 | 0.000 | 0.000 | 29 ký tự, không dấu |
| `ocr_plain_fields.pdf` | `GIAY KHAI SINH`, `Noi dang ky: Gia Lai`, `So: 123`, `Ngay sinh: 15/08/2025` | Khớp từng dòng | 0.000 | 0.000 | 0.000 | Số và ngày giữ đúng |
| `ocr_vi_fields.pdf` | `GIẤY KHAI SINH`, `Nơi đăng ký: Gia Lai`, `Số: 123`, `Ngày sinh: 15/08/2025` | `GIAY KHAI SINH`, `Noi däng ky: Gia Lai`, `S6: 123`, `Ngay sinh: 15/08/2025` | 0.108 (7 sửa) | 0.462 (6 từ) | 0.031 | Mất dấu. `Số` thành `S6`. Ngày `15/08/2025` và `123` còn đúng |

Ba trang cộng thời gian load model khoảng 2,7 giây trong một process. Không có ngưỡng CER để chấm smoke. Demo V1 nên dùng PDF có lớp chữ khi cần giữ dấu tiếng Việt. Kết quả này không phải độ chính xác pháp lý.

### Kiểm tra lại

| Lệnh | Exit | Kết quả |
|---|---|---|
| `compileall` và `pytest -q` trong `apps/ai-service` | 0 | 178 passed, 2 skipped |
| `pytest -q tests/test_p4b_services.py` với stack test | 0 | 2 passed, FakeEmbedder/FakeOCR |
| `gofmt -l`, `go vet`, `go test ./... -count=1`, `go test -race ./...` trong `apps/api` | 0 | không file gofmt |
| `npm run test:ci` trong `apps/web` | 0 | 18 tests, bundle hygiene PASS |
| `npm run build` trong `apps/web` | 0 | |
| `docker compose` config stack dev và `docker-compose.p4b-test.yml` | 0 | api, web, minio, postgres, qdrant, ai-service vẫn healthy |

## Hardening 2026-10-13

Guard ngày 2026-10-12 chỉ cấm vài tên development. Review độc lập chỉ ra một endpoint production khác tên đó vẫn được chấp nhận. Guard mới chỉ nhận Compose project `citizen-assistance-p4b-test` với đúng host, cổng, database, bucket, URL Qdrant và collection test. `CAS_P4B_INTEGRATION=1` một mình không đủ. Chi tiết và log chạy lại nằm ở [`PHASE_1_3_FINAL_BASELINE_REPORT.md`](PHASE_1_3_FINAL_BASELINE_REPORT.md).

Log `artifacts/phase_1_3_ocr/` của ngày 2026-10-12 nằm trên máy local và bị `.gitignore` loại khỏi ZIP review. Chúng không bị dựng lại từ báo cáo. Lần chạy 2026-10-13 ghi log mới trong `artifacts/phase_1_3_final_evidence/`.
