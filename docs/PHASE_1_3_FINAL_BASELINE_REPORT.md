# Phase 1–3 final baseline

Ngày chạy lại: 2026-10-13. Máy: macOS Apple Silicon, `uname -m` = `arm64`. Container AI: Linux `aarch64`, onnxruntime 1.20.1, paddle 3.0.0, paddleocr 2.9.1. Go trên host: `go1.26.3 darwin/arm64`. Project yêu cầu Go 1.25 trong `apps/api/go.mod`.

Đây là lần chạy lại trên cùng máy local, không phải xác nhận của máy review độc lập. Weight vẫn không vào Git. Log nằm ở `artifacts/phase_1_3_final_evidence/`, bị `.gitignore` loại, có `MANIFEST.tsv` ghi SHA-256 từng file.

```
PHASE_1_STATUS=PASS
PHASE_2_STATUS=PASS
PHASE_3_LOCAL_STATUS=PASS

P4B_MODEL_SMOKE=PASS
P4B_DONE=true
P4B_STORAGE_INTEGRATION=PASS
P4B_REAL_INDEXING=PASS

INTEGRATION_ISOLATION_GUARD=PASS
RUNBOOK_SYNCHRONIZED=PASS
TEST_EVIDENCE_COMPLETE=PASS

FP32_INT8_EQUIVALENCE=NOT_RUN
RUNPOD_S3_LIVE_SMOKE=NOT_RUN
GITHUB_ACTIONS=NOT_RUN
PRODUCTION_DEPLOYMENT=NOT_RUN
```

`P4B_DONE=true` vì script smoke in `P4B_DONE` và exit 0 trong lần chạy 2026-10-13. Marker này không có nghĩa production-ready, RunPod S3, GitHub Actions, hay FP32 tương đương INT8. `P4B_REAL_INDEXING=PASS` là kết quả index hai PDF ngày 2026-10-12. `pipeline.py` không đổi trong vòng này nên không index lại.

## Ba phase đang ở đâu

Phase 1: admin upload, MinIO, PostgreSQL, link và mock index đã chạy local. Admin đã có, không chạy `seed-demo`.

Phase 2: tách câu `và` giữ nguyên. `go test ./... -count=1` ngày 2026-10-13 exit 0. `go test -race` không chạy lại trong vòng này.

Phase 3: E5-small fp32, chiều 384, PaddleOCR 2.9.1 trên container ARM64. PDF chữ và PDF scan đã index ngày 2026-10-12. OCR có dấu vẫn hạn chế: fixture `ocr_vi_fields.pdf` CER 0.108, WER 0.462, `Số` thành `S6`. Demo chính nên dùng PDF có lớp chữ.

## Isolation guard

`isolated_p4b_targets()` chỉ chấp nhận đúng bộ đã chốt trong `deploy/docker-compose.p4b-test.yml`:

- Compose project `citizen-assistance-p4b-test`
- Postgres `127.0.0.1:5434` hoặc `localhost:5434`, database `citizen_assistance_p4b_test`
- MinIO `127.0.0.1:9010` hoặc `localhost:9010`, bucket `cas-p4b-integration-test`
- Qdrant `http://127.0.0.1:6335` hoặc `http://localhost:6335`
- Collection `p4b_test`, `p4b_partial`, `p4b_acknowledged`, `p4b_dimension`, `p4b_missing`, `p4b_restart`, `p4b_duplicate`

`prepare_isolated_target()` kiểm tra chuỗi đó trước. Chỉ khi chuỗi khớp mới kiểm tra `docker ps` của project test có publish `5434`, `9010`, `6335`, và `SELECT current_database()` trả về `citizen_assistance_p4b_test`. Guard fail thì hàm inspect không được gọi, nên không có kết nối và không có ghi.

Unit test từ chối database `citizen_assistance`, bucket `cas-documents`, MinIO cổng 9000, Qdrant cổng 6333, collection `knowledge_chunks`, URL `prod-db.internal` / `storage.prod.internal` / `https://qdrant.prod.internal`, database có chữ `test` nhưng host khác, đúng tên database nhưng cổng 5432, và biến `CAS_P4B_INTEGRATION=1` khi thiếu tài nguyên.

## Bằng chứng lần chạy 2026-10-13

| Test | Command | Environment | Passed | Failed | Skipped | Exit | Evidence |
|---|---|---|---|---|---|---|---|
| Model smoke | `python /opt/cas/scripts/p4b-model-smoke.py` | container `citizen-assistance-ai-service-1`, fp32 SHA `ca456c06…`, tokenizer `0b44a9d7…` | smoke in `P4B_DONE` | 0 | 0 | 0 | `model-smoke.txt` |
| Storage integration | `pytest -q tests/test_p4b_services.py` | Compose `citizen-assistance-p4b-test`, FakeEmbedder, FakeOCR | 2 | 0 | 0 | 0 | `storage-integration.txt` |
| Python suite | `compileall` rồi `pytest -q` | host venv 3.9, không bật integration | 178 | 0 | 2 | 0 | `pytest.txt` |
| Go | `gofmt -l`, `go vet ./...`, `go test ./... -count=1` | `apps/api`, go1.26.3 | ok | 0 | 0 | 0 | `go-test.txt`, `go-vet.txt`, `gofmt.txt` |
| Web | `npm run test:ci` | `apps/web` | 18 | 0 | 0 | 0 | `web-test.txt` |

Sau storage test: database development vẫn 163 document, `knowledge_chunks` vẫn 2 point, database test còn 0 document. `go test -race` không chạy lại. GitHub Actions không chạy.

Log ngày 2026-10-12 trong `artifacts/phase_1_3_ocr/` vẫn nằm trên máy này và đã bị loại khỏi ZIP vì `/artifacts/` nằm trong `.gitignore`. Lần này không chép nội dung báo cáo thành log mới. File trong `artifacts/phase_1_3_final_evidence/` là output của lệnh chạy lại.

## Câu hỏi chốt

1. Ba phase đạt local trên máy này: upload và mock index, tách câu, index model thật kèm smoke fp32. Chưa phải nghiệm thu production.
2. Guard của báo cáo này chặn đúng host, cổng và tên database theo `urlparse`. Cùng ngày, review chỉ ra query string của libpq vẫn đổi được đích kết nối. Bản vá và kết quả chạy lại nằm ở [`PHASE_1_3_ISOLATION_FINAL_REPORT.md`](PHASE_1_3_ISOLATION_FINAL_REPORT.md). Đoạn này giữ nguyên kết quả buổi sáng, không thay bằng lần vá sau.
3. Model smoke có log gốc exit 0 và dòng `P4B_DONE embedding_dim=384 ocr_chars=29 image_shape=(634, 1317, 3)` lúc `2026-10-13T03:15:05Z`.
4. Storage integration chạy trên project `citizen-assistance-p4b-test`, 2 passed. Đây là FakeEmbedder và FakeOCR.
5. Hạn chế còn lại: OCR dấu tiếng Việt, chưa so fp32 với int8, chưa RunPod S3, chưa GitHub Actions, chưa production. `go test -race` của vòng này chưa chạy lại.
6. Không có blocker trong pipeline P4B ngăn việc viết Phase 4 Activate. Không chạy integration test ghi dữ liệu nếu thiếu đúng bộ biến của stack test.
7. Blocker production nằm riêng: RunPod S3, TLS, backup, CPU x86 phải có AVX-512 VNNI nếu dùng graph int8, và Actions chưa xanh. Những mục đó không chặn code Activate trên stack local.
