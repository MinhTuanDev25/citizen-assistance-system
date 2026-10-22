# Các phase còn lại để xong V1

Ngày ghi: 2026-10-12. Cập nhật cùng ngày: phase 1, 2 và 3 đã chạy local và ghi tại [`PHASE_1_3_COMPLETION_REPORT.md`](PHASE_1_3_COMPLETION_REPORT.md). Ba phase đó PASS trên máy ARM64. Smoke model đã in `P4B_DONE` trên fixture chữ. Nền production của phase 1–3 được vá 2026-10-13 và ghi tại [`PRODUCTION_FOUNDATION_FINAL_REPORT.md`](PRODUCTION_FOUNDATION_FINAL_REPORT.md). Việc chức năng còn lại bắt đầu từ phase 4. RunPod S3, Actions, fp32/int8 và production vẫn chưa chạy.

Phần đã làm nằm ở [`tien-do.md`](tien-do.md). File này chỉ ghi việc còn lại, thứ tự làm, và thứ phải có trong tay trước mỗi phase: key, weight, checkpoint, cờ, tài khoản.

Xong phase 1 đến 5 thì cán bộ upload PDF, bật phiên bản, và người dân nhận checklist kèm đoạn trích từ đúng file đó. Phase 6 làm câu tự nhiên được nhận chắc hơn. Phase 7 là giọng Bahnar, làm cuối. Phase 8 là chạy demo và nghiệm thu.

Không cần key để viết code của phase 2, 4 và 5. Key và file model chỉ cần lúc bật chạy thật. Không commit key, weight, hay checkpoint.

| Phase | Việc | Thiếu để làm |
|---|---|---|
| 1 | Bật màn admin đã viết | Đã chạy local. Biên bản ở report phase 1–3 |
| 2 | Sửa nhận câu từ khóa | Đã chạy local. Câu mẫu chỉ còn hỏi người đi đăng ký |
| 3 | Index PDF bằng model thật | Đã index fp32 + Paddle trên ARM64. Smoke fixture chữ in `P4B_DONE`. RunPod S3 chưa chạy |
| 4 | Activate | Local acceptance 2026-10-13. `PHASE_4_STATUS=PASS` trên `cas_p4_test` + E5/Qdrant + browser UI. Chưa production-ready |
| 5 | RAG, citation từ PDF | Cùng weight embedding phase 3 |
| 6 | LLM extract thật | `OPENAI_API_KEY` hoặc `GEMINI_API_KEY` |
| 7 | Giọng Bahnar | Model đã chọn từ kết quả nghiên cứu, checkpoint đã đánh giá, inference thử trên audio Bahnar thật, rồi mới viết API |
| 8 | Demo và nghiệm thu | HTTPS, backup, Actions xanh, E2E chữ, E2E giọng Bahnar, ghi nhận Qdrant thay pgvector, UAT |

## Phase 1 — Bật màn admin đang có

Code upload, gắn link và index đã có. Cờ đang tắt nên UI hiện “chưa triển khai”.

Cần có:

- `ADMIN_INGESTION_ENABLED=true` và `VITE_ADMIN_INGESTION=true`
- `ADMIN_INDEXING_ENABLED=true` và `VITE_ADMIN_INDEXING=true`
- Build lại web. Vite gắn hai cờ `VITE_*` lúc build, sửa file env rồi refresh trình duyệt là chưa đủ
- Docker Compose profile `app`, `storage`, `ai`. Index pipeline thêm profile `qdrant`
- MinIO local: `OBJECT_STORAGE_ENDPOINT=minio:9000`, user và password trong `deploy/.env.example` (`minioadmin`). Không cần key RunPod cho bước này
- `AI_SERVICE_TOKEN` — chuỗi nội bộ Go gọi AI service, đã có giá trị local trong `.env.example`. Đây không phải key LLM
- Tài khoản admin đã có trong database local (`admin@chuse.vn`). Không chạy `seed-demo` trên database đang dùng. Máy mới chưa có user thì mới cần seed, và chỉ trên database trống.

`INDEX_MODE=mock` hoặc `pipeline` với `OCR_PROVIDER=fake` và `EMBEDDING_PROVIDER=fake` để thấy nút index mà chưa cần weight. Xong phase này là admin upload và index được trên máy local. Chat vẫn trả lời bằng JSON mẫu.

## Phase 2 — Sửa nhận câu từ khóa

Câu `Ở Gia Lai, đã kết hôn rồi và không có giấy` đang bị cắt thành hai mảnh vì chữ “và” không được tính là chỗ tách. Mảnh sau dính cả “đã kết hôn” và “không có”, nên ô kết hôn bị ghi là không và ô giấy chứng sinh bị bỏ.

Cần viết trong `apps/api/internal/decision/extract.go`. Không cần key, checkpoint, hay service ngoài. Xong phase này, câu một dòng theo kiểu trên điền đủ slot trên đường từ khóa hiện tại.

## Phase 3 — Index PDF bằng model thật

Pipeline đã viết. Thiếu file weight trên máy chạy AI service. Runtime không tự tải model.

Embedding, copy vào `EMBEDDING_MODEL_DIR` và không commit:

- Nguồn: `intfloat/multilingual-e5-small`, revision `6a0d452a575215f80b8f66276dd4ee5d504942c6`, chiều 384
- Máy ARM64, gồm Mac M4 và container Linux ARM64: file fp32 `onnx/model.onnx`, SHA-256 `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`. Không đổi tên file VNNI thành `model.onnx` trên máy này
- Linux x86_64: `onnx/model_qint8_avx512_vnni.onnx` chỉ khi CPU có đủ cờ AVX-512 VNNI, SHA-256 `dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88`. CPU x86 không có các cờ đó thì `embedding_cpu_unsupported`, không tự chuyển sang fp32
- `tokenizer.json`, SHA-256 `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`

OCR, copy vào `OCR_MODEL_DIR` và không commit:

- Weight PaddleOCR CPU: thư mục `det/`, `rec/`, `cls/`
- Thiếu file thì startup báo `ocr_model_missing`

Biến khi chạy thật:

- `INDEX_MODE=pipeline`
- `EMBEDDING_PROVIDER` và `OCR_PROVIDER` để trống để service bắt buộc đúng thư mục model. `fake` chỉ để thử, không phải model thật
- `INDEX_PIPELINE_TIMEOUT_SECONDS=120`, `INDEX_CLAIM_TTL_SECONDS=180`
- Qdrant: `QDRANT_URL`, `QDRANT_COLLECTION=knowledge_chunks`

RunPod S3 là kho file tùy chọn, không phải cửa trước OCR. Upload lưu PDF trước. Index tải PDF từ kho rồi mới OCR. Local dùng MinIO. Muốn kho RunPod thì điền trên máy chạy, không ghi vào git:

- `OBJECT_STORAGE_ENDPOINT=https://s3api-eu-ro-1.runpod.io` (đúng region đang có)
- `OBJECT_STORAGE_REGION`
- `OBJECT_STORAGE_BUCKET` = network volume id
- `OBJECT_STORAGE_ACCESS_KEY` và `OBJECT_STORAGE_SECRET_KEY`
- `OBJECT_STORAGE_USE_SSL=true`
- `OBJECT_STORAGE_AUTO_CREATE_BUCKET=false`
- Cổng 443. `http://` hoặc cổng khác làm service từ chối lúc khởi động

Xong phase này khi `scripts/p4b-model-smoke.py` đọc được weight local, và một PDF thật lên trạng thái `READY` trên Postgres cùng Qdrant. Hai test đang skip cần `CAS_P4B_INTEGRATION=1`. Chưa có activate và chưa có RAG.

## Phase 4 — Activate

Code đã có: API admin, nút kích hoạt trên `/admin/procedures`, candidate APPROVED, eligibility theo generation đã publish, transaction atomic, idempotency, audit, session pin. Chi tiết [`PHASE_4_ACTIVATION_COMPLETION_REPORT.md`](PHASE_4_ACTIVATION_COMPLETION_REPORT.md), vận hành [`PHASE_4_ACTIVATION_RUNBOOK.md`](PHASE_4_ACTIVATION_RUNBOOK.md).

`PHASE_4_CODE_COMPLETE=true`. `PHASE_4_STATUS=PASS` (local, isolated `cas_p4_test`). Integration A–H PASS. Real PDF native + E5-small 384 + Qdrant PASS. Browser UI hiển thị V2 ACTIVE / V1 ARCHIVED. PaddleOCR scanned PDF `NOT_RUN` (thiếu package). Không production-ready. Handoff retrieve: [`PHASE_4_ACTIVATION_RUNBOOK.md`](PHASE_4_ACTIVATION_RUNBOOK.md).

## Phase 5 — RAG và citation từ PDF

Chưa có `POST /v1/retrieve`. Khi lượt chat ra `DIRECT_ANSWER` hoặc `PROVIDE_FINAL_GUIDANCE`, Go gọi retrieve đúng `xa_id` và `procedure_version_id` đang ghim. Checklist JSON giữ nguyên. RAG chỉ thêm đoạn nguồn. Câu còn thiếu slot không gọi retrieve. Bubble chat hiện citation. Ghi `message_citations` theo chunk, document và page range.

Cần cùng bộ weight embedding phase 3, vì câu hỏi phải embed bằng đúng model đã embed PDF. Không cần key LLM cho retrieve. Qdrant đã có trong Compose. Thiếu chunk thì vẫn trả checklist JSON.

Xong phase 5 là chức năng “trả lời có căn cứ từ PDF chính thức” hoàn tất trên code.

## Phase 6 — LLM extract thật

`POST /v1/extract` đã có. Go chưa gọi vì `AI_EXTRACT_ENABLED=false`.

Để thử dây nối, không cần key: `AI_EXTRACT_ENABLED=true`, `LLM_PROVIDER=mock`, `AI_SERVICE_URL`, `AI_SERVICE_TOKEN`. Mock chỉ đúng các câu đã viết trong test, không hiểu mọi câu người dân.

Để nhận câu tự do trên model thật, đặt một key trên máy chạy, không commit:

- `LLM_PROVIDER=openai` và `OPENAI_API_KEY`, hoặc
- `LLM_PROVIDER=gemini` và `GEMINI_API_KEY`
- `LLM_MODEL` hợp lệ lúc startup
- Ngưỡng đã có: `AI_INTENT_SELECT_MIN=0.82`, `AI_INTENT_CONFIRM_MIN=0.55`, `AI_SLOT_CONFIDENCE_MIN=0.6`

Model lỗi, timeout hoặc hết hạn mức thì Go quay về từ khóa phase 2. Model không được sửa checklist hay tự đổi thủ tục đang ghim.

## Phase 7 — Giọng Bahnar

Làm sau cùng. Chưa có `POST /v1/speech/translate`, chưa có nút micro. Audio thành chữ Việt, rồi chữ đó vào turn như khi gõ. Không làm TTS.

API phục vụ chỉ được viết sau khi model đã được xác minh. Checkpoint mới nhất, hoặc checkpoint chưa qua đánh giá, không được lấy làm model cuối.

Thứ tự bắt buộc:

1. Xác định model được chọn từ kết quả nghiên cứu. Hai ứng viên là cascaded C0 (ASR + MT) và direct D0. Notebook 06 vẫn là placeholder. Notebook 06 chỉ mở sau khi NB03, NB04 và NB05 đều có best checkpoint bền và evaluate thành công (`SUCCESS_*_EVALUATE`), rồi chạy `verify → unlock_test → run_cascaded → run_direct → finalize` trên frozen test.
2. Kiểm tra đúng bộ đang được chọn: best checkpoint, config, tokenizer hoặc processor, và model weights. Điền đúng thư mục, không lấy bản mới nhất trên đĩa: `ASR_STATE_DIR` và `MT_STATE_DIR` cho C0, hoặc `DIRECT_STATE_DIR` cho D0. D0 là XLS-R 300M encoder và mBART-50 `vi_VN` decoder. File checkpoint nặng, để trên máy chạy model, không commit.
3. Chạy inference thử với audio Bahnar thật. Đầu ra phải là chữ Việt đọc được, trước khi có route hay nút micro.
4. Sau bước 3 mới xây API cho đúng model đã xác minh, gắn nút micro, và ghi `speech_translation_requests` cùng `model_versions` cho mỗi lượt nói.

Máy local không có GPU thì đặt service dịch giọng trên RunPod, cùng chỗ đang giữ checkpoint đã kiểm. Chat chữ không chờ phase này.

## Phase 8 — Demo và nghiệm thu

Cần:

- Một máy có HTTPS, health check, seed hộ tịch, một PDF `READY` đã activate
- Key LLM và key S3 chỉ trên server
- Backup Postgres hàng ngày cho bản demo
- GitHub Actions của nhánh hiện tại chạy xanh. File workflow đã có, lần chạy của vòng P4B chưa được ghi
- E2E chữ: upload, index, activate, chat đủ slot, bubble có citation đúng file
- E2E giọng Bahnar, đúng điều kiện nghiệm thu của proposal: audio Bahnar thật → Speech Translation → chữ Việt → Decision Engine → RAG → hướng dẫn hộ tịch tiếng Việt kèm citation từ PDF. Thiếu bài này thì chưa đủ bằng chứng nghiệm thu chức năng chính
- Ghi kiến trúc đang chạy: proposal mô tả PostgreSQL + pgvector, code P4B dùng PostgreSQL + Qdrant, chiều 384, model e5-small. Không viết lại P4B sang pgvector trong phase này. Cập nhật tài liệu kiến trúc triển khai thực tế, giải thích vì sao dùng Qdrant, và kiểm tra sự chấp thuận đối với thay đổi này
- UAT với cán bộ và người dân
- Metric nhận câu, slot, retrieval và citation. Performance và security theo tiêu chí nghiệm thu đã chốt

Xong phase 8 mới gọi là đã nghiệm thu. Xong phase 5 là đủ chức năng chữ và PDF. Xong phase 7 là đủ thêm giọng Bahnar trên code. Nghiệm thu proposal vẫn cần bài E2E giọng ở phase 8.
