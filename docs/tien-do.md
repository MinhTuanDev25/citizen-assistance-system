# Tiến độ Citizen Assistance System

Ngày ghi: 2026-10-08. Bảng API cập nhật 2026-10-12. Các phase còn lại nằm ở [`phase-con-lai.md`](phase-con-lai.md). Kết quả chạy phase 1–3 ngày 2026-10-12 nằm ở [`PHASE_1_3_COMPLETION_REPORT.md`](PHASE_1_3_COMPLETION_REPORT.md): ba phase local PASS. Cùng ngày, smoke E5 và PaddleOCR trên fixture chữ in `P4B_DONE`. Marker đó không có nghĩa production-ready.

Repo có hai việc song song. Phần này nói **artifact trợ lý thủ tục cấp xã** (`apps/`, `deploy/`, `docs/`). Cuối file có một mục ngắn về **luận văn Bahnar → tiếng Việt** (`bahnar-s2tt-thesis/`). Hai việc dùng chung repo, không thay thế nhau.

2026-10-13: production Compose truyền S3 credentials cho cả API và AI, `/ready` không tạo collection Qdrant, job CI `p4b-live` trỏ stack test, index-reaper có vòng lặp cấu hình được. Foundation chưa deploy. Chi tiết ở [`PRODUCTION_FOUNDATION_FINAL_REPORT.md`](PRODUCTION_FOUNDATION_FINAL_REPORT.md).

Tài liệu này mô tả code đang có trong repo. Nó không phải biên bản nghiệm thu. `P4B_DONE` trong report phase 1–3 chỉ là exit 0 của `scripts/p4b-model-smoke.py`. RunPod S3, GitHub Actions, so sánh fp32/int8 và deploy production vẫn `NOT_RUN`.

## Phạm vi V1 đang làm

Mục tiêu artifact: một xã (Chư Sê), chat tiếng Việt, nhận thủ tục, hỏi slot còn thiếu, rồi đưa checklist lấy từ JSON thủ tục. Tài liệu PDF được upload, đọc, OCR trang yếu, cắt chunk và gắn vector. Khi đủ điều kiện, admin kích hoạt một phiên bản; câu trả lời cuối mới được trích dẫn đoạn trong PDF của đúng phiên bản đó. Giọng Bahnar thành chữ Việt gắn vào cùng luồng chat, sau khi model luận văn đã chọn.

V1 **cố ý không làm**:

- Workspace nháp LLM, sửa draft, review và approve nhiều bước.
- Rollback phiên bản bắt buộc.
- Mở đất đai và bảo hiểm cho công dân trên chat.
- Nộp hồ sơ, thanh toán, chữ ký số, TTS Bahnar, nhiều xã, HA.

Ba domain vẫn có trong cơ sở dữ liệu để admin thấy catalog. Chat công dân chỉ được đọc domain `ho_tich_chung_thuc`.

## Bảng API

Đường dẫn công dân và admin nằm trên API Go, prefix `/api/v1`. AI service chỉ nhận token nội bộ từ Go, không mở cho trình duyệt. Route admin chỉ được gắn khi bật cờ tương ứng. Mặc định cả hai cờ tắt.

| API | Chức năng | Trạng thái | Actor |
|---|---|---|---|
| `GET /health` | Báo API Go còn sống | Đã làm | Vận hành |
| `GET /ready` | Báo API Go sẵn sàng phục vụ | Đã làm | Vận hành |
| `GET /swagger/*` | OpenAPI của API Go | Đã làm | Dev, Admin |
| `POST /api/v1/auth/register` | Đăng ký tài khoản công dân | Đã làm | Khách |
| `POST /api/v1/auth/login` | Đăng nhập, cấp JWT | Đã làm | Công dân, Admin |
| `GET /api/v1/auth/me` | Xem hồ sơ đang đăng nhập | Đã làm | Công dân, Admin |
| `POST /api/v1/auth/logout` | Đăng xuất | Đã làm | Công dân, Admin |
| `GET /api/v1/communes` | Danh sách xã | Đã làm | Khách, Công dân, Admin |
| `GET /api/v1/communes/:id` | Chi tiết một xã | Đã làm | Khách, Công dân, Admin |
| `GET /api/v1/domains` | Danh sách lĩnh vực | Đã làm | Khách, Công dân, Admin |
| `GET /api/v1/domains/:id` | Chi tiết một lĩnh vực | Đã làm | Khách, Công dân, Admin |
| `POST /api/v1/domains` | Tạo lĩnh vực | Đã làm | Admin |
| `PUT /api/v1/domains/:id` | Sửa lĩnh vực | Đã làm | Admin |
| `DELETE /api/v1/domains/:id` | Xóa lĩnh vực | Đã làm | Admin |
| `GET /api/v1/procedures` | Catalog thủ tục. Khách và công dân chỉ thấy hộ tịch và chứng thực. Admin thấy cả ba lĩnh vực | Đã làm | Khách, Công dân, Admin |
| `GET /api/v1/procedures/by-code/:code` | Đọc thủ tục theo mã. Ngoài phạm vi: khách nhận 404, công dân nhận 403, admin đọc được | Đã làm | Khách, Công dân, Admin |
| `GET /api/v1/procedures/:id/active-version` | Lấy definition phiên bản đang ACTIVE. Cùng luật phạm vi với by-code | Đã làm | Khách, Công dân, Admin |
| `POST /api/v1/sessions` | Mở hội thoại cho một xã. Khách nhận guest token. User đã login gắn `user_id` | Đã làm | Khách, Công dân, Admin |
| `GET /api/v1/sessions/:sessionId/messages` | Đọc lịch sử hội thoại của đúng session | Đã làm | Chủ session |
| `POST /api/v1/sessions/:sessionId/messages` | Gửi tin theo route cũ | Đã đóng, trả 410. Dùng route turns | Chủ session |
| `POST /api/v1/sessions/:sessionId/turns` | Một lượt chat: nhận thủ tục, điền slot, hỏi phần thiếu hoặc trả checklist JSON. Idempotent theo `X-Request-ID` | Đã làm. Nhận câu bằng từ khóa. LLM extract chỉ chạy khi `AI_EXTRACT_ENABLED=true` | Khách, Công dân, Admin |
| `POST /api/v1/admin/documents` | Upload PDF vào MinIO và bảng `documents` | Đã làm. Cần `ADMIN_INGESTION_ENABLED` | Admin |
| `GET /api/v1/admin/documents` | Liệt kê PDF theo domain và trạng thái | Đã làm. Cần cờ ingestion | Admin |
| `GET /api/v1/admin/documents/:id` | Xem metadata một PDF | Đã làm. Cần cờ ingestion | Admin |
| `GET /api/v1/admin/documents/:id/content` | Tải lại file PDF | Đã làm. Cần cờ ingestion | Admin |
| `GET /api/v1/admin/documents/link-targets` | Danh sách phiên bản thủ tục có thể gắn với PDF | Đã làm. Cần thêm `ADMIN_INDEXING_ENABLED` | Admin |
| `GET /api/v1/admin/documents/index-metrics` | Số liệu trạng thái indexing | Đã làm. Cần cờ indexing | Admin |
| `GET /api/v1/admin/documents/:id/links` | Các phiên bản đang gắn với PDF | Đã làm. Cần cờ indexing | Admin |
| `POST /api/v1/admin/documents/:id/links` | Gắn PDF với một phiên bản cùng xã và cùng domain | Đã làm. Cần cờ indexing | Admin |
| `POST /api/v1/admin/documents/:id/links/:versionId/index` | Chạy indexing cho link. `INDEX_MODE=mock` giả lập vòng đời. `INDEX_MODE=pipeline` đọc PDF, OCR, chunk, embed | Đã làm trên code. Smoke model thật và RunPod còn pending | Admin |
| `POST .../index/retry` | Index lại link `FAILED` | Đã làm. Cần cờ indexing | Admin |
| `POST .../index/reindex` | Index lại link `READY`, giữ generation cũ đến khi bản mới được kiểm | Đã làm. Cần cờ indexing | Admin |
| `DELETE /api/v1/admin/documents/:id/links/:versionId` | Gỡ link. Dòng còn trong DB, `unlinked_at` được ghi, generation active bị bỏ khỏi link | Đã làm. Cần cờ indexing | Admin |
| `GET /api/v1/admin/procedures` | Danh sách thủ tục của xã, kể cả chưa ACTIVE | Đã làm. ADMIN JWT | Admin |
| `GET /api/v1/admin/procedures/:procedureId/versions` | Versions, tài liệu, eligibility | Đã làm | Admin |
| `GET /api/v1/admin/procedures/:procedureId/versions/:versionId` | Chi tiết version và definition | Đã làm | Admin |
| `POST /api/v1/admin/procedures/:procedureId/versions` | Tạo candidate APPROVED, không kích hoạt | Đã làm. Cần `X-Request-ID` | Admin |
| `POST /api/v1/admin/procedures/:procedureId/versions/:versionId/activate` | Archive ACTIVE cũ, đặt version mới ACTIVE, audit | Đã làm. Atomic + idempotency. Không chạy OCR | Admin |
| `GET /api/v1/admin/procedures/:procedureId/activation-history` | Lịch sử `PROCEDURE_VERSION_ACTIVATED` | Đã làm | Admin |
| `GET /health` trên AI service | AI service còn sống | Đã làm | Go, vận hành |
| `GET /ready` trên AI service | Sẵn sàng nhận job. Trả 503 khi model, OCR hoặc slot pipeline chưa sẵn | Đã làm | Go, vận hành |
| `GET /v1/status` trên AI service | Phase hiện tại, chế độ index, và `rag: not_implemented` | Đã làm | Go, vận hành |
| `POST /v1/extract` trên AI service | Đề xuất mã thủ tục và slot. Go kiểm type rồi mới quyết định action | Đã làm. Token nội bộ. Provider mặc định `mock` | Go gọi hộ công dân |
| `POST /v1/index` trên AI service | `index.v1` mock hoặc `index.v2` pipeline. Pipeline không gọi mock | Đã làm. Token nội bộ. Weight thật chưa smoke | Go gọi hộ admin |
| `POST /v1/retrieve` trên AI service | Tìm chunk của đúng `xa_id` và phiên bản đang ghim, trả citation | Chưa có route | Go gọi khi công dân đã đủ slot |
| `POST /v1/speech/translate` trên AI service | Audio Bahnar thành chữ Việt, rồi đưa vào turn | Chưa có route | Go gọi hộ công dân |

## Đã làm được

### Nền dữ liệu và chạy local

- PostgreSQL kèm pgvector, migration `000001` đến `000017`. `000001`–`000016` đã đóng băng, không sửa. `000017` thêm metadata kích hoạt, không đụng seed ACTIVE.
- Seed một xã `xa_chu_se` và ba domain: hộ tịch và chứng thực, đất đai nhà ở quy hoạch, bảo hiểm và chính sách xã hội.
- Năm thủ tục seed, version `ACTIVE`: chứng thực bản sao, đăng ký khai sinh, tra cứu quy hoạch, xin giấy phép xây dựng, đăng ký BHYT hộ gia đình.
- MinIO cho PDF. Qdrant cho vector của pipeline indexing.
- Docker Compose theo profile: `app`, `storage`, `ai`, `qdrant`. Dockerfile của API, web và AI service đã build được trong vòng kiểm tra P4B.
- OpenAPI của API Go được sinh từ annotation và có test giữ contract.
- Upload và indexing **tắt mặc định**. Phải bật cờ mới thấy màn admin thật.

### Phase 1–3 — chat công dân và Decision Engine

Công dân mở web, chọn xã, chat bằng chữ tiếng Việt. API tạo session, ghim phiên bản thủ tục đang `ACTIVE`, và lưu từng lượt.

Go `decision.Decide` là nơi quyết định, không phải model:

| Action | Khi nào |
|---|---|
| `ASK_MISSING_SLOTS` | Còn slot bắt buộc. Câu hỏi lấy từ JSON, hỏi hết phần thiếu. |
| `DIRECT_ANSWER` | Thủ tục không có slot bắt buộc, ví dụ chứng thực bản sao. |
| `PROVIDE_FINAL_GUIDANCE` | Đủ slot. Checklist lấy từ JSON thủ tục. |
| `OUT_OF_SCOPE` | Câu không thuộc catalog công dân. |
| `CONFIRM_INTENT` | Đổi sang thủ tục khác khi đang ghim một thủ tục. |

Nhận câu có hai tầng:

- Mặc định là từ khóa trong Go. `AI_EXTRACT_ENABLED=false`, nên production hiện tại không gọi model.
- Khi bật cờ, Python `POST /v1/extract` đề xuất mã thủ tục và slot. Go kiểm type và enum. Model không được tự chọn action, không sửa checklist, không tự đổi thủ tục đang ghim. Lỗi, timeout hoặc rate-limit thì quay về từ khóa. Provider mặc định là `mock`, không cần API key.

Citation của lượt chat hiện ghi từ danh sách citation trong JSON seed (`message_citations`), không phải từ chunk PDF.

Auth: guest được chat trong phạm vi công dân; admin bắt buộc JWT. Công dân gọi thủ tục ngoài `ho_tich_chung_thuc` bị từ chối. Admin xem được cả ba domain.

Web đã có: chọn xã, chat, catalog, login, shell admin. Test Vitest của web đã chạy trong vòng P4B (18 passed) cùng production build.

### Phase 2 — LLM extract

Đã có FastAPI, contract extract, provider `mock` / `openai` / `gemini`, claim để hai request cùng lúc không gọi extractor hai lần, rate-limit sau auth, và test các câu kiểu “ba” → cha, “tôi” không đoán, câu gõ sai “chưa đăng ký kết hôn”.

Chưa bật mặc định. Chưa có key LLM trong source. Chưa diễn giải lại câu hỏi bằng LLM.

### Phase 3.1 — admin nhận PDF

Khi `ADMIN_INGESTION_ENABLED` và `VITE_ADMIN_INGESTION` cùng bật, admin đăng nhập rồi upload PDF vào bucket riêng, xem danh sách, chi tiết và tải lại file. Trùng checksum trong cùng xã bị chặn. Có metadata: tiêu đề, domain, số văn bản, cơ quan ban hành.

### Phase 4A — nền indexing

Admin gắn một PDF với một `procedure_version` cùng xã và cùng domain. Trạng thái nằm trên từng link: `UPLOADED`, `PROCESSING`, `READY`, `FAILED`. Có retry. Worker `INDEX_MODE=mock` chỉ giả lập vòng đời, không đọc PDF và không ghi vector.

Gỡ link là soft-unlink: dòng link còn, `unlinked_at` được ghi, generation đang active bị xóa khỏi link. Gắn lại đưa link về `UPLOADED` và không khôi phục generation cũ.

### Phase 4B — đọc PDF, OCR, chunk, vector

Đây là phần code mới nhất. Trạng thái đúng của vòng bundle 12:

```text
P4B_CODE_READY_FOR_RUNPOD_AND_REAL_MODEL_SMOKE
P4B_MODEL_SMOKE_PENDING
RUNPOD_S3_LIVE_SMOKE_PENDING
WORKFLOW_NOT_RUN
```

Pipeline `INDEX_MODE=pipeline` làm được:

1. Lấy PDF từ MinIO theo key nội bộ, kiểm checksum và chữ ký `%PDF-`.
2. Đọc text native. Trang yếu mới đưa qua OCR.
3. Cắt chunk xác định, embed, ghi staging trên PostgreSQL và điểm trên Qdrant.
4. API Go chỉ publish generation khi claim token còn khớp.
5. Reindex một link `READY` giữ generation cũ cho đến khi generation mới được kiểm xong.
6. Reaper dọn generation mồ côi. Xóa Qdrant phải trả đúng hợp đồng rồi mới xóa chunk PostgreSQL.
7. Mỗi request indexing chạy trong process group riêng. Hết hạn hoặc client ngắt thì chỉ giết nhóm đó, nhận diện bằng pid, thời điểm start và process group, không dùng `ps`.
8. Một pipeline một worker OCR. Slot chết khi đang ready được thay một lần rồi đóng. Shutdown có hạn thời gian và không để process mới start sau khi đã đóng.

Đoạn trạng thái phía trên là snapshot bundle 12, khi máy dev chưa có weight và smoke in `P4B_MODEL_SMOKE_PENDING`.

Ngày 2026-10-12 trên Mac Apple Silicon, biến thể đang chạy là fp32 của cùng revision `6a0d452a575215f80b8f66276dd4ee5d504942c6`, chiều 384, SHA-256 `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`. File INT8 AVX-512 VNNI chỉ dành cho Linux x86_64 có đủ cờ. OCR là PaddleOCR 2.9.1 CPU. Weight vẫn không nằm trong repo. Smoke sau khi đổi fixture chữ đã in `P4B_DONE`. Chi tiết và log nằm ở [`PHASE_1_3_COMPLETION_REPORT.md`](PHASE_1_3_COMPLETION_REPORT.md) và `docs/PHASE_1_3_FINAL_BASELINE_REPORT.md`. Admin local đã có, không chạy `seed-demo` trên database đang dùng.

Kiểm tra đã chạy trong vòng bundle 12, trên máy dev:

- pytest AI service: 171 passed, 2 skipped.
- Hai test skip vì chưa bật `CAS_P4B_INTEGRATION`: ghi Postgres + Qdrant từ MinIO thật, và lỗi Qdrant trên server thật.
- Ảnh production AI service, không cài `ps` và không đóng gói pytest: 26 test supervision passed.
- `go test`, `go vet`, `go test -race` của API.
- Một nhóm integration reaper trên database `citizen_assistance`.
- Vitest 18 passed và vite build.
- `docker compose` config và build các profile app, ai, storage, qdrant.

Chưa chạy: smoke model thật, smoke bucket RunPod, và workflow GitHub Actions của commit này. File workflow có sẵn. “Chưa chạy” nghĩa là chưa có lần Actions xanh được ghi trong vòng này, không phải là chưa viết CI.

P4B không có `POST /v1/retrieve` và không đổi câu trả lời của chatbot. Nút kích hoạt phiên bản nằm ở Phase 4 (2026-10-13), xem [`PHASE_4_ACTIVATION_COMPLETION_REPORT.md`](PHASE_4_ACTIVATION_COMPLETION_REPORT.md).

## Chưa làm

### 1. Kích hoạt phiên bản (activate)

Code Phase 4 đã có: API admin, candidate copy-from-ACTIVE, eligibility (generation READY đã publish; `PENDING` không chặn; identity definition khớp DB), transaction atomic, audit, idempotency, session pin, UI tách Activate vs refresh. Chi tiết [`PHASE_4_ACTIVATION_COMPLETION_REPORT.md`](PHASE_4_ACTIVATION_COMPLETION_REPORT.md) và [`PHASE_4_FINAL_ACCEPTANCE_REPORT.md`](PHASE_4_FINAL_ACCEPTANCE_REPORT.md).

Nghiệm thu local 2026-10-13: integration `cas_p4_test` PASS; E2E native PDF + E5/Qdrant + Activate + session pin PASS; browser UI V2 ACTIVE PASS. PaddleOCR scanned PDF `NOT_RUN`. JSON editor / `PENDING→VALID` vẫn backlog. Không production-ready.

### 2. RAG và citation từ tài liệu

Khi `Decide` ra `DIRECT_ANSWER` hoặc `PROVIDE_FINAL_GUIDANCE`, hệ thống chưa tìm chunk.

Việc còn lại:

- Retrieve top-k, lọc đúng `xa_id` và `procedure_version_id` đang ghim.
- Câu trả lời vẫn là summary và checklist trong JSON. RAG chỉ thêm đoạn dẫn nguồn, không được sửa checklist.
- Ghi `message_citations` theo chunk, document và page range của lần retrieve đó.
- Câu còn thiếu slot không được gọi retrieve.
- Test: chunk của version khác không lọt vào; không có chunk thì vẫn trả checklist JSON.
- Bubble chat hiện citation. UI này chờ retrieve có dữ liệu thật.

Vector hiện ở Qdrant, chiều 384. Plan cũ từng ghi pgvector 1536. Phần retrieve phải đi theo Qdrant và model e5-small đã chốt trong runbook P4B, không mở lại chiều 1536.

### 3. LLM extract trên model thật

Contract và fallback đã có. Chưa có:

- Chọn và cấu hình provider thật trên server, key chỉ ở môi trường chạy.
- Bộ hội thoại kiểm thử ngoài mock.
- Đo chất lượng nhận intent và slot trên câu người dân thật.

Keyword vẫn là đường chạy khi cờ tắt hoặc model lỗi.

### 4. Giọng Bahnar trên chat

Chưa có `POST /v1/speech/translate`, chưa ghi `speech_translation_requests` / `model_versions` cho lượt nói, chưa có nút micro.

Luồng đã chốt: audio Bahnar thành chữ Việt, rồi chữ đó đi vào turn như người dùng gõ. Không làm TTS.

Việc này chờ checkpoint RQ đã chọn. Nó không chặn chat chữ.

### 5. Một lệnh demo và nghiệm thu triển khai

Compose chạy được khi bật đúng profile và cờ. Chưa có một lệnh `docker compose up` mặc định ra cả chat lẫn admin indexing, vì các cờ đó cố ý tắt.

Chưa có:

- Smoke object storage RunPod (TLS, cổng 443, region, không tự tạo bucket).
- Một máy demo có HTTPS, health check, seed hộ tịch, một PDF `READY`, và key LLM chỉ trên server.
- Backup Postgres cho bản demo.
- Biên bản UAT.

### 6. Bộ bằng chứng kiểm thử còn thiếu

Đã có unit và một phần integration: Go (kể cả race), pytest AI (trừ hai test tích hợp P4B bị skip), Vitest web, contract OpenAPI, persistence MinIO trong job CI.

Chưa có, hoặc chưa được ghi là đạt:

- Hai test P4B tích hợp MinIO và Qdrant thật.
- Smoke embedding ONNX và PaddleOCR với weight local.
- GitHub Actions của vòng hiện tại (`WORKFLOW_NOT_RUN`).
- E2E trình duyệt từ upload đến câu trả lời có citation.
- Metric AI (intent, slot, retrieval, citation đúng nguồn).
- Performance và security test có tiêu chí nghiệm thu.
- UAT với cán bộ và người dân.

### 7. Những thứ proposal cũ nói nhưng V1 đã bỏ

Không nằm trong việc còn lại, trừ khi phạm vi được mở lại bằng văn bản mới:

- AI đọc PDF rồi sinh draft thủ tục để người sửa.
- Review, approve nhiều bước, rồi mới publish.
- Rollback version.
- Đưa đất đai và bảo hiểm vào chat công dân.
- OCR từng được ghi “không làm V1”, nhưng P4B đã làm OCR cho trang text yếu. Phần OCR còn lại chỉ là chạy weight thật, không phải viết lại pipeline.

## Luận văn Bahnar (RQ1 / RQ2)

Nằm ở `bahnar-s2tt-thesis/`. So sánh hệ cascaded ASR + MT với direct S2TT trên dữ liệu Bahnar–Việt.

Đã có code và contract cho notebook 01 đến 05: khóa split, ASR, MT, direct S2TT, loader và config (`rq1.yaml`, `asr.yaml`, `mt.yaml`, `direct.yaml`). Notebook 06, đánh giá frozen-test C0 và D0, vẫn là placeholder. Checkpoint không nằm trong gói review source. Kết quả luận văn chưa được cắm vào nút micro của chatbot.

## Các phase còn lại

Thứ tự làm, key, weight và checkpoint nằm ở [`phase-con-lai.md`](phase-con-lai.md).