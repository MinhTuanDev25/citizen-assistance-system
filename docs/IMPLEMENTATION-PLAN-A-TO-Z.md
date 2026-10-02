# Plan A→Z — artifact trợ lý xã (để review)

**Nguồn:** capstone proposal §1.6, §4, §5.  
**Repo:** `citizen-assistance-system` (`apps/`). RQ1/RQ2 nằm ở `bahnar-s2tt-thesis/` và **không** nằm trong plan này.  
**Ngày:** 2026-09-28.

Mục tiêu artifact: 1 xã, một domain hộ tịch & chứng thực, hỏi tiếng Việt (sau đó giọng Bahnar → chữ Việt), slot deterministic, hướng dẫn có citation từ PDF đã index.

**Không làm:** draft workspace, duyệt nhiều bước, OCR scan, đất đai/BHYT trên UI công dân, TTS Bahnar, nộp hồ sơ, thanh toán, HA, rollback bắt buộc.

---

## 0. Đã có — không làm lại

| Hạng | Việc đã chạy |
|------|----------------|
| DB | Postgres + pgvector image, migration `000001`–`000004` (schema, xã, seed thủ tục, user demo) |
| Go | Gin, JWT, guest session, catalog, `POST /api/v1/sessions/:id/turns` |
| Decision | `apps/api/internal/decision`: hỏi full slot thiếu, direct, final, out of scope, pin version, đổi thủ tục, sửa slot đã lưu |
| Nhận câu | **Keyword** trong `intent.go` + `extract.go`, **và** (P2, tùy chọn qua `AI_EXTRACT_ENABLED`) Python `POST /v1/extract` + `internal/aiextract` + `internal/chat/ai_plan.go` — keyword vẫn là fallback duy nhất khi AI tắt/lỗi/timeout |
| Web | Chat công dân, hội thoại mới, login, catalog chỉ hộ tịch |
| Admin UI | Upload, list, detail, download PDF khi cả hai cờ ingestion bật. Draft/OCR vẫn là trang “chưa triển khai” |
| P4A | Migration `000010`–`000013`. Trạng thái indexing trên từng link, mock index, retry. Không embedding, không activate |

Chưa có: embedding, RAG, activate, mic Bahnar, migration citation/speech (`message_citations`, `model_versions`, `speech_translation_requests`).

---

## 1. Luồng một lượt chat (chỗ LLM và RAG cắm vào)

Cửa duy nhất phía công dân: `POST /api/v1/sessions/:sessionId/turns` → `chat.Service.Turn` → `planTurn` → `decision.Decide` → `SaveTurn`.

```text
Câu người dân (hoặc chữ Việt từ speech)
        │
        ▼
[A] Python POST /v1/extract          ← LLM
        procedure_code + slots (allowed keys)
        │
        ▼
[B] Go validate type/enum
        │
        ▼
[C] decision.Decide                  ← không gọi model
        ASK | DIRECT | FINAL | OUT_OF_SCOPE
        │
        ├─ ASK: câu hỏi lấy từ JSON, dừng
        │
        └─ DIRECT / FINAL
              │
              ▼
        [D] Python retrieve            ← RAG + pgvector
              chỉ chunk của version đang pin
              │
              ▼
        Go ghép checklist JSON + citation
        RAG không được sửa checklist
```

Keyword `MatchIntent` / `Extract` giữ làm fallback khi Python timeout.

Giọng Bahnar (sau, khi có checkpoint): Python `POST /v1/speech/translate` **trước** bước [A]. Cùng `planTurn`.

---

## 2. Các pha còn lại

Làm đúng thứ tự. Pha sau không bắt đầu khi exit của pha trước chưa đạt.

### P2 — Extract LLM — **DONE** (2026-09-29, `phase_p2_llm_extract_bundle`)

**Vì sao trước RAG:** các lỗi “ba / tôi / gõ sai kết hôn” nằm ở nhận câu, không ở vector.

| # | Việc | File |
|---|------|------|
| 2.1 | FastAPI `apps/ai-service`, `GET /health` | `app/main.py` |
| 2.2 | `POST /v1/extract`: message, candidates (code, name, examples, slots), pinned procedure, allowed keys, slot state | `app/models/extract.py` |
| 2.3 | Provider `mock` \| `openai` \| `gemini` qua `LLM_PROVIDER` | `app/providers/{mock,openai,gemini}.py`, structured output, không free-text key |
| 2.4 | Go client timeout; lỗi thì keyword | `internal/aiextract/{client,validate}.go`, `internal/chat/ai_plan.go` |
| 2.5 | Test: “ba”→`cha`, “tôi” không đoán, “à quên chưa đưng kí kéth ôn”→`da_ket_hon=false` sau khi đã final | `apps/ai-service/tests/test_mock_provider.py` + `internal/chat/ai_plan_test.go` + `internal/decision/extraction_test.go` |

**Exit — đã đạt, xem `phase_p2_llm_extract_report.md`:** hội thoại mới, bốn câu trên đúng action; checklist vẫn từ JSON; toàn bộ test Phase 1–3 vẫn PASS không đổi; gọi AI không bao giờ diễn ra khi đang giữ transaction/row lock DB (kiến trúc 3 pha trong `chat.Service.turnWithAI`); AI không bao giờ tự chọn action, không tự bịa procedure/slot, không sửa guidance/checklist/citation, không tự đổi procedure đang pin (luôn qua `CONFIRM_INTENT`).

**Không:** paraphrase câu hỏi, RAG, speech, PDF upload, procedure draft, embedding, deploy/commit/push (giữ nguyên cho pha sau).

**Mặc định tắt:** `AI_EXTRACT_ENABLED=false` — API chạy nguyên luồng keyword-only cũ (`chat.Service.turnKeywordOnly`), không có gì đổi trừ khi bật cờ này.

**Sửa sau review (cùng bundle):** provider/model do server gán, không do model; claim `PENDING` (migration `000008`) để hai request đồng thời chỉ gọi extractor một lần; `ai-service` không publish cổng và yêu cầu `AI_SERVICE_TOKEN`; Go `DisallowUnknownFields` và config sai thì fail startup.

**Sửa blocker tiếp theo:** `POST /v1/extract` rate-limit sau auth (429 + `Retry-After`, Go không retry); ngưỡng Go và timeout Python từ chối `NaN`/`Inf`; `LLM_MODEL` và độ dài token kiểm tra lúc startup; trần payload chỉ còn hằng số Pydantic; Compose profile `ai` tùy chọn nên keyword-only (`--profile app`) không chờ `ai-service`.

### P3.1 — Admin document intake — upload PDF thật

Admin JWT upload PDF vào MinIO (bucket private) và bảng `documents` (migration `000009`: `mime_type`, `file_size_bytes`, unique `xa_id + checksum`). List, detail, và stream lại file. Chưa OCR, extract draft, embed, hay RAG. Mặc định tắt. Bật local cần cả hai cờ: `ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true docker compose --profile app --profile storage up --build`.

### P4A — Document indexing foundation

Migration `000010` (không sửa `000001`–`000009`). Thêm `READY` cạnh `PROCESSED`, `documents.supersedes_document_id` cùng xã, và `relationship_type` / `page_range` / `created_at` trên `procedure_version_documents`. Ràng buộc document, procedure và version cùng `xa_id` và `domain_id`.

Trạng thái indexing nằm trên từng link `procedure_version_documents` (`UPLOADED | PROCESSING | READY | FAILED`). Job thuộc `(document_id, procedure_version_id)` và `xa_id`. `documents.processing_status` là trạng thái tổng hợp. Worker P4A là mock `POST /v1/index`. Không ghi `knowledge_chunks`, không OCR, không embedding, không RAG, không activate. Migration thêm là `000011`–`000013` (không sửa `000001`–`000012`). `000013` chỉ chuẩn hóa response idempotency cũ để replay đúng schema mới.

Một lệnh chạy:

```bash
cd deploy && ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true ADMIN_INDEXING_ENABLED=true VITE_ADMIN_INDEXING=true \
  docker compose --profile app --profile storage --profile ai up --build
```

### P4B — Content processing and vector indexing

Migration `000014`. Native PDF text, OCR only for weak pages, deterministic chunks, embeddings in Qdrant, staging generations in PostgreSQL. Activate and retrieval stay out. See `docs/p4b-runbook.md`.

### P5 — RAG + citation

Chỉ khi `Decide` ra `DIRECT_ANSWER` hoặc `PROVIDE_FINAL_GUIDANCE`.

| # | Việc |
|---|------|
| 5.1 | Retrieve top-k, filter `xa_id` + `procedure_version_id` đã pin |
| 5.2 | Reply = summary/checklist JSON + đoạn dẫn nguồn |
| 5.3 | INSERT `message_citations` (chunk, document, page_range) |
| 5.4 | Test: chunk thủ tục khác version không lọt; thiếu chunk vẫn trả checklist JSON |

**Exit:** chứng thực và khai sinh (sau khi đủ slot) có nguồn; câu hỏi thiếu slot không gọi retrieve.

### P3b — UI còn lại (song song được sau P2)

Đã có chat text. Còn:

- Hiện citation trên bubble (khi P5 có)
- Admin đã có upload và trạng thái indexing mock (P4A). Activate để sau P4B
- Mic Bahnar để sau P6

### P6 — Speech (sau checkpoint RQ, không chặn text)

| # | Việc |
|---|------|
| 6.1 | `POST /v1/speech/translate` — audio → chữ Việt |
| 6.2 | Ghi `speech_translation_requests` + `model_versions` |
| 6.3 | Nút mic trên chat; chữ Việt đi vào turn như gõ |

**Exit:** một file Bahnar mẫu → guidance hộ tịch có citation. Không TTS.

### P7 — Chạy một lệnh local

Dockerfile `api`, `ai-service`, `web`. Compose: postgres, minio, api, ai, web. `docker compose up --build` ra chat + admin.

### P8 — CI tối thiểu

GitHub Actions: `go test`, pytest extract mock, web build. Không deploy prod trong pha này.

### P9 — Demo HTTPS

Một VPS, TLS, health, seed hộ tịch, một PDF READY, key LLM chỉ trên server. Backup Postgres hàng ngày là đủ cho demo.

---

## 3. Ai sở hữu logic

| Việc | Process | Hàm / API |
|------|---------|-----------|
| Session, pin, audit, persist | Go | `chat.Service.Turn`, `repository.SaveTurn` |
| Hỏi hay trả | Go | `decision.Decide` |
| Câu hỏi, checklist | JSON trong DB | `definition.slots`, `guidance` |
| Nhận thủ tục + slot | Python | `POST /v1/extract` |
| Embed + tìm đoạn | Python + pgvector | job embed, retrieve |
| Dịch giọng | Python | `POST /v1/speech/translate` |
| UI | React | `CitizenChatPage`, admin pages |

---

## 4. Thứ tự review

1. P2 extract (mock rồi LLM)  
2. P4A indexing foundation (`000010`, mock worker). Embed và activate là P4B, chưa làm  
3. P5 RAG  
4. P7 compose  
5. P6 speech khi có checkpoint  
6. P8–P9 demo

Ước lượng code (không gồm train RQ): P2 khoảng vài ngày, P4–P5 khoảng một tuần, speech thêm khi model đã chọn.

---

## 5. Cần bạn chốt trước khi code P2

- [ ] Primary LLM: OpenAI hay Gemini? (mock vẫn làm trước, không cần key để viết contract)
- [ ] Embedding giữ `text-embedding-3-small` / 1536? (schema đã là `vector(1536)`)
- [ ] P2 xong rồi mới P4, đúng thứ tự trên?
- [ ] Công dân demo vẫn guest được, admin bắt buộc login?

Khi đồng ý, bước code đầu là **P2.1–P2.4** (FastAPI extract + Go gọi vào `planTurn`).

> **Cập nhật 2026-09-29:** Mục 4–5 ở trên là ghi chú lập kế hoạch từ trước khi
> code P2, giữ lại làm lịch sử. P2 đã hoàn thành với `LLM_PROVIDER=mock` mặc
> định (không cần key), `AI_EXTRACT_ENABLED=false` mặc định (không đổi hành
> vi production hiện tại). Xem `phase_p2_llm_extract_report.md` để biết chi
> tiết exit-gate và kết quả kiểm thử.
