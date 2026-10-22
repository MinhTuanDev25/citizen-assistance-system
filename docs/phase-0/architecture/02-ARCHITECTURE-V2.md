# Architecture — Citizen assistance artifact

Khớp proposal §4.3–4.5. Sơ đồ cũ “Review Workspace / PDF→draft / approve nhiều bước” **không** còn là V1.

## 1. Hai pipeline

```text
Admin (simplified)                         Citizen runtime
------------------                         ---------------
Mở procedure version                       Text hoặc giọng Bahnar
   ↓                                          ↓
Upload PDF + metadata                      (nếu giọng) Python S2TT / Cascaded
   ↓                                          ↓
MinIO + documents                          chữ Việt → Go
   ↓                                          ↓
Tự tạo procedure_version_documents         Auth + session + pin version
   ↓                                          ↓
Job: extract text → chunk → embed          Decision Engine (Go, JSON)
   ↓                                          ↓
processing_status: uploaded→processing     thiếu slot → hỏi full missing
                 → ready | failed             ↓
   ↓                                       đủ slot → RAG scoped + citation
Admin xem index, kích hoạt version         conversation_messages + audit
```

Không có màn “gắn nguồn thủ công”, không có workspace nháp LLM, không có chuỗi duyệt nhiều bước.

## 2. Nguyên tắc

- Não nghiệp vụ = `procedure_versions.definition` (JSON). Go quyết định hỏi / trả, không hardcode từng thủ tục.
- Python: dịch giọng (ASR, MT, Direct S2TT đã chọn), embedding, retrieval. Không quyết định action.
- RAG chỉ lấy chunk gắn version **ACTIVE** (session pin version đó). Citation bắt buộc document + page range khi có.
- pgvector lưu embedding; Postgres lưu metadata giao dịch / version.

## 3. Layers (Citizen)

| Layer | Responsibility |
|-------|----------------|
| Citizen Portal | Chat, lịch sử, mic Bahnar (sau khi có model) |
| Auth | JWT admin bắt buộc; công dân guest hoặc login theo policy |
| Conversation | Session, pin procedure/version, gọi extract |
| **Decision Policy Engine** | `ASK_MISSING_SLOTS` / `DIRECT_ANSWER` / `PROVIDE_FINAL_GUIDANCE` / `OUT_OF_SCOPE` |
| Procedure Orchestrator | Load definition đã pin, merge slot, persist |
| Knowledge | Retrieve pgvector **theo version đã pin** + citation |
| Speech (Python) | Bahnar audio → Vietnamese text |

## 4. Layers (Admin)

| Layer | Responsibility |
|-------|----------------|
| Admin Portal | Tạo/mở version, upload PDF, metadata, xem index, activate |
| Auth | Role `ADMIN` |
| Knowledge Publisher | `procedure_version` ↔ `documents`, `knowledge_chunks`, `processing_status`, biên activate |
| Object storage | PDF gốc (và audio nghiên cứu nếu được phép) trên S3/MinIO |

## 5. Runtime topology (proposal §4.3)

```text
Citizen / Admin (React)
        │ HTTPS
        ▼
     Go backend     auth, session, workflow, procedure rules, audit
        │
        ├── PostgreSQL (metadata, sessions, definitions)
        ├── pgvector (`knowledge_chunks.embedding`)
        ├── S3/MinIO (PDF, audio)
        └── Python AI service
              ASR / MT / Direct S2TT (checkpoint đã chọn)
              embed + retrieve
```

CI/CD tối thiểu: build, test, image, deploy, health check, domain, HTTPS. Không HA enterprise.

## 6. Cố ý không làm V1

- Multi-xã
- TTS Bahnar / dịch Việt → Bahnar
- Nộp hồ sơ / thanh toán / chữ ký số
- AI draft workspace và approve nhiều bước
- OCR PDF scan
- Rollback version (optional nếu còn lịch)
- Mở domain đất đai / bảo hiểm cho công dân
