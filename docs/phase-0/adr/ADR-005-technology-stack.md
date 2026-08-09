# ADR-005: Technology Stack & Deployment (Pre-coding Lock)

## Status

**Accepted (partial)** — đã chốt theo Owner feedback; còn vài lựa chọn LLM vendor cụ thể / object storage provider.

## Context

- PostgreSQL đã có; vector dùng **pgvector**.
- Não nghiệp vụ = JSON definition; Decision Engine deterministic (Go).
- Owner FE chỉ dùng **React + JavaScript**.
- BE chốt **Go (Gin)**; AI module = **Python**.
- Cần làm rõ LLM abstraction, Voice, Deploy (CI/CD + containers), và các hạng mục còn thiếu (auth, file storage, OCR, embedding VN, cache, logging…).

---

## 1. Stack đã chốt

| Lớp | Quyết định | Ghi chú |
|-----|------------|---------|
| Frontend | **React.js + JavaScript** | Vite hoặc CRA; Citizen chat + Admin review |
| Backend API | **Go + Gin** | Auth, session, Decision Engine, CRUD admin, gọi AI service |
| AI module | **Python + FastAPI** | Slot extract, PDF→draft, RAG generate, embedding job |
| Database | **PostgreSQL** | Structured data (đã có) |
| Vector | **pgvector** (cùng Postgres) | Knowledge chunks |
| LLM access | **Provider abstraction** (OpenAI **và/hoặc** Gemini…) | Xem mục 2 |
| Voice | **V1.1: Voice → STT → text reply** | Xem mục 3; V1 text-only |
| Deploy | **Containers tách service + Reverse proxy + CI/CD** | Xem mục 4–5 |

### AI Python — lựa chọn chi tiết

| Item | Chọn |
|------|------|
| Framework | FastAPI |
| HTTP client LLM | abstraction `LLMProvider` interface |
| PDF parse | `pymupdf` / `pdfplumber` (+ OCR optional) |
| Embedding write | SQLAlchemy/psycopg + pgvector |
| Task nặng (embed/OCR) | FastAPI background task V1; Redis queue V1.1 nếu cần |

---

## 2. LLM: vì sao không “chỉ hardcode GPT hoặc Gemini”?

**Không phải bỏ GPT/Gemini** — mà **không gắn chết 1 vendor trong code**.

```text
Go / Python app
    │
    ▼
LLM Provider Interface (interface)
    ├── OpenAI GPT (primary hoặc fallback)
    ├── Gemini (primary hoặc fallback)
    └── (sau) Qwen / local
```

| Cách | Kết quả |
|------|---------|
| Hardcode OpenAI everywhere | Đổi sang Gemini = sửa nhiều chỗ, lock-in |
| **Abstraction** + config `LLM_PROVIDER=openai\|gemini` | Vẫn dùng GPT hoặc Gemini bình thường; đổi vendor bằng env |

**Đề xuất V1:**

- Primary: **OpenAI GPT** *hoặc* **Gemini** (chọn 1 theo key/budget/PII policy)
- Fallback: vendor còn lại (optional nhưng nên có)
- Mọi call qua interface: `ExtractSlots`, `GenerateDraft`, `ParaphraseReply` (nếu cần)

Structured output bắt buộc cho extract slots (JSON schema / function call).

**Open:** Owner chọn primary = `openai` hay `gemini` (cả hai đều fit abstraction).

---

## 3. Voice là gì? (giải thích đơn giản)

Bạn chưa từng làm voice — hiểu như sau:

| Khái niệm | Nghĩa |
|-----------|--------|
| **STT** (Speech-to-Text) | Mic nói → thành **chữ** |
| Pipeline hiện tại | Chữ vào Decision Engine như chat text |
| **TTS** (Text-to-Speech) | Chữ trả lời → thành **giọng nói** (không bắt buộc V1) |

### 2 kiểu

1. **Voice → Text reply** (đề xuất)  
   Dân nói → STT → cùng API chat → hiện chữ trên UI.  
   Giống người gõ, chỉ khác cách nhập.

2. **Voice → Voice**  
   Thêm TTS đọc câu trả lời. Phức tạp hơn (nghe checklist dài, latency).

```text
[Mic] → STT service → message text → Go API → reply_text → [màn hình chữ]
                                              └─(sau)→ TTS → [loa]
```

**V1:** chỉ Text.  
**V1.1:** thêm nút micro + STT (Whisper / Google / Azure — chốt khi làm voice).  
Không đổi kiến trúc lõi.

---

## 4. Deploy — tách container + CI/CD (đồng ý)

Không deploy “1 process lẫn lộn”. Topo V1:

```text
                    Internet
                       │
                       ▼
              [Reverse Proxy]
              Nginx hoặc Caddy
              (TLS, routing)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   web (React)    api (Go Gin)   ai-service (Python)
   container      container      container
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              PostgreSQL (+ pgvector)
              (managed hoặc container)
                       │
                       ▼
              Object Storage (PDF files)
              MinIO / S3 / GCS
                       │
                       ▼
              External LLM APIs
```

| Thành phần | Vai trò |
|------------|---------|
| Container riêng | `web`, `api`, `ai-service`, (optional) `postgres`, `minio` |
| Reverse proxy | HTTPS, `/` → web, `/api` → Go, `/ai` internal-only nếu cần |
| CI/CD | PR → test → build image → push registry → deploy staging/prod |
| Secrets | Env / secret manager — không commit API keys |

### CI/CD tối thiểu

```text
git push / PR
  → lint + unit test (Go, Python, FE)
  → build Docker images
  → push registry
  → deploy staging (compose/k8s)
  → manual approve → prod
```

Tool đề xuất: **GitHub Actions** (hoặc GitLab CI).  
Orchestration V1: **Docker Compose** trên 1 VPS; sau có thể K8s nếu cần.

---

## 5. Phân tích các hạng mục bạn nêu còn thiếu

### 5.1 Authentication — **CÓ, bắt buộc**

| Actor | Cơ chế đề xuất V1 |
|-------|-------------------|
| Citizen | JWT (login đơn giản hoặc guest token + session) — chốt có bắt buộc login hay không |
| Admin | JWT + role `admin` (bắt buộc) |

Go API: middleware Gin verify JWT.  
Admin upload/publish chỉ role admin.  
Refresh token optional V1.1.

### 5.2 Document Storage (PDF upload) — **CÓ, bắt buộc**

Upload PDF **không** chỉ lưu Postgres (bytea không scale).

| Lưu gì | Ở đâu |
|--------|--------|
| File PDF binary | **Object storage**: MinIO (self-host) hoặc S3/GCS |
| Metadata | Bảng `documents` (filename, storage_uri, checksum, effective_date, expire_date…) |

Flow:

```text
Admin upload → API nhận file → ghi Object Storage → INSERT documents.storage_uri
            → AI extract → procedure_drafts
```

**Đề xuất V1:** MinIO nếu self-host cùng VPS; S3 nếu dùng cloud.

### 5.3 Embedding Model — **FROZEN với data-model V1**

| Câu hỏi | Trả lời |
|---------|---------|
| Có cần embedding? | Có (Phase publish knowledge / RAG) |
| Model V1 | **OpenAI `text-embedding-3-small`** |
| Dimension | **1536** → cột `knowledge_chunks.embedding vector(1536)` |
| Đổi sau | Re-embed toàn bộ + migration |

Chat LLM vẫn abstraction OpenAI/Gemini; **embedding cố định** để schema/pgvector ổn định.

V1 chat với seed JSON chưa cần embedding runtime; Phase publish knowledge mới cần.

### 5.4 OCR — **Có điều kiện**

| PDF loại | Cần OCR? |
|----------|----------|
| PDF text (selectable) | Không — parse text thường đủ |
| PDF scan / ảnh | **Có OCR** |

**Đề xuất:**  
- V1: hỗ trợ PDF text-first; nếu extract text rỗng → báo admin “cần OCR / file scan”.  
- V1.1: gắn OCR (VD Tesseract `vie+eng` hoặc cloud Vision) trong `ai-service`.

Không bắt buộc OCR ngày 1 nếu xã cung cấp PDF chữ.

### 5.5 Cache — **Nên có mức tối thiểu**

| Cache gì | Tool | Phase |
|----------|------|-------|
| Procedure definition active | In-memory / Redis | V1 in-process cache trong Go; Redis V1.1 |
| Rate limit chat | Redis hoặc in-memory | V1.1 |
| Job queue embed/OCR | Redis | Khi có OCR/embed nặng |

V1 có thể **chưa Redis** nếu traffic 1 xã thấp; design interface sẵn.

### 5.6 Logging / Audit / Monitoring — **CÓ, bắt buộc**

| Loại | Mục đích | Đề xuất |
|------|----------|---------|
| Audit DB (`audit_logs`) | Ai publish, decision nào, version nào | Đã có trong data model |
| App logs structured | Debug, trace request_id | Zap/slog (Go), structlog (Python) |
| Metrics | Latency, error rate, LLM fail | Prometheus + Grafana (V1.1) hoặc cloud |
| Không log PII thừa | CCCD, nội dung nhạy cảm | Redaction policy |

### 5.7 Container — **CÓ**

Mỗi service 1 image/Dockerfile: `web`, `api`, `ai-service`. Compose file cho local + staging.

### 5.8 Reverse Proxy — **CÓ**

Nginx/Caddy: TLS, gzip, route, giới hạn upload size (PDF).

### 5.9 CI/CD — **CÓ**

Như mục 4. Pipeline tối thiểu trước prod.

---

## 6. Sơ đồ thành phần đầy đủ (sau khi bổ sung)

```text
[React Web]
    │
    ▼
[Nginx/Caddy]
    ├── /        → web
    └── /api/*   → Go Gin API
                      │
                      ├── PostgreSQL + pgvector
                      ├── Object Storage (PDF)
                      ├── AI Service (Python) ──► LLM (OpenAI/Gemini via abstraction)
                      └── (optional) Redis
```

---

## 7. Decision checklist (cập nhật)

| # | Hạng mục | Quyết định | Status |
|---|----------|------------|--------|
| 1 | API | Go Gin + Python FastAPI AI | **Chốt** |
| 2 | FE | React + JavaScript | **Chốt** |
| 3 | DB | PostgreSQL | **Chốt** |
| 4 | Vector | pgvector | **Chốt** |
| 5 | LLM | Abstraction; vendor = OpenAI và/hoặc Gemini | **Chốt hướng**; chọn primary còn mở |
| 6 | Voice | V1 text; V1.1 voice→STT→text | **Chốt hướng** |
| 7 | Object storage | MinIO hoặc S3 | **Cần chọn 1** |
| 8 | OCR | Text PDF first; OCR khi scan | **Chốt hướng** |
| 9 | Auth | JWT; admin bắt buộc | **Chốt hướng**; citizen login? mở |
| 10 | Cache | Optional Redis V1.1 | **Chốt hướng** |
| 11 | Logging/Audit | audit_logs + structured logs | **Chốt** |
| 12 | Deploy | Containers + Nginx + CI/CD | **Chốt** |
| 13 | Embedding VN | Có ở Phase 3; model multilingual | **Chốt hướng** |

---

## 8. Open questions còn lại (ngắn)

1. Primary LLM: OpenAI hay Gemini?  
2. Object storage: MinIO (VPS) hay S3/GCS?  
3. Citizen có bắt buộc đăng nhập V1 không?  
4. CI host: GitHub Actions / GitLab?

---

## Sign-off

| Role | Name | Date | Sign |
|------|------|------|------|
| Owner | | | |
| Backend | | | |
| AI | | | |
| Frontend | | | |
