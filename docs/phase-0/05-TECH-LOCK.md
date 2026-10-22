# Tech lock (khớp proposal)

Chi tiết: [`adr/ADR-005-technology-stack.md`](adr/ADR-005-technology-stack.md)  
Data model: [`db/data-model.md`](db/data-model.md)  
Scope: [`01-SCOPE-AND-CATALOG.md`](01-SCOPE-AND-CATALOG.md)

## Đã chốt theo proposal

| Hạng mục | Quyết định |
|----------|------------|
| FE | React.js + JavaScript |
| BE | Go (Gin) — session, decision, audit |
| AI | Python FastAPI — speech (ASR/MT/S2TT), embed, retrieve |
| DB | PostgreSQL |
| Vector | **pgvector** trên `knowledge_chunks` (RAG, không chọn thủ tục) |
| LLM chat/extract | Abstraction OpenAI và/hoặc Gemini; structured extract |
| Voice | Artifact: Bahnar speech → chữ Việt. Không TTS / Việt→Bahnar |
| Storage | S3/MinIO (PDF; audio nghiên cứu nếu được phép) |
| Deploy | Docker Compose + reverse proxy + CI/CD (build/test/image/deploy/health/HTTPS) |
| Domain công dân | Một nhóm hộ tịch & chứng thực |
| Admin | Upload + metadata + index + activate. Không draft workspace, không duyệt nhiều bước |
| OCR | Ngoài V1 |
| Rollback version | Optional |

Embedding schema V1: **`vector(1536)`** — hiện lock `text-embedding-3-small` để cột pgvector ổn định (chi tiết kỹ thuật, proposal chỉ yêu cầu embed vào pgvector).

## Còn mở (không chặn kiến trúc)

- [ ] Primary LLM extract: OpenAI hay Gemini?
- [ ] Object storage prod: MinIO hay S3?
- [ ] Công dân V1 bắt buộc login?
- [ ] CI host: GitHub Actions hay GitLab?
