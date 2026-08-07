# Phase 0.5 — Tech Lock (trước khi code)

## Đã chốt (Owner)

| Hạng mục | Quyết định |
|----------|------------|
| FE | React.js + JavaScript |
| BE | Go (Gin) |
| AI | Python FastAPI |
| DB | PostgreSQL |
| Vector | pgvector |
| LLM | Abstraction → OpenAI GPT và/hoặc Gemini |
| Voice | V1 text; V1.1 voice→STT→text |
| Deploy | Containers tách service + Reverse proxy + CI/CD |

Chi tiết + phân tích auth/storage/OCR/cache/logging:  
[`adr/ADR-005-technology-stack.md`](adr/ADR-005-technology-stack.md)

## Còn mở (chặn nhẹ)

- [ ] Primary LLM: OpenAI hay Gemini?
- [ ] Object storage: MinIO hay S3/GCS?
- [ ] Citizen bắt buộc login V1?
- [ ] CI: GitHub Actions / GitLab?

## Done tech lock khi

4 ô trên được trả lời → ký ADR-005 → mới code Phase 1.
