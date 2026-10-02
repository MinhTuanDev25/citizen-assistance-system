# P3.1 — Admin Document Intake (blocker fix)

P2 đã xong: `POST /v1/extract` chạy khi `AI_EXTRACT_ENABLED=true`, keyword vẫn là đường mặc định.

P3.1 chỉ gồm upload, danh sách, chi tiết và tải PDF. Chưa OCR, chưa bản nháp thủ tục, chưa embedding, chưa RAG, chưa activate.

Không deploy. Không gọi LLM thật. Không chạy OCR/RAG. Database `citizen_assistance` không bị `migrate down`. Chu kỳ migration chạy trên database tạm `cas_p31_fix` rồi xóa.

## Đã sửa

- Volume MinIO là `miniodata:/bitnami/minio/data`, cùng đường dẫn process ghi. Test tạo object, xóa container, giữ volume, tạo lại container, đọc lại đúng bytes và SHA-256.
- Ingestion vẫn tắt mặc định. Bật cùng lúc `ADMIN_INGESTION_ENABLED=true` và `VITE_ADMIN_INGESTION=true`, rồi `docker compose --profile app --profile storage up --build`. `VITE_ADMIN_INGESTION` có trong `deploy/.env.example`.
- OpenAPI đủ `POST/GET /api/v1/admin/documents`, `GET /{id}`, `GET /{id}/content`: multipart fields, mã phản hồi, `BearerAuth`. Đã regenerate `docs.go`, `swagger.json`, `swagger.yaml`. Contract test fail nếu thiếu path hoặc security.
- Ngày được `time.Parse("2006-01-02")`. `2026-02-31`, tháng 13, `2026-02-29` và expire trước effective trả 422, không gọi insert.
- Khi insert DB lỗi, xóa object bằng context mới, timeout 10 giây, không phụ thuộc request đã cancel.
- File đúng `DOCUMENT_MAX_BYTES` được nhận. Lớn hơn 1 byte trả 413. Trần request của route upload là 64 MiB cộng 256 KiB multipart, nên file đúng 64 MiB không bị overhead từ chối. Test Content-Length và chunked.
- CI chạy integration của `internal/document`, test MinIO persistence, và `docker compose config` cho profile `app`, `app+ai`, `app+storage`.
- Nút tải PDF bắt lỗi và hiện trên trang. Menu bản nháp luôn ghi “chưa triển khai”. Đã xóa `mockStore.js`, `DraftsPage.jsx`, `DraftReviewPage.jsx`.
- `gofmt` chỉnh khoảng trắng ở `procedure/handler.go`, `session/handler.go`, `db.go`, `health.go`, `logx.go`. Không đổi logic chat, auth hay decision engine. Không sửa migration `000001`–`000009`.

## Xác minh giao diện

Vite `VITE_ADMIN_INGESTION=true` tại `http://127.0.0.1:5174/admin/documents` hiện form thật: File PDF, tiêu đề, lĩnh vực, số hiệu, cơ quan ban hành, nút Upload. Menu bản nháp là “Bản nháp (chưa triển khai)”; `/admin/drafts` là trang deferred. API không chạy trong lần xem này nên danh sách báo lỗi tải, không phải trang giả. Bundle `false` không có “File PDF”. Bundle `true` có “File PDF” và “Đã lưu PDF”, không có `mockStore`.
