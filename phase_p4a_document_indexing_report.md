# P4A — idempotency từ v10 và migration test trên CI

Bản này đóng hai blocker còn lại của document indexing. Không có OCR, embedding, RAG, Activate, LLM thật, hay deploy production. Transaction, trạng thái từng link, claim token, recovery và API không bị thiết kế lại.

## Replay idempotency sau v10

`000001`–`000012` giữ nguyên. `000013_legacy_idempotency_replay` chỉ sửa JSON đã lưu:

- LINK thiếu `index_status`, `updated_at` hoặc `recoverable` được điền `UPLOADED`, `created_at` và `false`, rồi ghi đè bằng link còn sống nếu `document_id` và `procedure_version_id` là UUID hợp lệ.
- UNLINK chỉ có `{"status":"unlinked"}` được gắn `document_id` và `procedure_version_id` từ audit `DOCUMENT_UNLINKED` khi hai giá trị đó là UUID.
- Down của `000013` là `SELECT 1`. Response mới không bị cắt về dạng cũ.

Code vẫn chịu được body cũ nếu migration chưa ghi đủ field. UNLINK mà hash khớp và JSON không có UUID thì trả `200` và không xóa lần hai. UUID có trong JSON nhưng khác request thì vẫn `409 IDEMPOTENCY_CONFLICT`. LINK thiếu status hoặc `updated_at` được đọc từ link hiện tại; link đã mất thì dùng `UPLOADED` và `created_at` của dòng idempotency, không dùng `time.Now()`.

`payload_hash` vẫn bắt buộc khớp. Payload khác, kể cả `page_range` khác, trả `409`.

Database `citizen_assistance` đang ở version 12 nên chỉ được migrate **up** lên 13. Scratch database làm đủ up, down 13, up và dừng ở version 13, dirty false.

Các dòng idempotency đã bị bản `000011` cũ xóa trên database này không được dựng lại. golang-migrate không checksum file, nên `000011` trên đĩa không chạy lại.

## Migration test không gắn tên container

`migration_repair_test.go` không còn `docker exec citizen-assistance-postgres-1` hay `--network container:...`. Test đọc `DATABASE_URL`, tạo và xóa scratch database bằng pgx, rồi chạy migration trong process bằng `golang-migrate` v4.18.1. Cách này dùng được khi Postgres publish `127.0.0.1:5432`, cả Docker Compose local lẫn GitHub Actions `services.postgres`.

Ba test integration:

- `TestUpgradeFromV10RepairsUnlinkedReadyDocument`: document `READY` không link thành `UPLOADED`, version 13.
- `TestUpgradeFromV10FailsClosedWhenCommuneCannotBeDetermined`: dòng idempotency không xác định được xã làm migration dừng.
- `TestLegacyIdempotencyReplaysAfterUpgrade`: migrate tới v10, chèn LINK/UNLINK theo schema cũ với hash thật, migrate lên latest, rồi gọi HTTP. LINK replay `200` có `index_status`, `updated_at` khác zero và `recoverable`; vẫn đúng một link. UNLINK replay `200`, không xóa lần hai, không `404`. Payload khác `409`. Document xã khác dùng cùng request ID trả `404` và không lộ document gốc.

CI empty cycle đổi `down 12` thành `down 13`. Job integration chạy package `internal/index` với `CAS_INTEGRATION=1`, nên ba test trên chạy trên GitHub Actions. Compose job thêm `app + storage + ai` cùng lúc và kiểm tra service `ai-service`, `minio`, cùng volume `/bitnami/minio/data`.

## Giới hạn

Không khôi phục dữ liệu idempotency đã mất vì `000011` bản DELETE từng chạy trên `citizen_assistance`. Workflow GitHub Actions không được bấm chạy trên GitHub trong môi trường này; chu kỳ scratch up/down 13/up và `docker compose config` được chạy local. Không có thay đổi UI nên không mở lại trình duyệt.
