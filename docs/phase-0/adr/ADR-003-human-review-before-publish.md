# ADR-003: Publish không đi qua AI draft workspace

## Status

**Superseded** bởi capstone proposal §1.6 và §4.2 (2026). ADR Phase 0 cũ bắt buộc Review Workspace + approve nhiều bước — **không còn là V1**.

## Context

LLM extract PDF có thể sai slot. Bản ADR cũ chốt: mọi draft phải qua workspace duyệt rồi mới publish.

Proposal thu hẹp artifact: không workspace nháp, không chuỗi duyệt nhiều bước. Cán bộ vẫn chịu trách nhiệm nội dung JSON thủ tục (seed hoặc sửa tay) và **chỉ kích hoạt version khi nguồn PDF đã index xong**.

## Decision (V1 hiện hành)

1. Admin tạo/mở `procedure_version`, upload PDF + metadata.
2. Backend tự tạo `procedure_version_documents` (không màn gắn nguồn riêng).
3. Job cắt đoạn + embed; `processing_status`: `uploaded` → `processing` → `ready` | `failed`.
4. Admin xem trạng thái index rồi **activate**. Chỉ version ACTIVE vào retrieval công dân.
5. Không bảng `procedure_drafts` trong phạm vi artifact; không `source_draft_id` / `approved_by` / `approved_at` trên `procedure_versions`.
6. Rollback về version cũ: optional, không phải minimum.

Cán bộ không “duyệt draft LLM”. Họ soạn/giữ `definition` JSON và quyết định lúc nào version được ACTIVE.

## Consequences

- Artifact ngắn hơn, đúng lịch 12 tuần (tuần 11 gắn prototype).
- An toàn nguồn: PDF gốc trên MinIO, chunk + citation theo version.
- Không tự sinh procedure JSON từ PDF trong V1; seed / chỉnh JSON là đường chính cho hộ tịch.
