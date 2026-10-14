# Phase 4 — Procedure version activation runbook

## Mục đích

Đưa một phiên bản thủ tục đã được admin xác nhận và đã index đủ nguồn thành phiên bản ACTIVE chính thức của xã. Chat mới dùng version đó. Hội thoại đã ghim version cũ giữ nguyên.

Không chạy OCR, embedding, upload, hay ghi Qdrant khi Activate.

## Điều kiện

- JWT ADMIN.
- `X-Request-ID` là UUID.
- Procedure thuộc `XAID` của API (một xã; token ADMIN không mang quyền đa xã).
- Version `APPROVED` hoặc `INDEXING`.
- Mọi link SOURCE còn sống: `index_status=READY`, `active_generation_id` trỏ generation `READY` đã `published_at`, `chunk_count = vector_count ≥ 1`, `vector_dimension=384`, có `embedding_model_id`.
- `expected_active_version_id` đúng pointer hiện tại, hoặc `null` nếu chưa có ACTIVE.
- Definition parse được bằng Decision Engine.
- Nếu JSON có `procedure_code` / `name` thì phải khớp hàng procedure (tên so sánh sau khi gộp khoảng trắng). `domain`, `xa_id`, `version` chỉ bắt khớp khi field có mặt. JSON `status` không quyết định ACTIVE.
- `PENDING` trên `documents.validity_status` **không** chặn Activate. `EXPIRED` và `SUPERSEDED` chặn. `PENDING` là vòng đời tài liệu, không đồng nghĩa indexing lỗi.

**Activation readiness không phải xác nhận hiệu lực pháp lý của tài liệu.** Cán bộ chịu trách nhiệm xác nhận nội dung và tính phù hợp của nguồn trước khi kích hoạt. Workflow xác nhận `PENDING → VALID` là backlog riêng, không nằm trong Phase 4.

Phiên ACTIVE từ seed có thể không có PDF. Chúng không bị archive khi khởi động. UI đánh dấu `legacy_active` khi `status=ACTIVE` và `activated_at` rỗng.

UI hiện chỉ sao chép definition từ phiên ACTIVE. Thủ tục chưa có ACTIVE: giao diện báo rõ, không gửi request tạo version rỗng. JSON editor, AI drafting và phê duyệt nhiều bước là backlog, không chặn Phase 5.

## Tạo candidate

`POST /api/v1/admin/procedures/:procedureId/versions`

```json
{ "confirm": true, "copy_from_version_id": "<active-or-omit>", "version": "2.0.0" }
```

Bỏ `version` thì API tăng patch. Gắn PDF bằng API indexing hiện có, rồi index pipeline (không mock) trước khi kích hoạt.

## Activate

`POST /api/v1/admin/procedures/:procedureId/versions/:versionId/activate`

```json
{ "expected_active_version_id": "<current-active-uuid-or-null>" }
```

Sau COMMIT: version cũ `ARCHIVED` (nếu có), version mới `ACTIVE`, pointer khớp, một hàng audit `PROCEDURE_VERSION_ACTIVATED`.

## Lỗi thường gặp

| HTTP | code | Việc làm |
|---|---|---|
| 422 | `MOCK_INDEX_NOT_PUBLISHED` | Index lại bằng pipeline thật, đừng dùng mock READY |
| 422 | `PARTIAL_SOURCE_INDEX` | Đợi mọi SOURCE READY |
| 422 | `NO_SOURCE_DOCUMENT` | Gắn ít nhất một PDF SOURCE |
| 422 | `DEFINITION_IDENTITY_MISMATCH` | Definition không khớp mã thủ tục / xã / domain / version trong Postgres |
| 422 | `DOCUMENT_NOT_CURRENT` | Tài liệu EXPIRED hoặc SUPERSEDED; không dùng PENDING để chặn |
| 409 | `STALE_ACTIVE_VERSION` | Tải lại danh sách, gửi expected mới |
| 409 | `IDEMPOTENCY_CONFLICT` | Đổi `X-Request-ID` nếu payload khác |
| 409 | `ACTIVATION_CONFLICT` | Hai admin; tải lại trạng thái |

Retry sau timeout: gửi lại **cùng** `X-Request-ID` và cùng body. UI giữ key khi dialog còn mở.

## Chat

Hội thoại mới: catalog `LockTurn` chỉ lấy `pv.status='ACTIVE'` và `p.active_version_id`. Hội thoại cũ: `active_procedure_version_id` không đổi khi cùng procedure. Definition ARCHIVED được giữ. Không xóa session, slot, generation vì archive.

## Recovery

Đọc pointer và `procedure_versions.status` trên Postgres. Tra `admin_write_idempotency` theo `(xa_id, request_id, ACTIVATE_VERSION)`. Nếu COMMIT xong mà client timeout, replay trả kết quả cũ. Không rollback nghiệp vụ tự động về version trước — không có workflow phê duyệt ngược trong Phase 4.

## Rollback schema

`000017` down xóa cột `activated_at`/`activated_by`. Nếu đã có hàng idempotency `ACTIVATE_VERSION`/`CREATE_VERSION`, down bị chặn: giữ check action đã mở (forward-fix), không xóa audit.

## Isolation — Phase 4 integration tests

Chỉ chạy trên database `cas_p4_test`, host `127.0.0.1` hoặc `localhost`, cổng 5432 (hoặc `CAS_P4_TEST_PORT`). Guard parse DSN bằng `pgx`, exact match tên database, từ chối `PGHOST`/`PGDATABASE` và query key ngoài `sslmode`. Không fallback `citizen_assistance`.

```bash
export CAS_P4_TEST=1
export CAS_P4_TEST_DATABASE_URL='postgres://cas:cas@127.0.0.1:5432/cas_p4_test?sslmode=disable'
unset PGHOST PGHOSTADDR PGPORT PGDATABASE PGSERVICE PGSERVICEFILE PGSYSCONFDIR
cd apps/api
go test -tags=integration ./internal/activation/ -count=1 -v
```

Tạo database `cas_p4_test` trên instance test, migrate `000001`–`000017`, không dùng volume development, không `seed-demo` trên dữ liệu đang dùng.

## Phase 5 — Source Retention & Retrieval Invariants

1. Session pin một `procedure_version_id`. RAG phải lọc đúng version được pin, không lấy ACTIVE mới nếu session đã ghim ARCHIVED.
2. Cần chốt chính sách giữ generation/source cho hội thoại cũ: unlink PDF thuộc ACTIVE, reindex nguồn đang ACTIVE, generation SUPERSEDED, session cũ pin ARCHIVED, Qdrant cleanup/reaper.
3. Không tạo citation tới generation đã bị xóa.
4. Tài liệu đã kích hoạt phải có nguồn truy xuất nhất quán theo generation đã publish lúc Activate.
5. Nếu source không còn khả dụng, không bịa citation.
6. Phase 4 không xóa ARCHIVED version hay source generation chỉ vì Activate. Index-reaper vẫn chỉ dọn generation SUPERSEDED/FAILED đã claim.

Đây là handoff kỹ thuật cho Phase 5. Không triển khai RAG, rollback version, hay phức tạp hóa Activation trong Phase 4.
