# Phase 4 — Procedure Version Activation completion report

Ngày: 2026-10-13. Máy local Apple Silicon. Không deploy production. Không chạy seed-demo. Không sửa migration `000001`–`000016`. Phase 5–8 không được viết.

## 1. Đã implement những gì?

Admin quản lý toàn bộ phiên bản thủ tục trong xã (kể cả thủ tục chưa ACTIVE). Có luồng tạo candidate `APPROVED` tối thiểu (sao chép definition từ phiên đang ACTIVE). Activate chỉ khi mọi link SOURCE còn sống có generation `READY` đã publish (chunks = vectors ≥ 1, chiều 384, `embedding_model_id`, `published_at`). Mock READY không có generation không kích hoạt được. Transaction PostgreSQL archive phiên ACTIVE cũ, đặt phiên mới ACTIVE, cập nhật `procedures.active_version_id`, ghi audit và idempotency trong cùng một COMMIT. Catalog công dân đọc ACTIVE mới. Session đã ghim giữ nguyên version cũ.

## 2. API

Prefix `/api/v1`, JWT admin, commune scope theo `XAID` của server. Không tin `xa_id` từ body.

| Method | Path | Việc |
|---|---|---|
| GET | `/admin/procedures` | Danh sách thủ tục, kể cả chưa ACTIVE |
| GET | `/admin/procedures/:procedureId/versions` | Versions, documents, eligibility |
| GET | `/admin/procedures/:procedureId/versions/:versionId` | Chi tiết + definition |
| POST | `/admin/procedures/:procedureId/versions` | Tạo candidate APPROVED. `confirm=true`. `X-Request-ID` |
| POST | `/admin/procedures/:procedureId/versions/:versionId/activate` | Activate. Body `{ "expected_active_version_id": "<uuid\|null>" }` |
| GET | `/admin/procedures/:procedureId/activation-history` | Audit `PROCEDURE_VERSION_ACTIVATED` |

Mã lỗi ổn định: `VERSION_NOT_APPROVED`, `INVALID_DEFINITION`, `DEFINITION_IDENTITY_MISMATCH`, `NO_SOURCE_DOCUMENT`, `PARTIAL_SOURCE_INDEX`, `DOCUMENT_INDEX_*`, `ACTIVE_GENERATION_MISSING`, `MOCK_INDEX_NOT_PUBLISHED`, `DOCUMENT_NOT_CURRENT`, `SOURCE_SCOPE_MISMATCH`, `STALE_ACTIVE_VERSION`, `IDEMPOTENCY_CONFLICT`, `ACTIVATION_CONFLICT`.

## 3. UI

`apps/web/src/pages/admin/ProceduresPage.jsx`: danh sách, versions, eligibility tiếng Việt, xác nhận kích hoạt, tạo candidate, lịch sử. Nút disabled khi không đủ điều kiện. Double-click bị chặn. Activate success tách khỏi refresh; timeout giữ `X-Request-ID`. Thủ tục chưa ACTIVE không gửi create rỗng.

## 4. Migration

`deploy/migrations/000017_phase4_activation.up.sql`: `procedure_versions.activated_at`, `activated_by`; mở check `admin_write_idempotency.action` thêm `ACTIVATE_VERSION` và `CREATE_VERSION`; index audit. Không archive seed ACTIVE. Down không xóa lịch sử idempotency nếu còn hàng action mới (forward-fix).

## 5. Eligibility

Nguồn kiểm tra là link + generation, không phải `documents.processing_status`. Policy: mọi SOURCE còn `unlinked_at IS NULL` phải READY và generation READY đã publish. Một phần index → `PARTIAL_SOURCE_INDEX`. Version `APPROVED` hoặc `INDEXING` mới được kích hoạt. `ACTIVE`/`ARCHIVED` thì không. `PENDING` không chặn; `EXPIRED`/`SUPERSEDED` chặn. Definition parse bằng Go Decision Engine, rồi khớp identity với hàng Postgres (`DEFINITION_IDENTITY_MISMATCH` nếu lệch). Không sửa definition khi Activate.

## 6. Transaction

`SELECT procedures … FOR UPDATE` → lock version → lock links → kiểm expected active → archive ACTIVE cũ → set candidate ACTIVE → cập nhật pointer → audit → idempotency → COMMIT. Lỗi bất kỳ thì ROLLBACK.

## 7. Concurrency / idempotency

Hai admin cùng procedure serialize trên hàng procedure. `expected_active_version_id` lệch → 409 `STALE_ACTIVE_VERSION`. Cùng `X-Request-ID` + cùng payload → replay, không audit mới. Cùng ID khác payload → 409. Target đã ACTIVE đúng pointer → no-op, không audit mới.

## 8. Session pin

`LockTurn` catalog chỉ `pv.status = 'ACTIVE'` khớp pointer. Pinned load theo ID, mọi trạng thái. `SaveTurn` không đổi `active_procedure_version_id` khi cùng procedure. Regression `TestPinnedSessionIgnoresNewerActiveCatalog`.

## 9. Audit

`PROCEDURE_VERSION_ACTIVATED` trong `audit_logs`: request id, actor, procedure, previous/new version. Không ghi JWT, secret, definition đầy đủ.

## 10. Tests

| Hạng | Kết quả vòng final-fix 2026-10-13 |
|---|---|
| Go unit `go test ./... -count=1` | PASS |
| `go vet ./...` | PASS |
| `go test -race ./...` | PASS |
| Isolation guard unit | PASS |
| Integration `CAS_P4_TEST=1` trên `cas_p4_test:5433` | PASS |
| React `npm run test:ci` | PASS (30 tests + bundle) |
| Browser E2E UI V2 ACTIVE | PASS |
| Real PDF + E5/Qdrant Activate | PASS (native extract; PaddleOCR NOT_RUN) |

## 11. E2E

API E2E trên stack cô lập (`cas_p4_test:5433`, MinIO 9012, Qdrant 6336): upload `native.pdf`, E5-small dim 384, generation READY thật, Activate V2, session cũ pin V1 / session mới V2. Browser xác nhận UI ACTIVE 2.0.0 / ARCHIVED 1.0.0. PaddleOCR scanned PDF chưa chạy. Chi tiết [`PHASE_4_FINAL_ACCEPTANCE_REPORT.md`](PHASE_4_FINAL_ACCEPTANCE_REPORT.md).

## 12. Blocker Phase 5

Không có blocker cơ chế version. Phase 5 lấy `conversation_sessions.active_procedure_version_id` đã ghim. Index-reaper không xóa generation chỉ vì version ARCHIVED; cleanup vẫn theo generation SUPERSEDED/FAILED. Trước destructive unlink của nguồn phiên ghim, Phase 5 cần giữ generation đó.

```text
PHASE_4_BACKEND_API=PASS
PHASE_4_ADMIN_UI=PASS
PHASE_4_CANDIDATE_VERSION_MANAGEMENT=PASS
PHASE_4_ELIGIBILITY_VALIDATION=PASS
PHASE_4_ATOMIC_ACTIVATION=PASS
PHASE_4_CONCURRENCY=PASS
PHASE_4_IDEMPOTENCY=PASS
PHASE_4_AUDIT=PASS
PHASE_4_SESSION_PINNING=PASS
PHASE_4_DATABASE_MIGRATION=PASS

PHASE_4_UNIT_TESTS=PASS
PHASE_4_INTEGRATION_TESTS=PASS
PHASE_4_LOCAL_E2E=PASS
PHASE_4_REGRESSION=PASS

PHASE_4_CODE_COMPLETE=true
PHASE_4_STATUS=PASS

PHASE_1_3_LOCAL_BASELINE=PASS
PRODUCTION_FOUNDATION_READY=false
FULL_APPLICATION_READY=false
PRODUCTION_DEPLOYMENT=NOT_RUN
```
