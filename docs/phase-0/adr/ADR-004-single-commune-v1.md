# ADR-004: Single xã (V1)

## Status

Accepted (Phase 0)

## Context

V1 chỉ phục vụ 1 xã. Multi-tenant sớm sẽ làm phức tạp auth, data partition, citation và UAT.

## Decision

- Hard-config `xa_id` ở hệ thống
- Mọi procedure/knowledge chunk gắn `xa_id`
- Runtime reject / redirect nếu câu hỏi yêu cầu thủ tục ngoài phạm vi xã (khi phát hiện được)

Vẫn thiết kế schema có `xa_id` để sau mở multi-xã không phá model.

## Consequences

- Deploy đơn giản, UAT với 1 bộ văn bản
- Không optimize multi-tenant isolation ở V1
- Khi mở rộng: thêm tenant resolver, không đổi contract slot/decision
