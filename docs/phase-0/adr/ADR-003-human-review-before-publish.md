# ADR-003: Human review bắt buộc trước publish

## Status

Accepted (Phase 0)

## Context

LLM extract từ PDF có thể thiếu slot, sai điều kiện, hoặc hallucinate hồ sơ. Domain hành chính công không chấp nhận publish trực tiếp ra công dân.

## Decision

Mọi draft phải qua **Review Workspace**:

1. Source validation (đúng xã / hiệu lực / loại VB)
2. Schema validation
3. Admin edit slots/questions/guidance
4. Approve → publish version
5. Có rollback về version trước

## Consequences

- Latency cập nhật chậm hơn (có người duyệt) nhưng an toàn hơn
- Cần UI/admin API riêng
- Audit log bắt buộc: ai approve, version nào, lúc nào
