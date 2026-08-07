# ADR-002: Procedure definition JSON-driven

## Status

Accepted (Phase 0)

## Context

Có 18 thủ tục thuộc 3 domain. Hardcode workflow từng thủ tục trong code sẽ phình nhanh và khó cập nhật khi xã đổi quy định.

## Decision

Mỗi thủ tục lưu dạng `procedure_definition` JSON gồm:

- identity (id, domain, name)
- intent examples
- required / conditional slots
- slot questions
- guidance checklist + location rules
- citations / source metadata
- version + status

Admin upload PDF → sinh draft JSON → human review → publish version active.

## Consequences

- Thêm/sửa thủ tục không cần deploy code (chỉ publish version)
- Cần schema validation mạnh
- Seed tạm (`manual_seed`) được phép trước khi có PDF thật, nhưng không được active prod nếu chưa có nguồn xã review
