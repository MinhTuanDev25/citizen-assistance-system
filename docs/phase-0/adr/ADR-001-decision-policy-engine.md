# ADR-001: Decision Policy Engine tách riêng

## Status

Accepted (Phase 0)

## Context

Công dân có thể:

- hỏi thủ tục thiếu thông tin → cần hỏi bổ sung slot
- hỏi đủ thông tin / thủ tục đơn giản → trả lời thẳng
- hoàn tất đủ slot qua nhiều lượt → trả hướng dẫn cuối

Nếu nhúng logic này vào Conversation Manager hoặc LLM prompt thuần, sẽ khó kiểm soát, khó test, khó audit.

## Decision

Tách **Decision Policy Engine** giữa Conversation Manager và Procedure Orchestrator.

Engine chỉ trả về một trong 3 action:

- `ask_missing_slots`
- `direct_answer`
- `provide_final_guidance`

Input chính: `procedure_definition` active + `slot_state` session + slots vừa extract.

## Consequences

- Runtime deterministic hơn, test được bằng matrix
- LLM dùng để NLU/NLG, không tự quyết “có hỏi thêm hay không” ngoài rule JSON
- Cần maintain schema slot chuẩn
