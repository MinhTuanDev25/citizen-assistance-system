# Phase 0 — Design Lock (Pre-coding)

**Project:** AI Public Administrative Assistant (1 xã)  
**Status:** Draft for team review  
**Goal:** Khóa hợp đồng dữ liệu + kiến trúc trước khi code feature thật.

## Deliverables

| # | Artifact | Path | Status |
|---|----------|------|--------|
| 1 | Scope & catalog 18 thủ tục | `01-SCOPE-AND-CATALOG.md` | Ready |
| 2 | Architecture V2 summary | `architecture/02-ARCHITECTURE-V2.md` | Ready |
| 3 | ADR decisions | `adr/` | Ready |
| 4 | Procedure JSON Schema | `schemas/procedure_definition.schema.json` | Ready |
| 5 | Decision contract | `schemas/decision_contract.md` | Ready |
| 6 | Seed procedures | `seeds/` | Ready (4 seeds) |
| 7 | API contract | `api/openapi-notes.md` | Ready |
| 8 | DB model | `db/data-model.md` | Ready |
| 8b | Definition + full flows | `db/04-definition-and-flows.md` | Ready |
| 9 | Test matrix | `tests/test-matrix.md` | Ready |
| 10 | Done checklist | `99-PHASE-0-DONE-CHECKLIST.md` | Ready |
| 11 | Tech stack ADR | `adr/ADR-005-technology-stack.md` | Proposed |
| 12 | Tech lock gate | `05-TECH-LOCK.md` | Ready |

## Definition of Done — Phase 0

- [ ] Schema `procedure_definition` được approve
- [ ] Decision actions chỉ còn 3 loại: `ask_missing_slots` | `direct_answer` | `provide_final_guidance`
- [ ] Ít nhất 4 seed JSON review được (1–2/domain ưu tiên)
- [ ] API + DB model không còn open question blocker
- [ ] Test matrix đủ case thiếu-slot / đủ-slot / ngoài phạm vi
- [ ] ADR chính được ký duyệt (decision engine, human review, single xã)

## Next after Phase 0

→ Sprint 0 / Phase 1: Chat API + Decision Engine + seed `Đăng ký khai sinh` end-to-end.
