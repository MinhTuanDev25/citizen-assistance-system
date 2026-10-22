# Phase 0 — Design Lock (Pre-coding)

**Project:** Citizen assistance artifact (1 xã, hộ tịch) + nghiên cứu S2TT Bahnar→Việt  
**Nguồn khóa:** capstone proposal (2026) §1.6, §4  
**Status:** Docs đã align lại với proposal (không còn 18 thủ tục / draft workspace như bản Phase 0 cũ)

## Deliverables

| # | Artifact | Path | Status |
|---|----------|------|--------|
| 1 | Scope — 1 domain hộ tịch; đất đai/BHYT ngoài V1 | `01-SCOPE-AND-CATALOG.md` | Aligned |
| 2 | Architecture (upload→embed→activate; speech→text) | `architecture/02-ARCHITECTURE-V2.md` | Aligned |
| 3 | ADR | `adr/` | ADR-003 superseded; ADR-005 aligned |
| 4 | Procedure JSON Schema | `schemas/procedure_definition.schema.json` | Ready |
| 5 | Decision contract | `schemas/decision_contract.md` | Ready |
| 6 | Seed | `seeds/` | P0: khai sinh, chứng thực |
| 7 | API notes | `api/openapi-notes.md` | Ready |
| 8 | DB model + pgvector | `db/data-model.md` | Aligned; delta vs `000001` ghi ở đầu file |
| 8b | Definition + flows | `db/04-definition-and-flows.md` | Aligned |
| 9 | Test matrix | `tests/test-matrix.md` | Ready |
| 10 | Done checklist | `99-PHASE-0-DONE-CHECKLIST.md` | Aligned |
| 11 | Tech stack ADR | `adr/ADR-005-technology-stack.md` | Aligned |
| 12 | Tech lock | `05-TECH-LOCK.md` | Aligned |

## Definition of Done — Phase 0

- [x] Decision actions: `ASK_MISSING_SLOTS` \| `DIRECT_ANSWER` \| `PROVIDE_FINAL_GUIDANCE` (+ `OUT_OF_SCOPE`)
- [ ] Catalog công dân V1 = hộ tịch & chứng thực (không 3 nhóm)
- [ ] Admin V1 = upload + index + activate (không draft LLM)
- [ ] pgvector = RAG chunks, không chọn intent
- [ ] Voice artifact = Bahnar → chữ Việt; không TTS

## Next

Text chat + Decision Engine trên seed hộ tịch (đang làm). Speech gắn tuần 11 sau khi chọn checkpoint. RAG sau khi có PDF + embed.
