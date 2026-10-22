# Phase 0 Done Checklist

Khớp proposal §1.6 / §4. Bản checklist cũ (3 domain, 18 thủ tục, voice out-of-scope, ADR-003 review workspace) **không còn dùng**.

## A. Scope

- [ ] V1: 1 xã, **một** domain hộ tịch & chứng thực
- [ ] Out-of-scope: nộp hồ sơ, thanh toán, đất đai/BHYT công dân, OCR scan, draft workspace, TTS Bahnar, HA
- [ ] In-scope artifact: text + Bahnar→chữ Việt, slot, citation, upload PDF chữ, pgvector, MinIO, Docker
- [ ] Success: đủ slot → guidance + nguồn; demo giọng Bahnar → guidance Việt trên ít nhất một thủ tục hộ tịch

## B. Architecture & ADR

- [ ] Architecture V2 (simplified admin) reviewed
- [ ] ADR-001 Decision Policy Engine accepted
- [ ] ADR-002 JSON-driven procedures accepted
- [ ] ADR-003 **superseded** (không draft workspace)
- [ ] ADR-004 Single xã V1 accepted
- [ ] ADR-005 stack + pgvector + speech Python accepted

## C. Contracts

- [ ] `procedure_definition.schema.json` approved
- [ ] Decision contract approved
- [ ] DB model approved (sessions, slots, versions, documents, chunks, citations, speech tables)
- [ ] Biết delta `000001` vs model proposal

## D. Seeds (công dân V1)

- [ ] `dk_khai_sinh.json` reviewed
- [ ] `chung_thuc_ban_sao.json` reviewed
- [ ] Seed đất đai / BHYT **không** vào catalog công dân
- [ ] `manual_seed` chưa phải nguồn pháp lý

## E. Tests

- [ ] Test matrix: thiếu-slot / đủ-slot / direct / out-of-scope
- [ ] KS-01.. và CT-01 khớp JSON

## F. Open questions

1. `xa_id` triển khai? → seed `xa_chu_se`
2. Công dân bắt buộc login V1? → ________________
3. First-turn đủ slot: `DIRECT_ANSWER` (đã chốt trong engine)
4. PDF xã thật khi nào? → ________________
5. Checkpoint speech gắn artifact: sau RQ1 (proposal tuần 11)

## Sign-off

| Role | Name | Date | Sign |
|------|------|------|------|
| Product/Owner | | | |
| Backend | | | |
| AI/NLP | | | |

**Phase 0 status:** ☐ Pass  ☐ Pass with notes  ☐ Blocked
