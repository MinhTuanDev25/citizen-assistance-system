# Phase 0 Done Checklist

Dùng checklist này để **ký duyệt Phase 0** trước khi mở Sprint 0 code.

## A. Scope

- [ ] Xác nhận V1: 1 xã, 3 domain, 18 thủ tục
- [ ] Xác nhận out-of-scope: nộp hồ sơ online, thanh toán, multi-xã, voice
- [ ] Xác nhận success metric: hỏi đúng slot thiếu → đủ thì guidance + nguồn

## B. Architecture & ADR

- [ ] Architecture V2 reviewed
- [ ] ADR-001 Decision Policy Engine accepted
- [ ] ADR-002 JSON-driven procedures accepted
- [ ] ADR-003 Human review before publish accepted
- [ ] ADR-004 Single xã V1 accepted

## C. Contracts

- [ ] `procedure_definition.schema.json` approved
- [ ] Decision contract (3 actions + OUT_OF_SCOPE) approved
- [ ] API notes approved (chat + admin)
- [ ] DB model approved (sessions, slot_state, versions, audit)

## D. Seeds

- [ ] `dk_khai_sinh.json` reviewed (slots đúng nghiệp vụ)
- [ ] `chung_thuc_ban_sao.json` reviewed (direct answer)
- [ ] `xin_giay_phep_xay_dung.json` reviewed
- [ ] `tra_cuu_quy_hoach.json` reviewed
- [ ] `dk_bhyt_ho_gia_dinh.json` reviewed
- [ ] Thống nhất: seed = `manual_seed`, chưa được coi là nguồn pháp lý prod

## E. Tests

- [ ] Test matrix đủ case thiếu-slot / đủ-slot / direct / out-of-scope / publish
- [ ] Team BA/nghiệp vụ confirm expected của KS-01..KS-05 và CT-01

## F. Open questions còn blocker?

Ghi rõ nếu còn. Nếu trống → Phase 0 pass.

1. `xa_id` thật của xã triển khai là gì? → ________________
2. Có bắt buộc login công dân ở V1 không? → ________________
3. First-turn đủ slot: dùng `DIRECT_ANSWER` hay `PROVIDE_FINAL_GUIDANCE`? → **đề xuất: `DIRECT_ANSWER`**
4. Khi nào có PDF/văn bản chính thức để thay seed? → ________________
5. Voice mode: `voice→text` hay `voice↔voice`? → **đề xuất V1.1: voice→text**

## Sign-off

| Role | Name | Date | Sign |
|------|------|------|------|
| Product/Owner | | | |
| Backend | | | |
| AI/NLP | | | |
| Nghiệp vụ xã (nếu có) | | | |

**Phase 0 status:** ☐ Pass  ☐ Pass with notes  ☐ Blocked
