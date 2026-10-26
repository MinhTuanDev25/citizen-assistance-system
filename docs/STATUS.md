# Trạng thái Software V1

Cập nhật: 2026-10-26. Thay cho `tien-do.md`, `phase-con-lai.md` và
`IMPLEMENTATION-PLAN-A-TO-Z.md` (bản cũ ở `docs/archive/`).

File này ghi code đang có và mức bằng chứng của từng phần. Nó không phải biên
bản nghiệm thu. Bộ evidence chi tiết (`reports/`, gitignored) nằm trên máy dev và
bản sao lưu ngoài repo.

## Phạm vi V1

Một xã (Chư Sê, `xa_chu_se`), chat tiếng Việt về thủ tục hộ tịch và chứng thực:
nhận thủ tục, hỏi slot còn thiếu, trả checklist lấy từ JSON thủ tục đang ACTIVE,
kèm trích dẫn đoạn PDF của đúng phiên bản đó. Admin upload PDF, index, sinh bản
nháp thủ tục bằng AI, duyệt, rồi kích hoạt. Người dân nói tiếng Bahnar, hệ thống
dịch sang tiếng Việt, người dân xác nhận rồi câu đó đi vào cùng luồng chat.

Ngoài phạm vi V1: nộp hồ sơ, thanh toán, chữ ký số, TTS Bahnar, nhiều xã, HA.
Chat công dân chỉ đọc domain `ho_tich_chung_thuc`; hai domain còn lại chỉ admin thấy.

## Trạng thái theo phase

| Phase | Nội dung | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1–3 | Chat chữ, Decision Engine Go, admin nhận PDF, index PDF (native + OCR, E5 384 chiều, Qdrant) | Local PASS 2026-10-12/13 | `docs/archive/PHASE_1_3_*` |
| 4 | Candidate version, eligibility, Activate atomic + idempotent, session pin | Local PASS 2026-10-13 | `docs/archive/PHASE_4_*` |
| 4.5 | AI Draft định nghĩa thủ tục từ PDF, review, approve, activate | `PHASE45_LIFECYCLE_LOCAL_PASS` (provider mock grounded trên PDF thật). `PHASE45_GEMINI_DRAFT_E2E=NOT_VERIFIED` | `reports/phase45_*` |
| 5 | RAG: retrieve chunk đúng `xa_id` + phiên bản ghim, citation trên bubble | Chạy local; `PHASE_5_PASS_LOCAL=false`, `G3_CONTENT_SIGNOFF=PENDING` (chờ người duyệt nội dung) | `reports/phase5_final_acceptance` |
| 6 | LLM extract thật (Gemini), keyword fallback | `PHASE6_LOCAL_PASS` | `reports/phase6_final_closure` |
| 7 | Bahnar Speech C0 (ASR + MT) trên RunPod GPU, confirm → Chat Turn, audit không lưu transcript | Step 7.6 đóng; operator E2E PASS; Docker local → RunPod qua `speech-tunnel` dịch thật 2026-10-26 | `reports/phase7_step76_closure` |
| 8 | Demo HTTPS, backup, UAT, metric | **Chưa làm** | — |

Cơ chế chính đã có trong code:

- Go `decision.Decide` quyết định action (`ASK_MISSING_SLOTS`, `DIRECT_ANSWER`,
  `PROVIDE_FINAL_GUIDANCE`, `OUT_OF_SCOPE`, `CONFIRM_INTENT`). LLM chỉ đề xuất
  thủ tục và slot; lỗi, timeout hay rate-limit thì về từ khoá.
- Chat Turn idempotent theo `X-Request-ID` + hash message (`turn_idempotency`).
- Index pipeline: claim token, generation, reindex giữ bản cũ đến khi bản mới
  được kiểm, reaper dọn generation mồ côi.
- Speech: Go chỉ gọi Speech qua loopback; audit `speech_translation_requests`
  liên kết Chat Turn qua `chat_turn_request_id` (unique, migration `000022`).

Danh sách endpoint: Swagger của API Go (`/swagger/index.html`) và
[`apps/api/README.md`](../apps/api/README.md).

## Việc còn lại

1. **Phase 5 sign-off nội dung G3**: người duyệt xác nhận nội dung trích dẫn.
2. **Phase 4.5 với Gemini thật**: một lần admin HTTP sinh draft bằng Gemini, review, rồi chat trên definition đó.
3. **Go-live**: theo [`RUNPOD.md`](RUNPOD.md) §7 (kế hoạch, chưa kiểm chứng).
4. **Phase 8**:
   - HTTPS, health check, backup Postgres có thử restore.
   - E2E giọng đúng điều kiện proposal: audio Bahnar thật → dịch → Decision Engine
     → RAG → hướng dẫn tiếng Việt kèm citation PDF.
   - Metric intent, slot, retrieval, citation; performance và security; UAT với
     cán bộ và người dân.
   - Ghi nhận kiến trúc thực tế: proposal mô tả pgvector, code dùng Qdrant
     (384 chiều, e5-small). Không viết lại sang pgvector.

## Luận văn Bahnar (RQ1/RQ2)

Nằm ở `bahnar-s2tt-thesis/`, tách khỏi ứng dụng. So sánh cascaded C0 (ASR + MT)
với direct D0. Ứng dụng chỉ dùng `src/c0_speech` và checkpoint C0 đã chọn
(xem `RUNPOD.md` §1). Trạng thái nghiên cứu xem `bahnar-s2tt-thesis/README.md`.
