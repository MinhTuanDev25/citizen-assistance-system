# Scope V1 & Procedure Catalog

Nguồn khóa: capstone proposal *Quality-Aware Low-Resource Bahnar-to-Vietnamese Speech Translation*, §1.6 và §4.

## 1. Business scope

| Item | V1 (proposal) |
|------|----------------|
| Jurisdiction | 1 xã (`xa_id` cấu hình cố định, seed: `xa_chu_se`) |
| Product | Prototype trợ lý hành chính xã, gắn mô hình dịch giọng Bahnar → chữ Việt đã chọn |
| Administrative domain | **Một nhóm Hộ tịch & Chứng thực** (`ho_tich_chung_thuc`) với definition + workflow |
| Channels | Text tiếng Việt **và** giọng Bahnar → chữ Việt. Trả lời bằng chữ Việt |
| Success (artifact) | Một luồng hộ tịch hoàn chỉnh: hỏi/slot → hướng dẫn có citation; input giọng Bahnar vào được guidance tiếng Việt |
| Knowledge | PDF văn bản xã (chữ trích được) + seed tạm. RAG chỉ trên chunk của version **ACTIVE** |
| Operations | PostgreSQL + pgvector, S3/MinIO, `audit_logs`, Docker, CI/CD tối thiểu |

### Có trong V1

- Công dân: đăng nhập khi cần, hỏi text hoặc ghi giọng Bahnar, lịch sử hội thoại, hỏi slot còn thiếu một lượt, hướng dẫn có nguồn
- Cán bộ: tải PDF, metadata, gắn tự động `procedure_version_documents`, trạng thái index, **kích hoạt** version
- pgvector: embedding của `knowledge_chunks` (không dùng để chọn thủ tục)

### Không thuộc V1 (proposal §1.6)

| Loại | Excluded |
|------|----------|
| Nghiên cứu | Dịch Việt → Bahnar, TTS Bahnar, train foundation từ đầu |
| Domain | Đất đai, bảo hiểm / chính sách xã hội, mở rộng multi-domain |
| Công dân | Nộp hồ sơ online, thanh toán, chữ ký số, ra quyết định pháp lý |
| Admin | AI draft workspace, quy trình duyệt nhiều bước |
| Ops | HA enterprise, rollback phức tạp, tích hợp hệ thống nhà nước |
| PDF | OCR file scan — V1 chỉ PDF **trích được chữ**. Rollback version: làm nếu còn thời gian |

Hệ thống chỉ hướng dẫn thông tin. Quyết định hành chính vẫn thuộc cán bộ xã.

## 2. Catalog công dân V1 — Hộ tịch & Chứng thực

| procedure_code | Tên | Ghi chú |
|----------------|-----|---------|
| `dk_khai_sinh` | Đăng ký khai sinh | P0 — có slot bắt buộc + slot điều kiện |
| `chung_thuc_ban_sao` | Chứng thực bản sao | P0 — `required_slots` rỗng → `DIRECT_ANSWER` |

Các mã hộ tịch khác (`dk_ket_hon`, `dk_khai_tu`, …) có thể thêm **trong cùng domain** khi có definition; không mở domain mới.

### Ngoài phạm vi công dân V1 (seed DB có thể còn, không đưa catalog / matcher)

| Domain | procedure_code | Lý do |
|--------|----------------|-------|
| `dat_dai_nha_o_quy_hoach` | `tra_cuu_quy_hoach`, `xin_giay_phep_xay_dung`, … | Proposal loại đất đai |
| `bao_hiem_chinh_sach_xh` | `dk_bhyt_ho_gia_dinh`, … | Proposal loại bảo hiểm / phúc lợi |

## 3. Runtime (không đổi so với decision contract)

1. Input: chữ Việt **hoặc** giọng Bahnar đã dịch thành chữ Việt.
2. Nhận diện thủ tục trong catalog hộ tịch đang ACTIVE.
3. Decision Engine (Go, deterministic): `ASK_MISSING_SLOTS` / `DIRECT_ANSWER` / `PROVIDE_FINAL_GUIDANCE` / `OUT_OF_SCOPE`.
4. Thiếu slot → hỏi **hết** câu còn thiếu trong một lượt (câu lấy từ JSON).
5. Đủ slot → RAG chỉ trên chunk của **procedure version đã pin**, trả guidance + citation. RAG không đè checklist JSON.
6. Session pin `active_procedure_id` + `active_procedure_version_id` khi đã chọn thủ tục.

## 4. Voice trong artifact

Não hệ thống luôn chạy trên **chữ Việt + JSON**.

```text
[Text UI] ─────────────────────────────┐
                                       ├──► session + Decision Engine + definition
[Mic Bahnar] → S2TT/ASR+MT (Python) ──┘              │
                                                     ▼
                                              reply_text (tiếng Việt)
                                                     │
                                              không TTS Bahnar (V1)
```

TTS tiếng Bahnar và dịch Việt → Bahnar **không** thuộc V1. Thứ tự làm phần mềm có thể ship text trước, rồi gắn checkpoint đã chọn (proposal tuần 11).

## 5. Seed

Chưa có PDF xã → `source_type = manual_seed`, phải thay bằng văn bản xã trước demo/UAT. Nội dung seed là khung, không phải nguồn pháp lý.
