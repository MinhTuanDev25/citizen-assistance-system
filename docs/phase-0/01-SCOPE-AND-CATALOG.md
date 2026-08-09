# Scope V1 & Procedure Catalog

## 1. Business scope

| Item | V1 decision |
|------|-------------|
| Jurisdiction | **1 xã duy nhất** (`xa_id` cấu hình cố định) |
| Product | AI trợ lý hành chính công |
| Channels | **Text** (bắt buộc V1). **Voice** (có trong scope, mode chưa chốt — xem mục 5) |
| Success metric | Hỏi đúng slot thiếu (full missing) → đủ slot thì hướng dẫn đúng + có nguồn |
| Out of scope V1 | Nộp hồ sơ online, thanh toán phí, multi-xã |
| Knowledge source | PDF/văn bản xã (khi có) + seed thủ công tạm thời |

## 2. Domains & procedures (18)

### Domain A — Hộ tịch & Chứng thực (`ho_tich_chung_thuc`)

| procedure_id | Tên | Priority V1 |
|--------------|-----|-------------|
| `dk_khai_sinh` | Đăng ký khai sinh | P0 (seed) |
| `dk_ket_hon` | Đăng ký kết hôn | P1 |
| `dk_khai_tu` | Đăng ký khai tử | P1 |
| `xn_tinh_trang_hon_nhan` | Xác nhận tình trạng hôn nhân | P1 |
| `chung_thuc_ban_sao` | Chứng thực bản sao | P0 (seed) |
| `chung_thuc_chu_ky` | Chứng thực chữ ký | P1 |

### Domain B — Đất đai, Nhà ở & Quy hoạch (`dat_dai_nha_o_quy_hoach`)

| procedure_id | Tên | Priority V1 |
|--------------|-----|-------------|
| `cap_gcn_qsd_lan_dau` | Cấp GCN quyền sử dụng đất lần đầu | P1 |
| `tach_thua` | Tách thửa | P1 |
| `hop_thua` | Hợp thửa | P1 |
| `chuyen_muc_dich_sd_dat` | Chuyển mục đích sử dụng đất | P1 |
| `tra_cuu_quy_hoach` | Tra cứu quy hoạch | P0 (seed) |
| `xin_giay_phep_xay_dung` | Xin giấy phép xây dựng | P0 (seed) |

### Domain C — Bảo hiểm & Chính sách xã hội (`bao_hiem_chinh_sach_xh`)

| procedure_id | Tên | Priority V1 |
|--------------|-----|-------------|
| `dk_bhyt_tre_em` | Đăng ký BHYT trẻ em | P1 |
| `dk_bhyt_ho_gia_dinh` | Đăng ký BHYT hộ gia đình | P0 (seed) |
| `ho_ngheo` | Hộ nghèo | P1 |
| `ho_can_ngheo` | Hộ cận nghèo | P1 |
| `tro_cap_nguoi_cao_tuoi` | Trợ cấp người cao tuổi | P1 |
| `tro_cap_nguoi_khuyet_tat` | Trợ cấp người khuyết tật | P1 |

## 3. Runtime behavior (non-negotiable)

1. Nhận câu hỏi công dân → detect domain + procedure.
2. Load `procedure_definition` version **active**.
3. So khớp slot đã biết vs `required_slots` (và conditional slots nếu có).
4. Nếu thiếu → `ASK_MISSING_SLOTS` (**hỏi full danh sách slot thiếu trong 1 lượt**, không hỏi từng câu).
5. Nếu đủ → `PROVIDE_FINAL_GUIDANCE` (+ citation).
6. Nếu câu hỏi đã đủ thông tin / thủ tục không cần slot → `DIRECT_ANSWER`.

## 4. Assumptions (Phase 0)

- Chưa có PDF thật → seed dùng `source_type = manual_seed`, bắt buộc thay bằng văn bản xã trước UAT/prod.
- Nội dung hướng dẫn trong seed là **khung nghiệp vụ**, cán bộ xã phải review trước khi active production.
- Một số thủ tục đất đai/GPXD có thể cần chuyển cấp huyện; seed phải ghi rõ `authority_level`.

## 5. Channel: Text + Voice (chưa chốt mode voice)

| Input | Output | Ghi chú |
|-------|--------|---------|
| Text | Text | Core V1 — luôn có |
| Voice | Text | **Đề xuất V1.1**: STT → cùng pipeline text → reply chữ (dễ, ổn định, dễ audit) |
| Voice | Voice | Voice-to-voice: thêm TTS; phức tạp hơn (latency, giọng, đọc checklist dài) |

**Kiến trúc nên tách:**

```text
[Text UI] ────────┐
                  ├──► cùng Conversation + Decision + definition JSON
[Voice] → STT ────┘              │
                                 ▼
                            reply_text
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
                 hiện chữ              (optional) TTS → nghe
```

Nghĩa là **não hệ thống luôn chạy trên text + JSON**. Voice chỉ là lớp vào/ra.

**Khuyến nghị:**  
1) V1 ship text trước.  
2) Voice sớm thì làm **voice → text reply**.  
3) Voice-to-voice chỉ khi checklist/câu hỏi ổn định và có nhu cầu thực tế từ xã.
