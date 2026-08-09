# Test Matrix — Phase 0

Mục tiêu: khóa expected behavior trước khi code.  
Ký hiệu: **PASS** khi action + slots/guidance đúng contract.

## A. Đăng ký khai sinh (`dk_khai_sinh`)

| ID | User input | Expected action | Expected detail |
|----|------------|-----------------|-----------------|
| KS-01 | "Tôi muốn làm giấy khai sinh cho con tôi." | `ASK_MISSING_SLOTS` | missing = ask_now = đủ 3 slot; hỏi full 1 lượt |
| KS-02 | sau KS-01 trả đủ 3 ý trong 1 câu | `PROVIDE_FINAL_GUIDANCE` | extract 3 keys hợp lệ → đủ slot → guidance + citation |
| KS-03 | sau KS-01 chỉ trả được nơi sinh | `ASK_MISSING_SLOTS` | chỉ update `noi_sinh`; hỏi lại full missing còn lại |
| KS-04 | "Làm khai sinh, bé sinh BV tỉnh, bố mẹ đã kết hôn, có giấy chứng sinh" | `DIRECT_ANSWER` hoặc `PROVIDE_FINAL_GUIDANCE` | first-turn đủ slot → không hỏi lại |
| KS-05 | `co_giay_chung_sinh=false` | `ASK_MISSING_SLOTS` | kích hoạt conditional `nguoi_di_dang_ky`; hỏi full missing mới |

## B. Chứng thực bản sao (`chung_thuc_ban_sao`)

| ID | User input | Expected action | Expected detail |
|----|------------|-----------------|-----------------|
| CT-01 | "Tôi muốn chứng thực bằng đại học" | `DIRECT_ANSWER` | required_slots rỗng → trả checklist ngay + citation |
| CT-02 | "Chứng thực bản sao CCCD" | `DIRECT_ANSWER` | tương tự |

## C. Xin GPXD (`xin_giay_phep_xay_dung`)

| ID | User input | Expected action | Expected detail |
|----|------------|-----------------|-----------------|
| GP-01 | "Xin giấy phép xây dựng nhà" | `ASK_MISSING_SLOTS` | hỏi full required (`loai_cong_trinh`, `co_dat_dung_ten`) 1 lượt |
| GP-02 | chọn `nha_o_rieng_le` + đủ/thiếu diện tích | `ASK_MISSING_SLOTS` hoặc final | conditional `dien_tich_xay_dung` nếu thiếu |
| GP-03 | đủ slots | `PROVIDE_FINAL_GUIDANCE` | notes nhắc thẩm quyền mixed xã/huyện |

## D. Tra cứu quy hoạch (`tra_cuu_quy_hoach`)

| ID | User input | Expected action | Expected detail |
|----|------------|-----------------|-----------------|
| QH-01 | "Đất tôi có nằm quy hoạch không" | `ASK_MISSING_SLOTS` | hỏi `vi_tri_thua_dat` |
| QH-02 | cung cấp vị trí | `PROVIDE_FINAL_GUIDANCE` | hướng dẫn nơi tra cứu + citation |

## E. BHYT hộ gia đình (`dk_bhyt_ho_gia_dinh`)

| ID | User input | Expected action | Expected detail |
|----|------------|-----------------|-----------------|
| BH-01 | "Đăng ký BHYT hộ gia đình" | `ASK_MISSING_SLOTS` | hỏi full 2 required 1 lượt |
| BH-02 | đủ 2 required slots | `PROVIDE_FINAL_GUIDANCE` | checklist + citation |

## F. Guardrails / ngoài phạm vi

| ID | User input | Expected action | Expected detail |
|----|------------|-----------------|-----------------|
| OS-01 | "Làm hộ chiếu" | `OUT_OF_SCOPE` | không map 18 thủ tục |
| OS-02 | "Tư vấn đầu tư chứng khoán" | `OUT_OF_SCOPE` | ngoài domain |
| OS-03 | đổi topic giữa chừng từ khai sinh → chứng thực | switch procedure hoặc xác nhận lại | V1: hỏi confirm trước khi đổi `active_procedure_id` |

## G. Admin publish

| ID | Scenario | Expected |
|----|----------|----------|
| AD-01 | draft thiếu citation | validate fail |
| AD-02 | required_slot không có trong `slots` | validate fail |
| AD-03 | publish activate | version mới active, version cũ archived |
| AD-04 | rollback | active trỏ về version trước; chat dùng version mới active |

## H. Citation rule

| ID | Scenario | Expected |
|----|----------|----------|
| CI-01 | mọi `DIRECT_ANSWER` / `PROVIDE_FINAL_GUIDANCE` | `citations.length >= 1` |
| CI-02 | seed chưa có PDF | cho phép `source_type=manual_seed` ở staging; prod UAT yêu cầu thay official |
