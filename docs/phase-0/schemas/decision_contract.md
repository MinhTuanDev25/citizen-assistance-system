# Decision Contract (Runtime)

> Naming: JSON definition / API human field = `procedure_code` (e.g. `dk_khai_sinh`). DB technical FK = `procedure_id` (uuid → `procedures.id`).

## 1. Mục tiêu

Khóa output của Decision Policy Engine để backend/frontend/AI cùng một hợp đồng.

## 2. Actions (chỉ 3)

| action | Khi nào | Citizen UX |
|--------|---------|------------|
| `ask_missing_slots` | Còn required/conditional slot thiếu | Hỏi **full** danh sách slot thiếu trong **1 lượt** |
| `direct_answer` | Không cần slot, hoặc user đã cung cấp đủ ngay từ đầu và policy cho phép | Trả hướng dẫn ngay |
| `provide_final_guidance` | Session đã đủ slot (sau khi user trả lời bổ sung hoặc từ câu đầu) | Trả checklist + nơi nộp + citation |

## 3. Ai viết câu hỏi? Ai extract slot?

### 3.1 Câu hỏi bổ sung → lấy từ JSON (source of truth)

- Mỗi slot trong `definition.slots[slot].question` là câu hỏi chuẩn.
- Decision Engine trả về `questions[]` map 1-1 từ missing slots.
- **LLM (nếu dùng) chỉ được diễn đạt lại cho tự nhiên**, không được bịa thêm slot / đổi nghĩa câu hỏi.
- V1 khuyến nghị: ghép list câu hỏi từ JSON thành 1 reply (đánh số 1,2,3…) — deterministic, dễ test.

### 3.2 Extract câu trả lời → LLM có ràng buộc schema (không free-form)

Khi user trả lời, **không** để LLM tự nghĩ key tùy ý.

Pipeline:

1. Lấy danh sách **allowed keys** = missing slots hiện tại (hoặc toàn bộ slots của procedure).
2. Gọi LLM / NLU với **structured output** bắt buộc đúng schema, ví dụ:

```json
{
  "extracted": {
    "noi_sinh": "Bệnh viện Đa khoa tỉnh",
    "da_ket_hon": true,
    "co_giay_chung_sinh": null
  },
  "unresolved": ["co_giay_chung_sinh"]
}
```

3. Validator phía backend:
   - bỏ key không nằm trong allowed set
   - type-check theo `definition.slots[key].type` (boolean/enum/string/number)
   - enum phải thuộc `enum_values`
   - chỉ UPDATE `session_slot_states` với key hợp lệ
4. Slot vẫn `null` / thiếu → giữ `status=missing`, hỏi lại **các slot còn thiếu** (full missing còn lại).

**Vì sao biết đúng key?**  
Vì extract luôn bị **constrain** bởi `definition.slots` + `missing_slots` của procedure đang active — không phải đoán tự do.

## 4. Request context (input engine)

```json
{
  "session_id": "sess_001",
  "xa_id": "xa_demo_001",
  "user_message": "Tôi muốn làm giấy khai sinh cho con tôi.",
  "detected": {
    "domain": "ho_tich_chung_thuc",
    "procedure_code": "dk_khai_sinh",
    "confidence": 0.92
  },
  "extracted_slots": {},
  "slot_state": {
    "noi_sinh": { "value": null, "status": "missing" },
    "da_ket_hon": { "value": null, "status": "missing" },
    "co_giay_chung_sinh": { "value": null, "status": "missing" }
  },
  "procedure_version": "1.0.0"
}
```

> `slot_state`: map theo slot key; `status` ∈ `missing` | `known` | `confirmed`.  
> API response vẫn trả thêm `missing_slots[]` (derived) cho UI.

## 5. Response envelopes

### 5.1 ask_missing_slots (hỏi full missing)

```json
{
  "action": "ask_missing_slots",
  "procedure_code": "dk_khai_sinh",
  "procedure_version": "1.0.0",
  "missing_slots": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
  "ask_now": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
  "questions": [
    {
      "slot": "noi_sinh",
      "text": "Bé sinh ở đâu (bệnh viện/cơ sở y tế hay tại nhà, thuộc xã/phường nào)?"
    },
    {
      "slot": "da_ket_hon",
      "text": "Cha mẹ bé đã đăng ký kết hôn chưa?"
    },
    {
      "slot": "co_giay_chung_sinh",
      "text": "Anh/chị có giấy chứng sinh không?"
    }
  ],
  "reply_text": "Để hướng dẫn đăng ký khai sinh, anh/chị cho mình biết:\n1) Bé sinh ở đâu (bệnh viện/cơ sở y tế hay tại nhà, thuộc xã/phường nào)?\n2) Cha mẹ bé đã đăng ký kết hôn chưa?\n3) Anh/chị có giấy chứng sinh không?",
  "slot_state": {
    "noi_sinh": { "value": null, "status": "missing" },
    "da_ket_hon": { "value": null, "status": "missing" },
    "co_giay_chung_sinh": { "value": null, "status": "missing" }
  }
}
```

### 5.2 direct_answer

```json
{
  "action": "direct_answer",
  "procedure_code": "chung_thuc_ban_sao",
  "procedure_version": "1.0.0",
  "guidance": {
    "summary": "Để chứng thực bản sao, mang bản chính và bản photo đến bộ phận tiếp nhận của xã.",
    "checklist": [
      "Bản chính giấy tờ",
      "Bản photo cần chứng thực",
      "CCCD người yêu cầu"
    ],
    "where_to_submit": "Bộ phận tiếp nhận và trả kết quả — UBND xã"
  },
  "citations": [
    {
      "doc_id": "seed_chung_thuc_ban_sao",
      "title": "Hướng dẫn chứng thực bản sao (seed)",
      "source_type": "manual_seed"
    }
  ]
}
```

### 5.3 provide_final_guidance

```json
{
  "action": "provide_final_guidance",
  "procedure_code": "dk_khai_sinh",
  "procedure_version": "1.0.0",
  "filled_slots": {
    "noi_sinh": "Bệnh viện Đa khoa tỉnh",
    "da_ket_hon": true,
    "co_giay_chung_sinh": true
  },
  "guidance": {
    "summary": "Hồ sơ đăng ký khai sinh đã đủ thông tin định hướng.",
    "checklist": [
      "Giấy chứng sinh",
      "CCCD của cha/mẹ",
      "Giấy đăng ký kết hôn (bản sao/chứng thực nếu được yêu cầu)"
    ],
    "where_to_submit": "UBND xã theo nơi cư trú của cha/mẹ hoặc nơi sinh (theo hướng dẫn địa phương)",
    "notes": [
      "Nên đăng ký trong thời hạn quy định để tránh phát sinh thủ tục bổ sung."
    ]
  },
  "citations": [
    {
      "doc_id": "seed_dk_khai_sinh",
      "title": "Hướng dẫn đăng ký khai sinh (seed)",
      "source_type": "manual_seed"
    }
  ]
}
```

## 6. Pseudo-code quyết định

```text
load active procedure definition
extract_slots(user_message, allowed_keys=procedure.slots.keys)  # structured + validate
merge into slot_state (only valid keys/types)
recompute effective_required = required_slots + conditional_slots(matched)
missing = [s in effective_required if slot_state[s].status == missing or value empty]

if missing empty:
  if first turn complete: return direct_answer
  else: return provide_final_guidance
else:
  ask_now = missing                    # FULL missing, không cắt 1 slot
  questions = [definition.slots[s].question for s in ask_now]
  return ask_missing_slots
```

## 7. Quy ước V1

- `ask_mode = all_missing`: mỗi lượt hỏi **toàn bộ** slot đang thiếu.
- Câu hỏi lấy từ `definition.slots.*.question` (JSON), không để LLM tự tạo nội dung nghiệp vụ.
- Extract slot = structured output **constrained** theo allowed keys + type/enum; backend validate trước khi ghi `session_slot_states`.
- Không hỏi lại slot đã `confirmed`.
- Mọi final/direct answer **bắt buộc** có `citations`.
- Không detect được procedure → `out_of_scope`.
