  # Definition + Full Data Flows

## 1. `procedure_versions.definition` chứa gì?

Đây là **toàn bộ quy tắc nghiệp vụ của 1 thủ tục ở 1 version**, lưu dạng jsonb.  
Admin UI đọc/ghi object này qua form; runtime Decision Engine đọc để quyết định hỏi gì / trả gì.

### Cấu trúc (tóm tắt)

| Nhóm | Field | Dùng để làm gì |
|------|-------|----------------|
| Identity | `procedure_id`, `name`, `domain`, `xa_id`, `version`, `status` | Định danh thủ tục |
| Scope | `authority_level`, `description` | Xã/huyện/mixed; mô tả ngắn |
| NLU hints | `intent_examples` | Gợi ý nhận diện câu hỏi |
| Slot schema | `slots` | Mỗi slot: type, question, enum… |
| Rules | `required_slots`, `conditional_slots` | Slot bắt buộc / điều kiện |
| Ask policy | `ask_policy` | Hỏi mấy slot/lượt, khi nào xong |
| Answer | `guidance` | summary, checklist, nơi nộp, notes, phí, thời hạn |
| Provenance | `citations` | Nguồn (doc_id, title, source_type, effective_date) |

### Ví dụ rút gọn (khai sinh)

```json
{
  "procedure_id": "dk_khai_sinh",
  "domain": "ho_tich_chung_thuc",
  "name": "Đăng ký khai sinh",
  "xa_id": "xa_demo_001",
  "version": "1.0.0",
  "status": "active",
  "authority_level": "xa",
  "intent_examples": ["Tôi muốn làm giấy khai sinh cho con"],
  "slots": {
    "noi_sinh": { "type": "string", "question": "Bé sinh ở đâu...?" },
    "da_ket_hon": { "type": "boolean", "question": "Cha mẹ đã kết hôn chưa?" },
    "co_giay_chung_sinh": { "type": "boolean", "question": "Có giấy chứng sinh không?" }
  },
  "required_slots": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
  "conditional_slots": [],
  "ask_policy": { "ask_mode": "all_missing", "completion_policy": "all_required_slots" },
  "guidance": {
    "summary": "...",
    "checklist": ["Giấy chứng sinh", "CCCD cha/mẹ", "..."],
    "where_to_submit": "UBND xã ...",
    "notes": ["..."]
  },
  "citations": [
    {
      "doc_id": "...",
      "title": "...",
      "source_type": "pdf_official",
      "effective_date": "2026-01-01"
    }
  ]
}
```

**Không nhầm:**  
- `definition` = quy tắc thủ tục (catalog)  
- `slot_state` = trạng thái hội thoại của **1 user trong 1 session** (đã hỏi tới đâu)

---

## 2. Luồng Admin upload → lưu DB gì?

```text
Admin Upload PDF
      │
      ▼
 [1] documents                  ← file + metadata hiệu lực
      │
      ▼
 [2] LLM extract draft
      │
      ▼
 [3] procedure_drafts           ← draft_definition + validation_result
      │
      ▼
 [4] Admin Review UI (sửa form, không sửa raw JSON)
      │  validate lại → update validation_result
      ▼
 [5] Approve / Publish
      │
      ├─► procedure_versions    ← definition (snapshot), status=active
      ├─► procedures            ← active_version trỏ version mới
      ├─► (optional) archive version cũ
      └─► audit_logs            ← ai publish, version nào
```

### Chi tiết từng bước

| Bước | Hành động | Ghi vào bảng | Lưu những gì |
|------|-----------|--------------|--------------|
| 1 | Upload PDF | `documents` | `filename`, `storage_uri`, `checksum`, `xa_id`, `domain_id`, `effective_date`, `expire_date`, `status=active`, `uploaded_by` |
| 2 | Extract | (chưa publish) | LLM sinh object procedure; chưa active |
| 3 | Tạo draft | `procedure_drafts` | `document_id`, `draft_definition` (= bản definition nháp), `validation_result` (lần validate đầu), `status=draft` |
| 4 | Admin sửa trên UI | `procedure_drafts` | UPDATE `draft_definition`; mỗi lần Validate → UPDATE `validation_result` `{valid, errors[]}` |
| 5a | Publish | `procedure_versions` | INSERT: `procedure_id`, `version`, `definition` = copy từ `draft_definition`, `source_document_id`, `status=active`, `created_by`, `approved_by` |
| 5b | Trỏ active | `procedures` | UPSERT procedure; set `active_version`, `domain_id`, `name`, `xa_id` |
| 5c | Archive cũ | `procedure_versions` | version trước: `status=archived` |
| 5d | Đóng draft | `procedure_drafts` | `status=published` |
| 5e | Audit | `audit_logs` | `action=publish`, entity, payload `{procedure_id, version, document_id}` |

Phase 3 thêm: index embeddings từ document/definition → vector store (chưa có ở ER structured hiện tại).

---

## 3. Luồng Citizen hỏi → lưu DB gì?

Ví dụ: `"Tôi muốn làm giấy khai sinh cho con tôi."`

```text
User message
   │
   ▼
[A] Auth / session
   │  tạo hoặc tái sử dụng conversation_sessions
   ▼
[B] Conversation Manager
   │  detect domain + procedure_id (dk_khai_sinh)
   │  extract slots từ câu (nếu có)
   ▼
[C] Load procedures.active_version
   │  → procedure_versions.definition
   ▼
[D] Decision Policy Engine
   │  so definition.required_slots vs session slot_state
   │  → ask_missing_slots | direct_answer | provide_final_guidance
   ▼
[E] Persist + reply
```

### Turn 1 — thiếu slot (hỏi FULL missing một lượt)

| Bước | Đọc / Ghi | Nội dung |
|------|-----------|----------|
| A | INSERT/SELECT `conversation_sessions` | `user_id`, `xa_id`, `status=open`, `active_procedure_id=dk_khai_sinh`, `active_procedure_version=1.0.0` |
| A | INSERT `conversation_messages` | role=`user`, content=câu hỏi |
| C | READ `procedures` + `procedure_versions` | lấy `definition` active |
| D | UPSERT `session_slot_states` | `slot_state`: mọi required = `missing` |
| E | INSERT `conversation_messages` | role=`assistant`, `action=ask_missing_slots`, `reply` ghép **tất cả** `definition.slots.*.question` của missing |
| E | INSERT `audit_logs` | `action=chat_decision`, payload route + version |

`slot_state` sau turn 1:

```json
{
  "noi_sinh": { "value": null, "status": "missing" },
  "da_ket_hon": { "value": null, "status": "missing" },
  "co_giay_chung_sinh": { "value": null, "status": "missing" }
}
```

### Turn 2 — user trả lời (có thể trả nhiều ý trong 1 câu)

Pipeline extract:

1. Allowed keys = missing hiện tại (`noi_sinh`, `da_ket_hon`, `co_giay_chung_sinh`)
2. LLM structured output map vào đúng các key đó
3. Backend validate type/enum → UPDATE `session_slot_states`
4. Slot còn thiếu → hỏi lại **full missing còn lại**; đủ → `provide_final_guidance`

| Bước | Ghi | Nội dung |
|------|-----|----------|
| | INSERT message user | câu trả lời |
| | UPDATE `session_slot_states` | chỉ các key extract hợp lệ |
| | INSERT message assistant | `ask_missing_slots` (nếu còn thiếu) hoặc `provide_final_guidance` |

### Nhánh `direct_answer` (vd chứng thực bản sao)

- `definition.required_slots = []`
- Không cần hỏi thêm
- INSERT message assistant với `action=direct_answer` + guidance ngay
- `session_slot_states` có thể trống/`{}`

---

## 4. Bản đồ “ai sở hữu dữ liệu gì”

```text
┌─────────────────────────────┐
│ Catalog (admin publish)     │
│ domains                     │
│ procedures                  │
│ procedure_versions.definition │  ← quy tắc hỏi/trả
│ documents                   │  ← nguồn PDF + hiệu lực
│ procedure_drafts            │  ← nháp review UI
└─────────────────────────────┘

┌─────────────────────────────┐
│ Runtime (citizen chat)      │
│ conversation_sessions       │
│ conversation_messages       │
│ session_slot_states         │  ← tiến độ hỏi của user
└─────────────────────────────┘

┌─────────────────────────────┐
│ Cross-cutting               │
│ users / audit_logs          │
└─────────────────────────────┘
```

## 5. Một câu nhớ nhanh

- **Admin publish** ghi **định nghĩa thủ tục** (`definition`).  
- **User hỏi** ghi **tiến độ hội thoại** (`messages` + `slot_state`), rồi **đọc** `definition` để biết hỏi tiếp hay trả guidance.
