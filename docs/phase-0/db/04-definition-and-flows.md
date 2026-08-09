  # Definition + Full Data Flows

## 1. `procedure_versions.definition` chứa gì?

Đây là **toàn bộ quy tắc nghiệp vụ của 1 thủ tục ở 1 version**, lưu dạng jsonb.  
Admin UI đọc/ghi object này qua form; runtime Decision Engine đọc để quyết định hỏi gì / trả gì.

### Cấu trúc (tóm tắt)

| Nhóm | Field | Dùng để làm gì |
|------|-------|----------------|
| Identity | `procedure_code`, `name`, `domain`, `xa_id`, `version`, `status` | Mã thủ tục (human); DB PK là `procedures.id` uuid |
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
  "procedure_code": "dk_khai_sinh",
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
 [5] Approve / Publish (xem data-model §6)
      │
      ├─► procedure_versions       status=indexing + source_draft_id
      ├─► procedure_version_documents
      ├─► embed + knowledge_chunks (ngoài txn dài)
      └─► SHORT TXN: archive cũ → active + procedures.active_version_id
                     + draft=published + audit_logs
```

### Chi tiết từng bước

| Bước | Hành động | Ghi vào bảng | Lưu những gì |
|------|-----------|--------------|--------------|
| 1 | Upload PDF | `documents` | metadata + `processing_status` / `validity_status` |
| 2 | Extract | (chưa publish) | LLM sinh object procedure |
| 3 | Tạo draft | `procedure_drafts` | `draft_definition`, `validation_result`, `status=draft` |
| 4 | Admin sửa | `procedure_drafts` | UPDATE definition + validate |
| 5a | Tạo version | `procedure_versions` | `procedure_id` (uuid), `definition`, `status=indexing`, `source_draft_id` |
| 5b | Link docs | `procedure_version_documents` | N–N version ↔ document |
| 5c | Embed | `knowledge_chunks` | `chunk_index`, `document_id`, `embedding vector(1536)` |
| 5d | Activate (short txn) | versions + `procedures` + drafts + `audit_logs` | archive cũ; `active`; `active_version_id`; draft=`published`; audit |
| Fail embed | Cleanup | chunks + versions | DELETE chunks theo version → `status=approved` → retry |

Chi tiết đầy đủ: [`data-model.md`](data-model.md) §6.

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
   │  detect domain + procedure_code (dk_khai_sinh) → resolve procedures.id (uuid)
   │  extract slots từ câu (nếu có)
   ▼
[C] Load procedures.active_version_id
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
| A | INSERT/SELECT `conversation_sessions` | `user_id`, `xa_id`, `status=open`, `active_procedure_id` (uuid), `active_procedure_version_id` (uuid) |
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
