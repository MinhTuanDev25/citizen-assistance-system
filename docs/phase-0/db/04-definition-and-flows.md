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
  "xa_id": "xa_chu_se",
  "version": "1.0.0",
  "status": "ACTIVE",
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

Proposal §4.2 — **không** extract draft JSON.

```text
Admin mở procedure_version + Upload PDF
      │
      ▼
 [1] documents                  ← file MinIO + metadata hiệu lực
 [1b] procedure_version_documents  ← tự INSERT (relationship, page_range)
      │
      ▼
 [2] Job extract text → chunk → embed
      │  documents.processing_status UPLOADED → PROCESSING → READY | FAILED
      │  knowledge_chunks.embedding vector(1536)
      ▼
 [3] Admin thấy index status → activate
      │
      ├─► procedure_versions.status = ACTIVE
      ├─► procedures.active_version_id
      └─► audit_logs
```

| Bước | Hành động | Ghi vào bảng |
|------|-----------|--------------|
| 1 | Upload PDF | `documents` |
| 1b | Gắn nguồn tự động | `procedure_version_documents` |
| 2 | Embed | `knowledge_chunks` |
| 3 | Activate | `procedure_versions`, `procedures`, `audit_logs` |

OCR scan và rollback version **không** thuộc minimum. Chi tiết: [`data-model.md`](data-model.md) §6.

---

## 3. Luồng Citizen hỏi → lưu DB gì?

Ví dụ: `"Tôi muốn làm giấy khai sinh cho con tôi."`

```text
User message (chữ Việt, hoặc giọng Bahnar đã dịch)
   │
   ▼
[A] Auth / session
   │  tạo hoặc tái sử dụng conversation_sessions
   ▼
[B] Conversation Manager
   │  detect procedure trong catalog hộ tịch (extract structured / LLM)
   │  extract slots từ câu (allowed keys)
   ▼
[C] Load procedures.active_version_id
   │  → procedure_versions.definition
   ▼
[D] Decision Policy Engine
   │  so definition.required_slots vs session slot_state
   │  → ASK_MISSING_SLOTS | DIRECT_ANSWER | PROVIDE_FINAL_GUIDANCE
   ▼
[E] Persist + reply
```

### Turn 1 — thiếu slot (hỏi FULL missing một lượt)

| Bước | Đọc / Ghi | Nội dung |
|------|-----------|----------|
| A | INSERT/SELECT `conversation_sessions` | `user_id`, `xa_id`, `status=open`, `active_procedure_id` (uuid), `active_procedure_version_id` (uuid) |
| A | INSERT `conversation_messages` | role=`USER`, content=câu hỏi |
| C | READ `procedures` + `procedure_versions` | lấy `definition` active |
| D | UPSERT `session_slot_states` | `slot_state`: mọi required = `MISSING` |
| E | INSERT `conversation_messages` | role=`ASSISTANT`, `action=ASK_MISSING_SLOTS`, `reply` ghép **tất cả** `definition.slots.*.question` của missing |
| E | INSERT `audit_logs` | `action=chat_decision`, payload route + version |

`slot_state` sau turn 1:

```json
{
  "noi_sinh": { "value": null, "status": "MISSING" },
  "da_ket_hon": { "value": null, "status": "MISSING" },
  "co_giay_chung_sinh": { "value": null, "status": "MISSING" }
}
```

### Turn 2 — user trả lời (có thể trả nhiều ý trong 1 câu)

Pipeline extract:

1. Allowed keys = missing hiện tại (`noi_sinh`, `da_ket_hon`, `co_giay_chung_sinh`)
2. LLM structured output map vào đúng các key đó
3. Backend validate type/enum → UPDATE `session_slot_states`
4. Slot còn thiếu → hỏi lại **full missing còn lại**; đủ → `PROVIDE_FINAL_GUIDANCE`

| Bước | Ghi | Nội dung |
|------|-----|----------|
| | INSERT message user | câu trả lời |
| | UPDATE `session_slot_states` | chỉ các key extract hợp lệ |
| | INSERT message assistant | `ASK_MISSING_SLOTS` (nếu còn thiếu) hoặc `PROVIDE_FINAL_GUIDANCE` |

### Nhánh `DIRECT_ANSWER` (vd chứng thực bản sao)

- `definition.required_slots = []`
- Không cần hỏi thêm
- INSERT message assistant với `action=DIRECT_ANSWER` + guidance ngay
- `session_slot_states` có thể trống/`{}`

---

## 4. Bản đồ “ai sở hữu dữ liệu gì”

```text
┌─────────────────────────────┐
│ Catalog (admin activate)      │
│ domains                       │
│ procedures                    │
│ procedure_versions.definition │
│ documents + version_documents │
│ knowledge_chunks (pgvector)   │
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
