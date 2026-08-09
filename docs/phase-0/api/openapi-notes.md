# API Contract (Phase 0)

Base path giả định: `/api/v1`  
Auth: Bearer JWT (`citizen` | `admin`)

## 1. Citizen — Chat

### `POST /chat/turns`

Bắt đầu/tiếp tục một lượt hội thoại.

**Request**

```json
{
  "session_id": "sess_001",
  "message": "Tôi muốn làm giấy khai sinh cho con tôi."
}
```

`session_id` optional ở turn đầu — server tạo mới nếu thiếu.

**Response 200** (một trong các action)

```json
{
  "session_id": "sess_001",
  "action": "ask_missing_slots",
  "procedure_code": "dk_khai_sinh",
  "procedure_version": "1.0.0",
  "reply_text": "Bé sinh ở đâu (bệnh viện/cơ sở y tế hay tại nhà, thuộc xã/phường nào)?",
  "ask_now": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
  "missing_slots": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
  "questions": [
    { "slot": "noi_sinh", "text": "Bé sinh ở đâu...?" },
    { "slot": "da_ket_hon", "text": "Cha mẹ bé đã đăng ký kết hôn chưa?" },
    { "slot": "co_giay_chung_sinh", "text": "Anh/chị có giấy chứng sinh không?" }
  ],
  "citations": [],
  "debug": {
    "domain": "ho_tich_chung_thuc",
    "intent_confidence": 0.91
  }
}
```

> V1: `ask_now` = full `missing_slots` (không hỏi từng câu).
Khi final/direct:

```json
{
  "session_id": "sess_001",
  "action": "provide_final_guidance",
  "procedure_code": "dk_khai_sinh",
  "procedure_version": "1.0.0",
  "reply_text": "Anh/chị chuẩn bị các giấy tờ sau...",
  "guidance": { "checklist": ["..."], "where_to_submit": "..." },
  "citations": [{ "doc_id": "seed_dk_khai_sinh_v1", "title": "...", "source_type": "manual_seed" }]
}
```

### `GET /chat/sessions/{session_id}`

Trả conversation history + slot_state hiện tại.

### Out-of-scope response

```json
{
  "session_id": "sess_001",
  "action": "out_of_scope",
  "reply_text": "Hiện trợ lý chỉ hỗ trợ thủ tục hành chính của xã trong 3 nhóm: Hộ tịch & Chứng thực; Đất đai, Nhà ở & Quy hoạch; Bảo hiểm & Chính sách xã hội."
}
```

## 2. Admin — Knowledge pipeline

### `POST /admin/documents`

Upload PDF/text nguồn.

- multipart: `file` + `xa_id` + `domain` (optional)

**Response**

```json
{ "document_id": "doc_001", "status": "uploaded" }
```

### `POST /admin/documents/{document_id}/extract`

Sinh draft procedure JSON (LLM assist).

**Response**

```json
{
  "draft_id": "draft_001",
  "status": "draft",
  "procedure_draft": { "...procedure_definition..." }
}
```

### `GET /admin/drafts/{draft_id}`

Lấy draft để review.

### `PUT /admin/drafts/{draft_id}`

Admin sửa draft JSON.

### `POST /admin/drafts/{draft_id}/validate`

Chạy schema + rule checks.

**Response**

```json
{
  "valid": false,
  "errors": [
    { "path": "$.required_slots[0]", "message": "slot not defined in slots" }
  ]
}
```

### `POST /admin/drafts/{draft_id}/publish`

Approve + publish version.

**Request**

```json
{ "activate": true, "changelog": "Seed khai sinh v1" }
```

**Response**

```json
{
  "procedure_code": "dk_khai_sinh",
  "version": "1.0.0",
  "status": "active"
}
```

### `POST /admin/procedures/{procedure_id}/rollback`

```json
{ "to_version": "1.0.0" }
```

### `GET /admin/procedures`

List procedures + active version.

### `GET /admin/procedures/{procedure_id}/versions`

## 3. Error model

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid procedure draft",
    "details": []
  }
}
```

Codes chính: `UNAUTHORIZED`, `FORBIDDEN`, `VALIDATION_ERROR`, `NOT_FOUND`, `CONFLICT_VERSION`, `LLM_UNAVAILABLE`.

## 4. Non-goals API V1

- Streaming token-by-token (có thể thêm sau)
- Webhook nộp hồ sơ
- Multi-tenant path prefix theo xã
