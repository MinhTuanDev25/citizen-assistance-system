# Data Model (PostgreSQL) — Phase 0

`xa_id` xuất hiện ở bảng nghiệp vụ chính dù V1 chỉ 1 xã.

## 1. ER overview

```text
domains
  └── procedures
        └── procedure_versions   ←── active pointer trên procedures

users
  └── conversation_sessions
        └── conversation_messages
        └── session_slot_states   (PK: session_id + procedure_id)

documents
  └── procedure_drafts
        └── (publish) procedure_versions

audit_logs
```

Vector DB (tách): embeddings/chunks gắn `doc_id`, `procedure_id`, `xa_id`, `version`.

## 2. Tables

### `domains`

Danh mục domain — tách bảng để sau thêm domain không phải sửa enum trong code.

| Column | Type | Notes |
|--------|------|-------|
| id | text PK | e.g. `ho_tich_chung_thuc` |
| name | text | Tên hiển thị |
| description | text null | |
| sort_order | int | Thứ tự UI |
| is_active | boolean | default true |
| created_at | timestamptz | |

Seed V1:

| id | name |
|----|------|
| `ho_tich_chung_thuc` | Hộ tịch & Chứng thực |
| `dat_dai_nha_o_quy_hoach` | Đất đai, Nhà ở & Quy hoạch |
| `bao_hiem_chinh_sach_xh` | Bảo hiểm & Chính sách xã hội |

### `users`

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| role | text | `citizen` \| `admin` |
| full_name | text | |
| phone | text null | |
| created_at | timestamptz | |

### `conversation_sessions`

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | session_id |
| user_id | uuid FK null | anonymous allowed V1? → nên có user hoặc guest token |
| xa_id | text | |
| active_procedure_id | text null | |
| active_procedure_version | text null | |
| status | text | `open` \| `completed` \| `abandoned` |
| created_at / updated_at | timestamptz | |

### `conversation_messages`

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| session_id | uuid FK | |
| role | text | `user` \| `assistant` \| `system` |
| content | text | |
| action | text null | decision action if assistant |
| message_metadata | jsonb | extracted slots, confidence |
| created_at | timestamptz | |

### `session_slot_states`

Lưu trạng thái thu thập slot theo **từng thủ tục trong session** (dân hỏi nhiều lượt).

| Column | Type | Notes |
|--------|------|-------|
| session_id | uuid FK | |
| procedure_id | text | |
| slot_state | jsonb | một object duy nhất — xem cấu trúc bên dưới |
| updated_at | timestamptz | |
| **PK** | `(session_id, procedure_id)` | 1 session đổi thủ tục → nhiều row |

#### Cấu trúc `slot_state` (một cột, không tách 3 cột)

```json
{
  "noi_sinh": {
    "value": "Bệnh viện Đa khoa tỉnh",
    "status": "confirmed"
  },
  "da_ket_hon": {
    "value": true,
    "status": "known"
  },
  "co_giay_chung_sinh": {
    "value": null,
    "status": "missing"
  }
}
```

| `status` | Ý nghĩa |
|----------|---------|
| `missing` | chưa có giá trị, cần hỏi |
| `known` | đã extract nhưng chưa xác nhận chắc |
| `confirmed` | đã xác nhận, không hỏi lại |

**Vì sao gộp 1 cột thay vì 3 cột `known` / `missing` / `confirmed`?**

- 3 cột trước đó mô tả cùng một khái niệm (trạng thái từng slot) nhưng bị xé ra → dễ lệch (slot vừa có trong `known` vừa còn trong `missing`)
- 1 `slot_state` map theo `slot_key` là source of truth; `missing` list có thể derive khi cần
- Admin UI / debug đọc một object là đủ

Runtime vẫn có thể compute nhanh:

```text
missing_slots = [k for k,v in slot_state.items() if v.status == "missing"]
# hoặc: required_slots - keys(status in known|confirmed)
```

### `procedures`

| Column | Type | Notes |
|--------|------|-------|
| procedure_id | text PK | |
| domain_id | text FK → domains.id | |
| name | text | |
| xa_id | text | |
| active_version | text null | |
| updated_at | timestamptz | |

### `procedure_versions`

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| procedure_id | text FK | |
| version | text | semver |
| status | text | `draft` / `approved` / `active` / `archived` |
| definition | jsonb | full procedure_definition |
| source_document_id | uuid null | |
| created_by | uuid | |
| approved_by | uuid null | |
| created_at / approved_at | timestamptz | |
| unique(procedure_id, version) | | |

### `documents`

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| xa_id | text | |
| domain_id | text FK null → domains.id | |
| filename | text | |
| storage_uri | text | |
| checksum | text | |
| effective_date | date null | ngày có hiệu lực |
| expire_date | date null | null = chưa hết hạn |
| issued_date | date null | ngày ban hành (optional) |
| status | text | `active` / `expired` / `superseded` |
| uploaded_by | uuid | |
| created_at | timestamptz | |

**Rule:** runtime / publish chỉ ưu tiên document còn hiệu lực (`expire_date IS NULL OR expire_date >= today`). Snapshot citation vẫn lấy từ version đã publish.

### `procedure_drafts`

Draft cho **admin review trên UI** (form slots/câu hỏi/checklist). DB lưu jsonb bên dưới; admin không làm việc với “raw JSON”.

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| document_id | uuid FK null | |
| procedure_id | text null | gán sau khi review |
| draft_definition | jsonb | nội dung procedure draft |
| validation_result | jsonb | kết quả validate gần nhất `{ valid, errors[] }` |
| status | text | `draft` / `ready` / `published` / `rejected` |
| created_by / updated_by | uuid | |
| created_at / updated_at | timestamptz | |

### `audit_logs`

| Column | Type | Notes |
|--------|------|-------|
| id | bigserial PK | |
| actor_user_id | uuid null | |
| action | text | `publish`, `rollback`, `chat_decision`, ... |
| entity_type | text | |
| entity_id | text | |
| payload | jsonb | version, route, citations... |
| created_at | timestamptz | |

## 3. Constraints quan trọng

- Chỉ **1** `procedure_versions.status = active` / `procedure_id` (partial unique index hoặc transaction publish).
- `definition` phải pass schema trước khi `approved` / `active`.
- Mọi chat decision log vào `audit_logs` kèm `procedure_version`.
- `procedures.domain_id` và `documents.domain_id` phải trỏ `domains` đang `is_active` (khi tạo mới).

## 4. Indexes gợi ý

- `conversation_sessions(user_id, updated_at desc)`
- `session_slot_states(session_id)`
- `procedure_versions(procedure_id, status)`
- `documents(xa_id, status, effective_date, expire_date)`
- `audit_logs(created_at desc)`

## 5. Changelog từ bản trước

| Thay đổi | Lý do |
|----------|-------|
| Gộp `known`/`missing`/`confirmed` → `slot_state` | một source of truth, tránh lệch 3 cột |
| PK `(session_id, procedure_id)` | session có thể đổi thủ tục |
| Thêm bảng `domains` | mở rộng domain sau này |
| `documents.effective_date` / `expire_date` (+ status) | hiệu lực văn bản hành chính |
| Rename `draft_json` → `draft_definition`, `validation_json` → `validation_result` | đúng ngôn ngữ domain; UI admin không “sửa JSON” |
| Rename `definition_json` → `definition`, `payload_json` → `payload` | đồng bộ naming |
| `procedures.domain` → `domain_id` FK | gắn domains |
