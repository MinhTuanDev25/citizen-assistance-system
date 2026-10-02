# API Contract (Phase 0 → Phase 1–3)

Base path: `/api/v1`  
Auth: Bearer JWT (`CITIZEN` | `ADMIN`) or guest via `X-Guest-Token` on session routes.

## 1. Citizen — Chat

### `POST /sessions/{sessionId}/turns`

Canonical chat turn (Decision Engine). Replaces the obsolete `/chat/turns` sketch.

**Headers**

- `X-Guest-Token` — required for guest sessions
- `Authorization: Bearer …` — required for user-owned sessions
- `X-Request-ID` — UUID idempotency key; client generates once per send and reuses on retry

**Request**

```json
{
  "message": "Tôi muốn làm giấy khai sinh cho con tôi."
}
```

**Response 200** (one of: `ASK_MISSING_SLOTS`, `DIRECT_ANSWER`, `PROVIDE_FINAL_GUIDANCE`, `OUT_OF_SCOPE`, `CONFIRM_INTENT`)

```json
{
  "session_id": "…",
  "action": "ASK_MISSING_SLOTS",
  "procedure_code": "dk_khai_sinh",
  "procedure_version": "1.0.0",
  "reply_text": "…",
  "ask_now": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
  "missing_slots": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
  "questions": [
    { "slot": "noi_sinh", "text": "…" }
  ],
  "candidates": [],
  "citations": [],
  "user_message": { "id": "…", "role": "USER", "content": "…" },
  "assistant_message": { "id": "…", "role": "ASSISTANT", "action": "ASK_MISSING_SLOTS", "content": "…" }
}
```

Idempotent replay sets `idempotent_replay: true` and returns the prior turn for the same `(session_id, request_id)`.

### `GET /sessions/{sessionId}/messages`

Reload history. `limit=N` returns the **latest N** messages, ordered **oldest→newest**.

### `POST /sessions/{sessionId}/messages` — **gone (410)**

Deprecated write path. Clients must use `/turns`.

### `POST /sessions`

Create or resume session (`xa_id`, optional `guest_token`).

---

## 2. Deferred (admin ingestion)

P3.1 stores a real admin PDF in MinIO and `documents` (`mime_type`, `file_size_bytes`, SHA-256 checksum). Routes, admin JWT only: `POST/GET /api/v1/admin/documents`, `GET /api/v1/admin/documents/{id}`, `GET /api/v1/admin/documents/{id}/content`. Generated OpenAPI is in `apps/api/docs`. OCR, procedure draft, embedding, and RAG are still not implemented. The citizen UI stays unchanged. Both ingestion flags default off. Turn the real screen on together with `ADMIN_INGESTION_ENABLED=true` and `VITE_ADMIN_INGESTION=true`.

P4A adds admin JWT routes `GET /api/v1/admin/documents/link-targets`, `GET/POST /api/v1/admin/documents/{id}/links`, `DELETE /api/v1/admin/documents/{id}/links/{versionId}`, `POST /api/v1/admin/documents/{id}/links/{versionId}/index`, and `POST /api/v1/admin/documents/{id}/links/{versionId}/index/retry`. They mount only when `ADMIN_INDEXING_ENABLED=true` (which also requires ingestion and the AI service token). Indexing status is per link. The index call is a mock lifecycle. There is no activate route. Run it with `ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true ADMIN_INDEXING_ENABLED=true VITE_ADMIN_INDEXING=true docker compose --profile app --profile storage --profile ai up --build` from `deploy`.

---

## Legacy note

Do **not** call `POST /api/v1/chat/turns` — that path does not exist. Use `POST /api/v1/sessions/{sessionId}/turns`.
