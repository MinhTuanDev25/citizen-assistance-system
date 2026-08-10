# Web — Citizen + Admin (React + Vite + JavaScript)

## Run

Cần API đang chạy (`make api-run`, DB đã migrate qua `000004`) vì Vite proxy `/api` → `localhost:8080`.

```bash
make web-run
# http://localhost:5173
```

## Routes

| Path | Ai dùng | Chức năng |
|------|---------|-----------|
| `/chon-xa` | Mọi người | Chọn xã từ `GET /communes` |
| `/` | Citizen (guest OK) | Chat + catalog; session/messages API |
| `/login` | Admin / citizen | JWT login |
| `/admin` | Admin | Tổng quan |
| `/admin/documents` | Admin | Upload mock; domains từ API |
| `/admin/drafts` | Admin | Bản nháp mock |
| `/admin/drafts/:id` | Admin | Review mock |
| `/admin/procedures` | Admin | ACTIVE + definition API |

## API đã nối

- Auth: `POST /auth/login`, `POST /auth/logout`, (token trong `localStorage` `cas_auth`)
- Catalog: communes / domains / procedures
- Chat: `POST /sessions`, `GET|POST /sessions/:id/messages` (+ `X-Guest-Token` hoặc Bearer)
- Trả lời bot vẫn **mock local** cho đến Decision Engine

## Demo accounts (API)

- Admin: `admin@chuse.vn` / `admin123`
- Citizen: `citizen@example.com` / `citizen123`
