# Web — Citizen + Admin (React + Vite + JavaScript)

## Run

Cần API đang chạy (`make api-run`, DB đã migrate/seed) vì Vite proxy `/api` → `localhost:8080`.

```bash
make web-run
# http://localhost:5173
```

## Routes

| Path | Ai dùng | Chức năng |
|------|---------|-----------|
| `/chon-xa` | Mọi người | Chọn xã từ `GET /communes` (bắt buộc trước chat/login) |
| `/` | Citizen (guest OK) | Chat (mock) + danh mục thủ tục theo xã |
| `/login` | Admin / citizen demo | Đăng nhập |
| `/admin` | Admin | Tổng quan (ACTIVE count từ API) |
| `/admin/documents` | Admin | Upload mock; domain select từ API |
| `/admin/drafts` | Admin | Danh sách bản nháp (mock) |
| `/admin/drafts/:id` | Admin | Review form, validate, publish (mock) |
| `/admin/procedures` | Admin | ACTIVE procedures theo xã + xem definition |

## API đã nối

Client: `src/api/catalog.js` (proxy Vite). Xã đã chọn lưu `localStorage` (`cas.selectedCommune`).

- `GET /api/v1/communes` — chọn xã lần đầu / Đổi xã
- `GET /api/v1/procedures?xa_id=…` — catalog theo xã
- `GET /api/v1/procedures/:id/active-version` — xem definition
- `GET /api/v1/domains?active=true` — select domain khi upload

Chat vẫn mock (`src/api/chat.js`) cho đến khi có `POST /chat/turns`.

## Demo accounts

- Admin: `admin@chuse.vn` / `admin123`
- Citizen: `citizen@example.com` / `citizen123`

Auth vẫn mock (localStorage) — chưa nối JWT.
