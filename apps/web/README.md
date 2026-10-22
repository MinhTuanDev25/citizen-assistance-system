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
| `/commune` | Mọi người | Chọn xã từ `GET /communes` |
| `/` | Citizen (guest OK) | Trang chủ |
| `/chat` | Citizen (guest OK) | Hỏi đáp / hội thoại |
| `/procedures` | Citizen (guest OK) | Danh mục thủ tục |
| `/login` | Admin / citizen | JWT login |
| `/admin` | Admin | Tổng quan |
| `/admin/documents` | Admin | Upload; domains từ API |
| `/admin/drafts` | Admin | Bản nháp |
| `/admin/drafts/:id` | Admin | Review bản nháp |
| `/admin/procedures` | Admin | ACTIVE + definition API |

Legacy redirects: `/hoi` → `/chat`, `/thu-tuc` → `/procedures`, `/chon-xa` → `/commune`, `/giong-noi` → `/chat`, `/ho-so` → `/`.

## API đã nối

- Auth: `POST /auth/login`, `POST /auth/logout`, (token trong `localStorage` `cas_auth`)
- Catalog: communes / domains / procedures
- Chat: `POST /sessions`, `GET /sessions/:id/messages`, `POST /sessions/:id/turns` (+ `X-Guest-Token` hoặc Bearer)

## Demo accounts (API)

- Admin: `admin@chuse.vn` / `admin123`
- Citizen: `citizen@example.com` / `citizen123`
