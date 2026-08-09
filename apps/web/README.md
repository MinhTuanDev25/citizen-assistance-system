# Web — Citizen + Admin (React + Vite + JavaScript)

## Run

```bash
make web-run
# http://localhost:5173
```

## Routes

| Path | Ai dùng | Chức năng |
|------|---------|-----------|
| `/` | Citizen (guest OK) | Chat hỏi thủ tục (mock) |
| `/login` | Admin / citizen demo | Đăng nhập |
| `/admin` | Admin | Tổng quan pipeline |
| `/admin/documents` | Admin | Upload + extract draft |
| `/admin/drafts` | Admin | Danh sách bản nháp |
| `/admin/drafts/:id` | Admin | Review form, validate, publish |
| `/admin/procedures` | Admin | ACTIVE versions + rollback |

## Demo accounts

- Admin: `admin@chuse.vn` / `admin123`
- Citizen: `citizen@example.com` / `citizen123`

Auth + data đang **mock** (localStorage / memory) — chưa nối JWT/Go API.
