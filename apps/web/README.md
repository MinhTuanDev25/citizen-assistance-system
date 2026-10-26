# Web — Citizen + Admin (React + Vite + JavaScript)

## Run

Cần API đang chạy (`make api-run`, DB đã migrate) vì Vite proxy `/api` → `127.0.0.1:8080`.

```bash
make web-run
# http://localhost:5173
```

Speech (Step 7.6): native Go with `SPEECH_ENABLED=true` may listen on
`:8083`. Point the proxy at that process, never at RunPod:

```bash
VITE_API_PROXY=http://127.0.0.1:8083 npm run dev
```

See `docs/SPEECH.md`.

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
- Speech: `POST /sessions/:id/speech/translate` (WAV Bahnar). 🎙 trên ô nhập vẫn là nhận giọng **tiếng Việt** của trình duyệt. Micro Bahnar / tải WAV nằm dưới composer; phải xác nhận bản tiếng Việt trước khi gọi Chat Turn. Tự dừng 30s rồi **Dịch âm thanh**; lỗi tạm thời thì **Thử dịch lại**.

## Demo accounts (API)

- Admin: `admin@chuse.vn` / `admin123`
- Citizen: `citizen@example.com` / `citizen123`
