# Web — Citizen Portal (React + Vite + JavaScript)

## Run

```bash
# from repo root
make web-run

# or
cd apps/web && npm install && npm run dev
```

Mở http://localhost:5173

## Notes

- UI chat demo (mock reply) — chưa gọi Go API.
- Vite proxy sẵn `/api` → `http://localhost:8080` khi nối `POST /api/v1/chat/turns`.
