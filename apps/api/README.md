# Go Backend API (Gin)

## Layout

```text
apps/api/
├── cmd/api/main.go
├── configs/{local,prod}/config.yaml
├── internal/
│   ├── api/http/v1/          # MapRoutes + domain handlers
│   │   ├── routes.go
│   │   ├── commune/
│   │   └── domain/
│   ├── repository/
│   ├── response/
│   ├── httpserver/
│   ├── handler/              # health/ready
│   └── middleware/
└── README.md
```

## Run (local)

```bash
make db-up && make db-migrate
make api-run
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (DB ping) |
| GET | `/api/v1/communes` | List communes (`?active=true`) |
| GET | `/api/v1/communes/:id` | Get commune |
| GET | `/api/v1/domains` | List domains (`?active=true`) |
| GET | `/api/v1/domains/:id` | Get domain |
| POST | `/api/v1/domains` | Create domain |
| PUT | `/api/v1/domains/:id` | Update domain |
| DELETE | `/api/v1/domains/:id` | Soft-delete (`is_active=false`); `?hard=true` hard-delete |

## Logging

JSON logs → stdout (Promtail/Loki sau này scrape container). Mỗi request ghi:

- `service`, `env`, `request_id`, `method`, `path`, `query`, `status`, `latency_ms`
- `request_body` / `response_body` (truncate 4KB; mask password/token)
- lỗi handler: `handler_error` + `error` nội bộ

`/health` và `/ready` chỉ log ở mức debug khi thành công.

## Config

- Local: `configs/local/config.yaml`
- Prod: `APP_ENV=prod` + `DATABASE_URL` on VPS
