# Go Backend API (Gin)

## Layout

```text
apps/api/
├── cmd/api/main.go
├── configs/
│   ├── local/config.yaml   # local: đủ để chạy (như HDB)
│   └── prod/config.yaml    # prod: database.url trống → inject DATABASE_URL lúc deploy
├── Dockerfile
├── internal/
└── README.md
```

## Config (giống hướng HDB)

- **Local API:** `configs/local/config.yaml` — `make api-run`
- **Local DB (Compose):** `deploy/.env.example` → optional `deploy/.env` (gitignored)
- **Prod:** set `DATABASE_URL` (và secret) trên VPS lúc chạy container

## Run (local)

```bash
make db-up && make db-migrate   # Compose + migrate
make api-run                    # APP_ENV=local → configs/local/config.yaml
```

## Prod (VPS)

```bash
docker build -f apps/api/Dockerfile -t cas-api:latest .
docker run --rm -p 8080:8080 \
  -e APP_ENV=prod \
  -e DATABASE_URL='postgres://...' \
  cas-api:latest
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (DB ping) |

## Env overrides (chủ yếu prod)

| Variable | Role |
|----------|------|
| `APP_ENV` | `local` / `prod` |
| `DATABASE_URL` | Postgres DSN (bắt buộc khi prod YAML để trống) |
| `API_ADDR` | Listen addr |
| `XA_ID` | Commune id |
| `LOG_LEVEL` | Log level |
| `DB_MAX_CONNS` / `DB_MIN_CONNS` | Pool size |
| `CONFIG_FILE` | Optional explicit YAML path |
