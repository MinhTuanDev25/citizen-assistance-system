# Local database (Postgres + pgvector)

| File | Role |
|------|------|
| `docker-compose.yml` | Postgres / MinIO / migrate |
| `.env.example` | Mẫu — copy thành `.env` nếu muốn đổi user/port |
| `.env` | Local override Compose (**gitignored**) |
| `apps/api/configs/local/config.yaml` | DSN + config **Go API** (không để `DATABASE_URL` trong `.env`) |

LLM / prod secrets: chỉ trong `.env` local hoặc set trên VPS — không commit key thật.

## Quick start

```bash
cd deploy
cp .env.example .env   # optional
chmod +x scripts/migrate.sh
./scripts/migrate.sh up
```

From repo root: `make db-up` && `make db-migrate`

## DBeaver

| Field | Value |
|-------|--------|
| Host | `localhost` |
| Port | `5432` |
| Database | `citizen_assistance` |
| Username | `cas` |
| Password | `cas` |

MinIO (optional): `docker compose --profile full up -d minio`
