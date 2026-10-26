# Deploy (local Docker stack)

| File | Role |
|------|------|
| `docker-compose.yml` | Local stack: postgres, migrate, api, web, ai-service, minio, qdrant, speech-tunnel (profiles) |
| `docker-compose.p4b-test.yml` | Isolated Postgres/MinIO/Qdrant for CI storage tests (ports 5434/9010/6335) |
| `docker-compose.production.yml` | VPS reference, validated by CI `production-config` ([`README.production.md`](README.production.md)) |
| `.env.speech.example` | Mẫu Speech + SSH RunPod → copy thành `.env.speech.local` (**gitignored**) |
| `.env.example` | Mẫu — copy thành `.env` nếu muốn đổi user/port |
| `.env` | Local override Compose (**gitignored**) |
| `apps/api/configs/local/config.yaml` | DSN + config **Go API** (không để `DATABASE_URL` trong `.env`) |

LLM / prod secrets: chỉ trong `.env` local hoặc set trên VPS — không commit key thật.

## Chạy toàn bộ hệ thống (một môi trường duy nhất)

```bash
cp deploy/.env.speech.example deploy/.env.speech.local  # token Speech + SSH RunPod (gitignored)
make up      # build + chạy: postgres, migrate, api, web, ai-service, minio, qdrant, speech-tunnel
make logs    # xem log
make down    # dừng (giữ volume dữ liệu)
```

Web ở `127.0.0.1:${WEB_PORT}`, API ở `127.0.0.1:${API_PORT}` (theo `.env`).
Không có `.env.speech.local` thì mọi thứ vẫn chạy, chỉ Speech tắt.

## Quick start (chỉ database)

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
