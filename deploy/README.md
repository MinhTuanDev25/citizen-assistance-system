# Local database (Postgres + pgvector)

## Quick start

```bash
cd deploy
cp .env.example .env
chmod +x scripts/migrate.sh
./scripts/migrate.sh up
```

From repo root:

```bash
make db-up
make db-migrate
```

## Migrations (golang-migrate)

| File | Purpose |
|------|---------|
| `migrations/000001_init_schema.{up,down}.sql` | Full V1 schema |
| `migrations/000002_seed_master.{up,down}.sql` | 1 commune + 3 domains |

Procedure JSON seeds (`docs/phase-0/seeds/*.json`) load later via Phase 1 — not in SQL seed.

```bash
./scripts/migrate.sh version
./scripts/migrate.sh down 1
```

## DBeaver

1. Start Postgres: `docker compose up -d postgres`
2. New PostgreSQL connection:

| Field | Value |
|-------|--------|
| Host | `localhost` |
| Port | `5432` |
| Database | `citizen_assistance` |
| Username | `cas` |
| Password | `cas` |

```sql
SELECT * FROM communes;
SELECT * FROM domains ORDER BY sort_order;
SELECT version, dirty FROM schema_migrations;
```

MinIO (optional): `docker compose --profile full up -d minio`
