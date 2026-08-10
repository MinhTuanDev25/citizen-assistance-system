# Go Backend API (Gin)

## Run (local)

```bash
make db-up && make db-migrate
make api-run
```

Swagger UI: http://localhost:8080/swagger/index.html

Sau khi thêm/sửa API annotation:

```bash
make api-swagger   # regenerates apps/api/docs/
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (DB ping) |
| GET | `/swagger/*` | OpenAPI UI |
| GET | `/api/v1/communes` | List communes (`?active=true`) |
| GET | `/api/v1/communes/:id` | Get commune |
| GET | `/api/v1/domains` | List domains (`?active=true`) |
| GET | `/api/v1/domains/:id` | Get domain |
| POST | `/api/v1/domains` | Create domain |
| PUT | `/api/v1/domains/:id` | Update domain |
| DELETE | `/api/v1/domains/:id` | Soft-delete; `?hard=true` hard-delete |
| GET | `/api/v1/procedures` | List ACTIVE procedures grouped by domain |
| GET | `/api/v1/procedures/by-code/:code` | Resolve by `(xa_id, procedure_code)` |
| GET | `/api/v1/procedures/:id/active-version` | Active version + definition JSON |

## Logging

JSON stdout logs. `/health`, `/ready`, `/swagger` → debug when OK.

## Config

- Local: `configs/local/config.yaml`
- Prod: `APP_ENV=prod` + `DATABASE_URL`
