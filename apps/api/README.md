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
| POST | `/api/v1/auth/register` | Citizen register → JWT |
| POST | `/api/v1/auth/login` | Login → JWT |
| GET | `/api/v1/auth/me` | Current user (Bearer) |
| POST | `/api/v1/auth/logout` | Client discard token (Bearer) |
| GET | `/api/v1/communes` | List communes (`?active=true`) |
| GET | `/api/v1/communes/:id` | Get commune |
| GET | `/api/v1/domains` | List domains (`?active=true`) |
| GET | `/api/v1/domains/:id` | Get domain |
| POST | `/api/v1/domains` | Create domain (**ADMIN** JWT) |
| PUT | `/api/v1/domains/:id` | Update domain (**ADMIN** JWT) |
| DELETE | `/api/v1/domains/:id` | Soft-delete; `?hard=true` (**ADMIN** JWT) |
| GET | `/api/v1/procedures` | List ACTIVE procedures grouped by domain |
| GET | `/api/v1/procedures/by-code/:code` | Resolve by `(xa_id, procedure_code)` |
| GET | `/api/v1/procedures/:id/active-version` | Active version + definition JSON |
| POST | `/api/v1/sessions` | Create session (guest or Bearer JWT) |
| GET | `/api/v1/sessions/:sessionId/messages` | List messages |
| POST | `/api/v1/sessions/:sessionId/messages` | Save USER message |

### Auth demo (after migrate `000004`)

- Admin: `admin@chuse.vn` / `admin123`
- Citizen: `citizen@example.com` / `citizen123`

Header: `Authorization: Bearer <access_token>`. Prod: set `JWT_SECRET` (min 16 chars).

Guest chat: `X-Guest-Token`. Logged-in: Bearer → session `user_id` set. Chat turn / Decision Engine chưa gắn.

## Logging

JSON stdout logs. `/health`, `/ready`, `/swagger` → debug when OK.

## Config

- Local: `configs/local/config.yaml`
- Prod: `APP_ENV=prod` + `DATABASE_URL` + `JWT_SECRET`
