# Phase 1–3 final report

**Date:** 2026-09-29  
**Deliverable:** `phase_1_3_final_bundle.zip`  
**Scope:** Final Phase 1–3 remediation only. No next phase, no deploy, no commit/push.  
**Sole current verdict:** this file. `phase_1_3_audit.md` is historical/superseded and is **excluded** from the ZIP.

## Verdict

| Area | Status |
|------|--------|
| Catalog list auth | **PASS** |
| Detail auth (`by-code`, `active-version`) | **PASS** — guest/citizen → `CitizenDomainIDs`; ADMIN JWT for full |
| OpenAPI sync | **PASS** — regenerated from annotations; JSON/YAML/docs.go regression tested |
| Body limit 32 MiB (any method + chunked) | **PASS** → **413**; sensitive routes limited without body logs |
| Idempotency / CONFIRM pin / JWT / FE retry | **PASS** (unchanged) |
| Packaging | **PASS** |

## Files changed (this pass)

| File | Change |
|------|--------|
| `apps/api/internal/api/http/v1/procedure/handler.go` | OpenAPI annotations: Authorization, 403, guest/citizen/admin scope |
| `apps/api/docs/{docs.go,swagger.json,swagger.yaml}` | Regenerated with `swag init` (not hand-edited) |
| `apps/api/docs/openapi_contract_test.go` | Regression for both detail GETs in JSON + YAML |
| `apps/api/internal/middleware/request_id.go` | Limit every present body (no GET/DELETE method skip) |
| `apps/api/internal/middleware/request_id_test.go` | DELETE chunked + GET Content-Length oversized → 413 |
| `phase_1_3_audit.md` | Marked SUPERSEDED; not shipped in ZIP |
| `scripts/check-zip.sh` | Require OpenAPI artifacts; reject `phase_1_3_audit.md` |

## New tests

- OpenAPI `assertProcedureDetailGet` (Authorization, 403, CitizenDomainIDs / guest / citizen / admin)
- `TestDeleteWithOversizedChunkedBodyReturns413`
- `TestGetWithOversizedDeclaredContentLengthReturns413`

## Clean verification results

| Command | Exit |
|---------|------|
| `go test ./...` | **0** |
| `go vet ./...` | **0** |
| `go build` api + seed-demo + bootstrap-admin | **0** |
| migrate up / down2+up / down-all+up | **0** |
| `-tags=integration` chat | **0** |
| middleware body/DELETE/GET 413 | **0** |
| `go test ./docs/` OpenAPI | **0** |
| `npm ci` + Vitest + prod build + `check-bundle` | **0** |
| `npm audit` prod + all | **0** (0 vulns) |
| pytest + compileall | **0** |
| `docker compose --env-file .env.example config` | **0**; no `.env` created |

Skipped: **none**. Full log: `phase_1_3_test_output.txt`.

## Self-check

1. OpenAPI YAML/JSON for by-code + active-version include `Authorization`, response `403`, and CitizenDomainIDs guest/citizen/admin policy text.  
2. DELETE (and GET) with real oversized / declared-oversize body → **413**.  
3. ZIP excludes `phase_1_3_audit.md`; `SHA256SUMS.txt` covers all shipped files except itself; `<zip>.sha256` matches the ZIP.
