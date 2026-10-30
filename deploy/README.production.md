# Production deployment

This file is the deploy contract for the application tier. It is not a record of a completed production deployment. Phase 4–8 are not implemented. `PRODUCTION_DEPLOYMENT` stays `NOT_RUN` until a separate, approved deploy on real infrastructure.

## What is production-ready here

The local Compose file `docker-compose.yml` is for development. It publishes Postgres, uses development passwords, and defaults indexing to mock. Do not point that file at a public host.

`docker-compose.production.yml` runs only the API, the AI service, and the web image. PostgreSQL, S3-compatible object storage, and Qdrant are external services configured by endpoint. Copy `deploy/.env.production.example` into a root-only file or a secret store. Do not commit the filled file.

`docker compose -f docker-compose.production.yml config` refuses to render while a required variable is empty. With the variables set, the rendered file keeps `APP_ENV=prod`, `INDEX_MODE=pipeline`, ONNX and Paddle providers, `OBJECT_STORAGE_AUTO_CREATE_BUCKET=false`, and publishes the API and web only on `127.0.0.1`. Postgres, Qdrant, and MinIO ports are not published by this file.

## Images

Build from the repository root. Tags below are local labels, not a release digest.

```bash
docker build -f apps/api/Dockerfile -t cas-api:phase13-hardening .
docker build -f apps/web/Dockerfile -t cas-web:phase13-hardening .
docker build -f apps/ai-service/Dockerfile -t cas-ai:phase13-hardening .
```

The API image built on this Mac is `linux/arm64` and runs as user `cas` (uid 10001). Starting it with `APP_ENV=prod` and no `DATABASE_URL` exits before it listens, with `DATABASE_URL missing for APP_ENV=prod`. The web image built here is also `linux/arm64`. The AI image was not rebuilt in this change.

A Linux x86_64 host needs its own build. `EMBEDDING_VARIANT` selects the E5 graph: `fp32` runs on any x86_64 or ARM64 CPU, `int8-vnni` needs AVX-512 VNNI and fails closed without it, `auto` (the ai-service default) picks INT8 with VNNI and FP32 otherwise. Fetch the model of the variant you run. Do not treat the Mac image as proof for another architecture.

## Startup rules

`APP_ENV=prod` on the API rejects:

- a database URL without `sslmode=require`, `verify-ca`, or `verify-full`
- JWT or AI service tokens that contain `local-dev`, `change-me`, or `minioadmin`
- `minioadmin` object-storage keys and automatic bucket creation when ingestion is on
- `INDEX_MODE=mock` when admin indexing is on

`APP_ENV=prod` on the AI service rejects `INDEX_MODE=mock` and fake embedding or OCR providers. Real ONNX and Paddle directories must match their manifests or the process does not become ready. There is no fallback from a real provider to a fake one.

`LLM_PROVIDER=mock` and `AI_EXTRACT_ENABLED=false` remain the production default because real LLM extraction is Phase 6 and is not in this build. Keyword extraction and the Go Decision Engine stay in the API.

## Object storage

Go API and the AI service must receive the same endpoint, bucket, region, access key, secret key, TLS flag, and `OBJECT_STORAGE_AUTO_CREATE_BUCKET=false`. The production Compose file now injects those variables into both containers. `APP_ENV=prod` plus `INDEX_MODE=pipeline` refuses a missing, empty, or whitespace `OBJECT_STORAGE_BUCKET`, empty keys, `http://` endpoints, auto-create, and `minioadmin`. The AI process applies that check itself; Compose interpolation is not the only gate. Local and test still default the bucket to `cas-documents`.

Create the Qdrant collection once before the first `/ready` check:

```bash
QDRANT_URL=http://qdrant.internal:6333 QDRANT_COLLECTION=knowledge_chunks \
  python scripts/provision-qdrant.py
```

`/ready` only reads that collection. It does not create it.

## Index reaper

The API image also contains `./index-reaper`. Compose starts one replica of that binary with `INDEX_REAPER_INTERVAL_SECONDS=60`. `SIGTERM` and `SIGINT` stop new cycles and let the in-flight cleanup finish for `INDEX_REAPER_SHUTDOWN_SECONDS` (default 30, Compose `stop_grace_period` 45s). Claim uses `SKIP LOCKED`, so a second replica does not take the same generation. Qdrant delete and Postgres state are not one transaction; a retry after an interrupted cycle is idempotent and does not pick `READY` generations or live claims. Set the interval to empty and run the binary once for a one-shot cleanup. Do not point it at the development database.

## Web feature flags

Admin upload and indexing screens are compiled into the web image. Rebuild with `--build-arg VITE_ADMIN_INGESTION=true --build-arg VITE_ADMIN_INDEXING=true` only when the matching Go flags are true. Those args are not a security boundary.

## Network

Put a TLS reverse proxy on the host. It should terminate HTTPS and proxy to `127.0.0.1:8088` (web) which proxies `/api/` to the API container. Do not publish the AI service. The AI container is reachable only on the Compose network at port 8001.

The web image sends `X-Content-Type-Options`, `X-Frame-Options`, and `Referrer-Policy`, and forwards `X-Request-ID`.

## Data services

- PostgreSQL: TLS required by the API check above. Run migrations with `migrate` against that database once, from an operator shell, not from every replica at once. Frozen migrations `000001`–`000015` stay unchanged. Do not run seed-demo.
- Object storage: MinIO is valid for a private network. RunPod S3 needs the region, TLS on port 443, and an existing bucket id. The application must not create that bucket.
- Qdrant: private endpoint only. Vectors are 384-dimensional cosine in `knowledge_chunks`. The Postgres `embedding` column is unused.

## Backup and rollback

Take a Postgres dump and confirm a restore on a separate database before calling backup verified. That restore has not been run. Object storage durability is the provider's. Qdrant can be rebuilt by reindexing documents; a snapshot procedure is not verified here. Roll back by redeploying the previous immutable image tag. Do not roll schema backward on a production database that already applied a migration.

## Still blocked

Staging credentials, a production host, a RunPod bucket, an x86_64 runner, a GitHub Actions run of the new `production-config` job, and a restore drill are not available in this change. Phase 4 through Phase 8 are not part of this deploy.
