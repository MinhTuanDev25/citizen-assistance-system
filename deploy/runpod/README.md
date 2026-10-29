# All-in-one image for one RunPod GPU pod

Phase 8.1-A. One image runs the whole Citizen Assistance System:

- nginx serving the web build;
- Go API and index-reaper;
- Python AI service;
- Speech C0;
- PostgreSQL 16 with pgvector;
- Qdrant 1.13.4;
- MinIO.

`tini` starts `entrypoint.sh`, which then hands over to `supervisord`. Docker/Compose is not used inside the pod, because RunPod GPU pods cannot run nested containers.

Data persists only on the volume at `/workspace/cas`. Only port **8088** listens beyond loopback; every other service binds `127.0.0.1`.

## Layout

| Path in repo | Role |
|---|---|
| `Dockerfile` | Multi-stage build. Web and Go build on `$BUILDPLATFORM` (no QEMU for esbuild/Go); the runtime is `python:3.11-slim-bookworm` |
| `Dockerfile.dockerignore` | Build-context allowlist for this Dockerfile only |
| `stage_c0.sh`, `c0_files.txt` | Copies the Speech C0 runtime from **committed Git blobs** into `build/c0/` and writes `SOURCE_REVISION`, `SOURCE_FILES` and `c0_code.sha256` |
| `rootfs/opt/cas/bin/entrypoint.sh` | Root pre-start: mode and secret checks, ownership, internal CA, Postgres first init, Speech checks, boot id |
| `rootfs/opt/cas/bin/bootstrap.sh` | One-shot as `cas`: waits for the data services, runs migrations (with a `pg_dump` first when migrations are pending), creates the bucket, writes `bootstrap.ok` |
| `rootfs/opt/cas/bin/wait-for.sh` | Startup gate for app programs: waits for this boot's `bootstrap.ok` and/or an HTTP 2xx, then `exec`s the program |
| `rootfs/opt/cas/bin/healthcheck.sh` | Operator report and image `HEALTHCHECK` (`--quiet`) |
| `rootfs/opt/cas/bin/backup.sh` | `pg_dump` and a Qdrant snapshot to `/workspace/cas/backups` with rotation |
| `rootfs/opt/cas/bin/restore_drill.sh` | Restores the newest backups into a scratch DB and a scratch collection, compares row and point counts with live data, then removes the scratch copies |
| `rootfs/opt/cas/bin/pg-lease.sh`, `pg-start.sh`, `cas-pg-recover` | Single-writer workspace lease: heartbeat and fencing; data services refuse to start without a fresh lease; verified recovery |
| `fetch_models.sh`, `models/embedding.manifest.json` | Downloads the pinned e5 ONNX model (repository, revision, SHA-256 per file). All-or-nothing: every file is verified in a staging directory before it replaces the destination; a model that is already complete is left untouched |
| `rootfs/opt/cas/bin/cas-run` | Runs operator tools (e.g. `bootstrap-admin`) with this boot's environment, as a given user |
| `rootfs/opt/cas/bin/svc-run` | Starts every supervisord program with `env -i` plus its own allowlist (`/run/cas-env/<service>.env`), then drops to the program's user |
| `rootfs/etc/cas/*` | `supervisord.conf`, `nginx.conf`, `postgresql.cas.conf`, `pg_hba.conf` |
| `env.staging.example` | Variable names for the RunPod template (no values) |
| `tests/smoke_cpu.sh`, `tests/c0_in_image.sh`, `tests/c0_in_image_negative.sh`, `tests/fetch_models_test.sh` | CPU smoke, C0 level (b) checks, proof that the C0 checker fails on broken images, offline all-or-nothing test of `fetch_models.sh`. All exit non-zero on any FAIL |

## Build

```bash
deploy/runpod/stage_c0.sh            # refuses untracked/modified C0 files or a changed import closure
docker buildx build --platform linux/amd64 -f deploy/runpod/Dockerfile \
  --build-arg CAS_REVISION=$(git rev-parse HEAD) -t cas-aio:$(git rev-parse --short HEAD) --load .
```

The build fails if `CAS_REVISION` is not the commit C0 was staged from.

RunPod needs `linux/amd64`. On Apple Silicon the amd64 runtime stage runs under QEMU emulation: the image builds there, and the C0 checks pass, but Qdrant and MinIO crash with SIGSEGV under QEMU (their official amd64 images do too), so the CPU smoke must run on native amd64.

## Run (CPU smoke, no GPU)

```bash
docker volume create cas-aio-test
docker run -d --name cas-aio -p 127.0.0.1:18088:8088 \
  -v cas-aio-test:/workspace -v "$PWD/models:/workspace/cas/models:ro" \
  -e CAS_SPEECH_MODE=off -e LLM_PROVIDER=mock -e CAS_ALLOW_MOCK_LLM=true \
  -e JWT_SECRET=... -e AI_SERVICE_TOKEN=... -e POSTGRES_PASSWORD=... \
  -e MINIO_ROOT_USER=... -e MINIO_ROOT_PASSWORD=... cas-aio:<tag>
docker exec cas-aio healthcheck.sh
```

Full automated run on throwaway containers and volumes: `deploy/runpod/tests/smoke_cpu.sh <image>`.

## Startup order

1. **`entrypoint.sh`** (root, no server is started):
   - checks the mode and secrets;
   - creates directories and sets ownership;
   - checks models and C0 (in `gpu` mode: checkpoints, tokenizer cache and GPU, fail-closed);
   - sets up the CA and leaf certificates;
   - on the first start only, runs `initdb` plus a temporary unix-socket-only server that creates role `cas`, database `citizen_assistance` and the `pgcrypto`/`vector` extensions (both need superuser);
   - writes a new boot id and removes any stale `bootstrap.ok`;
   - `exec supervisord`.
2. **supervisord** starts `postgres`, `qdrant` and `minio` (priority 10).
3. **`bootstrap`** (priority 20, user `cas`):
   - waits until a TLS query succeeds against Postgres, Qdrant answers `/readyz`, and MinIO answers HTTPS with the internal CA;
   - refuses a dirty migration state or a DB that is newer than the image;
   - if migrations are pending, writes a `pg_dump` to `backups/pre-migrate/`, then runs `migrate up`;
   - creates the bucket;
   - writes `bootstrap.ok` and exits 0. Its expected state is `EXITED`.
4. **`ai-service`** and **`speech`** (gpu mode only) start once `bootstrap.ok` exists. **`api`** additionally waits for ai-service `/ready`, and **`nginx`** waits for api `/ready`.

`wait-for.sh` only gates startup. If Postgres or MinIO dies later, supervisord restarts it after a crash. A manual `supervisorctl stop` keeps it down. The apps keep running, report `/ready` 503 and fail requests until the dependency is back, then reconnect without a restart.

## Speech mode

| | `CAS_SPEECH_MODE=off` | `CAS_SPEECH_MODE=gpu` |
|---|---|---|
| program `speech` | not started | started after `bootstrap.ok` |
| Go `SPEECH_ENABLED` | forced `false`; `true` in env is refused | `true`; `false` in env is refused |
| `c0_code.sha256` + `import src.c0_speech.app` | checked | checked |
| checkpoints (`/workspace/cas/manifests/c0_checkpoints.sha256`), `C0_INFERENCE_CONFIG`, `HF_HOME`, `nvidia-smi` | not checked | **required, fail-closed** (container exits 64) |

There is no default: an unset or unknown value is refused.

## Per-service environment

`entrypoint.sh` computes the whole runtime configuration, but no program inherits it as-is. It writes one allowlist per program to `/run/cas-env/<service>.env` (root `0700`/`0600`). supervisord starts every program as `svc-run <service> <user> <command>`, which does `env -i` plus that list, then `gosu <user>`.

| Program | Secrets it receives |
|---|---|
| nginx, postgres, qdrant, pg-lease | none |
| minio | `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` |
| index-reaper | `DATABASE_URL` |
| bootstrap | `DATABASE_URL`, object storage keys |
| api | `DATABASE_URL`, `JWT_SECRET`, `AI_SERVICE_TOKEN`, object storage keys, `SPEECH_SERVICE_TOKEN` |
| ai-service | `DATABASE_URL`, `AI_SERVICE_TOKEN`, object storage keys, `GEMINI_API_KEY` / `OPENAI_API_KEY` |
| speech | `SPEECH_SERVICE_TOKEN` |

`POSTGRES_PASSWORD` itself reaches no program; it is only used to build `DATABASE_URL` and at first init. A variable set in the RunPod template reaches a program only if it is in that program's allowlist in `entrypoint.sh`. Smoke test E1 reads `/proc/<pid>/environ` of every program and checks this table.

## Users and permissions

| What | User | Mode |
|---|---|---|
| entrypoint, supervisord, tini | root | — |
| Postgres, `pgdata` | `postgres` (10010) | `0700` |
| MinIO, `minio/` | `minio` (10011) | `0700` |
| Qdrant, `qdrant/` | `qdrant` (10012) | `0700` |
| Go API, index-reaper, ai-service, bootstrap, `backups/`, `/run/cas` | `cas` (10001) | `0700` / `/run/cas` `0755` |
| Speech C0 | `speech` (10013) | — |
| nginx | `web` (10014) | — |
| `tls/` / `tls/ca/` / `tls/postgres/` / `tls/minio/` | root / root / postgres / minio | `0711` / `0700` / `0700` / `0700`; keys `0600` |
| `tls/ca.crt` | root | `0644` |

UIDs are fixed in the image. On every start, `entrypoint.sh` repairs ownership on the volume if another pod or image wrote it.

## TLS

- The internal CA lives in `tls/ca/` (EC P-256, 10 years). Leaf certs are signed by it, valid 825 days, with SAN `IP:127.0.0.1,DNS:localhost`.
- Certificates are created only when absent. If a key or certificate is half-present, mismatched, expired, or signed by another CA, startup stops with an error. Nothing is regenerated or overwritten silently.
- The CA is added to the system trust store, and `SSL_CERT_FILE` points to that bundle. Go and minio-py verify MinIO with it.
- Postgres accepts TCP only with TLS (`hostssl`). Apps connect with `sslmode=verify-full&sslrootcert=/workspace/cas/tls/ca.crt`, so the client checks the chain and the `127.0.0.1` SAN. A wrong CA is refused (smoke T6).
- Rotation is manual: stop the pod, move the old files away, start again.

## Operations

```bash
docker exec <c> healthcheck.sh                                   # status table + supervisor
docker exec <c> supervisorctl -c /etc/cas/supervisord.conf status
docker exec -e BOOTSTRAP_ADMIN_EMAIL=admin@chuse.vn -e BOOTSTRAP_ADMIN_PASSWORD='<strong>' \
  <c> cas-run cas /opt/cas/api/bootstrap-admin                   # set the admin password
docker exec <c> backup.sh                                        # pg_dump + Qdrant snapshot
docker exec <c> restore_drill.sh                                 # full drill: Postgres AND Qdrant must restore (missing snapshot = FAIL)
docker exec <c> restore_drill.sh --postgres-only                 # explicit partial drill; result line says PARTIAL
```

`backup.sh` writes only to the volume. Copy `/workspace/cas/backups`, and the MinIO data, out of RunPod.

## Rules

- **Single writer per volume.** Postgres, Qdrant and MinIO data must never have two writers; see "Single-writer lease" below.
- First Postgres init writes `pgdata/.cas-init-complete` last. A `pgdata` without it (interrupted first init) is refused instead of being used half-initialised.
- `POSTGRES_PASSWORD` is applied once, at first init. To rotate it, run `ALTER ROLE cas PASSWORD ...` as `postgres` first, then change the secret.
- Migrations only move up. A DB newer than the image is refused, and so is a dirty state. Nothing is forced or rolled back automatically.
- No procedure is activated automatically. The Admin activates in the UI.

## Single-writer lease

`/workspace/cas/locks/` protects `pgdata`, `qdrant` and `minio` with two layers:

Owner identity: `runpod-<RUNPOD_POD_ID>`, else `docker-<container id>` (stable across restarts of the same container, different for every new container even with the same `--hostname`), else `host-<hostname>`.

1. **`flock` on `workspace.flock`.** `entrypoint.sh` takes it before touching any data, and supervisord keeps it (inherited fd) for the container's lifetime. A second container on the same host is refused immediately.
2. **Lease directory `workspace.lease/`.** It is created with `mkdir`, which is atomic. It holds `owner.json` (owner = `RUNPOD_POD_ID`, or the container hostname) and a `heartbeat`, which `pg-lease` rewrites every 10 s. `pg-lease` stops the data services at once if the lease names another owner or the heartbeat cannot be written. Postgres, Qdrant and MinIO start through `pg-start.sh`, which refuses to run without this container's fresh lease.

Behaviour on start:

| Situation | Result |
|---|---|
| no lease, no `postmaster.pid` | new lease |
| lease of the **same** owner (container or pod restarted) | reclaimed; that owner's stale `postmaster.pid` is removed (the same container/pod cannot run twice) |
| lease of **another** owner, or `postmaster.pid` without a lease | **maintenance mode**: no service starts; the container stays up for `docker exec` / SSH; the health check fails |

On a clean stop, `pg-lease` releases the lease (renamed `workspace.lease.released-<ts>-<owner>` for audit), but only after the data services have stopped.

### Recovery after an unclean stop of another owner

1. **Make sure the old pod or container is really gone.** Check the RunPod console or `docker ps`.
2. In the container that is in maintenance mode, run:

   ```bash
   docker exec <c> cas-pg-recover <old owner id>   # id from the MAINTENANCE log line
   ```

3. `cas-pg-recover` checks, with no way to skip any step:
   - the owner id matches the lease exactly;
   - no data service runs here, and nothing listens on 5432/6333/9000;
   - the heartbeat (or `postmaster.pid`) does not change for at least 3× the heartbeat interval **recorded in the lease** + 5 s. `CAS_PG_RECOVER_WAIT_S` can only make the wait longer. The owner fences its data services after 2 missed beats, so they are down before this wait can pass;
   - the flock can be taken.

   It then renames, never deletes, the lease and the stale `postmaster.pid`, and appends an audit line to `locks/recovery.log`.
4. Restart the container. It takes a new lease.

**UNVERIFIED on RunPod:** both layers are proven on a local Docker volume, where both containers share one kernel. Whether `flock` works between **two pods** on a RunPod network volume is unknown; the lease and heartbeat layer does not depend on it. A two-pod test on the real volume is mandatory in 8.1-B.

## Upload limits

| Layer | Limit | Answer |
|---|---|---|
| API (`DOCUMENT_MAX_BYTES`) | 20 MiB (default, same as the local stack) | `413` JSON `PAYLOAD_TOO_LARGE` |
| nginx (`client_max_body_size`) | 65 MiB = 64 MiB hard max of `DOCUMENT_MAX_BYTES` + multipart overhead | `413` HTML from nginx |

The gap is intentional: nginx only cuts bodies that the API could never accept, so raising `DOCUMENT_MAX_BYTES` (for example for the 451-page source PDF) needs no nginx change. Both limits are tested (smoke A6/A7).

## LLM provider

`LLM_PROVIDER` is required: `gemini` (needs `GEMINI_API_KEY`), `openai` (needs `OPENAI_API_KEY`), or `mock`. `mock` is refused unless `CAS_ALLOW_MOCK_LLM=true`, which only tests set.

## Known limits

- **MinIO credentials:** the app uses the MinIO root credentials; a scoped service account is a Go-live item.
- **Qdrant:** no API key, because the code has no client support for one. Qdrant is reachable on loopback only.
- **Client IP for rate limiting:** nginx uses `CF-Connecting-IP`, falling back to the TCP peer. Which header RunPod forwards is unverified until 8.1-B.
- **Legacy `/ai-drafts` call:** this synchronous call can exceed the 100 s proxy limit. Drafting jobs are asynchronous and are not affected.
- **Restore drill timing:** `restore_drill.sh` compares the restored copy with the live data at drill time. Run it right after `backup.sh`, when nothing is writing; writes in between show up as a count mismatch.
- **Replacing the model:** when `fetch_models.sh` installs a *different* model, it swaps the directory, so restart any container that bind-mounts it. A model that is already complete and valid is never touched.
- **Model swap interrupted:**
  - If publishing fails, the previous model is put back.
  - If that also fails, the previous model is kept intact at `<parent>/.<name>.old.<pid>` and the script prints the path. It is never deleted.
  - A `SIGKILL` exactly between the two renames cannot run any handler. In that case the destination is missing and the previous model sits in `.<name>.old.<pid>`. Recovery: `mv <parent>/.<name>.old.<pid> <dest>`, then rerun `fetch_models.sh`.
- **Single writer across pods:** both lease layers are proven only on a local Docker volume. A two-pod test on the RunPod network volume is mandatory in 8.1-B.
