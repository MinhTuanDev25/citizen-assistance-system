# All-in-one image for one RunPod GPU pod

Phase 8.1-A. One image runs the whole Citizen Assistance System:

- nginx serving the web build;
- Go API and index-reaper;
- Python AI service;
- Speech C0;
- PostgreSQL 16 with pgvector;
- Qdrant 1.13.4.

Documents are plain files on the volume (no MinIO).

`tini` starts `entrypoint.sh`, which then hands over to `supervisord`. Docker/Compose is not used inside the pod, because RunPod GPU pods cannot run nested containers.

PostgreSQL and Qdrant run on the container disk (`/var/lib/cas`), where POSIX owners and modes work. Their durable copy is on the volume at `/workspace/cas`: synchronous WAL, verified base backups, Qdrant snapshots with fingerprints, and the document files. A new container restores from the volume before it opens anything. See "Storage". Only port **8088** listens beyond loopback; every other service binds `127.0.0.1`.

## Layout

| Path in repo | Role |
|---|---|
| `Dockerfile` | Multi-stage build. Web and Go build on `$BUILDPLATFORM` (no QEMU for esbuild/Go); the runtime is `python:3.11-slim-bookworm` |
| `Dockerfile.dockerignore` | Build-context allowlist for this Dockerfile only |
| `stage_c0.sh`, `c0_files.txt` | Copies the Speech C0 runtime from **committed Git blobs** into `build/c0/` and writes `SOURCE_REVISION`, `SOURCE_FILES` and `c0_code.sha256` |
| `rootfs/opt/cas/bin/entrypoint.sh` | Root pre-start: mode and secret checks, directories, internal CA, Postgres reuse / restore-from-volume / first init, Speech checks, boot id |
| `rootfs/opt/cas/bin/bootstrap.sh` | One-shot as `cas`: waits for Postgres, Qdrant and the synchronous WAL receiver, runs migrations (with a `pg_dump` first when migrations are pending), verifies the document files, restores/verifies Qdrant, waits for a verified base backup, writes `bootstrap.ok` |
| `rootfs/opt/cas/bin/cas-pg-restore.sh`, `pg-restore-wal.sh` | Restore-on-boot of Postgres from the volume (base backup + WAL), fail-closed; also used by the restore drill |
| `rootfs/opt/cas/bin/pg-receivewal.sh` | `pg_receivewal --synchronous` to the volume (program `pg-receivewal`) |
| `rootfs/opt/cas/lib/cas_storage.py`, `bin/cas-storage` | Program `cas-keeper` (base backups, Qdrant snapshots, storage and receiver guard) and the checks bootstrap and the drill share |
| `rootfs/opt/cas/bin/wait-for.sh` | Startup gate for app programs: waits for this boot's `bootstrap.ok` and/or an HTTP 2xx, then `exec`s the program |
| `rootfs/opt/cas/bin/healthcheck.sh` | Operator report and image `HEALTHCHECK` (`--quiet`) |
| `rootfs/opt/cas/bin/backup.sh` | `pg_dump`, a verified base backup and a Qdrant snapshot now, on the volume |
| `rootfs/opt/cas/bin/restore_drill.sh` | Recovers base backup + WAL into a scratch cluster and compares every table (rows and content) with live; restores the newest snapshot into a scratch collection and compares its fingerprint; verifies the document files; removes the scratch copies |
| `rootfs/opt/cas/bin/pg-lease.sh`, `pg-start.sh`, `cas-pg-recover` | Single-writer workspace lease: heartbeat and fencing; data services refuse to start without a fresh lease; verified recovery |
| `fetch_models.sh`, `models/embedding.manifest.{int8-vnni,fp32}.json` | Downloads the pinned e5 ONNX model (repository, revision, SHA-256 per file) in the **variant the target CPU needs** (see "Embedding model and CPU"). All-or-nothing: every file is verified in a staging directory before it replaces the destination; a model that is already complete is left untouched |
| `rootfs/opt/cas/bin/cas-run` | Runs operator tools (e.g. `bootstrap-admin`) with this boot's environment, as a given user |
| `rootfs/opt/cas/bin/svc-run` | Starts every supervisord program with `env -i` plus its own allowlist (`/run/cas-env/<service>.env`), then drops to the program's user |
| `rootfs/etc/cas/*` | `supervisord.conf`, `nginx.conf.template` (rendered by `entrypoint.sh` to `/run/cas-nginx.conf` with the exact upload limits), `postgresql.cas.conf`, `pg_hba.conf` |
| `env.staging.example` | Variable names for the RunPod template (no values) |
| `staging/` | 8.1-B: switching the existing RTX 4090 pod to this image on volume `buppvu4d48`. `pod_stage.sh check/preserve/apply` (C0 config, checkpoint manifest, tokenizer cache, e5 fp32), `verify_after_switch.sh`, and the rollback. See `staging/README.md` |
| `tests/smoke_cpu.sh`, `tests/c0_in_image.sh`, `tests/c0_in_image_negative.sh`, `tests/fetch_models_test.sh` | CPU smoke, C0 level (b) checks, proof that the C0 checker fails on broken images, offline all-or-nothing test of `fetch_models.sh`. All exit non-zero on any FAIL |

## Build

```bash
deploy/runpod/stage_c0.sh            # refuses untracked/modified C0 files or a changed import closure
docker buildx build --platform linux/amd64 -f deploy/runpod/Dockerfile \
  --build-arg CAS_REVISION=$(git rev-parse HEAD) -t cas-aio:$(git rev-parse --short HEAD) --load .
```

The build fails if `CAS_REVISION` is not the commit C0 was staged from.

RunPod needs `linux/amd64`. On Apple Silicon the amd64 runtime stage runs under QEMU emulation: the image builds there, and the C0 checks pass, but Qdrant crashes with SIGSEGV under QEMU (its official amd64 image does too), so the CPU smoke must run on native amd64 (or on a native arm64 build for development).

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
   - checks models and C0 (in `gpu` mode: checkpoints, tokenizer cache and GPU, fail-closed);
   - takes the workspace lease, creates the directories (container disk: owners and modes; volume: plain `mkdir`, writer must be able to write);
   - sets up the CA and leaf certificates on the container disk;
   - Postgres: reuses this container's `pgdata` if it belongs to the volume's cluster; otherwise restores it from the volume (`cas-pg-restore.sh`); or, only when the volume holds no cluster at all, runs `initdb` (with data checksums) plus a temporary unix-socket-only server that creates role `cas` (`pg_monitor`), database `citizen_assistance` and the `pgcrypto`/`vector` extensions;
   - writes a new boot id and removes any stale `bootstrap.ok`;
   - `exec supervisord`.
2. **supervisord** starts `pg-receivewal` (priority 8), `postgres` and `qdrant` (priority 10), and `cas-keeper` (25).
3. **`bootstrap`** (priority 20, user `cas`):
   - waits until a TLS query succeeds against Postgres and Qdrant answers `/readyz`, then until the WAL receiver streams **synchronously** (no write before WAL can reach the volume);
   - refuses a dirty migration state or a DB that is newer than the image;
   - if migrations are pending, writes a `pg_dump` to `backups/pre-migrate/`, then runs `migrate up`;
   - verifies every document file (size and SHA-256 against the `documents` row);
   - restores an empty Qdrant collection from the newest snapshot and verifies it, or verifies the existing one (see "Storage");
   - waits until `cas-keeper` has a verified base backup of this cluster and timeline;
   - writes `bootstrap.ok` and exits 0. Its expected state is `EXITED`.
4. **`ai-service`** and **`speech`** (gpu mode only) start once `bootstrap.ok` exists. **`api`** additionally waits for ai-service `/ready`, and **`nginx`** waits for api `/ready`.

`wait-for.sh` only gates startup. If Postgres or the WAL receiver dies later, supervisord restarts it after a crash. A manual `supervisorctl stop` keeps it down. The apps keep running, report `/ready` 503 and fail requests until the dependency is back, then reconnect without a restart.

## Storage

| Where | What | Survives a new container |
|---|---|---|
| container disk `/var/lib/cas` | `pgdata`, Qdrant storage, TLS keys and CA | no: rebuilt from the volume |
| volume `pg/state/system_identifier` | identity of the cluster this volume belongs to | yes |
| volume `pg/state/flushed_lsn.json` | the WAL position the receiver had flushed on the volume, recorded about every second by `cas-keeper` | yes |
| volume `pg/wal` | every WAL record, written by `pg_receivewal --synchronous` | yes |
| volume `pg/base/<ts>` | base backups (`pg_basebackup -Fp -X stream`, checked by `pg_verifybackup`); newest `CAS_BASE_KEEP` (2) kept | yes |
| volume `backups/` | `pg_dump` archives (`backup.sh`, before migrations) | yes |
| volume `qdrant/snapshots/<collection>` | snapshots + `.json`: SHA-256 and fingerprint (per generation: point count, SHA-256 of ids, payloads, float32 vectors), Postgres READY set | yes |
| volume `qdrant/baselines/<collection>` | `<generation>.json`: the fingerprint of each READY generation when first snapshotted; written once, never replaced, not pruned with the snapshots | yes |
| volume `objects/<bucket>/<key>` | document files written by the API's file store, read by the ai-service | yes |

**Synchronous WAL.** `synchronous_standby_names = 'cas_vol'` with `synchronous_commit = on`: a COMMIT is acknowledged only after `pg_receivewal --synchronous` has written and fsynced its WAL on the volume. While the receiver is down, commits wait (`SyncRep`); they are never silently asynchronous. A client that gives up while waiting gets an error, but its transaction stays committed locally and reaches the volume when the receiver is back. `max_slot_wal_keep_size = 4GB` bounds the WAL kept on the container disk; a lost slot is reported by the keeper and needs a new base backup.

**What this claims, and what it does not.** Acknowledged commits are designed to survive the loss of the container disk (tests: `tests/persist_test.sh` PT1–PT3, PT9). Whether an `fsync` that the RunPod network volume acknowledged survives a host or power failure of RunPod's storage is not something these tests can show.

**Restore on boot (`cas-pg-restore.sh`).** Fail-closed, in order: newest complete base backup, `pg_verifybackup`; its system identifier equals the volume's; WAL continuity (every segment from the base backup's start to the newest file present; tests PT4); recovery on the container disk with `restore_command` reading the volume, promotion at the end. Where recovery ended (an exact LSN) is then checked three ways, any failure refusing the restore:
- it is at or after the durable LSN recorded in `pg/state/flushed_lsn.json` (read before recovery starts): every commit acknowledged before that record is replayed (PT5c, PT5e: damage or lost writes inside the newest segment);
- it is in or after the newest WAL segment on the volume (PT5: a corrupt older segment);
- in the copy of the last segment that recovery read, nothing replayable follows it: only zeros, or one record cut at a page boundary with consistent continuation headers (`cas-storage wal-end-check`; PT5d: damage hidden behind a lagging durable-LSN record). A truncated WAL file is refused by Postgres itself (PT5f).

Not provable from the volume: a commit acknowledged after the newest durable-LSN record (at most about one second) whose WAL bytes the volume then lost. No RPO 0 is claimed. A missing `flushed_lsn.json` refuses the restore; an operator who knows the WAL is complete can set `CAS_RESTORE_WITHOUT_DURABLE_LSN=1` for that one boot. The keeper then takes a fresh base backup before `bootstrap.ok`, and bootstrap also waits for this boot's first durable-LSN record.

**Qdrant.** Engine data on the container disk; `cas-keeper` takes a snapshot whenever the set of READY generations in Postgres changes, re-reading the fingerprint after the snapshot (a collection that changed meanwhile is retried). On a new container, bootstrap uploads the newest snapshot, checks its SHA-256 and that the restored collection's fingerprint equals the recorded one (PT6), then that every READY generation has exactly `vector_count` points (PT7). On every boot, also with the collection still on the container disk, each READY generation's ids, payloads and vectors must equal its **baseline** (PT7c: one vector changed, same count). A READY generation never changes (re-indexing writes a new generation), so its baseline is the fingerprint the keeper recorded with its first snapshot, written once and kept after that snapshot is pruned. Before every snapshot (keeper or `backup.sh`), the live collection is checked: a READY generation with a point count other than Postgres `vector_count`, or points other than its baseline, refuses the snapshot, so the change can never become the baseline or the copy that a new container restores (PT7e). The running keeper also re-checks the live collection every `CAS_QDRANT_VERIFY_SECONDS` (300) without any READY change (PT7r: point deleted, PT7s: vector changed, both without a reboot). Either finding is a problem (`/readyz` 503) that is cleared only by a later check that passes (re-checked every 60 s), never because the READY set went back to an earlier one. A snapshot that fails for other reasons (Qdrant API, disk) is a warning, retried every tick, and a problem after 10 minutes without a snapshot of the current READY set. Baselines are written after their snapshot is published; if that write fails, the keeper retries, and bootstrap completes a missing baseline from the newest snapshot holding the generation as READY, only when the live points equal that snapshot's fingerprint. Between an index publish and the keeper's first snapshot of it (normally one keeper interval, 10 s) a generation has no baseline yet: a change in that window is recorded as its baseline. A READY generation without any recorded fingerprint (indexed, then the container stopped before the keeper's snapshot) makes bootstrap fail: reindex that document, or, after checking the collection yourself, record the current state with `cas-storage qdrant-snapshot` (as root).

**Objects.** `OBJECT_STORAGE_BACKEND=file`: `apps/api/internal/storage/file.go` writes, `apps/ai-service/app/indexing/file_objects.py` reads, both under the key contract `deploy/contracts/object_store_keys.json`. Every path component is opened relative to its parent with `O_NOFOLLOW` from a descriptor of the root, so neither `..` nor a symlink anywhere below the root can lead outside it. Writes: temporary file, `fsync`, `rename`, directory `fsync`. Bootstrap checks every document's file (size, SHA-256).

**Guard (`cas-keeper`, every 10 s; status in `/run/cas/storage-status.json`).** Problems: WAL receiver not streaming synchronously; replication slot lost; durable LSN not recorded for 30 s; a READY Qdrant generation incomplete or different from its baseline (snapshot refused, or the periodic check); no Qdrant snapshot of the current READY set for 10 minutes; free space below `CAS_VOLUME_MIN_FREE_GB` (10) of `CAS_VOLUME_QUOTA_GB` (200), measured with `du` of the whole volume every `CAS_VOLUME_SCAN_SECONDS` (1800) plus this system's own growth in between. Any problem: `/readyz` answers 503 and `healthcheck.sh` fails; low free space also sets the database read-only (`default_transaction_read_only`) until free space is back above the minimum plus `CAS_VOLUME_RESUME_MARGIN_GB` (5) (PT8). Warnings: WAL above `CAS_WAL_BUDGET_GB` (10). Base backups every `CAS_BASEBACKUP_HOURS` (24) or after `CAS_BASEBACKUP_WAL_GB` (1) of WAL; WAL older than the oldest kept base backup is removed (`pg_archivecleanup`); `CAS_SNAPSHOT_KEEP` (3) snapshots kept.

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
| nginx, postgres, qdrant, pg-lease, pg-receivewal | none |
| index-reaper, bootstrap, cas-keeper | `DATABASE_URL` |
| api | `DATABASE_URL`, `JWT_SECRET`, `AI_SERVICE_TOKEN`, `SPEECH_SERVICE_TOKEN` |
| ai-service | `DATABASE_URL`, `AI_SERVICE_TOKEN`, `GEMINI_API_KEY` / `OPENAI_API_KEY` |
| speech | `SPEECH_SERVICE_TOKEN` |

`POSTGRES_PASSWORD` itself reaches no program; it is only used to build `DATABASE_URL` and at first init. A variable set in the RunPod template reaches a program only if it is in that program's allowlist in `entrypoint.sh`. Smoke test E1 reads `/proc/<pid>/environ` of every program and checks this table.

## Users and permissions

| What | User | Mode |
|---|---|---|
| entrypoint, supervisord, tini | root | — |
| Postgres, `/var/lib/cas/pgdata`; pg_receivewal | `postgres` (10010) | `0700` |
| Qdrant, `/var/lib/cas/qdrant` | `qdrant` (10012) | `0700` |
| Go API, index-reaper, ai-service, bootstrap, `/run/cas` | `cas` (10001) | `/run/cas` `0755` |
| cas-keeper | root (drops to postgres for `pg_basebackup`) | — |
| Speech C0 | `speech` (10013) | — |
| nginx | `web` (10014) | — |
| `/var/lib/cas/tls/` / `tls/ca/` / `tls/postgres/` | root / root / postgres | `0711` / `0700` / `0700`; keys `0600` |
| `/var/lib/cas/tls/ca.crt` | root | `0644` |
| volume: `pg/`, `pg/wal`, `pg/base` / `backups/`, `objects/` / `pg/state`, `qdrant/` | postgres / cas / root (writer) | `0750` (`qdrant/` `0755`) where the volume keeps modes |

UIDs are fixed in the image. Owners and modes on the **volume** are applied where the filesystem supports them and are never relied on: the RunPod network volume (MooseFS over FUSE) reports `0777` and refuses `chown`. Nothing secret is written there (smoke F1): every key, the CA and `pgdata` stay on the container disk.

## TLS

- The internal CA lives in `/var/lib/cas/tls/ca/` on the container disk (EC P-256, 10 years). A new container gets a new CA; only loopback clients of this container use it. Leaf certs are signed by it, valid 825 days, with SAN `IP:127.0.0.1,DNS:localhost`.
- Certificates are created only when absent. If a key or certificate is half-present, mismatched, expired, or signed by another CA, startup stops with an error. Nothing is regenerated or overwritten silently.
- The CA is added to the system trust store, and `SSL_CERT_FILE` points to that bundle.
- Postgres accepts TCP only with TLS (`hostssl`). Apps connect with `sslmode=verify-full&sslrootcert=/var/lib/cas/tls/ca.crt`, so the client checks the chain and the `127.0.0.1` SAN. A wrong CA is refused (smoke T6).
- Rotation is manual: stop the pod, move the old files away, start again.

## Operations

```bash
docker exec <c> healthcheck.sh                                   # status table + supervisor
docker exec <c> supervisorctl -c /etc/cas/supervisord.conf status
docker exec -e BOOTSTRAP_ADMIN_EMAIL=admin@chuse.vn -e BOOTSTRAP_ADMIN_PASSWORD='<strong>' \
  <c> cas-run cas /opt/cas/api/bootstrap-admin                   # set the admin password
docker exec <c> backup.sh                                        # pg_dump + verified base backup + Qdrant snapshot
docker exec <c> restore_drill.sh                                 # full drill: Postgres, Qdrant and objects must restore/verify
docker exec <c> restore_drill.sh --postgres-only                 # explicit partial drill; result line says PARTIAL
docker exec <c> cas-storage status                               # keeper status: receiver, slot, free space, backups
```

Everything stays on the volume. Copy `/workspace/cas/pg`, `backups`, `qdrant/snapshots`, `qdrant/baselines` and `objects` out of RunPod (off-site copy: not automated yet).

## Rules

- **Single writer per volume.** The WAL, base backups, snapshots and objects of the volume must never have two writers; see "Single-writer lease" below.
- First Postgres init and every restore write `pgdata/.cas-init-complete` last. A `pgdata` without it is refused instead of being used half-initialised. A `pgdata` of another cluster than the volume's (`pg/state/system_identifier`) is refused too.
- `POSTGRES_PASSWORD` is applied once, at first init. To rotate it, run `ALTER ROLE cas PASSWORD ...` as `postgres` first, then change the secret.
- Migrations only move up. A DB newer than the image is refused, and so is a dirty state. Nothing is forced or rolled back automatically.
- No procedure is activated automatically. The Admin activates in the UI.

## Single-writer lease

`/workspace/cas/locks/` protects the volume (WAL, base backups, snapshots, objects) with two layers:

Owner identity: `runpod-<RUNPOD_POD_ID>`, else `docker-<container id>` (stable across restarts of the same container, different for every new container even with the same `--hostname`), else `host-<hostname>`.

1. **`flock` on `workspace.flock`.** `entrypoint.sh` takes it before touching any data, and supervisord keeps it (inherited fd) for the container's lifetime. A second container on the same host is refused immediately.
2. **Lease directory `workspace.lease/`.** It is created with `mkdir`, which is atomic. It holds `owner.json` (owner = `RUNPOD_POD_ID`, or the container hostname) and a `heartbeat`, which `pg-lease` rewrites every 10 s. `pg-lease` stops the data services at once if the lease names another owner or the heartbeat cannot be written. Postgres, Qdrant, the WAL receiver and `cas-keeper` start through `pg-start.sh`, which refuses to run without this container's fresh lease; fencing also stops the API, which writes document files.

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

| Endpoint | Limit enforced by nginx | Answer when exceeded |
|---|---|---|
| `POST /api/v1/admin/documents` (PDF upload) | `DOCUMENT_MAX_BYTES` (default 20 MiB, max 64 MiB) + 256 KiB multipart overhead | `413` JSON `PAYLOAD_TOO_LARGE`, from nginx |
| `POST /api/v1/sessions/<id>/speech/translate` | `SPEECH_MAX_UPLOAD_BYTES` (default and max 10 MiB) + 256 KiB | `413` JSON `PAYLOAD_TOO_LARGE`, from nginx |
| rest of `/api` | 65 MiB | `413` HTML, from nginx |

- `entrypoint.sh` renders these limits into the nginx config from the same variables, and the same multipart overhead, that the Go API uses. The two layers therefore agree to the byte.
- nginx must reject an oversized upload itself. If it streamed the body to Go instead, Go would answer 413 from `Content-Length` and close the connection while nginx was still sending. The client then gets a **502**, which is what the first native amd64 CI run showed.
- Smoke tests:
  - A6: 21 MiB PDF → 413 JSON;
  - A6b: a body just under the limit is not cut by nginx;
  - A7: 66 MiB elsewhere → 413 from nginx;
  - A8: oversized audio → 413 JSON.

## Embedding model and CPU

`EMBEDDING_VARIANT` chooses the ONNX graph of `intfloat/multilingual-e5-small` (same pinned revision, tokenizer, 384 dimensions, `query:` / `passage:` prefixes, mean pooling and normalization in both). The ai-service applies it in `apps/ai-service/app/indexing/embed.py` (`select_embedding_variant`), from the real CPU flags:

| `EMBEDDING_VARIANT` | CPU with AVX-512 VNNI | CPU without it (e.g. AMD EPYC 7B13 / Zen 3, arm64) |
|---|---|---|
| `fp32` (**image default**) | fp32 (`onnx/model.onnx`, 470 MB) | fp32 |
| `int8-vnni` | int8 (`onnx/model_qint8_avx512_vnni.onnx`, 118 MB) | **refused** (`embedding_cpu_unsupported`) |
| `auto` | int8 | fp32 |

Manifests: `models/embedding.manifest.fp32.json`, `models/embedding.manifest.int8-vnni.json` (SHA-256 of every file).

- **Why fp32 is the default:** a RunPod pod can restart on another host CPU. With `auto`, a pod that moves between a VNNI and a non-VNNI host would need the other model. fp32 runs on every x86_64 (AVX2) and arm64 host, so the model and the vector identity stay the same.
- **Vectors of the two variants are not mixed:** each index generation stores the model checksum (`embedding_checksum`). Retrieval only uses chunks whose checksum equals the running model's, and answers `embedding_identity_mismatch` otherwise. Changing the variant on a volume that already has indexed documents therefore needs a reindex (see below).
- Fetch the model for the variant you run: `deploy/runpod/fetch_models.sh --variant fp32 /workspace/cas/models/embedding`. `--variant auto` applies the rule to the machine running the script. `EMBEDDING_VARIANT` is read when `--variant` and `CAS_MODEL_VARIANT` are unset.
- `entrypoint.sh` validates `EMBEDDING_VARIANT`, then checks the installed model with the ai-service's own rules (variant on this CPU + SHA-256 of every file). The log line `embedding model verified ...` shows the variant, the CPU model and `avx512_vnni=yes|no`.
- `EMBEDDING_CPU_FLAGS` replaces the real CPU flags in the ai-service. It is for tests only: `entrypoint.sh` refuses it unless `CAS_ALLOW_CPU_FLAGS_OVERRIDE=true`.
- Tests: smoke N9–N13 and V1, `tests/embedding_in_image.sh` (real inference of both graphs on the CI runner), `tests/fetch_models_test.sh` FM10–FM12, and the retrieval comparison in `eval/`.

### Changing the variant on a volume with indexed documents

Never switch in place:

1. `backup.sh` (pg_dump, base backup, Qdrant snapshot).
2. Fetch the new variant's model into a new directory, set `EMBEDDING_VARIANT` and `EMBEDDING_MODEL_DIR`, restart.
3. Reindex each document from the admin page. Until a document is reindexed, retrieval answers `embedding_identity_mismatch` for it instead of returning vectors of the other model.
4. Rollback: set the previous `EMBEDDING_VARIANT` and `EMBEDDING_MODEL_DIR` back and reindex again, or restore the backup of step 1. A reindex marks the previous generation `SUPERSEDED`; only failed generations are cleaned from Qdrant, so do not count on the old vectors being usable: the backup is the rollback.
5. Procedure activation is not touched by any of these steps; review and Activate stay manual.

## LLM provider

`LLM_PROVIDER` is required: `gemini` (needs `GEMINI_API_KEY`), `openai` (needs `OPENAI_API_KEY`), or `mock`. `mock` is refused unless `CAS_ALLOW_MOCK_LLM=true`, which only tests set.

## Known limits

- **Volume isolation:** on the RunPod network volume every file is `0777`: any process in this container could read or change any file there, the research data included. Nothing secret is stored on it, and every write of this system goes under `/workspace/cas` (file store confined to its root, no symlink followed), but full separation needs a volume of its own.
- **Off-site copy:** WAL, base backups, snapshots and objects are all on the same network volume. A copy outside RunPod is not automated.
- **Qdrant:** no API key, because the code has no client support for one. Qdrant is reachable on loopback only.
- **Client IP for rate limiting:** nginx uses `CF-Connecting-IP`, falling back to the TCP peer. Which header RunPod forwards is unverified until 8.1-B.
- **Legacy `/ai-drafts` call:** this synchronous call can exceed the 100 s proxy limit. Drafting jobs are asynchronous and are not affected.
- **Restore drill timing:** `restore_drill.sh` compares the restored copy with the live data at drill time. Run it when nothing is writing; writes in between show up as a content mismatch.
- **Replacing the model:** when `fetch_models.sh` installs a *different* model, it swaps the directory, so restart any container that bind-mounts it. A model that is already complete and valid is never touched.
- **Model swap interrupted:**
  - If publishing fails, the previous model is put back.
  - If that also fails, the previous model is kept intact at `<parent>/.<name>.old.<pid>` and the script prints the path. It is never deleted.
  - A `SIGKILL` exactly between the two renames cannot run any handler. In that case the destination is missing and the previous model sits in `.<name>.old.<pid>`. Recovery: `mv <parent>/.<name>.old.<pid> <dest>`, then rerun `fetch_models.sh`.
- **Single writer across pods:** both lease layers are proven only on a local Docker volume. A two-pod test on the RunPod network volume is mandatory in 8.1-B.
