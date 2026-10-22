# P4B — content processing and vector indexing

P4B reads a PDF from MinIO, extracts text, OCRs only weak pages, chunks, embeds, and stages a generation. The Go API publishes that generation only while the claim token still matches. There is no retrieval, no chatbot, and no Activate button.

## Local command

Prepare models only if you select the ONNX and Paddle adapters. The process does not download them.

```bash
cd deploy && ADMIN_INGESTION_ENABLED=true VITE_ADMIN_INGESTION=true ADMIN_INDEXING_ENABLED=true VITE_ADMIN_INDEXING=true \
  INDEX_MODE=pipeline EMBEDDING_PROVIDER=fake OCR_PROVIDER=fake \
  INDEX_PIPELINE_TIMEOUT_SECONDS=120 INDEX_CLAIM_TTL_SECONDS=180 \
  docker compose --profile app --profile storage --profile ai --profile qdrant up --build
```

Pipeline timeout is at most 15 minutes. Claim TTL must be longer than that timeout by at least one second. Mock mode still uses `INDEX_TIMEOUT_MS` with a 10 second ceiling.

Unlink is soft: the link row stays, `unlinked_at` is set, and `active_generation_id` is cleared. Jobs, generations, and chunks remain. Relink clears `unlinked_at` and returns the link to `UPLOADED` without restoring the old generation. Reindex of a `READY` link keeps the active generation until the new one is verified.

`scripts/p4b-model-smoke.py` reads only local model directories. Without weights it prints `P4B_MODEL_SMOKE_PENDING` and exits 2. Exit 0 prints `P4B_DONE` only after a real E5 forward pass and a real PaddleOCR read of `apps/ai-service/tests/fixtures/ocr_text_scan.pdf`. That fixture is an image of `GIAY KHAI SINH`, `GIA LAI`, and `SO 123`, with no text layer. `mixed.pdf` page 2 stays the blank-stripe negative fixture. `P4B_DONE` is the smoke marker. It does not mean RunPod S3, GitHub Actions, fp32/int8 equivalence, or production deployment have run.

## Local Apple Silicon

Verified layout: macOS on Apple Silicon, Compose project `citizen-assistance`, Linux ARM64 containers. `deploy/.env` is a local override and stays gitignored. `deploy/.env.example` leaves ingestion and indexing off. Do not run `seed-demo` when `admin@chuse.vn` is already in the database. Local MinIO does not need a RunPod key.

1. Prerequisites: Docker Desktop, Go 1.25 for host API tests, Node for the web package, and the Python venv in `apps/ai-service/.venv` for unit tests. Real OCR smoke runs in the `ai-service` container, which has PaddleOCR and Pdfium.
2. Environment: copy `deploy/.env.example` to `deploy/.env` only on a new machine. Set `ADMIN_INGESTION_ENABLED`, `VITE_ADMIN_INGESTION`, `ADMIN_INDEXING_ENABLED`, and `VITE_ADMIN_INDEXING` to `true` for the admin UI. Set `INDEX_MODE=pipeline`, `EMBEDDING_PROVIDER=onnx`, `OCR_PROVIDER=paddle`, `INDEX_PIPELINE_TIMEOUT_SECONDS=120`, and `INDEX_CLAIM_TTL_SECONDS=180`. The verified local file publishes the API at `API_PORT=8082` and the web UI at `WEB_PORT=5174` so they do not collide with `make api-run` on 8080 and Vite on 5173.
3. Model directories: `models/embedding` and `models/ocr` at the repo root. Compose mounts them with `EMBEDDING_MODEL_DIR_HOST` and `OCR_MODEL_DIR_HOST`, defaulting to those paths, at `/models/embedding` and `/models/ocr`.
4. On this ARM64 machine the embedding manifest must say `quantization_method` `fp32`, revision `6a0d452a575215f80b8f66276dd4ee5d504942c6`, `model.onnx` SHA-256 `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`, and `tokenizer.json` SHA-256 `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`. Check with `shasum -a 256 models/embedding/model.onnx models/embedding/tokenizer.json`. Do not rename the AVX-512 VNNI file to `model.onnx` on this machine.
5. From `deploy/`: `docker compose --env-file .env --profile app --profile storage --profile ai --profile qdrant up -d`. Rebuild the web image after changing either `VITE_*` flag.
6. Health: `docker compose ps`. `api`, `web`, `minio`, and `postgres` expose healthchecks. `ai-service` uses `/ready` when `INDEX_MODE=pipeline`. Qdrant listens on `${QDRANT_PORT:-6333}`.
7. The same Compose command starts the Go API, the Python AI service, and the web container. `make api-run` and `make web-run` are the host processes, not required once the Compose stack is up.
8. Open `http://127.0.0.1:${WEB_PORT}/admin/documents` and sign in as the existing admin. Do not run `go run ./cmd/seed-demo`.
9. Upload a PDF from that page. Bytes go to the MinIO bucket `cas-documents`. Metadata goes to PostgreSQL `documents`.
10. Link the document to a procedure version in the same xa and domain. The local khai sinh version is `dk_khai_sinh` `1.0.0`.
11. `INDEX_MODE=mock` exercises the admin status flow and does not read the PDF or write vectors.
12. `INDEX_MODE=pipeline` with `EMBEDDING_PROVIDER=onnx` and `OCR_PROVIDER=paddle` reads the PDF, OCRs weak pages, and writes chunks plus Qdrant points.
13. Chunks are rows in `knowledge_chunks`. The `embedding` column is unused. Vectors are points in Qdrant collection `knowledge_chunks`, size 384, cosine.
14. Model smoke, inside the running `ai-service` container after copying the current script, `app/`, and `ocr_text_scan.pdf` to `/opt/cas`: `EMBEDDING_MODEL_DIR=/models/embedding OCR_MODEL_DIR=/models/ocr python /opt/cas/scripts/p4b-model-smoke.py`. Exit 0 prints `P4B_DONE`. Exit 2 prints `P4B_MODEL_SMOKE_PENDING` when a manifest is missing.
15. If `/ready` stays unavailable, compare each model file with the manifest SHA-256. An ARM64 container must be on the fp32 graph. A VNNI graph on this CPU does not load.

Storage integration tests are separate. They stay on `deploy/docker-compose.p4b-test.yml` (project `citizen-assistance-p4b-test`, Postgres `127.0.0.1:5434` database `citizen_assistance_p4b_test`, MinIO `127.0.0.1:9010` bucket `cas-p4b-integration-test`, Qdrant `http://127.0.0.1:6335`). `CAS_P4B_INTEGRATION=1` alone does not make a target safe. The guard also requires `CAS_P4B_TEST_COMPOSE_PROJECT=citizen-assistance-p4b-test` and those exact endpoints. It parses the Postgres URL with libpq and rejects query parameters other than `sslmode=disable`, including `host`, `hostaddr`, `port`, `dbname`, `service`, and `servicefile`. It also rejects a non-empty `PGHOST`, `PGHOSTADDR`, `PGPORT`, `PGDATABASE`, `PGSERVICE`, `PGSERVICEFILE`, or `PGSYSCONFDIR`. The test URL pins `hostaddr=127.0.0.1` with `?` when the URL has no query string and `&` when it already has one. That pin is test-only. A production database URL does not pass through it. The effective target is checked again immediately before `psycopg.connect`. Docker must publish `5434`, `9010`, `9011`, and `6335` on `127.0.0.1` only. `current_database()` is an extra check, not the only one.

`GET /ready` does not create a Qdrant collection. Create `knowledge_chunks` with `scripts/provision-qdrant.py` before the first pipeline start. A 404 collection makes the service not ready.

The two tests in `apps/ai-service/tests/test_p4b_services.py` use `FakeEmbedder` and `FakeOCR`. They refuse the development database `citizen_assistance`, bucket `cas-documents`, MinIO port 9000, and Qdrant port 6333. Run them only with `CAS_P4B_INTEGRATION=1` and `CAS_P4B_TEST_*` pointed at `deploy/docker-compose.p4b-test.yml`. A pass there is storage integration, not real-model inference.

`EMBEDDING_PROVIDER=fake` is an explicit test double. It is not the mock lifecycle from P4A. Omit `EMBEDDING_PROVIDER` and `OCR_PROVIDER` when `INDEX_MODE=pipeline` and the service refuses to start unless the local ONNX and Paddle directories match their manifests.

`INDEX_MODE=mock` remains the P4A worker. Pipeline mode never calls that mock.

The orphan reaper requires `DATABASE_URL`, `QDRANT_URL`, and `QDRANT_COLLECTION`. A missing value exits non-zero and does not claim a generation. HTTP 404 from Qdrant is a failure, not an empty collection. The Python pipeline uses the same cleanup contract: delete must return HTTP 200, `result.status` must be `completed`, and the exact generation count must be 0. A 404, 409, 500, malformed body, or leftover points is `qdrant_failed` and the reaper retries later. `INDEX_CLEANUP_BATCH_SIZE` defaults to 8 and cannot exceed 32. Each claimed row renews its cleanup lease before delete. A run prints `cleanup_claimed`, `cleanup_completed`, `cleanup_retryable_failed`, `cleanup_terminal_failed`, and `cleanup_lost`. Any failure exits non-zero. `COMPLETED` is written only after Qdrant delete reports `result.status=completed`, the generation count is exactly 0, and the PostgreSQL chunk delete commits with the same claim token.

Each `index.v2` request creates one monotonic deadline of `INDEX_PIPELINE_TIMEOUT_SECONDS`. The pipeline runs in its own process group and already holds the models. ONNX and PaddleOCR children stay in that group. When the deadline expires, or the ASGI client sends `http.disconnect`, only that call's process group is killed within `INDEX_TERMINATION_GRACE_SECONDS`. Identity is the kernel tuple of pid, start time, and verified process group, read from `/proc` or `proc_pidinfo`. The supervisor does not call `ps`. A slot is ready only after that identity is verified. SIGTERM is followed by SIGKILL of the same group even if the leader has already exited. Independent groups are cleaned up concurrently: the shared lock only updates bookkeeping, and each identity has its own lock. The grace wait polls the group and returns as soon as it is empty. Recovery starts only after the group is confirmed empty. A ready slot whose pipeline leader has died is detected by the slot monitor, the snapshotted group is cleared, and one replacement is spawned. A second death of that slot is cleaned and left failed so recovery cannot loop. `/ready` is 503 while that replacement is starting. Shutdown publishes closed, registers every recovery thread, and starts a process under the same lock, so `proc.start` cannot run after `shutdown` returns. Recovery joins are bounded by `timeout_s`. A thread still alive at that deadline leaves shutdown failed, and the thread is a daemon. A recycled PID is not signaled. `/ready` stays 503 while any slot is recovering, failed, or missing a live OCR state. Each pipeline process starts one OCR worker. That worker does not arm a parent-death signal, because OCR recovery runs on a short-lived thread and Linux would kill the replacement when the thread exits. Cleanup after a partial upsert uses a separate 2 second budget and still requires a verified empty count.

Object storage accepts MinIO and RunPod S3. Local MinIO may auto-create the bucket and may use `minio:9000` without TLS. RunPod requires `OBJECT_STORAGE_REGION`, TLS on port 443, and `OBJECT_STORAGE_AUTO_CREATE_BUCKET=false`. A RunPod hostname with `OBJECT_STORAGE_USE_SSL=false`, an `http://` RunPod URL, or an explicit RunPod port other than 443 fails at startup. The service does not call `MakeBucket` or `GetBucketLocation` when the region is set and auto-create is false. The bucket value is the RunPod network volume id. Do not put access keys, secret keys, or presigned URLs in source or in this runbook.

## Model preparation

Both graphs are `intfloat/multilingual-e5-small` at commit `6a0d452a575215f80b8f66276dd4ee5d504942c6`. The previous pin `6e0d5e48e6120c11e986346003f54bca39b85422` is not a commit of this repository. Dimension stays 384. `select_embedding_variant` picks one graph. It does not let an Apple Silicon machine load the VNNI file.

ARM64 and aarch64, including this Mac and the Linux ARM64 `ai-service` image, use ONNX Runtime CPU Execution Provider with the fp32 file `onnx/model.onnx`. SHA-256 `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`. `tokenizer.json` SHA-256 `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`.

Linux x86_64 uses `onnx/model_qint8_avx512_vnni.onnx` only when every required AVX-512 VNNI flag is present. SHA-256 `dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88`. A x86_64 CPU without those flags raises `embedding_cpu_unsupported`. FP32 is not a silent fallback on that CPU. Copy the selected file to `model.onnx` and copy `tokenizer.json`. Do not commit either file.

The JSON below is the x86_64 VNNI manifest. The ARM64 manifest on this machine uses `source_file` `onnx/model.onnx`, `quantization` `fp32`, and the fp32 SHA-256 above.

```json
{
  "model_id": "intfloat/multilingual-e5-small",
  "source_repository": "https://huggingface.co/intfloat/multilingual-e5-small",
  "revision": "6a0d452a575215f80b8f66276dd4ee5d504942c6",
  "source_file": "onnx/model_qint8_avx512_vnni.onnx",
  "dimension": 384,
  "quantization": "int8",
  "quantization_method": "qint8_avx512_vnni",
  "opset": null,
  "files": {
    "model.onnx": "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
  }
}
```

4. Mount the directory at `EMBEDDING_MODEL_DIR`. A missing file or a bad checksum stops startup.
5. PaddleOCR DBNet + CRNN CPU weights go in `OCR_MODEL_DIR` (`det/`, `rec/`, `cls/`). An empty directory is `ocr_model_missing`.

## Error codes

| Code | Meaning |
|---|---|
| `source_not_found` | Bucket or object key is not the internal key for this document |
| `source_timeout` | MinIO did not answer |
| `checksum_mismatch` | Bytes do not match the stored SHA-256 |
| `mime_invalid` | File does not start with `%PDF-` |
| `pdf_corrupt` | Parser cannot read the PDF |
| `pdf_encrypted` | PDF is encrypted |
| `pdf_oversized` | File exceeds the size cap |
| `page_range_invalid` | Range is outside the page count |
| `no_extractable_text` | Native text and OCR produced nothing |
| `ocr_timeout` / `ocr_failed` / `ocr_model_missing` | OCR adapter failed closed |
| `embedding_model_missing` / `embedding_checksum_mismatch` / `embedding_dimension` / `embedding_non_finite` | Embedding adapter failed closed |
| `postgres_failed` / `qdrant_failed` | Staging write failed |
| `publish_mismatch` | Counts, manifest, or generation do not match |
| `worker_mismatch` | Response identity does not match the claim |
| `claim_rejected` | Staging row was no longer `STAGING` |
| `page_limit` | PDF or OCR page count exceeds the configured maximum |
| `pipeline_busy` | Concurrent OCR slots are full |
| `orphan_expired` | Reaper marked an expired staging generation failed |
| `timeout` / `worker_failed` | Existing P4A worker failures |

Codes are at most 64 characters. A link stays `FAILED` with `last_error_code`. A previous `READY` generation is kept until a newer one publishes.

## Recovery

Retry uses the existing retry endpoint and a new request ID. Reindex of a READY link is `POST /api/v1/admin/documents/{id}/links/{versionId}/index/reindex`. It opens a new staging generation and does not delete the active one first. Unlink returns 409 while any job for that link is still `CLAIMED` and unexpired, including a reindex that leaves the link `READY`. A finish after `unlinked_at` is set does not publish and does not change `active_generation_id`. `go run ./cmd/index-reaper` claims expired non-active `STAGING` or `FAILED` generations with `UPDATE ... RETURNING` (`cleanup_status` `PENDING` to `CLAIMED`). It deletes chunks and Qdrant points only for ids that claim returned. Qdrant delete must parse `result.status=completed` and an exact count of 0 before `cleanup_status` becomes `COMPLETED`. A later run does not select `COMPLETED` rows. `INDEX_ORPHAN_GRACE_SECONDS` (default 120) is the age threshold. A live claim, an active generation, a `READY` row, and a generation younger than the grace period are not claimed. `Finish()` publishes only when `claim_expires_at > now()` and `cleanup_status` is still `PENDING`. OCR and ONNX initialization run in a child process that is killed if it exceeds its timeout. `select_embedding_variant` loads `qint8_avx512_vnni` only when the CPU advertises every required AVX-512 VNNI flag. `arm64` and `aarch64` load the fp32 graph. Any other CPU without those flags fails with `embedding_cpu_unsupported`. There is no invented fallback file. `/health` is liveness. In pipeline mode `/ready` probes loaded adapters, PostgreSQL, the Qdrant collection dimension and distance, and the MinIO bucket, and it returns only status codes. P4C should read `procedure_version_documents.active_generation_id` and filter Qdrant by `xa_id`, `procedure_version_id`, and `generation_id`. PostgreSQL and Qdrant are not one transaction. If PostgreSQL fails after a Qdrant upsert, the pipeline deletes that generation's points and accepts the delete only when the operation status is `completed` and the exact count is 0.

Logs use `index_finish` and `index_pipeline` with counts only. They do not include page text.
