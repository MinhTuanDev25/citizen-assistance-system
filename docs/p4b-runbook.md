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

`scripts/p4b-model-smoke.py` reads only local model directories. Without weights it prints `P4B_MODEL_SMOKE_PENDING` and does not claim `P4B_DONE`.

`EMBEDDING_PROVIDER=fake` is an explicit test double. It is not the mock lifecycle from P4A. Omit `EMBEDDING_PROVIDER` and `OCR_PROVIDER` when `INDEX_MODE=pipeline` and the service refuses to start unless the local ONNX and Paddle directories match their manifests.

`INDEX_MODE=mock` remains the P4A worker. Pipeline mode never calls that mock.

The orphan reaper requires `DATABASE_URL`, `QDRANT_URL`, and `QDRANT_COLLECTION`. A missing value exits non-zero and does not claim a generation. HTTP 404 from Qdrant is a failure, not an empty collection. The Python pipeline uses the same cleanup contract: delete must return HTTP 200, `result.status` must be `completed`, and the exact generation count must be 0. A 404, 409, 500, malformed body, or leftover points is `qdrant_failed` and the reaper retries later. `INDEX_CLEANUP_BATCH_SIZE` defaults to 8 and cannot exceed 32. Each claimed row renews its cleanup lease before delete. A run prints `cleanup_claimed`, `cleanup_completed`, `cleanup_retryable_failed`, `cleanup_terminal_failed`, and `cleanup_lost`. Any failure exits non-zero. `COMPLETED` is written only after Qdrant delete reports `result.status=completed`, the generation count is exactly 0, and the PostgreSQL chunk delete commits with the same claim token.

Each `index.v2` request creates one monotonic deadline of `INDEX_PIPELINE_TIMEOUT_SECONDS`. MinIO, OCR, token counting, ONNX, Qdrant, and PostgreSQL all spend that same budget. A stage does not receive the full timeout again. After the deadline, the worker does not start another Qdrant or PostgreSQL write. Cleanup after a partial upsert uses a separate 2 second budget and still requires a verified empty count.

## Model preparation

Production embedding is `intfloat/multilingual-e5-small` at commit `6a0d452a575215f80b8f66276dd4ee5d504942c6` on `https://huggingface.co/intfloat/multilingual-e5-small`. That commit adds `onnx/model_qint8_avx512_vnni.onnx`. The previous pin `6e0d5e48e6120c11e986346003f54bca39b85422` is not a commit of this repository.

`embedding_checksum` is the SHA-256 of the local `model.onnx` file. It must equal the upstream LFS oid `dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88`. `tokenizer.json` must equal `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`. Changing only the revision string does not make another file valid.

Copy those two files to `model.onnx` and `tokenizer.json`. Do not commit them. The ONNX graph opset is inside the file and was not downloaded here, so `opset` stays unverified until `scripts/p4b-model-smoke.py` can read the local graph.

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

Retry uses the existing retry endpoint and a new request ID. Reindex of a READY link is `POST /api/v1/admin/documents/{id}/links/{versionId}/index/reindex`. It opens a new staging generation and does not delete the active one first. Unlink returns 409 while any job for that link is still `CLAIMED` and unexpired, including a reindex that leaves the link `READY`. A finish after `unlinked_at` is set does not publish and does not change `active_generation_id`. `go run ./cmd/index-reaper` claims expired non-active `STAGING` or `FAILED` generations with `UPDATE ... RETURNING` (`cleanup_status` `PENDING` to `CLAIMED`). It deletes chunks and Qdrant points only for ids that claim returned. Qdrant delete must parse `result.status=completed` and an exact count of 0 before `cleanup_status` becomes `COMPLETED`. A later run does not select `COMPLETED` rows. `INDEX_ORPHAN_GRACE_SECONDS` (default 120) is the age threshold. A live claim, an active generation, a `READY` row, and a generation younger than the grace period are not claimed. `Finish()` publishes only when `claim_expires_at > now()` and `cleanup_status` is still `PENDING`. OCR and ONNX initialization run in a child process that is killed if it exceeds its timeout. The pinned graph is `qint8_avx512_vnni`; startup refuses to load it unless the CPU advertises `avx512vnni`. There is no invented fallback file. `/health` is liveness. In pipeline mode `/ready` probes loaded adapters, PostgreSQL, the Qdrant collection dimension and distance, and the MinIO bucket, and it returns only status codes. P4C should read `procedure_version_documents.active_generation_id` and filter Qdrant by `xa_id`, `procedure_version_id`, and `generation_id`. PostgreSQL and Qdrant are not one transaction. If PostgreSQL fails after a Qdrant upsert, the pipeline deletes that generation's points and accepts the delete only when the operation status is `completed` and the exact count is 0.

Logs use `index_finish` and `index_pipeline` with counts only. They do not include page text.
