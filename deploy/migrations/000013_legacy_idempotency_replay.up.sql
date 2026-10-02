-- 000013_legacy_idempotency_replay.up.sql
-- v10 stored LINK responses without per-link status fields, and UNLINK
-- responses as {"status":"unlinked"}. Rewrite those blobs so a replay
-- matches the current response shape. Rows that already have the fields
-- are left unchanged.

UPDATE admin_write_idempotency
SET response_json = response_json || jsonb_build_object(
        'index_status', 'UPLOADED',
        'updated_at', created_at,
        'recoverable', false
    )
WHERE action = 'LINK'
  AND (
      COALESCE(response_json->>'index_status', '') = ''
      OR response_json->>'updated_at' IS NULL
      OR response_json->>'recoverable' IS NULL
  );

WITH eligible AS MATERIALIZED (
    SELECT request_id, xa_id,
        CASE
            WHEN response_json->>'document_id' ~ '^[0-9a-fA-F-]{36}$'
            THEN (response_json->>'document_id')::uuid
        END AS document_id,
        CASE
            WHEN response_json->>'procedure_version_id' ~ '^[0-9a-fA-F-]{36}$'
            THEN (response_json->>'procedure_version_id')::uuid
        END AS procedure_version_id
    FROM admin_write_idempotency
    WHERE action = 'LINK'
      AND response_json->>'document_id' ~ '^[0-9a-fA-F-]{36}$'
      AND response_json->>'procedure_version_id' ~ '^[0-9a-fA-F-]{36}$'
)
UPDATE admin_write_idempotency AS idem
SET response_json = idem.response_json || jsonb_build_object(
        'index_status', link.index_status,
        'updated_at', link.updated_at,
        'recoverable', (
            link.index_status = 'FAILED'
            OR (
                link.index_status = 'PROCESSING'
                AND (job.claim_expires_at IS NULL OR job.claim_expires_at <= now())
            )
        )
    )
FROM eligible
JOIN procedure_version_documents AS link
    ON link.document_id = eligible.document_id
   AND link.procedure_version_id = eligible.procedure_version_id
   AND link.xa_id = eligible.xa_id
LEFT JOIN LATERAL (
    SELECT claim_expires_at
    FROM document_index_jobs
    WHERE document_id = link.document_id
      AND procedure_version_id = link.procedure_version_id
      AND xa_id = link.xa_id
      AND status = 'CLAIMED'
    ORDER BY claimed_at DESC
    LIMIT 1
) AS job ON true
WHERE idem.action = 'LINK'
  AND idem.request_id = eligible.request_id
  AND idem.xa_id = eligible.xa_id;

UPDATE admin_write_idempotency AS idem
SET response_json = idem.response_json || jsonb_build_object(
        'status', COALESCE(idem.response_json->>'status', 'unlinked'),
        'document_id', audit.entity_id,
        'procedure_version_id', audit.payload->>'procedure_version_id'
    )
FROM audit_logs AS audit
WHERE idem.action = 'UNLINK'
  AND COALESCE(idem.response_json->>'document_id', '') = ''
  AND audit.request_id = idem.request_id
  AND audit.action = 'DOCUMENT_UNLINKED'
  AND audit.entity_id ~ '^[0-9a-fA-F-]{36}$'
  AND COALESCE(audit.payload->>'procedure_version_id', '') ~ '^[0-9a-fA-F-]{36}$';
