-- 000011_link_index_status.up.sql
-- Indexing status lives on each document↔version link.
-- A job belongs to one (document, procedure version) inside one commune.
-- documents.processing_status stays an aggregate of the current links.

ALTER TABLE procedure_version_documents
    ADD COLUMN index_status text NOT NULL DEFAULT 'UPLOADED',
    ADD COLUMN last_error_code text,
    ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();

ALTER TABLE procedure_version_documents
    ADD CONSTRAINT pvd_index_status_check CHECK (
        index_status IN ('UPLOADED', 'PROCESSING', 'READY', 'FAILED')
    ),
    ADD CONSTRAINT pvd_last_error_code_check CHECK (
        last_error_code IS NULL OR char_length(last_error_code) BETWEEN 1 AND 64
    );

ALTER TABLE procedure_version_documents DROP CONSTRAINT pvd_page_range_check;
ALTER TABLE procedure_version_documents
    ADD CONSTRAINT pvd_page_range_check CHECK (
        page_range IS NULL
        OR (
            page_range ~ '^[1-9][0-9]{0,3}(-[1-9][0-9]{0,3})?$'
            AND (
                strpos(page_range, '-') = 0
                OR split_part(page_range, '-', 1)::int <= split_part(page_range, '-', 2)::int
            )
        )
    );

ALTER TABLE document_index_jobs
    ADD COLUMN procedure_version_id uuid,
    ADD COLUMN claim_token uuid;

UPDATE document_index_jobs AS job
SET procedure_version_id = link.procedure_version_id
FROM (
    SELECT DISTINCT ON (document_id, xa_id) document_id, xa_id, procedure_version_id
    FROM procedure_version_documents
    ORDER BY document_id, xa_id, created_at
) AS link
WHERE job.document_id = link.document_id
  AND job.xa_id = link.xa_id
  AND job.procedure_version_id IS NULL;

DELETE FROM document_index_jobs WHERE procedure_version_id IS NULL;

UPDATE document_index_jobs SET claim_token = gen_random_uuid() WHERE claim_token IS NULL;

ALTER TABLE document_index_jobs
    ALTER COLUMN procedure_version_id SET NOT NULL,
    ALTER COLUMN claim_token SET NOT NULL;

ALTER TABLE document_index_jobs DROP CONSTRAINT ux_document_index_jobs_request;
DROP INDEX ux_document_index_jobs_one_claim;

ALTER TABLE document_index_jobs
    ADD CONSTRAINT fk_index_jobs_link
        FOREIGN KEY (procedure_version_id, document_id)
        REFERENCES procedure_version_documents (procedure_version_id, document_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT document_index_jobs_error_code_check CHECK (
        error_code IS NULL OR char_length(error_code) BETWEEN 1 AND 64
    ),
    ADD CONSTRAINT ux_document_index_jobs_request UNIQUE (xa_id, request_id);

CREATE UNIQUE INDEX ux_document_index_jobs_one_claim
    ON document_index_jobs (document_id, procedure_version_id)
    WHERE status = 'CLAIMED';

ALTER TABLE admin_write_idempotency ADD COLUMN xa_id text;

-- Keep existing rows. xa_id comes from the audit payload, or from the
-- document named by that audit, and only when exactly one commune matches.
WITH src AS (
    SELECT idem.request_id,
           idem.action,
           resolved.xa
    FROM admin_write_idempotency AS idem
    JOIN (
        SELECT request_id,
               CASE action
                   WHEN 'DOCUMENT_LINKED' THEN 'LINK'
                   WHEN 'DOCUMENT_UNLINKED' THEN 'UNLINK'
               END AS action,
               COALESCE(
                   NULLIF(btrim(payload->>'xa_id'), ''),
                   (
                       SELECT d.xa_id
                       FROM documents AS d
                       WHERE d.id::text = audit_logs.entity_id
                   )
               ) AS xa
        FROM audit_logs
        WHERE action IN ('DOCUMENT_LINKED', 'DOCUMENT_UNLINKED')
    ) AS resolved
        ON resolved.request_id = idem.request_id
       AND resolved.action = idem.action
    WHERE resolved.xa IS NOT NULL AND btrim(resolved.xa) <> ''
),
unique_xa AS (
    SELECT request_id, action, min(xa) AS xa
    FROM src
    GROUP BY request_id, action
    HAVING count(DISTINCT xa) = 1
)
UPDATE admin_write_idempotency AS idem
SET xa_id = unique_xa.xa
FROM unique_xa
WHERE idem.request_id = unique_xa.request_id
  AND idem.action = unique_xa.action
  AND idem.xa_id IS NULL;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM admin_write_idempotency
        WHERE xa_id IS NULL OR btrim(xa_id) = ''
    ) THEN
        RAISE EXCEPTION 'admin_write_idempotency xa_id cannot be determined';
    END IF;
END $$;

ALTER TABLE admin_write_idempotency ALTER COLUMN xa_id SET NOT NULL;
ALTER TABLE admin_write_idempotency DROP CONSTRAINT admin_write_idempotency_pkey;
ALTER TABLE admin_write_idempotency ADD PRIMARY KEY (xa_id, request_id, action);

UPDATE procedure_version_documents AS link
SET index_status = CASE job.status
        WHEN 'SUCCEEDED' THEN 'READY'
        WHEN 'FAILED' THEN 'FAILED'
        WHEN 'CLAIMED' THEN 'PROCESSING'
        ELSE 'UPLOADED'
    END,
    last_error_code = CASE WHEN job.status = 'FAILED' THEN job.error_code ELSE NULL END,
    updated_at = now()
FROM (
    SELECT DISTINCT ON (document_id, procedure_version_id)
        document_id, procedure_version_id, status, error_code
    FROM document_index_jobs
    ORDER BY document_id, procedure_version_id, claimed_at DESC
) AS job
WHERE link.document_id = job.document_id
  AND link.procedure_version_id = job.procedure_version_id;

UPDATE documents AS doc
SET processing_status = CASE
        WHEN NOT EXISTS (
            SELECT 1 FROM procedure_version_documents AS link
            WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id AND link.index_status <> 'READY'
        ) THEN 'READY'
        WHEN EXISTS (
            SELECT 1 FROM procedure_version_documents AS link
            WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id AND link.index_status = 'PROCESSING'
        ) THEN 'PROCESSING'
        WHEN EXISTS (
            SELECT 1 FROM procedure_version_documents AS link
            WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id AND link.index_status = 'FAILED'
        ) THEN 'FAILED'
        ELSE 'UPLOADED'
    END,
    updated_at = now()
WHERE EXISTS (
    SELECT 1 FROM procedure_version_documents AS link
    WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id
);
