-- Reverse 000011. In-flight claims are closed so the older one-claim-per-document
-- index can be restored.

UPDATE document_index_jobs
SET status = 'FAILED',
    error_code = 'migration_down',
    finished_at = now(),
    claim_expires_at = NULL
WHERE status = 'CLAIMED';

ALTER TABLE document_index_jobs DROP CONSTRAINT IF EXISTS fk_index_jobs_link;
ALTER TABLE document_index_jobs DROP CONSTRAINT IF EXISTS document_index_jobs_error_code_check;
ALTER TABLE document_index_jobs DROP CONSTRAINT IF EXISTS ux_document_index_jobs_request;
DROP INDEX IF EXISTS ux_document_index_jobs_one_claim;

ALTER TABLE document_index_jobs
    DROP COLUMN IF EXISTS claim_token,
    DROP COLUMN IF EXISTS procedure_version_id;

ALTER TABLE document_index_jobs
    ADD CONSTRAINT ux_document_index_jobs_request UNIQUE (document_id, request_id);

CREATE UNIQUE INDEX ux_document_index_jobs_one_claim
    ON document_index_jobs (document_id)
    WHERE status = 'CLAIMED';

ALTER TABLE admin_write_idempotency DROP CONSTRAINT admin_write_idempotency_pkey;
ALTER TABLE admin_write_idempotency DROP COLUMN IF EXISTS xa_id;
ALTER TABLE admin_write_idempotency ADD PRIMARY KEY (request_id, action);

ALTER TABLE procedure_version_documents DROP CONSTRAINT IF EXISTS pvd_index_status_check;
ALTER TABLE procedure_version_documents DROP CONSTRAINT IF EXISTS pvd_last_error_code_check;
ALTER TABLE procedure_version_documents DROP CONSTRAINT IF EXISTS pvd_page_range_check;

ALTER TABLE procedure_version_documents
    DROP COLUMN IF EXISTS updated_at,
    DROP COLUMN IF EXISTS last_error_code,
    DROP COLUMN IF EXISTS index_status;

ALTER TABLE procedure_version_documents
    ADD CONSTRAINT pvd_page_range_check CHECK (
        page_range IS NULL
        OR page_range ~ '^[1-9][0-9]{0,3}(-[1-9][0-9]{0,3})?$'
    );
