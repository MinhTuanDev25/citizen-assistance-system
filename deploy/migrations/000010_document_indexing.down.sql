DROP TABLE IF EXISTS admin_write_idempotency;

DROP INDEX IF EXISTS ux_document_index_jobs_one_claim;
DROP TABLE IF EXISTS document_index_jobs;

DROP INDEX IF EXISTS ix_pvd_document;

ALTER TABLE procedure_version_documents
    DROP CONSTRAINT IF EXISTS fk_pvd_document_xa_domain,
    DROP CONSTRAINT IF EXISTS fk_pvd_procedure_xa_domain,
    DROP CONSTRAINT IF EXISTS fk_pvd_version,
    DROP CONSTRAINT IF EXISTS pvd_page_range_check,
    DROP CONSTRAINT IF EXISTS pvd_relationship_type_check;

ALTER TABLE procedure_version_documents
    DROP COLUMN IF EXISTS created_at,
    DROP COLUMN IF EXISTS page_range,
    DROP COLUMN IF EXISTS relationship_type,
    DROP COLUMN IF EXISTS domain_id,
    DROP COLUMN IF EXISTS xa_id,
    DROP COLUMN IF EXISTS procedure_id;

ALTER TABLE procedures DROP CONSTRAINT IF EXISTS ux_procedures_id_xa_domain;

ALTER TABLE documents DROP CONSTRAINT IF EXISTS fk_documents_supersedes;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_not_self_supersede;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS ux_documents_id_xa_domain;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS ux_documents_id_xa;
ALTER TABLE documents DROP COLUMN IF EXISTS supersedes_document_id;

UPDATE documents SET processing_status = 'PROCESSED' WHERE processing_status = 'READY';

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_processing_status_check;
ALTER TABLE documents
    ADD CONSTRAINT documents_processing_status_check CHECK (
        processing_status IN ('UPLOADED', 'PROCESSING', 'PROCESSED', 'FAILED')
    );
