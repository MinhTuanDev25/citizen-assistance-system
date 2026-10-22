ALTER TABLE knowledge_chunks DROP CONSTRAINT IF EXISTS fk_chunks_generation_link;
ALTER TABLE document_index_generations DROP CONSTRAINT IF EXISTS fk_generations_job_link;
ALTER TABLE document_index_jobs DROP CONSTRAINT IF EXISTS ux_index_jobs_id_link;

ALTER TABLE procedure_version_documents DROP CONSTRAINT IF EXISTS pvd_unlinked_has_no_active_generation;
ALTER TABLE procedure_version_documents DROP CONSTRAINT IF EXISTS pvd_reindex_error_code_check;
ALTER TABLE procedure_version_documents
    DROP COLUMN IF EXISTS reindex_error_code,
    DROP COLUMN IF EXISTS unlinked_at;
