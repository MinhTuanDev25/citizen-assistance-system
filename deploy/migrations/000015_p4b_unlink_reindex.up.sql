-- 000015_p4b_unlink_reindex.up.sql
-- Soft-unlink keeps job, generation, and chunk history.
-- A generation must belong to the same job, link, and commune.

ALTER TABLE procedure_version_documents
    ADD COLUMN unlinked_at timestamptz,
    ADD COLUMN reindex_error_code text;

ALTER TABLE procedure_version_documents
    ADD CONSTRAINT pvd_reindex_error_code_check CHECK (
        reindex_error_code IS NULL OR char_length(reindex_error_code) BETWEEN 1 AND 64
    ),
    ADD CONSTRAINT pvd_unlinked_has_no_active_generation CHECK (
        unlinked_at IS NULL OR active_generation_id IS NULL
    );

ALTER TABLE document_index_jobs
    ADD CONSTRAINT ux_index_jobs_id_link UNIQUE (id, document_id, procedure_version_id, xa_id);

ALTER TABLE document_index_generations
    ADD CONSTRAINT fk_generations_job_link
        FOREIGN KEY (job_id, document_id, procedure_version_id, xa_id)
        REFERENCES document_index_jobs (id, document_id, procedure_version_id, xa_id)
        ON DELETE RESTRICT;

ALTER TABLE knowledge_chunks
    ADD CONSTRAINT fk_chunks_generation_link
        FOREIGN KEY (generation_id, document_id, procedure_version_id, xa_id)
        REFERENCES document_index_generations (id, document_id, procedure_version_id, xa_id)
        ON DELETE RESTRICT;
