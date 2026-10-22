-- 000019_index_job_page_range.up.sql
-- Reindex range is claimed with the job. Empty string = whole PDF.
-- NULL = pre-migration job (worker still reads procedure_version_documents.page_range).

ALTER TABLE document_index_jobs
    ADD COLUMN IF NOT EXISTS page_range text;
