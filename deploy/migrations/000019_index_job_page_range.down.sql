-- 000019_index_job_page_range.down.sql

ALTER TABLE document_index_jobs
    DROP COLUMN IF EXISTS page_range;
