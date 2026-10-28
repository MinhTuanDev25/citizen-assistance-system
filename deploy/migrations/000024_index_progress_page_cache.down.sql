-- Reverse 000024. The page cache is derived data and can be rebuilt by indexing again.

DROP TABLE IF EXISTS document_page_extractions;

ALTER TABLE document_index_jobs DROP CONSTRAINT IF EXISTS document_index_jobs_progress_check;
ALTER TABLE document_index_jobs
    DROP COLUMN IF EXISTS report,
    DROP COLUMN IF EXISTS progress_updated_at,
    DROP COLUMN IF EXISTS pages_rejected,
    DROP COLUMN IF EXISTS pages_ocr,
    DROP COLUMN IF EXISTS pages_done,
    DROP COLUMN IF EXISTS pages_total,
    DROP COLUMN IF EXISTS progress_stage;
