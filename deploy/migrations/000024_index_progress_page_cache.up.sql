-- 000024_index_progress_page_cache.up.sql
-- Background indexing for long PDFs.
-- 1) Per-job progress and a small report (e.g. pages rejected by the OCR quality
--    gate). Written by the AI worker while the job is CLAIMED, guarded by claim_token.
-- 2) Per-page extraction cache keyed by the PDF bytes (source_sha256) and the
--    extraction settings (extraction_key), so a retry resumes instead of
--    re-running OCR on pages already processed. Derived data: deleting a document
--    deletes its cache.

ALTER TABLE document_index_jobs
    ADD COLUMN progress_stage text,
    ADD COLUMN pages_total int,
    ADD COLUMN pages_done int,
    ADD COLUMN pages_ocr int,
    ADD COLUMN pages_rejected int,
    ADD COLUMN progress_updated_at timestamptz,
    ADD COLUMN report jsonb;

ALTER TABLE document_index_jobs ADD CONSTRAINT document_index_jobs_progress_check CHECK (
    (progress_stage IS NULL OR progress_stage IN ('queued', 'extract', 'chunk', 'embed', 'write'))
    AND (pages_total IS NULL OR pages_total >= 0)
    AND (pages_done IS NULL OR (pages_done >= 0 AND (pages_total IS NULL OR pages_done <= pages_total)))
    AND (pages_ocr IS NULL OR pages_ocr >= 0)
    AND (pages_rejected IS NULL OR pages_rejected >= 0)
);

CREATE TABLE document_page_extractions (
    document_id    uuid NOT NULL,
    xa_id          text NOT NULL,
    source_sha256  text NOT NULL,
    extraction_key text NOT NULL,
    page_number    int  NOT NULL,
    source         text NOT NULL,
    accepted       boolean NOT NULL,
    reason         text,
    text           text NOT NULL DEFAULT '',
    quality        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, source_sha256, extraction_key, page_number),
    CONSTRAINT fk_page_extractions_document FOREIGN KEY (document_id, xa_id)
        REFERENCES documents (id, xa_id) ON DELETE CASCADE,
    CONSTRAINT document_page_extractions_source_check CHECK (source IN ('native', 'ocr', 'blank')),
    CONSTRAINT document_page_extractions_page_check CHECK (page_number >= 1),
    CONSTRAINT document_page_extractions_reason_check CHECK (
        (accepted AND reason IS NULL) OR (NOT accepted AND reason IS NOT NULL AND char_length(reason) <= 64)
    )
);
