-- 000025_procedure_draft_jobs.up.sql
-- Multi-batch AI drafting for long legal PDFs.
-- A job covers the selected chunks of ONE procedure section (never a whole
-- multi-procedure document). The chunks are split into batches that fit one
-- model call. Every batch result is stored before the next batch runs, so a
-- retry only re-runs failed batches and never overwrites a finished one.
-- The aggregated Definition is written as a NEW procedure_drafts row; existing
-- drafts are never updated by a job (SUPPLEMENT jobs reference their base draft).

CREATE TABLE procedure_draft_jobs (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    xa_id            text NOT NULL,
    procedure_id     uuid NOT NULL,
    document_id      uuid NOT NULL,
    generation_id    uuid NOT NULL REFERENCES document_index_generations (id) ON DELETE RESTRICT,
    mode             text NOT NULL,
    base_draft_id    uuid REFERENCES procedure_drafts (id) ON DELETE RESTRICT,
    request_id       uuid NOT NULL,
    request_hash     text NOT NULL,
    status           text NOT NULL DEFAULT 'QUEUED',
    plan             jsonb NOT NULL,
    batches_total    int NOT NULL,
    batches_done     int NOT NULL DEFAULT 0,
    batches_failed   int NOT NULL DEFAULT 0,
    draft_id         uuid REFERENCES procedure_drafts (id) ON DELETE RESTRICT,
    error_code       text,
    error_message    text,
    lease_owner      uuid,
    lease_expires_at timestamptz,
    created_by       uuid NOT NULL REFERENCES users (id) ON DELETE RESTRICT,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    started_at       timestamptz,
    finished_at      timestamptz,
    CONSTRAINT fk_draft_jobs_procedure FOREIGN KEY (procedure_id, xa_id)
        REFERENCES procedures (id, xa_id) ON DELETE RESTRICT,
    CONSTRAINT fk_draft_jobs_document FOREIGN KEY (document_id, xa_id)
        REFERENCES documents (id, xa_id) ON DELETE RESTRICT,
    CONSTRAINT ux_draft_jobs_request UNIQUE (xa_id, request_id),
    CONSTRAINT procedure_draft_jobs_mode_check CHECK (mode IN ('FULL', 'SUPPLEMENT')),
    CONSTRAINT procedure_draft_jobs_base_check CHECK ((mode = 'SUPPLEMENT') = (base_draft_id IS NOT NULL)),
    CONSTRAINT procedure_draft_jobs_status_check CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')),
    CONSTRAINT procedure_draft_jobs_counts_check CHECK (
        batches_total >= 1
        AND batches_done >= 0 AND batches_failed >= 0
        AND batches_done + batches_failed <= batches_total
    ),
    -- SUCCEEDED means every planned batch finished and the aggregate draft exists.
    CONSTRAINT procedure_draft_jobs_success_check CHECK (
        status <> 'SUCCEEDED' OR (draft_id IS NOT NULL AND batches_done = batches_total)
    )
);

CREATE INDEX ix_draft_jobs_procedure ON procedure_draft_jobs (procedure_id, created_at DESC);
CREATE INDEX ix_draft_jobs_runnable ON procedure_draft_jobs (status, lease_expires_at)
    WHERE status IN ('QUEUED', 'RUNNING');

CREATE TABLE procedure_draft_job_batches (
    job_id        uuid NOT NULL REFERENCES procedure_draft_jobs (id) ON DELETE CASCADE,
    batch_index   int NOT NULL,
    page_start    int NOT NULL,
    page_end      int NOT NULL,
    chunk_start   int NOT NULL,
    chunk_end     int NOT NULL,
    excerpt_ids   uuid[] NOT NULL,
    status        text NOT NULL DEFAULT 'PENDING',
    attempts      int NOT NULL DEFAULT 0,
    result        jsonb,
    error_code    text,
    error_message text,
    started_at    timestamptz,
    finished_at   timestamptz,
    PRIMARY KEY (job_id, batch_index),
    CONSTRAINT procedure_draft_job_batches_status_check CHECK (status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')),
    CONSTRAINT procedure_draft_job_batches_range_check CHECK (
        batch_index >= 0 AND page_start >= 1 AND page_end >= page_start
        AND chunk_start >= 0 AND chunk_end >= chunk_start
        AND cardinality(excerpt_ids) >= 1 AND attempts >= 0
    ),
    -- A finished batch keeps its model output; it is the job's intermediate result.
    CONSTRAINT procedure_draft_job_batches_result_check CHECK (status <> 'SUCCEEDED' OR result IS NOT NULL)
);
