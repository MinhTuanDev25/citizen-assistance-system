-- Reverse 000025. Drafts created by jobs stay in procedure_drafts; only the
-- job bookkeeping and per-batch intermediate results are removed.

DROP TABLE IF EXISTS procedure_draft_job_batches;
DROP TABLE IF EXISTS procedure_draft_jobs;
